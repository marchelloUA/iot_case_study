"""Bronze layer: raw ingestion with metadata and dead-letter routing.

Reads the source JSON array, stamps every element with ingestion metadata,
and writes a CSV where Attributes is stored as a JSON string. Records that
cannot be JSON-serialized (e.g. they contain datetime objects, circular
references, or non-serializable custom objects) are written to a separate
dead-letter CSV rather than crashing the run.

Azure equivalent:
  Source: Azure Event Hubs (Kafka endpoint) or ADLS Gen2 raw zone
  Sink:   ADLS Gen2 bronze container, Delta format (write.format("delta"))
  Dead-letter: separate Delta table "device_stream_demo_dead_letter" as shown
  in the Databricks notebook (IoT case study 150326_0002.ipynb, Cell 0).
  The _ingested_at column is equivalent to EventData.enqueued_time from
  Event Hubs. The _source_file column maps to the ADLS path or Event Hub
  partition/offset metadata.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

logger = logging.getLogger(__name__)

_IDENTITY_COLS = ("DeviceId", "TimestampUTC", "DeviceType", "Status")


class IngestResult(NamedTuple):
    """Counts returned by ingest() for observability."""
    rows_written: int
    dead_letter_count: int


def safe_serialize(record: Any) -> tuple[str | None, str | None]:
    """Try to JSON-serialize a record.

    Returns (json_str, None) on success or (None, error_json) on failure.
    The error_json is a JSON object with '_error' (the exception message) and
    '_raw' (the str() representation of the bad record), matching the schema
    used by the Databricks notebook's dead-letter table.
    """
    try:
        return json.dumps(record, ensure_ascii=False), None
    except (TypeError, ValueError) as exc:
        error_doc = json.dumps({"_error": str(exc), "_raw": str(record)}, ensure_ascii=False)
        return None, error_doc


def _record_to_row(
    record: Any, source_file: str, ingested_at: str
) -> dict[str, Any]:
    """Convert one raw JSON element to a bronze row dict.

    For non-dict elements (null, int, string) the identity columns are None
    and the raw element is JSON-serialized into Attributes. This preserves
    every element in the source array without silently losing data.
    """
    if isinstance(record, dict):
        row: dict[str, Any] = {col: record.get(col) for col in _IDENTITY_COLS}
        attrs = record.get("Attributes")
    else:
        row = {col: None for col in _IDENTITY_COLS}
        attrs = record

    # Always store Attributes as a JSON string for a uniform column type
    # across all device types (dicts for WM/DW, lists for coffee machines).
    row["Attributes"] = json.dumps(attrs, ensure_ascii=False) if attrs is not None else None
    row["_ingested_at"] = ingested_at
    row["_source_file"] = source_file
    return row


def ingest(input_path: Path, output_path: Path) -> IngestResult:
    """Read a JSON array from input_path, add metadata columns, write CSVs.

    Records that cannot be JSON-serialized are written to a dead-letter CSV
    at <output_stem>_dead_letter.csv alongside the main output. The
    dead-letter schema is (_raw_error, _ingested_at).

    Returns IngestResult(rows_written, dead_letter_count).

    Raises ValueError if the JSON root is not an array.
    Raises FileNotFoundError if input_path does not exist.
    """
    with input_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Expected a JSON array at the root of the input file")

    ingested_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    source_file = str(input_path.resolve())

    good_rows: list[dict[str, Any]] = []
    dead_rows: list[dict[str, Any]] = []

    for record in raw:
        _, err = safe_serialize(record)
        if err is not None:
            dead_rows.append({"_raw_error": err, "_ingested_at": ingested_at})
            logger.warning("Bronze: dead-letter record: %s", err[:120])
            continue
        good_rows.append(_record_to_row(record, source_file, ingested_at))

    col_order = list(_IDENTITY_COLS) + ["Attributes", "_ingested_at", "_source_file"]

    if good_rows:
        df = pd.DataFrame(good_rows, columns=col_order)
    else:
        df = pd.DataFrame(columns=col_order)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Bronze: wrote %d row(s) to %s", len(df), output_path)

    if dead_rows:
        dl_path = output_path.with_name(output_path.stem + "_dead_letter.csv")
        dl_df = pd.DataFrame(dead_rows, columns=["_raw_error", "_ingested_at"])
        dl_df.to_csv(dl_path, index=False, encoding="utf-8")
        logger.warning(
            "Bronze: %d dead-letter record(s) written to %s", len(dead_rows), dl_path
        )

    return IngestResult(rows_written=len(df), dead_letter_count=len(dead_rows))


def main(argv: list[str] | None = None) -> int:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Bronze layer: ingest raw IoT JSON events.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        result = ingest(args.input, args.output)
    except FileNotFoundError:
        print(f"ERROR: input file not found: {args.input}", flush=True)
        return 1
    except ValueError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    except OSError as exc:
        print(f"ERROR: could not write output: {exc}", flush=True)
        return 1

    print(f"Bronze: wrote {result.rows_written} row(s) to {args.output}")
    if result.dead_letter_count:
        print(f"Bronze: {result.dead_letter_count} dead-letter record(s) written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
