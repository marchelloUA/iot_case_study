"""Medallion pipeline runner: bronze -> silver -> gold.

Orchestrates the full three-layer pipeline for a single source JSON file.

Output structure:
  <output-dir>/bronze/<input-stem>.csv   raw records + ingestion metadata
  <output-dir>/silver/<input-stem>.csv   validated, deduplicated, normalised
  <output-dir>/gold/gold_<device>.csv    per-device flattened tables
  <output-dir>/gold/gold_summary.csv     event counts + failure rates

Usage:
  python -m src.run_pipeline --input data/sample_input.json --output-dir output
  # or after pip install -e .:
  iot-pipeline-layers --input data/sample_input.json --output-dir output
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src import bronze, silver, gold

logger = logging.getLogger(__name__)


def run(input_path: Path, output_dir: Path) -> int:
    """Execute bronze -> silver -> gold for input_path.

    Returns 0 on success, 1 on any error.
    Layer outputs land under output_dir/bronze/, output_dir/silver/,
    and output_dir/gold/.
    """
    stem = input_path.stem
    bronze_path = output_dir / "bronze" / f"{stem}.csv"
    silver_path = output_dir / "silver" / f"{stem}.csv"
    gold_dir = output_dir / "gold"

    # Bronze
    try:
        n_bronze = bronze.ingest(input_path, bronze_path)
    except FileNotFoundError:
        print(f"ERROR: input file not found: {input_path}", flush=True)
        return 1
    except (ValueError, OSError) as exc:
        print(f"ERROR: bronze layer failed: {exc}", flush=True)
        return 1
    print(f"Bronze: {n_bronze} row(s) -> {bronze_path}")

    # Silver
    try:
        result = silver.cleanse(bronze_path, silver_path)
    except (FileNotFoundError, OSError) as exc:
        print(f"ERROR: silver layer failed: {exc}", flush=True)
        return 1
    print(
        f"Silver: {result.rows_in} in -> {result.rows_out} out "
        f"({result.dropped_invalid} invalid, {result.dropped_duplicate} duplicate dropped) "
        f"-> {silver_path}"
    )

    if result.rows_out == 0:
        logger.warning(
            "Silver produced 0 rows; gold output will be empty. "
            "Check for data quality issues in the source file."
        )

    # Gold
    try:
        gold_result = gold.build_gold(silver_path, gold_dir)
    except (FileNotFoundError, OSError) as exc:
        print(f"ERROR: gold layer failed: {exc}", flush=True)
        return 1

    for dt, count in gold_result.rows_by_device_type.items():
        print(f"Gold:   {dt}: {count} row(s)")
    if gold_result.unknown_device_types:
        print(f"Gold:   unknown device types (fallback): {gold_result.unknown_device_types}")
    print(f"Gold:   summary -> {gold_dir / 'gold_summary.csv'}")

    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="iot-pipeline-layers",
        description=(
            "Run the full medallion pipeline (bronze -> silver -> gold) "
            "for an IoT event JSON file."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the source JSON file (array of device events)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Base directory for layer outputs (bronze/, silver/, gold/ are created here)",
    )
    args = parser.parse_args(argv)

    return run(args.input, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
