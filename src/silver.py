"""Silver layer: cleansing, deduplication, and dead-letter routing.

Reads bronze data and applies data quality rules in two passes:

Structural pass (always runs):
  - Required fields (DeviceId, TimestampUTC, DeviceType, Status) must be
    present, non-blank strings. Rows that fail are written to a dead-letter
    CSV and excluded from the silver output.
  - TimestampUTC is normalized to a canonical UTC ISO-8601 string so that
    "2023-01-01T07:11:29" and "2023-01-01T07:11:29Z" are treated as the same
    event during deduplication.
  - Exact duplicates on (DeviceId, TimestampUTC, DeviceType, Status) are
    dropped; the first occurrence is kept.

Fuzzy dedup pass (runs when optional ml libraries are installed):
  - Uses recordlinkage to block candidate pairs on DeviceId, then compares
    DeviceType and TimestampUTC exactly and Status with RapidFuzz
    token_sort_ratio. Pairs where all exact fields match and Status similarity
    is above FUZZY_SIMILARITY_THRESHOLD are considered duplicates; the later
    record in each pair is dropped.
  - This mirrors the approach in the Databricks notebook
    (IoT case study 150326_0002.ipynb, Cell 2).

All silver rows receive a _silver_processed_at audit column (UTC timestamp,
same value for all rows in one cleanse() call).

Azure equivalent:
  Source: ADLS Gen2 bronze / Delta bronze
  Sink:   ADLS Gen2 silver container, Delta format
  Dead-letter: separate Delta table "device_stream_silver_dead_letter"
  On Databricks: exact dedup uses window row_number(); fuzzy dedup uses
  recordlinkage toPandas() + RapidFuzz on the driver, then joins back.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional ML imports -- install with:  pip install -e .[ml]
# ---------------------------------------------------------------------------
try:
    import recordlinkage as _rl
    _RECORDLINKAGE_AVAILABLE = True
except ImportError:
    _RECORDLINKAGE_AVAILABLE = False

try:
    from rapidfuzz import fuzz as _fuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_REQUIRED_FIELDS = ("DeviceId", "TimestampUTC", "DeviceType", "Status")
_DEDUP_KEY = list(_REQUIRED_FIELDS)
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# RapidFuzz token_sort_ratio threshold for treating two Status strings as
# equivalent (e.g. "Program start" vs "Program Start"). 90 gives a tight
# match while tolerating minor casing or whitespace differences.
FUZZY_SIMILARITY_THRESHOLD = 90


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class CleanseResult(NamedTuple):
    """Counts returned by cleanse() for observability and alerting."""
    rows_in: int
    rows_out: int
    dropped_invalid: int
    dropped_duplicate: int
    dropped_fuzzy: int = 0      # recordlinkage + RapidFuzz: near-duplicate rows dropped


# ---------------------------------------------------------------------------
# Structural cleansing (always runs)
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (valid_df, invalid_df).

    A row is invalid when any required field is absent (NaN), not a string,
    or a whitespace-only string.
    """
    valid = pd.Series(True, index=df.index)
    for field in _REQUIRED_FIELDS:
        if field not in df.columns:
            logger.error(
                "Silver: required column %r is missing from the bronze data; "
                "all rows will be dropped",
                field,
            )
            return df.iloc[0:0].copy(), df.copy()
        valid &= (
            df[field].notna()
            & df[field].astype(str).str.strip().ne("")
        )
    n_invalid = int((~valid).sum())
    if n_invalid:
        logger.warning(
            "Silver: %d row(s) with missing/blank required fields -> dead-letter", n_invalid
        )
    return df[valid].copy(), df[~valid].copy()


def _normalize_timestamps(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Parse TimestampUTC to UTC and reformat as a canonical ISO-8601 string.

    Rows where the timestamp cannot be parsed are dropped (silver is strict).
    Returns (updated_df, n_dropped).
    """
    parsed = pd.to_datetime(df["TimestampUTC"], format="ISO8601", utc=True, errors="coerce")
    bad_mask = parsed.isna()
    n_bad = int(bad_mask.sum())
    if n_bad:
        logger.warning("Silver: dropped %d row(s) with unparseable timestamp(s)", n_bad)
    df = df[~bad_mask].copy()
    parsed = parsed[~bad_mask]
    df["TimestampUTC"] = parsed.dt.strftime(_TS_FORMAT)
    return df, n_bad


def _deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop duplicate rows on the four identity fields, keeping the first.

    Returns (deduped_df, n_dropped).
    """
    before = len(df)
    df = df.drop_duplicates(subset=_DEDUP_KEY, keep="first")
    n_dropped = before - len(df)
    if n_dropped:
        logger.info("Silver: removed %d exact duplicate event(s)", n_dropped)
    return df, n_dropped


# ---------------------------------------------------------------------------
# Fuzzy dedup pass (runs when recordlinkage + rapidfuzz are installed)
# ---------------------------------------------------------------------------

def _drop_fuzzy_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop near-duplicate events using recordlinkage + RapidFuzz.

    Approach (mirrors the Databricks notebook Cell 2):
      1. recordlinkage.Index().block("DeviceId") generates candidate pairs
         only within the same device, avoiding an O(n^2) full cross-join.
      2. recordlinkage.Compare() checks DeviceType and TimestampUTC exactly.
      3. fuzz.token_sort_ratio() scores Status similarity for each candidate
         pair. Catches "Program start" vs "Program Start" casing variants.
      4. Pairs where all exact fields match AND status_fuzzy >= threshold are
         duplicates. The record with the higher pandas index (arrived later)
         is dropped.

    recordlinkage is designed for record linkage across datasets, but its
    blocking and comparison infrastructure works equally well for intra-dataset
    near-duplicate detection at the scale of a single bronze batch.

    Skips silently when either library is not installed.
    Returns (clean_df, n_dropped).
    """
    if not _RECORDLINKAGE_AVAILABLE or not _RAPIDFUZZ_AVAILABLE:
        if not _RECORDLINKAGE_AVAILABLE:
            logger.debug("Silver: recordlinkage not installed; skipping fuzzy dedup")
        if not _RAPIDFUZZ_AVAILABLE:
            logger.debug("Silver: rapidfuzz not installed; skipping fuzzy dedup")
        return df, 0

    pdf = df[list(_DEDUP_KEY)].copy().reset_index(drop=False)
    # Keep original index in a column so we can map back after filtering
    orig_idx_col = "__orig_idx__"
    pdf = pdf.rename(columns={"index": orig_idx_col})
    pdf_work = pdf.drop(columns=[orig_idx_col])

    indexer = _rl.Index()
    indexer.block("DeviceId")
    candidate_pairs = indexer.index(pdf_work)

    if len(candidate_pairs) == 0:
        return df, 0

    compare = _rl.Compare()
    compare.exact("DeviceId",    "DeviceId",    label="DeviceId_exact")
    compare.exact("DeviceType",  "DeviceType",  label="DeviceType_exact")
    compare.exact("TimestampUTC","TimestampUTC", label="timestamp_exact")
    features = compare.compute(candidate_pairs, pdf_work)

    fuzzy_scores = [
        _fuzz.token_sort_ratio(
            str(pdf_work.loc[a, "Status"]),
            str(pdf_work.loc[b, "Status"]),
        )
        for a, b in candidate_pairs
    ]
    features["status_fuzzy"] = fuzzy_scores

    dup_mask = (
        (features["DeviceId_exact"]   == 1)
        & (features["DeviceType_exact"] == 1)
        & (features["status_fuzzy"]    >= FUZZY_SIMILARITY_THRESHOLD)
    )
    dup_pairs = features[dup_mask]

    indices_to_drop: set[int] = set()
    for a, b in dup_pairs.index:
        # Drop the later record (higher positional index = arrived later)
        indices_to_drop.add(max(a, b))

    if not indices_to_drop:
        return df, 0

    # Map positional indices back to the original DataFrame index
    orig_indices_to_drop = pdf.loc[list(indices_to_drop), orig_idx_col].tolist()
    clean_df = df.drop(index=orig_indices_to_drop)
    n_dropped = len(orig_indices_to_drop)
    logger.warning("Silver: recordlinkage/RapidFuzz dropped %d fuzzy duplicate(s)", n_dropped)
    return clean_df, n_dropped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cleanse(bronze_path: Path, output_path: Path) -> CleanseResult:
    """Read a bronze CSV, apply quality rules, and write a silver CSV.

    Runs structural cleansing (validate, normalize, deduplicate) followed by
    fuzzy dedup (recordlinkage + RapidFuzz) when the [ml] optional
    dependencies are installed.

    Invalid rows (missing/blank required fields) are written to a dead-letter
    CSV at <output_stem>_dead_letter.csv.

    All output rows carry a _silver_processed_at column (same UTC timestamp
    for every row in one call), matching the Databricks notebook audit column.

    Returns a CleanseResult with row counts for observability. Callers can
    check dropped_invalid + dropped_duplicate + dropped_fuzzy against a
    threshold and alert when the drop rate is unexpectedly high.

    Raises FileNotFoundError if bronze_path does not exist.
    """
    df = pd.read_csv(bronze_path, encoding="utf-8", dtype=str)
    rows_in = len(df)

    silver_processed_at = datetime.now(tz=timezone.utc).strftime(_TS_FORMAT)

    # Structural pass
    df, invalid_df = _validate(df)
    df, n_bad_ts = _normalize_timestamps(df)
    df, n_dupes = _deduplicate(df)

    # Write dead-letter rows (invalid required fields + unparseable timestamps)
    n_invalid = len(invalid_df) + n_bad_ts
    if not invalid_df.empty:
        dl_path = output_path.with_name(output_path.stem + "_dead_letter.csv")
        invalid_df["_rejection_reason"] = "missing_required_field"
        invalid_df["_silver_processed_at"] = silver_processed_at
        dl_path.parent.mkdir(parents=True, exist_ok=True)
        invalid_df.to_csv(dl_path, index=False, encoding="utf-8")
        logger.warning(
            "Silver: %d invalid row(s) written to dead-letter: %s", len(invalid_df), dl_path
        )

    # Fuzzy dedup pass (optional)
    df, n_fuzzy = _drop_fuzzy_duplicates(df)

    # Add audit column
    if not df.empty:
        df["_silver_processed_at"] = silver_processed_at
        df = df.sort_values(["DeviceId", "TimestampUTC"], na_position="last")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Silver: wrote %d row(s) to %s", len(df), output_path)

    return CleanseResult(
        rows_in=rows_in,
        rows_out=len(df),
        dropped_invalid=n_invalid,
        dropped_duplicate=n_dupes,
        dropped_fuzzy=n_fuzzy,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Silver layer: cleanse and deduplicate IoT events.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        result = cleanse(args.input, args.output)
    except FileNotFoundError:
        print(f"ERROR: input file not found: {args.input}", flush=True)
        return 1
    except OSError as exc:
        print(f"ERROR: could not write output: {exc}", flush=True)
        return 1

    print(
        f"Silver: {result.rows_in} in -> {result.rows_out} out "
        f"({result.dropped_invalid} invalid, {result.dropped_duplicate} exact-dup, "
        f"{result.dropped_fuzzy} fuzzy-dup dropped)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
