"""Tests for the fuzzy dedup pass in src/silver.py.

Covers:
  - _drop_fuzzy_duplicates: near-dup detection and dropping
  - CleanseResult.dropped_fuzzy field
  - Full cleanse() integration: dropped_fuzzy, row counts, _silver_processed_at column

Tests are written against the internal helper function so each behaviour
can be verified in isolation without needing a full bronze CSV on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("recordlinkage", reason="recordlinkage not installed; install with pip install -e .[ml]")
pytest.importorskip("rapidfuzz", reason="rapidfuzz not installed; install with pip install -e .[ml]")

from src.silver import CleanseResult, _drop_fuzzy_duplicates, FUZZY_SIMILARITY_THRESHOLD, cleanse as silver_cleanse
from src.bronze import ingest as bronze_ingest
from src.generate_synthetic import generate_events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal silver-ready DataFrame with the DEDUP_KEY columns."""
    base = {
        "DeviceId":     "syn_aaabbbccc0001",
        "TimestampUTC": "2024-01-01T00:00:00Z",
        "DeviceType":   "Washing machine",
        "Status":       "Program End",
        "Attributes":   "{}",
        "_ingested_at": "2024-01-01T00:00:00Z",
        "_source_file": "test",
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def _write_json(path: Path, data: list) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. _drop_fuzzy_duplicates: near-duplicate detection
# ---------------------------------------------------------------------------

def test_fuzzy_dup_same_device_same_ts_similar_status_dropped():
    """Two rows with same DeviceId, same DeviceType, same TimestampUTC, near-identical Status are dropped."""
    df = pd.DataFrame([
        {
            "DeviceId": "syn_aaabbbccc0001", "TimestampUTC": "2024-01-01T10:00:00Z",
            "DeviceType": "Washing machine", "Status": "Program Start",
            "Attributes": "{}", "_ingested_at": "t", "_source_file": "t",
        },
        {
            "DeviceId": "syn_aaabbbccc0001", "TimestampUTC": "2024-01-01T10:00:00Z",
            "DeviceType": "Washing machine", "Status": "program start",
            "Attributes": "{}", "_ingested_at": "t", "_source_file": "t",
        },
    ])
    clean_df, n_dropped = _drop_fuzzy_duplicates(df)
    assert n_dropped == 1
    assert len(clean_df) == 1


def test_fuzzy_dup_return_value_is_tuple():
    """_drop_fuzzy_duplicates returns a (DataFrame, int) tuple."""
    df = _make_df([{}])
    result = _drop_fuzzy_duplicates(df)
    assert isinstance(result, tuple)
    assert len(result) == 2
    clean_df, n_dropped = result
    assert isinstance(clean_df, pd.DataFrame)
    assert isinstance(n_dropped, int)


def test_fuzzy_dup_dropped_count_is_correct():
    """n_dropped equals the number of rows removed from the DataFrame."""
    df = pd.DataFrame([
        {
            "DeviceId": "syn_aaabbbccc0001", "TimestampUTC": "2024-01-01T10:00:00Z",
            "DeviceType": "Washing machine", "Status": "Program Start",
            "Attributes": "{}", "_ingested_at": "t", "_source_file": "t",
        },
        {
            "DeviceId": "syn_aaabbbccc0001", "TimestampUTC": "2024-01-01T10:00:00Z",
            "DeviceType": "Washing machine", "Status": "program start",
            "Attributes": "{}", "_ingested_at": "t", "_source_file": "t",
        },
    ])
    original_len = len(df)
    clean_df, n_dropped = _drop_fuzzy_duplicates(df)
    assert len(clean_df) == original_len - n_dropped


def test_fuzzy_dup_different_device_ids_not_paired():
    """Events from different DeviceIds are not compared (blocked on DeviceId)."""
    df = pd.DataFrame([
        {
            "DeviceId": "syn_aaabbbccc0001", "TimestampUTC": "2024-01-01T10:00:00Z",
            "DeviceType": "Washing machine", "Status": "Program Start",
            "Attributes": "{}", "_ingested_at": "t", "_source_file": "t",
        },
        {
            "DeviceId": "syn_zzz9998887770", "TimestampUTC": "2024-01-01T10:00:00Z",
            "DeviceType": "Washing machine", "Status": "Program Start",
            "Attributes": "{}", "_ingested_at": "t", "_source_file": "t",
        },
    ])
    clean_df, n_dropped = _drop_fuzzy_duplicates(df)
    assert n_dropped == 0
    assert len(clean_df) == 2


def test_fuzzy_dup_same_device_very_different_status_not_dropped():
    """Same DeviceId and TimestampUTC but very different Status strings: not a fuzzy dup."""
    df = pd.DataFrame([
        {
            "DeviceId": "syn_aaabbbccc0001", "TimestampUTC": "2024-01-01T10:00:00Z",
            "DeviceType": "Washing machine", "Status": "Program Start",
            "Attributes": "{}", "_ingested_at": "t", "_source_file": "t",
        },
        {
            "DeviceId": "syn_aaabbbccc0001", "TimestampUTC": "2024-01-01T10:00:00Z",
            "DeviceType": "Washing machine", "Status": "XXXXXXXXXXXXXXXXXXX",
            "Attributes": "{}", "_ingested_at": "t", "_source_file": "t",
        },
    ])
    clean_df, n_dropped = _drop_fuzzy_duplicates(df)
    assert n_dropped == 0
    assert len(clean_df) == 2


def test_fuzzy_dup_single_row_returns_unchanged():
    """A single-row DataFrame is returned unchanged with n_dropped == 0."""
    df = _make_df([{}])
    clean_df, n_dropped = _drop_fuzzy_duplicates(df)
    assert n_dropped == 0
    assert len(clean_df) == 1


def test_fuzzy_dup_empty_df_returns_unchanged():
    """An empty DataFrame is returned unchanged with n_dropped == 0."""
    df = _make_df([])
    clean_df, n_dropped = _drop_fuzzy_duplicates(df)
    assert n_dropped == 0
    assert len(clean_df) == 0


def test_fuzzy_dup_different_device_type_exact_match_required():
    """Same DeviceId and TimestampUTC but different DeviceType: not a fuzzy dup (exact DeviceType required)."""
    df = pd.DataFrame([
        {
            "DeviceId": "syn_aaabbbccc0001", "TimestampUTC": "2024-01-01T10:00:00Z",
            "DeviceType": "Washing machine", "Status": "Program Start",
            "Attributes": "{}", "_ingested_at": "t", "_source_file": "t",
        },
        {
            "DeviceId": "syn_aaabbbccc0001", "TimestampUTC": "2024-01-01T10:00:00Z",
            "DeviceType": "Dishwasher", "Status": "program start",
            "Attributes": "{}", "_ingested_at": "t", "_source_file": "t",
        },
    ])
    clean_df, n_dropped = _drop_fuzzy_duplicates(df)
    assert n_dropped == 0


# ---------------------------------------------------------------------------
# 2. CleanseResult has dropped_fuzzy field
# ---------------------------------------------------------------------------

def test_cleanse_result_has_dropped_fuzzy_field():
    """CleanseResult has a dropped_fuzzy attribute."""
    r = CleanseResult(rows_in=10, rows_out=10, dropped_invalid=0, dropped_duplicate=0)
    assert hasattr(r, "dropped_fuzzy")


def test_cleanse_result_dropped_fuzzy_defaults_to_zero():
    """dropped_fuzzy defaults to 0 so existing callers are not broken."""
    r = CleanseResult(rows_in=5, rows_out=5, dropped_invalid=0, dropped_duplicate=0)
    assert r.dropped_fuzzy == 0


def test_cleanse_result_does_not_have_old_ml_fields():
    """CleanseResult no longer has flagged_anomaly, flagged_fuzzy_dup, or imputed_temperature."""
    r = CleanseResult(rows_in=5, rows_out=5, dropped_invalid=0, dropped_duplicate=0)
    assert not hasattr(r, "flagged_anomaly")
    assert not hasattr(r, "flagged_fuzzy_dup")
    assert not hasattr(r, "imputed_temperature")


def test_fuzzy_similarity_threshold_constant_exists():
    """FUZZY_SIMILARITY_THRESHOLD is exported from silver and equals 90."""
    assert FUZZY_SIMILARITY_THRESHOLD == 90


# ---------------------------------------------------------------------------
# 3. Full cleanse() integration with recordlinkage
# ---------------------------------------------------------------------------

def test_cleanse_dropped_fuzzy_non_negative(tmp_path: Path):
    """cleanse() result.dropped_fuzzy is non-negative for any input."""
    events = generate_events(100, seed=9001)
    src = _write_json(tmp_path / "ev.json", events)
    bronze_csv = tmp_path / "b.csv"
    bronze_ingest(src, bronze_csv)
    result = silver_cleanse(bronze_csv, tmp_path / "s.csv")
    assert result.dropped_fuzzy >= 0


def test_cleanse_rows_out_le_rows_in(tmp_path: Path):
    """cleanse() never adds rows: rows_out <= rows_in."""
    events = generate_events(100, seed=9002)
    src = _write_json(tmp_path / "ev.json", events)
    bronze_csv = tmp_path / "b.csv"
    bronze_ingest(src, bronze_csv)
    result = silver_cleanse(bronze_csv, tmp_path / "s.csv")
    assert result.rows_out <= result.rows_in


def test_cleanse_silver_processed_at_column_present(tmp_path: Path):
    """Silver CSV output includes the _silver_processed_at audit column."""
    events = generate_events(50, seed=9003)
    src = _write_json(tmp_path / "ev.json", events)
    bronze_csv = tmp_path / "b.csv"
    silver_csv = tmp_path / "s.csv"
    bronze_ingest(src, bronze_csv)
    silver_cleanse(bronze_csv, silver_csv)
    df = pd.read_csv(silver_csv)
    assert "_silver_processed_at" in df.columns, "_silver_processed_at column missing from silver output"


def test_cleanse_synthetic_data_dropped_fuzzy_is_zero(tmp_path: Path):
    """Synthetic data has unique DeviceIds per event; no fuzzy dups expected."""
    events = generate_events(200, seed=9004)
    src = _write_json(tmp_path / "ev.json", events)
    bronze_csv = tmp_path / "b.csv"
    bronze_ingest(src, bronze_csv)
    result = silver_cleanse(bronze_csv, tmp_path / "s.csv")
    assert result.dropped_fuzzy == 0
