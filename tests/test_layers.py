"""Tests for the bronze, silver, gold, and run_pipeline modules."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.bronze import ingest as bronze_ingest
from src.gold import build_gold, _build_summary
from src.run_pipeline import run
from src.silver import cleanse as silver_cleanse, CleanseResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_EVENTS = [
    {
        "DeviceId": "aaaa000000000001",
        "TimestampUTC": "2023-01-01T07:00:00Z",
        "DeviceType": "Washing machine",
        "Status": "Program Start",
        "Attributes": {
            "Temperature": 60, "Temperature_Unit": "Celsius",
            "SpinningSpeed": 900, "SpinningSpeed_Unit": "Rotations per Minute",
        },
    },
    {
        "DeviceId": "aaaa000000000002",
        "TimestampUTC": "2023-01-01T08:00:00Z",
        "DeviceType": "Coffee Machine",
        "Status": "Program End",
        "Attributes": [{"Id": "Temperature", "Value": 86, "Unit": "Celsius"}],
    },
    {
        "DeviceId": "aaaa000000000003",
        "TimestampUTC": "2023-01-01T09:00:00Z",
        "DeviceType": "Dishwasher",
        "Status": "Program End",
        "Attributes": {
            "Temperature": 40, "Temperature_Unit": "Celsius",
            "QuickPowerWashActive": 1, "QuickPowerWashActive_Unit": "Bit",
        },
    },
    {
        "DeviceId": "aaaa000000000001",
        "TimestampUTC": "2023-01-01T07:30:00Z",
        "DeviceType": "Washing machine",
        "Status": "Program Failure",
        "Attributes": {"FailureIds": [10, 17]},
    },
]


def _write_events(tmp_path: Path, events: list, filename: str = "events.json") -> Path:
    p = tmp_path / filename
    p.write_text(json.dumps(events), encoding="utf-8")
    return p


def _make_bronze(tmp_path: Path, events: list | None = None) -> tuple[Path, Path]:
    """Write sample events and run bronze ingest. Returns (source, bronze_csv)."""
    events = events if events is not None else SAMPLE_EVENTS
    src = _write_events(tmp_path, events)
    out = tmp_path / "bronze.csv"
    bronze_ingest(src, out)
    return src, out


def _make_silver(tmp_path: Path, events: list | None = None) -> tuple[Path, Path]:
    """Run bronze then silver. Returns (bronze_csv, silver_csv)."""
    _, bronze_path = _make_bronze(tmp_path, events)
    out = tmp_path / "silver.csv"
    silver_cleanse(bronze_path, out)
    return bronze_path, out


# ---------------------------------------------------------------------------
# Bronze tests
# ---------------------------------------------------------------------------

def test_bronze_adds_metadata_columns(tmp_path: Path):
    """Every row has _ingested_at and _source_file columns."""
    _, bronze_path = _make_bronze(tmp_path)
    df = pd.read_csv(bronze_path)
    assert "_ingested_at" in df.columns
    assert "_source_file" in df.columns
    assert df["_ingested_at"].notna().all()
    assert df["_source_file"].notna().all()


def test_bronze_ingested_at_is_consistent_within_run(tmp_path: Path):
    """All rows from one ingest call share the same _ingested_at timestamp."""
    _, bronze_path = _make_bronze(tmp_path)
    df = pd.read_csv(bronze_path)
    assert df["_ingested_at"].nunique() == 1


def test_bronze_attributes_serialized_to_string(tmp_path: Path):
    """Attributes column is a non-empty string for all rows."""
    _, bronze_path = _make_bronze(tmp_path)
    df = pd.read_csv(bronze_path, dtype=str)
    for i, val in enumerate(df["Attributes"]):
        assert isinstance(val, str) and val.strip(), f"Row {i}: Attributes not a string: {val!r}"


def test_bronze_coffee_machine_attributes_is_json_list(tmp_path: Path):
    """Coffee machine Attributes round-trips as a list, not a dict."""
    _, bronze_path = _make_bronze(tmp_path)
    df = pd.read_csv(bronze_path, dtype=str)
    coffee_row = df[df["DeviceType"] == "Coffee Machine"].iloc[0]
    parsed = json.loads(coffee_row["Attributes"])
    assert isinstance(parsed, list)


def test_bronze_non_dict_element_stored_as_nan_identity_row(tmp_path: Path):
    """A null element in the array is stored as a row with NaN identity fields."""
    events: list = [None, SAMPLE_EVENTS[0]]
    _, bronze_path = _make_bronze(tmp_path, events)
    df = pd.read_csv(bronze_path, dtype=str)
    assert len(df) == 2
    # First row (null element) has NaN identity fields
    assert pd.isna(df.loc[0, "DeviceId"]) or df.loc[0, "DeviceId"] == "" or str(df.loc[0, "DeviceId"]) == "nan"


def test_bronze_empty_array_writes_header_only(tmp_path: Path):
    """An empty JSON array produces a CSV with headers but no data rows."""
    src = _write_events(tmp_path, [])
    out = tmp_path / "bronze.csv"
    n = bronze_ingest(src, out)
    assert n.rows_written == 0
    df = pd.read_csv(out)
    assert len(df) == 0
    assert "_ingested_at" in df.columns


def test_bronze_raises_for_non_list_json(tmp_path: Path):
    """A JSON object at the root raises ValueError."""
    src = tmp_path / "bad.json"
    src.write_text('{"not": "an array"}', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array"):
        bronze_ingest(src, tmp_path / "out.csv")


# ---------------------------------------------------------------------------
# Silver tests
# ---------------------------------------------------------------------------

def test_silver_returns_cleanse_result(tmp_path: Path):
    """cleanse() returns a CleanseResult NamedTuple."""
    _, bronze_path = _make_bronze(tmp_path)
    result = silver_cleanse(bronze_path, tmp_path / "silver.csv")
    assert isinstance(result, CleanseResult)
    assert result.rows_in == len(SAMPLE_EVENTS)
    assert result.rows_out == len(SAMPLE_EVENTS)
    assert result.dropped_invalid == 0
    assert result.dropped_duplicate == 0
    assert result.dropped_fuzzy == 0


def test_silver_drops_blank_required_field(tmp_path: Path):
    """A row with a blank DeviceId is dropped; skipped count is 1."""
    events = [
        {**SAMPLE_EVENTS[0], "DeviceId": "   "},
        SAMPLE_EVENTS[1],
    ]
    _, bronze_path = _make_bronze(tmp_path, events)
    result = silver_cleanse(bronze_path, tmp_path / "silver.csv")
    assert result.dropped_invalid == 1
    assert result.rows_out == 1


def test_silver_deduplicates_on_key(tmp_path: Path):
    """Two records with identical (DeviceId, TimestampUTC, DeviceType, Status) become one."""
    events = [SAMPLE_EVENTS[0], SAMPLE_EVENTS[0]]
    _, bronze_path = _make_bronze(tmp_path, events)
    result = silver_cleanse(bronze_path, tmp_path / "silver.csv")
    assert result.dropped_duplicate == 1
    assert result.rows_out == 1


def test_silver_deduplication_treats_z_and_no_z_as_same_event(tmp_path: Path):
    """'2023-01-01T07:00:00' and '2023-01-01T07:00:00Z' are the same event after normalization."""
    e1 = {**SAMPLE_EVENTS[0], "TimestampUTC": "2023-01-01T07:00:00"}
    e2 = {**SAMPLE_EVENTS[0], "TimestampUTC": "2023-01-01T07:00:00Z"}
    _, bronze_path = _make_bronze(tmp_path, [e1, e2])
    result = silver_cleanse(bronze_path, tmp_path / "silver.csv")
    assert result.dropped_duplicate == 1
    assert result.rows_out == 1


def test_silver_drops_unparseable_timestamp(tmp_path: Path):
    """A row with timestamp 'not-a-date' is dropped, not converted to NaT."""
    events = [
        {**SAMPLE_EVENTS[0], "TimestampUTC": "not-a-date"},
        SAMPLE_EVENTS[1],
    ]
    _, bronze_path = _make_bronze(tmp_path, events)
    result = silver_cleanse(bronze_path, tmp_path / "silver.csv")
    assert result.dropped_invalid == 1
    assert result.rows_out == 1


def test_silver_preserves_attributes_as_string(tmp_path: Path):
    """Attributes column passes through silver unchanged as a JSON string."""
    _, bronze_path = _make_bronze(tmp_path)
    out = tmp_path / "silver.csv"
    silver_cleanse(bronze_path, out)
    df = pd.read_csv(out, dtype=str)
    for i, val in enumerate(df["Attributes"]):
        assert isinstance(val, str), f"Row {i}: Attributes lost string type in silver"


def test_silver_empty_bronze_produces_empty_silver(tmp_path: Path):
    """An empty bronze CSV produces an empty silver CSV with zero rows."""
    _, bronze_path = _make_bronze(tmp_path, [])
    out = tmp_path / "silver.csv"
    result = silver_cleanse(bronze_path, out)
    assert result.rows_in == 0
    assert result.rows_out == 0
    df = pd.read_csv(out)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# Gold tests
# ---------------------------------------------------------------------------

def test_gold_produces_per_device_csvs(tmp_path: Path):
    """gold directory contains per-device CSVs and a summary."""
    _, silver_path = _make_silver(tmp_path)
    gold_dir = tmp_path / "gold"
    build_gold(silver_path, gold_dir)
    assert (gold_dir / "gold_washing_machine.csv").exists()
    assert (gold_dir / "gold_coffee_machine.csv").exists()
    assert (gold_dir / "gold_dishwasher.csv").exists()
    assert (gold_dir / "gold_summary.csv").exists()


def test_gold_washing_machine_has_temperature_c(tmp_path: Path):
    """Washing machine gold table has Temperature_C, not raw Temperature."""
    _, silver_path = _make_silver(tmp_path)
    gold_dir = tmp_path / "gold"
    build_gold(silver_path, gold_dir)
    wm_df = pd.read_csv(gold_dir / "gold_washing_machine.csv")
    assert "Temperature_C" in wm_df.columns
    assert "Temperature" not in wm_df.columns


def test_gold_coffee_machine_has_pivoted_columns(tmp_path: Path):
    """Coffee machine gold table has Temperature_Value and Temperature_Unit."""
    _, silver_path = _make_silver(tmp_path)
    gold_dir = tmp_path / "gold"
    build_gold(silver_path, gold_dir)
    cm_df = pd.read_csv(gold_dir / "gold_coffee_machine.csv")
    assert "Temperature_Value" in cm_df.columns
    assert "Temperature_Unit" in cm_df.columns


def test_gold_summary_has_correct_columns(tmp_path: Path):
    """Summary table has the five expected columns."""
    _, silver_path = _make_silver(tmp_path)
    gold_dir = tmp_path / "gold"
    build_gold(silver_path, gold_dir)
    summary = pd.read_csv(gold_dir / "gold_summary.csv")
    expected = {
        "DeviceId", "DeviceType", "total_events", "failure_events", "failure_rate_pct",
        "first_seen_utc", "last_seen_utc", "days_active",
    }
    assert expected.issubset(set(summary.columns))


def test_gold_summary_failure_rate_100_pct(tmp_path: Path):
    """A device with only failure events gets failure_rate_pct = 100.0."""
    events = [
        {**SAMPLE_EVENTS[0], "Status": "Program Failure", "Attributes": {"FailureIds": [1]}},
        {**SAMPLE_EVENTS[0], "TimestampUTC": "2023-01-01T08:00:00Z",
         "Status": "Program Failure", "Attributes": {"FailureIds": [2]}},
    ]
    _, silver_path = _make_silver(tmp_path, events)
    gold_dir = tmp_path / "gold"
    build_gold(silver_path, gold_dir)
    summary = pd.read_csv(gold_dir / "gold_summary.csv")
    row = summary[summary["DeviceId"] == events[0]["DeviceId"]].iloc[0]
    assert row["failure_rate_pct"] == pytest.approx(100.0)
    assert int(row["failure_events"]) == 2


def test_gold_summary_zero_failures(tmp_path: Path):
    """A device with no failure events gets failure_rate_pct = 0.0."""
    _, silver_path = _make_silver(tmp_path, [SAMPLE_EVENTS[0]])
    gold_dir = tmp_path / "gold"
    build_gold(silver_path, gold_dir)
    summary = pd.read_csv(gold_dir / "gold_summary.csv")
    row = summary.iloc[0]
    assert row["failure_rate_pct"] == pytest.approx(0.0)
    assert int(row["failure_events"]) == 0


def test_gold_unknown_device_type_uses_fallback(tmp_path: Path):
    """An unknown DeviceType row is included in the gold output with Attributes as JSON."""
    events = [{
        "DeviceId": "zzzz000000000099",
        "TimestampUTC": "2023-01-01T10:00:00Z",
        "DeviceType": "Refrigerator",
        "Status": "Door Open",
        "Attributes": {"DoorSensor": 1},
    }]
    _, silver_path = _make_silver(tmp_path, events)
    gold_dir = tmp_path / "gold"
    result = build_gold(silver_path, gold_dir)
    assert "Refrigerator" in result.unknown_device_types
    assert (gold_dir / "gold_refrigerator.csv").exists()


def test_gold_empty_silver_produces_empty_outputs(tmp_path: Path):
    """Empty silver produces an empty summary and no per-device CSVs."""
    _, silver_path = _make_silver(tmp_path, [])
    gold_dir = tmp_path / "gold"
    result = build_gold(silver_path, gold_dir)
    assert result.rows_by_device_type == {}
    summary = pd.read_csv(gold_dir / "gold_summary.csv")
    assert len(summary) == 0


# ---------------------------------------------------------------------------
# run_pipeline integration test
# ---------------------------------------------------------------------------

def test_run_pipeline_end_to_end(tmp_path: Path):
    """run() with sample_input.json produces all expected output directories and files."""
    src = _write_events(tmp_path, SAMPLE_EVENTS)
    out_dir = tmp_path / "output"
    rc = run(src, out_dir)
    assert rc == 0
    assert (out_dir / "bronze" / "events.csv").exists()
    assert (out_dir / "silver" / "events.csv").exists()
    assert (out_dir / "gold" / "gold_summary.csv").exists()
    assert (out_dir / "gold" / "gold_washing_machine.csv").exists()

    summary = pd.read_csv(out_dir / "gold" / "gold_summary.csv")
    assert len(summary) == len({e["DeviceId"] for e in SAMPLE_EVENTS})


def test_run_pipeline_missing_file_returns_1(tmp_path: Path):
    """run() returns 1 when the input file does not exist."""
    rc = run(tmp_path / "nonexistent.json", tmp_path / "out")
    assert rc == 1
