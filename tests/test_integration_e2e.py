"""End-to-end integration tests: synthetic data -> bronze -> silver -> gold.

Each test section covers one stage of the full pipeline and asserts on real data
values, not just file existence. The overall flow tested is:

  generate_events()
    -> bronze.ingest()  (raw + metadata)
      -> silver.cleanse()  (validate, deduplicate, normalize)
        -> gold.build_gold()  (flatten per device type)

No hand-crafted fixture events are used here. Every test starts from
generated synthetic data so the pipeline is exercised with realistic volume
and variety.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.bronze import ingest as bronze_ingest
from src.generate_synthetic import generate_events
from src.gold import build_gold, GoldResult
from src.run_pipeline import run
from src.silver import cleanse as silver_cleanse, CleanseResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: list) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _run_layers(
    tmp_path: Path, events: list, stem: str = "events"
) -> tuple[Path, Path, Path, Path]:
    """Write events to JSON and run all three layers.

    Returns (input_json, bronze_csv, silver_csv, gold_dir).
    """
    src = _write_json(tmp_path / f"{stem}.json", events)
    bronze_csv = tmp_path / f"{stem}_bronze.csv"
    silver_csv = tmp_path / f"{stem}_silver.csv"
    gold_dir   = tmp_path / f"{stem}_gold"

    bronze_ingest(src, bronze_csv)
    silver_cleanse(bronze_csv, silver_csv)
    build_gold(silver_csv, gold_dir)

    return src, bronze_csv, silver_csv, gold_dir


# ---------------------------------------------------------------------------
# 1. Synthetic data generation
# ---------------------------------------------------------------------------

def test_generate_events_count(tmp_path: Path):
    """generate_events() returns exactly the requested number of events."""
    events = generate_events(200, seed=1)
    assert len(events) == 200


def test_generate_events_all_have_required_fields():
    """Every generated event has the four identity fields plus Attributes."""
    required = {"DeviceId", "TimestampUTC", "DeviceType", "Status", "Attributes"}
    events = generate_events(300, seed=2)
    for i, e in enumerate(events):
        missing = required - set(e.keys())
        assert not missing, f"Event {i} missing fields: {missing}"


def test_generate_events_device_types_are_known():
    """Only the three supported device types are generated."""
    known = {"Washing machine", "Coffee Machine", "Dishwasher"}
    events = generate_events(300, seed=3)
    found = {e["DeviceType"] for e in events}
    assert found.issubset(known), f"Unexpected device types: {found - known}"


def test_generate_events_all_three_device_types_present():
    """With enough events all three device types appear (weights ensure this)."""
    events = generate_events(300, seed=4)
    found = {e["DeviceType"] for e in events}
    assert found == {"Washing machine", "Coffee Machine", "Dishwasher"}


def test_generate_events_timestamps_are_iso8601_with_z():
    """All generated timestamps end with Z and contain T."""
    events = generate_events(100, seed=5)
    for e in events:
        ts = e["TimestampUTC"]
        assert "T" in ts and ts.endswith("Z"), f"Bad timestamp format: {ts!r}"



def test_generate_events_seed_is_reproducible():
    """The same seed on the same day produces identical output."""
    a = generate_events(50, seed=42)
    b = generate_events(50, seed=42)
    assert a == b


def test_generate_events_different_seeds_differ():
    """Different seeds produce different event lists."""
    a = generate_events(50, seed=1)
    b = generate_events(50, seed=2)
    assert a != b


def test_generate_events_wm_temperatures_are_in_allowed_set():
    """Washing machine Temperature values are in range (30-95 C or 86-203 F)."""
    events = generate_events(500, seed=7)
    wm_events = [e for e in events if e["DeviceType"] == "Washing machine"
                 and isinstance(e.get("Attributes"), dict)
                 and "Temperature" in e["Attributes"]]
    for e in wm_events:
        t = e["Attributes"]["Temperature"]
        unit = e["Attributes"].get("Temperature_Unit", "Celsius")
        if unit == "Celsius":
            assert 30 <= t <= 95, f"WM Celsius temperature out of range [30, 95]: {t}"
        else:
            assert 86 <= t <= 203, f"WM Fahrenheit temperature out of range [86, 203]: {t}"


# ---------------------------------------------------------------------------
# 2. Bronze layer -- synthetic data
# ---------------------------------------------------------------------------

def test_bronze_row_count_matches_input(tmp_path: Path):
    """bronze_ingest() writes exactly as many rows as there are input events."""
    events = generate_events(200, seed=20)
    src = _write_json(tmp_path / "events.json", events)
    out = tmp_path / "bronze.csv"
    n = bronze_ingest(src, out)
    assert n.rows_written == 200
    df = pd.read_csv(out)
    assert len(df) == 200


def test_bronze_metadata_columns_are_fully_populated(tmp_path: Path):
    """_ingested_at and _source_file are non-null for every row."""
    _, bronze_csv, _, _ = _run_layers(tmp_path, generate_events(100, seed=21))
    df = pd.read_csv(bronze_csv)
    assert df["_ingested_at"].notna().all()
    assert df["_source_file"].notna().all()


def test_bronze_ingested_at_same_for_all_rows_in_run(tmp_path: Path):
    """All rows in one ingest run share the same _ingested_at value."""
    _, bronze_csv, _, _ = _run_layers(tmp_path, generate_events(100, seed=22))
    df = pd.read_csv(bronze_csv)
    assert df["_ingested_at"].nunique() == 1


def test_bronze_attributes_valid_json_for_all_rows(tmp_path: Path):
    """The Attributes column is a valid JSON string for every row."""
    _, bronze_csv, _, _ = _run_layers(tmp_path, generate_events(200, seed=23))
    df = pd.read_csv(bronze_csv, dtype=str)
    for i, val in enumerate(df["Attributes"]):
        try:
            json.loads(val)
        except (ValueError, TypeError) as exc:
            pytest.fail(f"Row {i}: Attributes is not valid JSON: {val!r} ({exc})")


def test_bronze_coffee_machine_attributes_is_json_list(tmp_path: Path):
    """Coffee machine Attributes round-trips as a list (not a dict)."""
    _, bronze_csv, _, _ = _run_layers(tmp_path, generate_events(200, seed=24))
    df = pd.read_csv(bronze_csv, dtype=str)
    coffee_rows = df[df["DeviceType"] == "Coffee Machine"]
    assert len(coffee_rows) > 0, "No coffee machine rows generated -- increase seed count"
    for _, row in coffee_rows.iterrows():
        parsed = json.loads(row["Attributes"])
        assert isinstance(parsed, list), f"Coffee Attributes is not a list: {parsed!r}"


def test_bronze_identity_columns_non_blank(tmp_path: Path):
    """All four identity columns are non-null and non-blank in bronze."""
    _, bronze_csv, _, _ = _run_layers(tmp_path, generate_events(200, seed=25))
    df = pd.read_csv(bronze_csv, dtype=str)
    for col in ("DeviceId", "TimestampUTC", "DeviceType", "Status"):
        blank = df[col].isna() | (df[col].str.strip() == "")
        assert not blank.any(), f"Blank values in bronze.{col}"


# ---------------------------------------------------------------------------
# 3. Silver layer -- synthetic data
# ---------------------------------------------------------------------------

def test_silver_rows_out_le_rows_in(tmp_path: Path):
    """Silver never adds rows; rows_out <= rows_in."""
    events = generate_events(200, seed=30)
    src = _write_json(tmp_path / "ev.json", events)
    bronze_csv = tmp_path / "b.csv"
    bronze_ingest(src, bronze_csv)
    result = silver_cleanse(bronze_csv, tmp_path / "s.csv")
    assert result.rows_out <= result.rows_in


def test_silver_clean_synthetic_data_has_zero_drops(tmp_path: Path):
    """Synthetic data is always valid -- silver should drop nothing."""
    events = generate_events(200, seed=31)
    src = _write_json(tmp_path / "ev.json", events)
    bronze_csv = tmp_path / "b.csv"
    bronze_ingest(src, bronze_csv)
    result = silver_cleanse(bronze_csv, tmp_path / "s.csv")
    assert result.dropped_invalid == 0
    assert result.dropped_duplicate == 0
    assert result.rows_out == result.rows_in
    assert result.dropped_fuzzy == 0


def test_silver_no_blank_required_fields_in_output(tmp_path: Path):
    """Silver output has no blank or null values in any required column."""
    _, _, silver_csv, _ = _run_layers(tmp_path, generate_events(200, seed=32))
    df = pd.read_csv(silver_csv, dtype=str)
    for col in ("DeviceId", "TimestampUTC", "DeviceType", "Status"):
        blank = df[col].isna() | (df[col].str.strip() == "")
        assert not blank.any(), f"Blank values in silver.{col}"


def test_silver_timestamps_canonical_utc_format(tmp_path: Path):
    """Every TimestampUTC in silver matches YYYY-MM-DDTHH:MM:SSZ exactly."""
    _, _, silver_csv, _ = _run_layers(tmp_path, generate_events(200, seed=33))
    df = pd.read_csv(silver_csv, dtype=str)
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
    bad = ~df["TimestampUTC"].str.match(pattern)
    assert not bad.any(), (
        f"{bad.sum()} non-canonical timestamp(s) found:\n"
        f"{df.loc[bad, 'TimestampUTC'].tolist()}"
    )


def test_silver_no_duplicates_in_output(tmp_path: Path):
    """The dedup key (DeviceId, TimestampUTC, DeviceType, Status) is unique in silver."""
    _, _, silver_csv, _ = _run_layers(tmp_path, generate_events(300, seed=34))
    df = pd.read_csv(silver_csv, dtype=str)
    key = ["DeviceId", "TimestampUTC", "DeviceType", "Status"]
    dupes = df.duplicated(subset=key)
    assert not dupes.any(), f"{dupes.sum()} duplicate event(s) found in silver"


def test_silver_deduplicates_injected_duplicates(tmp_path: Path):
    """Silver drops exact duplicates injected into the input."""
    events = generate_events(50, seed=35)
    doubled = events + events          # 100 events, every one duplicated
    src = _write_json(tmp_path / "ev.json", doubled)
    bronze_csv = tmp_path / "b.csv"
    bronze_ingest(src, bronze_csv)
    result = silver_cleanse(bronze_csv, tmp_path / "s.csv")
    assert result.dropped_duplicate == 50
    assert result.rows_out == 50


def test_silver_attributes_pass_through_as_string(tmp_path: Path):
    """Attributes column is still a string in silver -- gold parses it back."""
    _, _, silver_csv, _ = _run_layers(tmp_path, generate_events(100, seed=36))
    df = pd.read_csv(silver_csv, dtype=str)
    for i, val in enumerate(df["Attributes"]):
        assert isinstance(val, str), f"Row {i}: Attributes is not a string in silver"


# ---------------------------------------------------------------------------
# 4. Gold layer -- synthetic data
# ---------------------------------------------------------------------------

def test_gold_all_files_exist(tmp_path: Path):
    """All four expected gold output files are created."""
    _, _, _, gold_dir = _run_layers(tmp_path, generate_events(200, seed=40))
    assert (gold_dir / "gold_washing_machine.csv").exists()
    assert (gold_dir / "gold_coffee_machine.csv").exists()
    assert (gold_dir / "gold_dishwasher.csv").exists()
    assert (gold_dir / "gold_summary.csv").exists()


def test_gold_per_device_files_are_non_empty(tmp_path: Path):
    """Each per-device gold CSV has at least one row."""
    _, _, _, gold_dir = _run_layers(tmp_path, generate_events(300, seed=41))
    for name in ("gold_washing_machine.csv", "gold_coffee_machine.csv",
                 "gold_dishwasher.csv"):
        df = pd.read_csv(gold_dir / name)
        assert len(df) > 0, f"{name} is empty"


def test_gold_wm_temperature_c_is_celsius(tmp_path: Path):
    """All washing machine Temperature_C values are in [0, 100] -- Fahrenheit converted."""
    _, _, _, gold_dir = _run_layers(tmp_path, generate_events(400, seed=42))
    wm = pd.read_csv(gold_dir / "gold_washing_machine.csv")
    with_temp = wm[wm["Temperature_C"].notna()]
    assert len(with_temp) > 0, "No WM rows have Temperature_C"
    # Synthetic generator uses 40/60/90 C. Fahrenheit equivalents are 104/140/194.
    # If conversion fails, values > 100 appear here.
    outside = with_temp[~with_temp["Temperature_C"].between(0, 100)]
    assert len(outside) == 0, (
        f"Washing machine Temperature_C out of [0, 100] -- likely Fahrenheit conversion failed:\n"
        f"{outside[['DeviceId', 'Temperature_C']].to_string()}"
    )


def test_gold_wm_temperature_c_values_match_expected_set(tmp_path: Path):
    """WM temperatures are in [0, 100] C after conversion."""
    _, _, _, gold_dir = _run_layers(tmp_path, generate_events(400, seed=43))
    wm = pd.read_csv(gold_dir / "gold_washing_machine.csv")
    with_temp = wm["Temperature_C"].dropna()
    assert all(0 <= t <= 100 for t in with_temp), (
        f"WM Temperature_C values out of [0, 100]: "
        f"{[t for t in with_temp if not (0 <= t <= 100)]}"
    )


def test_gold_wm_has_expected_columns(tmp_path: Path):
    """Washing machine gold table has all columns matching _flatten_washing_machine()."""
    _, _, _, gold_dir = _run_layers(tmp_path, generate_events(200, seed=44))
    wm = pd.read_csv(gold_dir / "gold_washing_machine.csv")
    expected = {
        "DeviceId", "TimestampUTC", "DeviceType", "Status",
        "Temperature_C",
        "SpinningSpeed", "SpinningSpeed_Unit",
        "TwinDos_Colour", "TwinDos_White", "TwinDos_Unit",
        "FailureId",
    }
    assert expected.issubset(set(wm.columns)), (
        f"Missing columns: {expected - set(wm.columns)}"
    )


def test_gold_wm_spinning_speed_values_are_valid(tmp_path: Path):
    """SpinningSpeed values are one of {600,800,900,1000,1100,1200,1400} or null (failure events)."""
    _, _, _, gold_dir = _run_layers(tmp_path, generate_events(400, seed=45))
    wm = pd.read_csv(gold_dir / "gold_washing_machine.csv")
    with_speed = wm["SpinningSpeed"].dropna()
    allowed = {600.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1400.0}
    unexpected = set(with_speed.unique()) - allowed
    assert not unexpected, f"Unexpected SpinningSpeed values: {unexpected}"


def test_gold_cm_has_dynamic_attribute_columns(tmp_path: Path):
    """Coffee machine gold table contains dynamically discovered {Id}_Value / {Id}_Unit columns."""
    _, _, _, gold_dir = _run_layers(tmp_path, generate_events(300, seed=46))
    cm = pd.read_csv(gold_dir / "gold_coffee_machine.csv")
    # Temperature appears in ~70% of CM events; with 300 total events and ~30%
    # coffee share (~90 events) it should always be present.
    assert "Temperature_Value" in cm.columns, "Temperature_Value missing from CM gold"
    assert "Temperature_Unit" in cm.columns,  "Temperature_Unit missing from CM gold"


def test_gold_cm_temperature_values_in_range(tmp_path: Path):
    """Coffee machine temperatures are between 80 and 96 Celsius (generator range)."""
    _, _, _, gold_dir = _run_layers(tmp_path, generate_events(300, seed=47))
    cm = pd.read_csv(gold_dir / "gold_coffee_machine.csv")
    with_temp = cm["Temperature_Value"].dropna()
    assert len(with_temp) > 0
    assert with_temp.between(80, 96).all(), (
        f"CM temperature out of [80, 96]: {with_temp[~with_temp.between(80, 96)].tolist()}"
    )


def test_gold_dw_has_expected_columns(tmp_path: Path):
    """Dishwasher gold table has all columns matching _flatten_dishwasher()."""
    _, _, _, gold_dir = _run_layers(tmp_path, generate_events(200, seed=48))
    dw = pd.read_csv(gold_dir / "gold_dishwasher.csv")
    expected = {
        "DeviceId", "TimestampUTC", "DeviceType", "Status",
        "Temperature_C", "QuickPowerWashActive",
    }
    assert expected.issubset(set(dw.columns)), (
        f"Missing columns: {expected - set(dw.columns)}"
    )


def test_gold_dw_temperature_c_is_celsius(tmp_path: Path):
    """Dishwasher Temperature_C values are in [0, 100]."""
    _, _, _, gold_dir = _run_layers(tmp_path, generate_events(300, seed=49))
    dw = pd.read_csv(gold_dir / "gold_dishwasher.csv")
    with_temp = dw["Temperature_C"].dropna()
    assert len(with_temp) > 0
    outside = with_temp[~with_temp.between(0, 100)]
    assert len(outside) == 0, (
        f"Dishwasher Temperature_C out of [0, 100]: {outside.tolist()}"
    )


def test_gold_dw_temperature_values_match_expected_set(tmp_path: Path):
    """DW temperatures are in [0, 100] C after conversion."""
    _, _, _, gold_dir = _run_layers(tmp_path, generate_events(300, seed=50))
    dw = pd.read_csv(gold_dir / "gold_dishwasher.csv")
    with_temp = dw["Temperature_C"].dropna()
    assert all(0 <= t <= 100 for t in with_temp), (
        f"DW Temperature_C values out of [0, 100]: "
        f"{[t for t in with_temp if not (0 <= t <= 100)]}"
    )


def test_gold_summary_has_correct_columns(tmp_path: Path):
    """Summary table has all expected columns."""
    _, _, _, gold_dir = _run_layers(tmp_path, generate_events(200, seed=51))
    summary = pd.read_csv(gold_dir / "gold_summary.csv")
    expected = {
        "DeviceId", "DeviceType", "total_events", "failure_events", "failure_rate_pct",
        "first_seen_utc", "last_seen_utc", "days_active",
    }
    assert expected.issubset(set(summary.columns))


def test_gold_summary_failure_rate_in_valid_range(tmp_path: Path):
    """failure_rate_pct is between 0 and 100 for every device."""
    _, _, _, gold_dir = _run_layers(tmp_path, generate_events(300, seed=52))
    summary = pd.read_csv(gold_dir / "gold_summary.csv")
    assert (summary["failure_rate_pct"] >= 0).all()
    assert (summary["failure_rate_pct"] <= 100).all()


def test_gold_summary_total_events_positive(tmp_path: Path):
    """Every device in the summary has at least one event."""
    _, _, _, gold_dir = _run_layers(tmp_path, generate_events(300, seed=53))
    summary = pd.read_csv(gold_dir / "gold_summary.csv")
    assert (summary["total_events"] > 0).all()
    assert (summary["failure_events"] >= 0).all()


def test_gold_summary_covers_all_device_types(tmp_path: Path):
    """Summary includes all three device types from the synthetic generator."""
    _, _, _, gold_dir = _run_layers(tmp_path, generate_events(300, seed=54))
    summary = pd.read_csv(gold_dir / "gold_summary.csv")
    found = set(summary["DeviceType"])
    assert "Washing machine" in found
    assert "Coffee Machine" in found
    assert "Dishwasher" in found


def test_gold_total_rows_equals_silver_rows(tmp_path: Path):
    """Sum of all per-device gold rows equals the silver row count -- no data loss."""
    events = generate_events(300, seed=55)
    src = _write_json(tmp_path / "ev.json", events)
    bronze_csv = tmp_path / "b.csv"
    silver_csv = tmp_path / "s.csv"
    gold_dir   = tmp_path / "gold"

    bronze_ingest(src, bronze_csv)
    silver_result = silver_cleanse(bronze_csv, silver_csv)
    gold_result   = build_gold(silver_csv, gold_dir)

    n_silver = silver_result.rows_out
    n_gold   = sum(gold_result.rows_by_device_type.values())
    assert n_gold >= n_silver, (
        f"Data loss between silver and gold: silver={n_silver}, gold={n_gold} "
        f"({gold_result.rows_by_device_type})"
    )


# ---------------------------------------------------------------------------
# 5. Large volume and batch-size tests
# ---------------------------------------------------------------------------

def test_large_volume_run(tmp_path: Path):
    """1000-event run completes with no drops and sensible output sizes."""
    events = generate_events(1000, seed=200)
    src = _write_json(tmp_path / "large.json", events)

    out_dir = tmp_path / "out"
    rc = run(src, out_dir)
    assert rc == 0

    silver_df  = pd.read_csv(out_dir / "silver" / "large.csv")
    summary_df = pd.read_csv(out_dir / "gold"   / "gold_summary.csv")

    # Clean synthetic data has no drops
    assert len(silver_df) == 1000, (
        f"Expected 1000 rows in silver, got {len(silver_df)} -- "
        "synthetic data should not trigger any drops"
    )

    # All three device types appear in the summary
    assert set(summary_df["DeviceType"]).issuperset(
        {"Washing machine", "Coffee Machine", "Dishwasher"}
    )

    # Gold rows >= silver rows (WM explodes FailureIds so gold may exceed silver)
    wm  = pd.read_csv(out_dir / "gold" / "gold_washing_machine.csv")
    cm  = pd.read_csv(out_dir / "gold" / "gold_coffee_machine.csv")
    dw  = pd.read_csv(out_dir / "gold" / "gold_dishwasher.csv")
    assert len(wm) + len(cm) + len(dw) >= 1000
