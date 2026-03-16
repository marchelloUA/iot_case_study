"""Tests for the synthetic data generator."""
import json
from datetime import datetime
from pathlib import Path

import pytest

from src.generate_synthetic import generate_events


def test_generate_events_returns_correct_count():
    """generate_events returns exactly the requested number of events."""
    assert len(generate_events(50, seed=1)) == 50


def test_generate_events_zero_count():
    """count=0 returns an empty list."""
    assert generate_events(0, seed=1) == []


def test_generate_events_all_required_fields_present():
    """Every generated event has the four required pipeline fields plus Attributes."""
    required = {"DeviceId", "TimestampUTC", "DeviceType", "Status", "Attributes"}
    for i, event in enumerate(generate_events(100, seed=1)):
        assert required.issubset(event.keys()), f"Record {i} missing fields: {required - event.keys()}"


def test_generate_events_device_ids_have_syn_prefix():
    """DeviceId starts with 'syn_' followed by 13 lowercase-alphanumeric chars."""
    import re
    pattern = re.compile(r"^syn_[a-z0-9]{13}$")
    for event in generate_events(50, seed=1):
        did = event["DeviceId"]
        assert pattern.match(did), f"DeviceId does not match expected format: {did!r}"


def test_generate_events_timestamps_have_z_suffix():
    """All timestamps carry a Z suffix so the UTC intent is explicit."""
    for event in generate_events(50, seed=1):
        assert event["TimestampUTC"].endswith("Z"), f"Missing Z: {event['TimestampUTC']!r}"


def test_generate_events_device_types_are_known():
    """Only the three known device types appear in generated output."""
    known = {"Washing machine", "Coffee Machine", "Dishwasher"}
    types = {e["DeviceType"] for e in generate_events(200, seed=1)}
    assert types.issubset(known)


def test_generate_events_same_seed_same_output():
    """Two calls with the same seed on the same run produce identical output."""
    assert generate_events(30, seed=7) == generate_events(30, seed=7)


def test_generate_events_days_changes_timestamp_window():
    """Wider days value produces a wider timestamp spread."""
    def span_seconds(events):
        times = [
            datetime.strptime(e["TimestampUTC"], "%Y-%m-%dT%H:%M:%SZ")
            for e in events
        ]
        return (max(times) - min(times)).total_seconds()

    narrow = span_seconds(generate_events(500, seed=1, days=1))
    wide = span_seconds(generate_events(500, seed=1, days=60))
    assert wide > narrow


@pytest.mark.parametrize("bad_days", [0, -5])
def test_generate_events_invalid_days_raises(bad_days: int):
    """days=0 or negative raises ValueError."""
    with pytest.raises(ValueError, match="days must be at least 1"):
        generate_events(10, days=bad_days)


def test_generate_events_wm_temperatures_in_valid_range():
    """WM Temperature values are within the documented generator ranges."""
    events = generate_events(500, seed=7)
    wm_events = [
        e for e in events
        if e["DeviceType"] == "Washing machine"
        and isinstance(e.get("Attributes"), dict)
        and "Temperature" in e["Attributes"]
    ]
    for e in wm_events:
        t = e["Attributes"]["Temperature"]
        unit = e["Attributes"].get("Temperature_Unit", "Celsius")
        if unit == "Celsius":
            assert 30 <= t <= 95, f"Celsius temperature out of range: {t}"
        else:
            assert 86 <= t <= 203, f"Fahrenheit temperature out of range: {t}"


def test_generate_events_dw_can_have_failure_events():
    """Dishwasher generates some failure events (non-zero in large sample)."""
    events = generate_events(500, seed=10)
    dw_failures = [
        e for e in events
        if e["DeviceType"] == "Dishwasher" and e["Status"] == "Program Failure"
    ]
    assert len(dw_failures) > 0, "Expected at least some DW failure events in 500-event sample"


def test_generate_events_cm_always_has_temperature():
    """Coffee machine events always include a Temperature attribute entry."""
    events = generate_events(200, seed=11)
    cm_events = [e for e in events if e["DeviceType"] == "Coffee Machine"]
    for e in cm_events:
        attrs = e["Attributes"]
        assert isinstance(attrs, list), "CM Attributes should be a list"
        ids = [item.get("Id") for item in attrs if isinstance(item, dict)]
        assert "Temperature" in ids, f"CM event missing Temperature attribute: {attrs!r}"


def test_generate_events_round_trip_through_medallion(tmp_path: Path):
    """Every generated event is accepted by bronze+silver with zero drops."""
    from src.bronze import ingest as bronze_ingest
    from src.silver import cleanse as silver_cleanse

    events = generate_events(100, seed=42)
    p = tmp_path / "synthetic.json"
    p.write_text(json.dumps(events), encoding="utf-8")

    bronze_result = bronze_ingest(p, tmp_path / "bronze.csv")
    assert bronze_result.rows_written == 100
    assert bronze_result.dead_letter_count == 0

    silver_result = silver_cleanse(tmp_path / "bronze.csv", tmp_path / "silver.csv")
    assert silver_result.dropped_invalid == 0
    assert silver_result.dropped_duplicate == 0
    assert silver_result.rows_out == 100
