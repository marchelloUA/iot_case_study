"""Generate synthetic IoT event JSON for volume/load testing.

This script generates a JSON array of events matching the notebook's synthetic
data generators. Device types, status weights, attribute pools, and device ID
format all mirror the Databricks notebook (IoT case study 150326_0002.ipynb).

Device IDs use the "syn_" prefix plus 13 random lowercase-alphanumeric characters
so they can be distinguished from the real initial_data device IDs in both local
runs and Databricks Delta table queries.

Example:
    python -m src.generate_synthetic --records 1000 --output data/synthetic.json
"""

from __future__ import annotations

import argparse
import json
import random
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _random_suffix(rng: random.Random, n: int = 13) -> str:
    return "".join(rng.choices(string.ascii_lowercase + string.digits, k=n))


def _random_timestamp(rng: random.Random, start: datetime, end: datetime) -> str:
    span = end - start
    seconds = rng.randint(0, int(span.total_seconds()))
    return (start + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gen_washing_machine(rng: random.Random, device_id: str, event_time: str) -> dict[str, Any]:
    """Mirrors the real washing machine message patterns.

    Weights reflect real device telemetry distributions -- failures are rare,
    which matters for downstream anomaly detection logic.
    """
    status = rng.choices(
        ["Program Start", "Program End", "Program Failure"],
        weights=[0.40, 0.45, 0.15],
    )[0]

    if status == "Program Failure":
        failure_pool = [10, 17, 23, 31, 42, 55]
        attrs: dict[str, Any] = {
            "FailureIds": rng.sample(failure_pool, k=rng.randint(1, 3))
        }
    else:
        unit = rng.choice(["Celsius", "Fahrenheit"])
        temp = rng.randint(30, 95) if unit == "Celsius" else rng.randint(86, 203)
        attrs = {
            "Temperature": temp,
            "Temperature_Unit": unit,
            "SpinningSpeed": rng.choice([600, 800, 900, 1000, 1100, 1200, 1400]),
            "SpinningSpeed_Unit": "Rotations per Minute",
        }
        if rng.random() > 0.4:
            attrs["TwinDos"] = [rng.randint(70, 100), rng.randint(70, 100)]
            attrs["TwinDos_Unit"] = "Percentage"

    return {
        "DeviceId": device_id,
        "TimestampUTC": event_time,
        "DeviceType": "Washing machine",
        "Status": status,
        "Attributes": attrs,
    }


def _gen_coffee_machine(rng: random.Random, device_id: str, event_time: str) -> dict[str, Any]:
    """Mirrors the real coffee machine message pattern (list-style Attributes)."""
    status = rng.choices(
        ["Program Start", "Program End", "Program Failure"],
        weights=[0.35, 0.55, 0.10],
    )[0]

    attr_pool = [
        {"Id": "Grinding",      "Value": rng.randint(60, 100), "Unit": "Percentage"},
        {"Id": "Temperature",   "Value": rng.randint(80, 96),  "Unit": "Celsius"},
        {"Id": "WaterPressure", "Value": round(rng.uniform(8, 10), 1), "Unit": "Bar"},
        {"Id": "CupSize",       "Value": rng.choice([30, 60, 120, 240]), "Unit": "Millilitre"},
    ]
    # Always include Temperature; optionally add 0-2 other attributes
    attrs_list = [attr_pool[1]] + rng.sample(
        [attr_pool[0], attr_pool[2], attr_pool[3]], k=rng.randint(0, 2)
    )

    return {
        "DeviceId": device_id,
        "TimestampUTC": event_time,
        "DeviceType": "Coffee Machine",
        "Status": status,
        "Attributes": attrs_list,
    }


def _gen_dishwasher(rng: random.Random, device_id: str, event_time: str) -> dict[str, Any]:
    """Mirrors the real dishwasher message pattern."""
    status = rng.choices(
        ["Program Start", "Program End", "Program Failure"],
        weights=[0.38, 0.50, 0.12],
    )[0]

    if status == "Program Failure":
        failure_pool = [5, 11, 14, 22, 33]
        attrs: dict[str, Any] = {
            "FailureIds": rng.sample(failure_pool, k=rng.randint(1, 2))
        }
    else:
        unit = rng.choice(["Celsius", "Fahrenheit"])
        temp = rng.randint(40, 75) if unit == "Celsius" else rng.randint(104, 167)
        attrs = {
            "Temperature": temp,
            "Temperature_Unit": unit,
            "QuickPowerWashActive": rng.randint(0, 1),
            "QuickPowerWashActive_Unit": "Bit",
        }

    return {
        "DeviceId": device_id,
        "TimestampUTC": event_time,
        "DeviceType": "Dishwasher",
        "Status": status,
        "Attributes": attrs,
    }


_GENERATORS = {
    "Washing machine": _gen_washing_machine,
    "Coffee Machine":  _gen_coffee_machine,
    "Dishwasher":      _gen_dishwasher,
}

_DEVICE_TYPE_WEIGHTS = {"Washing machine": 0.50, "Coffee Machine": 0.30, "Dishwasher": 0.20}


def generate_events(count: int, seed: int | None = None, days: int = 365) -> list[dict[str, Any]]:
    """Generate count synthetic IoT events spread over the given number of days.

    Uses a local random.Random instance so repeated calls in the same process
    are independent and do not interfere with each other's state.

    Device IDs are "syn_" + 13-character random lowercase-alphanumeric strings,
    matching the notebook's format so they can be filtered from initial_data rows.

    Passing the same seed on the same calendar day gives the same output.
    days must be at least 1.
    """
    if days < 1:
        raise ValueError(f"days must be at least 1, got {days!r}")
    rng = random.Random(seed)

    end = datetime.now(tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None
    )
    start = end - timedelta(days=days)

    device_types = list(_DEVICE_TYPE_WEIGHTS.keys())
    weights = list(_DEVICE_TYPE_WEIGHTS.values())

    events: list[dict[str, Any]] = []
    for _ in range(count):
        event_time = _random_timestamp(rng, start, end)
        device_id = "syn_" + _random_suffix(rng)
        kind = rng.choices(device_types, weights=weights, k=1)[0]
        event = _GENERATORS[kind](rng, device_id, event_time)
        events.append(event)

    return events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic IoT event JSON.")
    parser.add_argument("--records", type=int, default=1000,
                        help="Number of events to generate (must be >= 0)")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic.json"),
                        help="Output file path")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible output.")
    parser.add_argument("--days", type=int, default=365,
                        help="Time window in days for generated timestamps (minimum 1).")

    args = parser.parse_args(argv)

    if args.records < 0:
        print("ERROR: --records must be >= 0", flush=True)
        return 1

    events = generate_events(args.records, seed=args.seed, days=args.days)
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: could not write output: {exc}", flush=True)
        return 1

    print(f"Wrote {len(events)} synthetic events to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
