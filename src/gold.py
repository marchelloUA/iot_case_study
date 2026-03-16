"""Gold layer: BI / reporting ready tables.

Reads silver data and produces:
  - Per-device-type wide tables with all type-specific attributes flattened
    into columns.
  - A summary table with event counts, failure rates, and active-days per
    device, suitable for dashboards and operational monitoring.

Device-type flatten logic mirrors the Databricks notebook
(IoT case study 150326_0002.ipynb, Cell 3):

  - Washing machine: one row per FailureId for failure events (explode);
    Program Start/End rows have FailureId = null. Temperature always in Celsius.
  - Coffee machine: dynamically discovers all attribute Ids present in the
    silver data and produces {Id}_Value / {Id}_Unit column pairs.
  - Dishwasher: QuickPowerWashActive is a boolean (True/False), matching the
    notebook's cast from the raw Bit (0/1) value.

Azure equivalent:
  Source: ADLS Gen2 silver / Delta silver
  Sink:   ADLS Gen2 gold, Databricks SQL tables, or Delta tables readable by
          Power BI via Direct Query or import mode.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

logger = logging.getLogger(__name__)

_RESERVED_COLUMNS: frozenset[str] = frozenset(
    {"DeviceId", "TimestampUTC", "DeviceType", "Status"}
)


class GoldResult(NamedTuple):
    """Counts returned by build_gold() for observability."""
    rows_by_device_type: dict[str, int]
    unknown_device_types: list[str]


def _parse_attributes(attrs_str: str | None) -> Any:
    """Parse the Attributes JSON string back to its original Python type."""
    if attrs_str is None or (isinstance(attrs_str, float) and pd.isna(attrs_str)):
        return None
    if attrs_str in ("null", ""):
        return None
    try:
        return json.loads(attrs_str)
    except (ValueError, TypeError):
        logger.warning("Gold: could not parse Attributes JSON: %r", str(attrs_str)[:80])
        return attrs_str


def _audit_cols(row: pd.Series) -> dict[str, Any]:
    """Extract audit columns present in silver."""
    result = {}
    for col in ("_ingested_at", "_source_file", "_silver_processed_at"):
        if col in row.index:
            result[col] = row[col]
    return result


# ---------------------------------------------------------------------------
# Washing machine flatten -- one row per FailureId (explode pattern)
# ---------------------------------------------------------------------------

def _flatten_washing_machine_rows(row: pd.Series) -> list[dict[str, Any]]:
    """Flatten a washing machine silver row into one or more gold rows.

    For Program Failure events, one row is produced per FailureId so that
    each failure code is individually queryable (matches notebook explode_outer).
    For Program Start/End events, one row is produced with FailureId = null.
    Temperature is always in Celsius.
    """
    attrs: dict[str, Any] = _parse_attributes(row.get("Attributes")) or {}
    base: dict[str, Any] = {
        "DeviceId":   row["DeviceId"],
        "TimestampUTC": row["TimestampUTC"],
        "DeviceType": row["DeviceType"],
        "Status":     row["Status"],
    }
    base.update(_audit_cols(row))

    # Temperature -> Celsius
    temp = attrs.get("Temperature")
    unit = attrs.get("Temperature_Unit", "")
    if temp is not None:
        try:
            temp_f = float(temp)
            if unit.lower() in {"f", "fahrenheit"}:
                temperature_c = round((temp_f - 32.0) * 5.0 / 9.0, 2)
            else:
                temperature_c = temp_f
        except (ValueError, TypeError):
            temperature_c = None
    else:
        temperature_c = None

    base["Temperature_C"] = temperature_c
    base["SpinningSpeed"] = attrs.get("SpinningSpeed")
    base["SpinningSpeed_Unit"] = attrs.get("SpinningSpeed_Unit")

    twin = attrs.get("TwinDos")
    if isinstance(twin, list):
        base["TwinDos_Colour"] = twin[0] if len(twin) > 0 else None
        base["TwinDos_White"]  = twin[1] if len(twin) > 1 else None
    else:
        base["TwinDos_Colour"] = None
        base["TwinDos_White"]  = None
    base["TwinDos_Unit"] = attrs.get("TwinDos_Unit")

    # Explode FailureIds -- one row per failure, null for non-failure events
    failures = attrs.get("FailureIds")
    if isinstance(failures, list) and failures:
        return [{**base, "FailureId": fid} for fid in failures]
    return [{**base, "FailureId": None}]


# ---------------------------------------------------------------------------
# Coffee machine flatten -- dynamic Id discovery
# ---------------------------------------------------------------------------

def _flatten_coffee_machine_row(row: pd.Series, cm_attrs: list[tuple[str, str]]) -> dict[str, Any]:
    """Flatten a coffee machine silver row using the discovered attribute Id list."""
    attrs_raw = _parse_attributes(row.get("Attributes"))
    attrs_list = attrs_raw if isinstance(attrs_raw, list) else (
        [attrs_raw] if isinstance(attrs_raw, dict) else []
    )
    # Build a lookup: raw_id -> {Value, Unit}
    attr_lookup: dict[str, dict[str, Any]] = {}
    for item in attrs_list:
        if isinstance(item, dict) and isinstance(item.get("Id"), str):
            attr_lookup[item["Id"].strip()] = item

    out: dict[str, Any] = {
        "DeviceId":     row["DeviceId"],
        "TimestampUTC": row["TimestampUTC"],
        "DeviceType":   row["DeviceType"],
        "Status":       row["Status"],
    }
    out.update(_audit_cols(row))
    for raw_id, safe_id in cm_attrs:
        item = attr_lookup.get(raw_id, {})
        out[f"{safe_id}_Value"] = item.get("Value")
        out[f"{safe_id}_Unit"]  = item.get("Unit")
    return out


def _discover_cm_attrs(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Discover all distinct attribute Ids across coffee machine rows."""
    seen_safe: set[str] = set()
    result: list[tuple[str, str]] = []
    for attrs_str in df["Attributes"]:
        attrs_raw = _parse_attributes(attrs_str)
        if not isinstance(attrs_raw, list):
            continue
        for item in attrs_raw:
            if not isinstance(item, dict):
                continue
            raw_id = item.get("Id")
            if not isinstance(raw_id, str) or not raw_id.strip():
                continue
            raw_id = raw_id.strip()
            safe_id = re.sub(r"[^\w]", "_", raw_id)
            if safe_id in _RESERVED_COLUMNS:
                logger.warning(
                    "Gold CM: attribute Id %r collides with reserved column; skipping", raw_id
                )
                continue
            if safe_id not in seen_safe:
                seen_safe.add(safe_id)
                result.append((raw_id, safe_id))
    return result


# ---------------------------------------------------------------------------
# Dishwasher flatten
# ---------------------------------------------------------------------------

def _flatten_dishwasher_row(row: pd.Series) -> dict[str, Any]:
    """Flatten a dishwasher silver row."""
    attrs: dict[str, Any] = _parse_attributes(row.get("Attributes")) or {}

    temp = attrs.get("Temperature")
    unit = attrs.get("Temperature_Unit", "")
    if temp is not None:
        try:
            temp_f = float(temp)
            if unit.lower() in {"f", "fahrenheit"}:
                temperature_c = round((temp_f - 32.0) * 5.0 / 9.0, 2)
            else:
                temperature_c = temp_f
        except (ValueError, TypeError):
            temperature_c = None
    else:
        temperature_c = None

    # QuickPowerWashActive as boolean (Bit 0/1 -> False/True)
    qpw_raw = attrs.get("QuickPowerWashActive")
    try:
        quick_power_wash = bool(int(qpw_raw)) if qpw_raw is not None else None
    except (ValueError, TypeError):
        quick_power_wash = None

    out: dict[str, Any] = {
        "DeviceId":             row["DeviceId"],
        "TimestampUTC":         row["TimestampUTC"],
        "DeviceType":           row["DeviceType"],
        "Status":               row["Status"],
        "Temperature_C":        temperature_c,
        "QuickPowerWashActive": quick_power_wash,
        "FailureId":            None,
    }

    # Explode FailureIds same as WM
    failures = attrs.get("FailureIds")
    if isinstance(failures, list) and failures:
        return [{**out, "FailureId": fid} for fid in failures]
    return [out]


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-device KPIs from the silver DataFrame.

    Columns match the Databricks notebook gold_summary table:
      total_events, failure_events, failure_rate_pct,
      first_seen_utc, last_seen_utc, days_active.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "DeviceId", "DeviceType", "total_events", "failure_events",
            "failure_rate_pct", "first_seen_utc", "last_seen_utc", "days_active",
        ])

    ts = pd.to_datetime(df["TimestampUTC"], format="ISO8601", utc=True, errors="coerce")
    df = df.copy()
    df["_ts"] = ts

    agg = (
        df.groupby(["DeviceId", "DeviceType"], sort=False)
        .agg(
            total_events=("DeviceId", "count"),
            first_seen_utc=("_ts", "min"),
            last_seen_utc=("_ts", "max"),
        )
        .reset_index()
    )

    failure_mask = df["Status"].str.contains("failure", case=False, na=False)
    failures = (
        df[failure_mask]
        .groupby(["DeviceId", "DeviceType"], sort=False)
        .size()
        .reset_index(name="failure_events")
    )

    summary = agg.merge(failures, on=["DeviceId", "DeviceType"], how="left")
    summary["failure_events"] = summary["failure_events"].fillna(0).astype(int)
    summary["failure_rate_pct"] = (
        summary["failure_events"] / summary["total_events"].replace(0, float("nan")) * 100
    ).round(2).fillna(0.0)

    summary["days_active"] = (
        (summary["last_seen_utc"] - summary["first_seen_utc"])
        .dt.days.fillna(0).astype(int)
    )
    # Format timestamps as strings for CSV output
    summary["first_seen_utc"] = summary["first_seen_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    summary["last_seen_utc"]  = summary["last_seen_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    col_order = [
        "DeviceId", "DeviceType", "total_events", "failure_events",
        "failure_rate_pct", "first_seen_utc", "last_seen_utc", "days_active",
    ]
    return summary[col_order].sort_values(["DeviceType", "DeviceId"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_gold(silver_path: Path, output_dir: Path) -> GoldResult:
    """Read a silver CSV, flatten records per device type, and write gold tables.

    Writes:
      output_dir/gold_<device_type>.csv  - one per known device type
      output_dir/gold_summary.csv        - aggregated per-device statistics

    Returns a GoldResult with row counts and any unrecognised device types.
    Raises FileNotFoundError if silver_path does not exist.
    """
    df = pd.read_csv(silver_path, encoding="utf-8", dtype=str)

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_dir = output_dir.resolve()
    rows_by_type: dict[str, int] = {}
    unknown_types: list[str] = []

    # ---- Washing machine ----
    wm_df = df[df["DeviceType"].str.contains("washing", case=False, na=False)]
    if not wm_df.empty:
        wm_rows: list[dict[str, Any]] = []
        for _, row in wm_df.iterrows():
            wm_rows.extend(_flatten_washing_machine_rows(row))
        wm_out = pd.DataFrame(wm_rows)
        out_path = (output_dir / "gold_washing_machine.csv").resolve()
        wm_out.to_csv(out_path, index=False, encoding="utf-8")
        rows_by_type["Washing machine"] = len(wm_out)
        logger.info("Gold WM: wrote %d row(s) to %s", len(wm_out), out_path)

    # ---- Coffee machine ----
    cm_df = df[df["DeviceType"].str.contains("coffee", case=False, na=False)]
    if not cm_df.empty:
        cm_attrs = _discover_cm_attrs(cm_df)
        cm_rows = [_flatten_coffee_machine_row(row, cm_attrs) for _, row in cm_df.iterrows()]
        cm_out = pd.DataFrame(cm_rows)
        out_path = (output_dir / "gold_coffee_machine.csv").resolve()
        cm_out.to_csv(out_path, index=False, encoding="utf-8")
        rows_by_type["Coffee Machine"] = len(cm_out)
        logger.info("Gold CM: wrote %d row(s) to %s", len(cm_out), out_path)

    # ---- Dishwasher ----
    dw_df = df[df["DeviceType"].str.contains("dishwasher", case=False, na=False)]
    if not dw_df.empty:
        dw_rows: list[dict[str, Any]] = []
        for _, row in dw_df.iterrows():
            dw_rows.extend(_flatten_dishwasher_row(row))
        dw_out = pd.DataFrame(dw_rows)
        out_path = (output_dir / "gold_dishwasher.csv").resolve()
        dw_out.to_csv(out_path, index=False, encoding="utf-8")
        rows_by_type["Dishwasher"] = len(dw_out)
        logger.info("Gold DW: wrote %d row(s) to %s", len(dw_out), out_path)

    # ---- Unknown device types (safety net) ----
    known_filter = (
        df["DeviceType"].str.contains("washing",    case=False, na=False)
        | df["DeviceType"].str.contains("coffee",   case=False, na=False)
        | df["DeviceType"].str.contains("dishwasher", case=False, na=False)
    )
    unknown_df = df[~known_filter]
    if not unknown_df.empty:
        for dt_val in unknown_df["DeviceType"].dropna().unique():
            dt_str = str(dt_val)
            if dt_str not in unknown_types:
                unknown_types.append(dt_str)
                logger.warning("Gold: unrecognised DeviceType %r -> fallback", dt_str)

        safe_name = re.sub(r"[^\w]", "_", unknown_df["DeviceType"].iloc[0] or "unknown").lower()
        out_path = (output_dir / f"gold_{safe_name}.csv").resolve()
        if out_path.is_relative_to(resolved_dir):
            unk_out = unknown_df[[
                "DeviceId", "TimestampUTC", "DeviceType", "Status", "Attributes"
            ] + [c for c in ("_ingested_at", "_source_file", "_silver_processed_at")
                 if c in unknown_df.columns]].copy()
            unk_out.to_csv(out_path, index=False, encoding="utf-8")
            rows_by_type[str(unknown_df["DeviceType"].iloc[0])] = len(unk_out)
            logger.info("Gold unknown: wrote %d row(s) to %s", len(unk_out), out_path)

    # ---- Summary ----
    summary_df = _build_summary(df)
    summary_path = output_dir / "gold_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    logger.info("Gold: summary (%d device(s)) -> %s", len(summary_df), summary_path)

    return GoldResult(rows_by_device_type=rows_by_type, unknown_device_types=unknown_types)


def main(argv: list[str] | None = None) -> int:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Gold layer: flatten silver events to BI tables.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        result = build_gold(args.input, args.output_dir)
    except FileNotFoundError:
        print(f"ERROR: silver file not found: {args.input}", flush=True)
        return 1
    except OSError as exc:
        print(f"ERROR: could not write output: {exc}", flush=True)
        return 1

    for dt, count in result.rows_by_device_type.items():
        print(f"Gold: {dt}: {count} row(s)")
    if result.unknown_device_types:
        print(f"Gold: unknown device types (fallback): {result.unknown_device_types}")
    print(f"Gold: summary written to {args.output_dir}/gold_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
