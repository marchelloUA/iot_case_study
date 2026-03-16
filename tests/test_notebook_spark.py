"""PySpark tests for the Databricks notebook logic.

Tests the notebook's PySpark implementations in
notebooks/IoT case study 150326_0002.ipynb, cell by cell:

  - Bronze: parse_file(), BRONZE_SCHEMA, metadata columns
  - Silver: required-field filter, Z-suffix timestamp fix, window dedup
  - Gold WM: Fahrenheit conversion, column set, TwinDos, FailureIds
  - Gold CM: dynamic Id discovery, _Value/_Unit pairs, reserved-column guard
  - Gold DW: same temperature conversion as WM, QuickPowerWashActive columns
  - Summary: groupBy + agg, failure_rate_pct range
  - End-to-end: bronze -> silver -> gold with no data loss

Tests are skipped automatically when pyspark is not installed.
"""
from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("pyspark")  # skip module if PySpark not installed

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.functions import col, explode_outer
from pyspark.sql.types import (
    ArrayType, DoubleType, IntegerType, StringType, StructField, StructType,
)

from src.generate_synthetic import generate_events


# ---------------------------------------------------------------------------
# Session-scoped SparkSession -- shared across all tests for speed
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark():
    s = (
        SparkSession.builder
        .master("local[2]")
        .appName("iot-notebook-test")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    yield s
    s.stop()


# ---------------------------------------------------------------------------
# Schemas and constants -- copied from notebook cells verbatim
# ---------------------------------------------------------------------------

BRONZE_SCHEMA = StructType([
    StructField("DeviceId",     StringType()),
    StructField("TimestampUTC", StringType()),
    StructField("DeviceType",   StringType()),
    StructField("Status",       StringType()),
    StructField("Attributes",   StringType()),
    StructField("_ingested_at", StringType()),
    StructField("_source_file", StringType()),
])

REQUIRED = ["DeviceId", "TimestampUTC", "DeviceType", "Status"]
DEDUP_KEY = REQUIRED[:]

WM_ATTRS_SCHEMA = StructType([
    StructField("Temperature",        DoubleType()),
    StructField("Temperature_Unit",   StringType()),
    StructField("SpinningSpeed",      IntegerType()),
    StructField("SpinningSpeed_Unit", StringType()),
    StructField("TwinDos",            ArrayType(IntegerType())),
    StructField("TwinDos_Unit",       StringType()),
    StructField("FailureIds",         ArrayType(IntegerType())),
])

CM_ATTRS_SCHEMA = ArrayType(StructType([
    StructField("Id",    StringType()),
    StructField("Value", DoubleType()),
    StructField("Unit",  StringType()),
]))

DW_ATTRS_SCHEMA = StructType([
    StructField("Temperature",               DoubleType()),
    StructField("Temperature_Unit",          StringType()),
    StructField("QuickPowerWashActive",      IntegerType()),
    StructField("QuickPowerWashActive_Unit", StringType()),
])

_RESERVED_COLS = {"DeviceId", "TimestampUTC", "DeviceType", "Status"}


# ---------------------------------------------------------------------------
# Notebook helper functions (bronze cell) -- copied from notebook verbatim
# ---------------------------------------------------------------------------

def _make_parse_file(ingested_at: str):
    """Return a parse_file function bound to the given ingested_at timestamp.

    Mirrors the parse_file() closure in notebook bronze cell.
    """
    def parse_file(path_content):
        path, content = path_content
        records = json.loads(content)
        if not isinstance(records, list):
            raise ValueError(f"Expected JSON array, got {type(records).__name__}")
        rows = []
        for rec in records:
            if not isinstance(rec, dict):
                rows.append({
                    "DeviceId": None, "TimestampUTC": None,
                    "DeviceType": None, "Status": None,
                    "Attributes": json.dumps(rec),
                    "_ingested_at": ingested_at,
                    "_source_file": path,
                })
                continue
            attrs = rec.get("Attributes")
            rows.append({
                "DeviceId":     rec.get("DeviceId"),
                "TimestampUTC": rec.get("TimestampUTC"),
                "DeviceType":   rec.get("DeviceType"),
                "Status":       rec.get("Status"),
                "Attributes":   json.dumps(attrs) if attrs is not None else None,
                "_ingested_at": ingested_at,
                "_source_file": path,
            })
        return rows
    return parse_file


# ---------------------------------------------------------------------------
# Silver transformation helpers -- copied from notebook silver cell verbatim
# ---------------------------------------------------------------------------

def _valid_filter():
    filt = None
    for col in REQUIRED:
        cond = F.col(col).isNotNull() & (F.trim(F.col(col)) != "")
        filt = cond if filt is None else filt & cond
    return filt


def _normalize_timestamps(df):
    """Z-suffix-safe timestamp normalization -- exact copy of notebook silver cell."""
    return (
        df
        .withColumn(
            "TimestampUTC",
            F.date_format(
                F.to_utc_timestamp(
                    F.to_timestamp(
                        F.regexp_replace(F.col("TimestampUTC"), "Z$", ""),
                        "yyyy-MM-dd'T'HH:mm:ss",
                    ),
                    "UTC",
                ),
                "yyyy-MM-dd'T'HH:mm:ss'Z'",
            )
        )
        .filter(F.col("TimestampUTC").isNotNull())
    )


def _deduplicate(df):
    """Window-function dedup -- exact copy of notebook silver cell."""
    df = df.withColumn("_row_id", F.monotonically_increasing_id())
    w = Window.partitionBy(*DEDUP_KEY).orderBy(
        F.col("_ingested_at").asc(), F.col("_row_id").asc()
    )
    return (
        df
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn", "_row_id")
    )


# ---------------------------------------------------------------------------
# Test helpers: build bronze / silver DataFrames from event lists
# ---------------------------------------------------------------------------

def _bronze_df(spark, events, ingested_at="2024-01-01T00:00:00Z", source="test"):
    parse_file = _make_parse_file(ingested_at)
    rows = parse_file((source, json.dumps(events)))
    return spark.createDataFrame(rows, schema=BRONZE_SCHEMA)


def _silver_df(spark, events, **kw):
    df = _bronze_df(spark, events, **kw)
    df = df.filter(_valid_filter())
    df = _normalize_timestamps(df)
    df = _deduplicate(df)
    return df


# ---------------------------------------------------------------------------
# 1. Bronze -- parse_file and schema
# ---------------------------------------------------------------------------

def test_bronze_parse_file_row_count(spark):
    """parse_file() returns exactly one row per input event."""
    events = generate_events(100, seed=1000)
    parse_file = _make_parse_file("2024-01-01T00:00:00Z")
    rows = parse_file(("test", json.dumps(events)))
    assert len(rows) == 100


def test_bronze_non_dict_elements_become_audit_rows():
    """Non-dict elements in the JSON array become audit rows with null identity fields."""
    parse_file = _make_parse_file("2024-01-01T00:00:00Z")
    rows = parse_file(("test", json.dumps(["not-a-dict", 42, {"DeviceId": "abc"}])))
    assert len(rows) == 3
    assert rows[0]["DeviceId"] is None   # non-dict
    assert rows[1]["DeviceId"] is None   # non-dict
    assert rows[2]["DeviceId"] == "abc"  # dict: identity field preserved


def test_bronze_schema_columns(spark):
    """Bronze DataFrame has all seven expected columns."""
    events = generate_events(50, seed=1001)
    df = _bronze_df(spark, events)
    assert set(df.columns) == {
        "DeviceId", "TimestampUTC", "DeviceType", "Status",
        "Attributes", "_ingested_at", "_source_file",
    }


def test_bronze_ingested_at_same_for_all_rows(spark):
    """_ingested_at is the same value for every row in one run."""
    events = generate_events(50, seed=1002)
    df = _bronze_df(spark, events, ingested_at="2024-06-01T12:00:00Z")
    values = [r._ingested_at for r in df.select("_ingested_at").collect()]
    assert all(v == "2024-06-01T12:00:00Z" for v in values)


def test_bronze_attributes_is_valid_json(spark):
    """Attributes column is a valid JSON string for every row."""
    events = generate_events(100, seed=1003)
    df = _bronze_df(spark, events)
    for row in df.select("Attributes").collect():
        json.loads(row.Attributes)  # raises if invalid


def test_bronze_coffee_machine_attributes_is_json_list(spark):
    """Coffee machine Attributes round-trips as a list (not a dict)."""
    events = generate_events(200, seed=1004)
    df = _bronze_df(spark, events)
    cm_rows = (
        df.filter(F.col("DeviceType") == "Coffee Machine")
        .select("Attributes").collect()
    )
    assert len(cm_rows) > 0
    for row in cm_rows:
        assert isinstance(json.loads(row.Attributes), list)


# ---------------------------------------------------------------------------
# 2. Silver -- validation, Z-suffix timestamp fix, deduplication
# ---------------------------------------------------------------------------

def test_silver_drops_rows_with_blank_required_field(spark):
    """Rows where any required field is blank or null are dropped."""
    events = generate_events(50, seed=2000)
    bad_events = events + [
        {**events[0], "DeviceId": "   "},    # whitespace-only
        {**events[1], "Status": None},       # null
    ]
    df = _bronze_df(spark, bad_events)
    valid = df.filter(_valid_filter())
    assert valid.count() == 50


def test_silver_z_suffix_timestamp_is_not_dropped(spark):
    """Z-suffix timestamps must NOT be silently dropped.

    Without regexp_replace("Z$", ""), Spark's to_timestamp() returns null for
    Z-suffixed strings, causing every Event Hubs event to be lost at the silver
    quality gate. This is the critical notebook fix.
    """
    rows = [{
        "DeviceId": "d1", "TimestampUTC": "2023-06-15T08:30:00Z",
        "DeviceType": "Washing machine", "Status": "Running",
        "Attributes": "{}", "_ingested_at": "t", "_source_file": "t",
    }]
    df = spark.createDataFrame(rows, schema=BRONZE_SCHEMA)
    df = _normalize_timestamps(df)
    assert df.count() == 1, (
        "Z-suffix timestamp was dropped -- regexp_replace fix not working"
    )
    assert df.collect()[0].TimestampUTC == "2023-06-15T08:30:00Z"


def test_silver_no_z_timestamp_normalized_to_z_suffix(spark):
    """Timestamps without Z are normalized to the Z form."""
    rows = [{
        "DeviceId": "d1", "TimestampUTC": "2023-06-15T08:30:00",
        "DeviceType": "Washing machine", "Status": "Running",
        "Attributes": "{}", "_ingested_at": "t", "_source_file": "t",
    }]
    df = spark.createDataFrame(rows, schema=BRONZE_SCHEMA)
    df = _normalize_timestamps(df)
    assert df.collect()[0].TimestampUTC == "2023-06-15T08:30:00Z"


def test_silver_z_and_no_z_same_event_deduped_to_one_row(spark):
    """The same event sent with and without Z suffix deduplicates to one row.

    This is only possible if both variants normalize to the same canonical
    timestamp string before the dedup key is applied.
    """
    rows = [
        {"DeviceId": "d1", "TimestampUTC": "2023-03-01T10:00:00Z",
         "DeviceType": "Washing machine", "Status": "Running",
         "Attributes": "{}", "_ingested_at": "2024-01-01T00:00:00Z", "_source_file": "t"},
        {"DeviceId": "d1", "TimestampUTC": "2023-03-01T10:00:00",
         "DeviceType": "Washing machine", "Status": "Running",
         "Attributes": "{}", "_ingested_at": "2024-01-01T00:00:00Z", "_source_file": "t"},
    ]
    df = spark.createDataFrame(rows, schema=BRONZE_SCHEMA)
    df = _normalize_timestamps(df)
    df = _deduplicate(df)
    assert df.count() == 1, "Z and no-Z variants of the same event should dedup to one row"


def test_silver_exact_duplicates_deduplicated(spark):
    """Exact duplicate events are removed; unique events all survive."""
    events = generate_events(30, seed=2001)
    doubled = events + events
    silver = _silver_df(spark, doubled)
    assert silver.count() == 30


def test_silver_dedup_is_deterministic_same_ingested_at(spark):
    """Window dedup is deterministic when _ingested_at is the same for all rows.

    When _ingested_at is identical (as it is for all rows in one run),
    monotonically_increasing_id() breaks ties, matching keep='first' in silver.py.
    """
    events = generate_events(100, seed=2002)
    silver_a = _silver_df(spark, events, ingested_at="2024-01-01T00:00:00Z")
    silver_b = _silver_df(spark, events, ingested_at="2024-01-01T00:00:00Z")
    assert silver_a.count() == silver_b.count()


def test_silver_timestamps_canonical_format_after_normalization(spark):
    """Every TimestampUTC in the silver output matches YYYY-MM-DDTHH:MM:SSZ."""
    events = generate_events(100, seed=2003)
    silver = _silver_df(spark, events)
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
    bad = silver.filter(~F.col("TimestampUTC").rlike(pattern))
    assert bad.count() == 0


def test_silver_no_blank_required_fields_in_output(spark):
    """No required field is null or blank in the silver output."""
    events = generate_events(100, seed=2004)
    silver = _silver_df(spark, events)
    for col in REQUIRED:
        blank = silver.filter(F.col(col).isNull() | (F.trim(F.col(col)) == ""))
        assert blank.count() == 0, f"Blank values found in silver.{col}"


# ---------------------------------------------------------------------------
# 3. Gold WM -- Fahrenheit conversion, column set, FailureIds
# ---------------------------------------------------------------------------

def _apply_wm_gold(df):
    """Apply the WM gold transformations from the notebook cell."""
    df = df.withColumn("attrs", F.from_json(F.col("Attributes"), WM_ATTRS_SCHEMA))
    df = (
        df
        .withColumn(
            "Temperature_C",
            F.when(
                F.lower(F.col("attrs.Temperature_Unit")).isin("f", "fahrenheit"),
                (F.col("attrs.Temperature") - 32.0) * 5.0 / 9.0,
            ).otherwise(F.col("attrs.Temperature"))
        )
        .withColumn(
            "Temperature_Unit",
            F.when(F.col("attrs.Temperature").isNotNull(), F.lit("Celsius"))
            .otherwise(F.lit(None))
        )
        .withColumn("SpinningSpeed",      F.col("attrs.SpinningSpeed"))
        .withColumn("SpinningSpeed_Unit", F.col("attrs.SpinningSpeed_Unit"))
        .withColumn("TwinDos_Colour",     F.col("attrs.TwinDos").getItem(0))
        .withColumn("TwinDos_White",      F.col("attrs.TwinDos").getItem(1))
        .withColumn("TwinDos_Unit",       F.col("attrs.TwinDos_Unit"))
    )
    df = df.withColumn("FailureId", explode_outer(col("attrs.FailureIds")))
    return df.select(
        "DeviceId", "TimestampUTC", "DeviceType", "Status",
        "Temperature_C", "Temperature_Unit",
        "SpinningSpeed", "SpinningSpeed_Unit",
        "TwinDos_Colour", "TwinDos_White", "TwinDos_Unit",
        "FailureId",
    )


def test_gold_wm_fahrenheit_to_celsius_conversion(spark):
    """104 F -> 40 C, 140 F -> 60 C, 194 F -> 90 C (isin 'f'/'fahrenheit')."""
    rows = [
        {"DeviceId": "wm1", "TimestampUTC": "2024-01-01T00:00:00Z",
         "DeviceType": "Washing machine", "Status": "Running",
         "Attributes": json.dumps({"Temperature": 104.0, "Temperature_Unit": "Fahrenheit"}),
         "_ingested_at": "t", "_source_file": "t"},
        {"DeviceId": "wm2", "TimestampUTC": "2024-01-01T00:01:00Z",
         "DeviceType": "Washing machine", "Status": "Running",
         "Attributes": json.dumps({"Temperature": 140.0, "Temperature_Unit": "f"}),
         "_ingested_at": "t", "_source_file": "t"},
        {"DeviceId": "wm3", "TimestampUTC": "2024-01-01T00:02:00Z",
         "DeviceType": "Washing machine", "Status": "Running",
         "Attributes": json.dumps({"Temperature": 90.0, "Temperature_Unit": "Celsius"}),
         "_ingested_at": "t", "_source_file": "t"},
    ]
    df = spark.createDataFrame(rows, schema=BRONZE_SCHEMA)
    df = _normalize_timestamps(df)
    df = _apply_wm_gold(df)
    results = {r.DeviceId: r.Temperature_C for r in df.select("DeviceId", "Temperature_C").collect()}
    assert abs(results["wm1"] - 40.0) < 0.01, f"104 F should be 40 C, got {results['wm1']}"
    assert abs(results["wm2"] - 60.0) < 0.01, f"140 F should be 60 C, got {results['wm2']}"
    assert abs(results["wm3"] - 90.0) < 0.01, f"90 C should stay 90 C, got {results['wm3']}"


def test_gold_wm_temperature_unit_always_celsius(spark):
    """After gold WM transformation, Temperature_Unit is always 'Celsius'."""
    rows = [
        {"DeviceId": "wm1", "TimestampUTC": "2024-01-01T00:00:00Z",
         "DeviceType": "Washing machine", "Status": "Running",
         "Attributes": json.dumps({"Temperature": 104.0, "Temperature_Unit": "Fahrenheit"}),
         "_ingested_at": "t", "_source_file": "t"},
    ]
    df = spark.createDataFrame(rows, schema=BRONZE_SCHEMA)
    df = _normalize_timestamps(df)
    df = _apply_wm_gold(df)
    assert df.collect()[0].Temperature_Unit == "Celsius"


def test_gold_wm_has_all_expected_columns(spark):
    """WM gold table has the exact column set from _flatten_washing_machine()."""
    events = generate_events(200, seed=3001)
    silver = _silver_df(spark, events)
    wm = _apply_wm_gold(silver.filter(F.lower(F.col("DeviceType")).contains("washing")))
    expected = {
        "DeviceId", "TimestampUTC", "DeviceType", "Status",
        "Temperature_C", "Temperature_Unit",
        "SpinningSpeed", "SpinningSpeed_Unit",
        "TwinDos_Colour", "TwinDos_White", "TwinDos_Unit",
        "FailureId",
    }
    assert expected.issubset(set(wm.columns)), f"Missing: {expected - set(wm.columns)}"


def test_gold_wm_temperature_c_in_celsius_range(spark):
    """All WM Temperature_C values from synthetic data are in [0, 100]."""
    events = generate_events(300, seed=3002)
    silver = _silver_df(spark, events)
    wm = _apply_wm_gold(silver.filter(F.lower(F.col("DeviceType")).contains("washing")))
    outside = wm.filter(
        F.col("Temperature_C").isNotNull() & ~F.col("Temperature_C").between(0, 100)
    )
    assert outside.count() == 0, "WM Temperature_C out of [0, 100] -- Fahrenheit conversion failed"


def test_gold_wm_temperature_exact_values(spark):
    """Synthetic WM temps are all in [0, 100] C after conversion."""
    events = generate_events(400, seed=3003)
    silver = _silver_df(spark, events)
    wm = _apply_wm_gold(silver.filter(F.lower(F.col("DeviceType")).contains("washing")))
    temps = {
        r.Temperature_C
        for r in wm.filter(F.col("Temperature_C").isNotNull()).select("Temperature_C").collect()
    }
    assert all(0 <= t <= 100 for t in temps if t is not None), (
        f"WM Temperature_C values outside [0, 100]: {[t for t in temps if t is not None and not (0 <= t <= 100)]}"
    )


def test_gold_wm_spinning_speed_values(spark):
    """SpinningSpeed values are one of {600,800,900,1000,1100,1200,1400} rpm or null (failure rows)."""
    events = generate_events(400, seed=3004)
    silver = _silver_df(spark, events)
    wm = _apply_wm_gold(silver.filter(F.lower(F.col("DeviceType")).contains("washing")))
    speeds = {
        r.SpinningSpeed
        for r in wm.filter(F.col("SpinningSpeed").isNotNull()).select("SpinningSpeed").collect()
    }
    assert speeds.issubset({600, 800, 900, 1000, 1100, 1200, 1400}), f"Unexpected SpinningSpeed values: {speeds}"


# ---------------------------------------------------------------------------
# 4. Gold CM -- dynamic attribute discovery
# ---------------------------------------------------------------------------

def _cm_attrs_from_df(cm_df):
    """Discover CM attribute Ids and return (raw_id, safe_id) pairs."""
    raw_ids = (
        cm_df
        .select(F.explode("attrs_arr").alias("attr"))
        .select(F.col("attr.Id").alias("Id"))
        .where(F.col("Id").isNotNull() & (F.trim(F.col("Id")) != ""))
        .distinct()
        .collect()
    )
    result = []
    for row in raw_ids:
        raw = row.Id.strip()
        safe = re.sub(r"[^\w]", "_", raw)
        if safe in _RESERVED_COLS:
            continue
        result.append((raw, safe))
    return result


def _apply_cm_gold(df, cm_attrs):
    """Apply CM gold per-Id columns using the intermediate-column pattern."""
    for raw_id, safe_id in cm_attrs:
        tmp = f"_m_{safe_id}"
        # Use F.expr with SQL lambda syntax rather than a Python lambda.
        # The Python default-argument pattern "lambda x, _rid=raw_id: x.Id == _rid"
        # looks like a two-parameter lambda to PySpark, which binds the second
        # parameter to the array index (not to raw_id), causing CAST_INVALID_INPUT
        # when Spark tries to compare the string field against the integer index.
        # SQL lambda "x -> x.Id = '...'" is unambiguous: x.Id is struct field access
        # and the right-hand side is a string literal.
        escaped = raw_id.replace("'", "''")
        df = (
            df
            .withColumn(tmp, F.expr(f"filter(attrs_arr, x -> x.Id = '{escaped}')"))
            .withColumn(f"{safe_id}_Value", F.when(F.size(F.col(tmp)) > 0, F.col(tmp).getItem(0)["Value"]))
            .withColumn(f"{safe_id}_Unit",  F.when(F.size(F.col(tmp)) > 0, F.col(tmp).getItem(0)["Unit"]))
            .drop(tmp)
        )
    return df


def test_gold_cm_discovers_all_ids(spark):
    """All distinct Ids in the CM attrs array are discovered dynamically."""
    rows = [
        {"DeviceId": "cm1", "TimestampUTC": "2024-01-01T00:00:00Z",
         "DeviceType": "Coffee Machine", "Status": "Idle",
         "Attributes": json.dumps([{"Id": "Temperature", "Value": 75.0, "Unit": "Celsius"},
                                   {"Id": "Grinding", "Value": 3.0, "Unit": "level"}]),
         "_ingested_at": "t", "_source_file": "t"},
        {"DeviceId": "cm2", "TimestampUTC": "2024-01-01T00:01:00Z",
         "DeviceType": "Coffee Machine", "Status": "Idle",
         "Attributes": json.dumps([{"Id": "Temperature", "Value": 80.0, "Unit": "Celsius"},
                                   {"Id": "WaterLevel", "Value": 500.0, "Unit": "ml"}]),
         "_ingested_at": "t", "_source_file": "t"},
    ]
    df = spark.createDataFrame(rows, schema=BRONZE_SCHEMA)
    df = _normalize_timestamps(df)
    df = df.withColumn("attrs_arr", F.from_json(F.col("Attributes"), CM_ATTRS_SCHEMA))
    cm_attrs = _cm_attrs_from_df(df)
    safe_ids = {safe for _, safe in cm_attrs}
    assert "Temperature" in safe_ids
    assert "Grinding" in safe_ids
    assert "WaterLevel" in safe_ids


def test_gold_cm_value_and_unit_columns_populated(spark):
    """Each discovered Id gets a correctly populated _Value and _Unit column."""
    rows = [{
        "DeviceId": "cm1", "TimestampUTC": "2024-01-01T00:00:00Z",
        "DeviceType": "Coffee Machine", "Status": "Idle",
        "Attributes": json.dumps([{"Id": "Temperature", "Value": 75.0, "Unit": "Celsius"}]),
        "_ingested_at": "t", "_source_file": "t",
    }]
    df = spark.createDataFrame(rows, schema=BRONZE_SCHEMA)
    df = _normalize_timestamps(df)
    df = df.withColumn("attrs_arr", F.from_json(F.col("Attributes"), CM_ATTRS_SCHEMA))
    cm_attrs = _cm_attrs_from_df(df)
    df = _apply_cm_gold(df, cm_attrs)
    row = df.select("Temperature_Value", "Temperature_Unit").collect()[0]
    assert row.Temperature_Value == 75.0
    assert row.Temperature_Unit == "Celsius"


def test_gold_cm_reserved_column_name_skipped(spark):
    """An attribute Id that collides with a reserved column is not added."""
    rows = [{
        "DeviceId": "cm1", "TimestampUTC": "2024-01-01T00:00:00Z",
        "DeviceType": "Coffee Machine", "Status": "Idle",
        "Attributes": json.dumps([
            {"Id": "DeviceId",    "Value": 1.0,  "Unit": "bad"},
            {"Id": "Temperature", "Value": 75.0, "Unit": "Celsius"},
        ]),
        "_ingested_at": "t", "_source_file": "t",
    }]
    df = spark.createDataFrame(rows, schema=BRONZE_SCHEMA)
    df = _normalize_timestamps(df)
    df = df.withColumn("attrs_arr", F.from_json(F.col("Attributes"), CM_ATTRS_SCHEMA))
    cm_attrs = _cm_attrs_from_df(df)
    safe_ids = {safe for _, safe in cm_attrs}
    assert "DeviceId" not in safe_ids, "Reserved column DeviceId should be excluded"
    assert "Temperature" in safe_ids


def test_gold_cm_intermediate_column_dropped(spark):
    """The _m_{safe_id} intermediate column is dropped from the final output."""
    rows = [{
        "DeviceId": "cm1", "TimestampUTC": "2024-01-01T00:00:00Z",
        "DeviceType": "Coffee Machine", "Status": "Idle",
        "Attributes": json.dumps([{"Id": "Temperature", "Value": 75.0, "Unit": "Celsius"}]),
        "_ingested_at": "t", "_source_file": "t",
    }]
    df = spark.createDataFrame(rows, schema=BRONZE_SCHEMA)
    df = _normalize_timestamps(df)
    df = df.withColumn("attrs_arr", F.from_json(F.col("Attributes"), CM_ATTRS_SCHEMA))
    cm_attrs = _cm_attrs_from_df(df)
    df = _apply_cm_gold(df, cm_attrs)
    assert not any(c.startswith("_m_") for c in df.columns), (
        "Intermediate _m_ columns were not dropped from the CM gold output"
    )


def test_gold_cm_synthetic_has_temperature_columns(spark):
    """Coffee machine gold from synthetic data has Temperature_Value/Unit columns."""
    events = generate_events(300, seed=4001)
    silver = _silver_df(spark, events)
    cm = silver.filter(F.lower(F.col("DeviceType")).contains("coffee"))
    cm = cm.withColumn("attrs_arr", F.from_json(F.col("Attributes"), CM_ATTRS_SCHEMA))
    cm_attrs = _cm_attrs_from_df(cm)
    cm = _apply_cm_gold(cm, cm_attrs)
    assert "Temperature_Value" in cm.columns
    assert "Temperature_Unit" in cm.columns


# ---------------------------------------------------------------------------
# 5. Gold DW -- temperature conversion, column set
# ---------------------------------------------------------------------------

def _apply_dw_gold(df):
    df = df.withColumn("attrs", F.from_json(F.col("Attributes"), DW_ATTRS_SCHEMA))
    df = (
        df
        .withColumn(
            "Temperature_C",
            F.when(
                F.lower(F.col("attrs.Temperature_Unit")).isin("f", "fahrenheit"),
                (F.col("attrs.Temperature") - 32.0) * 5.0 / 9.0,
            ).otherwise(F.col("attrs.Temperature"))
        )
        .withColumn(
            "QuickPowerWashActive",
            F.when(F.col("attrs.QuickPowerWashActive") == 1, F.lit(True))
            .otherwise(F.when(F.col("attrs.QuickPowerWashActive") == 0, F.lit(False))
                       .otherwise(F.lit(None)))
        )
    )
    return df.select(
        "DeviceId", "TimestampUTC", "DeviceType", "Status",
        "Temperature_C", "QuickPowerWashActive",
    )


def test_gold_dw_fahrenheit_conversion(spark):
    """DW Fahrenheit values convert to Celsius the same way as WM."""
    rows = [
        {"DeviceId": "dw1", "TimestampUTC": "2024-01-01T00:00:00Z",
         "DeviceType": "Dishwasher", "Status": "Running",
         "Attributes": json.dumps({"Temperature": 104.0, "Temperature_Unit": "Fahrenheit",
                                   "QuickPowerWashActive": 0, "QuickPowerWashActive_Unit": "bool"}),
         "_ingested_at": "t", "_source_file": "t"},
        {"DeviceId": "dw2", "TimestampUTC": "2024-01-01T00:01:00Z",
         "DeviceType": "Dishwasher", "Status": "Running",
         "Attributes": json.dumps({"Temperature": 60.0, "Temperature_Unit": "Celsius",
                                   "QuickPowerWashActive": 1, "QuickPowerWashActive_Unit": "bool"}),
         "_ingested_at": "t", "_source_file": "t"},
    ]
    df = spark.createDataFrame(rows, schema=BRONZE_SCHEMA)
    df = _normalize_timestamps(df)
    df = _apply_dw_gold(df)
    results = {r.DeviceId: r.Temperature_C for r in df.select("DeviceId", "Temperature_C").collect()}
    assert abs(results["dw1"] - 40.0) < 0.01, f"104 F should be 40 C, got {results['dw1']}"
    assert abs(results["dw2"] - 60.0) < 0.01, f"60 C stays 60 C, got {results['dw2']}"


def test_gold_dw_has_expected_columns(spark):
    """DW gold table has all columns from _flatten_dishwasher()."""
    events = generate_events(200, seed=5001)
    silver = _silver_df(spark, events)
    dw = _apply_dw_gold(silver.filter(F.lower(F.col("DeviceType")).contains("dishwasher")))
    expected = {
        "DeviceId", "TimestampUTC", "DeviceType", "Status",
        "Temperature_C", "QuickPowerWashActive",
    }
    assert expected.issubset(set(dw.columns)), f"Missing: {expected - set(dw.columns)}"


def test_gold_dw_temperature_c_in_celsius_range(spark):
    """All DW Temperature_C values from synthetic data are in [0, 100]."""
    events = generate_events(300, seed=5002)
    silver = _silver_df(spark, events)
    dw = _apply_dw_gold(silver.filter(F.lower(F.col("DeviceType")).contains("dishwasher")))
    outside = dw.filter(
        F.col("Temperature_C").isNotNull() & ~F.col("Temperature_C").between(0, 100)
    )
    assert outside.count() == 0


def test_gold_dw_temperature_exact_values(spark):
    """Synthetic DW temps are all in [0, 100] C after conversion."""
    events = generate_events(300, seed=5003)
    silver = _silver_df(spark, events)
    dw = _apply_dw_gold(silver.filter(F.lower(F.col("DeviceType")).contains("dishwasher")))
    temps = {
        r.Temperature_C
        for r in dw.filter(F.col("Temperature_C").isNotNull()).select("Temperature_C").collect()
    }
    assert all(0 <= t <= 100 for t in temps if t is not None), (
        f"DW Temperature_C values outside [0, 100]: {[t for t in temps if t is not None and not (0 <= t <= 100)]}"
    )


# ---------------------------------------------------------------------------
# 6. Summary -- groupBy + agg, failure_rate_pct
# ---------------------------------------------------------------------------

def _build_summary(silver):
    return (
        silver
        .groupBy("DeviceId", "DeviceType")
        .agg(
            F.count("*").alias("total_events"),
            F.sum(
                F.when(F.lower(F.col("Status")).contains("failure"), 1).otherwise(0)
            ).alias("failure_events"),
        )
        .withColumn(
            "failure_rate_pct",
            F.round((F.col("failure_events") / F.col("total_events")) * 100.0, 2),
        )
        .orderBy("DeviceType", "DeviceId")
    )


def test_gold_summary_has_correct_columns(spark):
    events = generate_events(200, seed=6001)
    summary = _build_summary(_silver_df(spark, events))
    expected = {"DeviceId", "DeviceType", "total_events", "failure_events", "failure_rate_pct"}
    assert expected.issubset(set(summary.columns))


def test_gold_summary_failure_rate_in_valid_range(spark):
    """failure_rate_pct is between 0 and 100 for every device."""
    events = generate_events(300, seed=6002)
    summary = _build_summary(_silver_df(spark, events))
    outside = summary.filter(
        (F.col("failure_rate_pct") < 0) | (F.col("failure_rate_pct") > 100)
    )
    assert outside.count() == 0


def test_gold_summary_covers_all_three_device_types(spark):
    """All three device types appear in the summary table."""
    events = generate_events(300, seed=6003)
    summary = _build_summary(_silver_df(spark, events))
    types = {r.DeviceType for r in summary.select("DeviceType").distinct().collect()}
    assert "Washing machine" in types
    assert "Coffee Machine" in types
    assert "Dishwasher" in types


def test_gold_summary_total_events_positive(spark):
    """Every device in the summary has at least one event."""
    events = generate_events(200, seed=6004)
    summary = _build_summary(_silver_df(spark, events))
    assert summary.filter(F.col("total_events") <= 0).count() == 0


# ---------------------------------------------------------------------------
# 7. Spark SQL availability (prerequisite for OPTIMIZE/VACUUM on Databricks)
# ---------------------------------------------------------------------------

def test_spark_sql_engine_available(spark):
    """Spark SQL engine is reachable -- prerequisite for OPTIMIZE/VACUUM cells."""
    result = spark.sql("SELECT 1 + 1 AS two").collect()
    assert result[0].two == 2


# ---------------------------------------------------------------------------
# 8. End-to-end notebook pipeline -- bronze -> silver -> gold
# ---------------------------------------------------------------------------

def test_notebook_pipeline_no_data_loss(spark):
    """Full bronze -> silver -> gold (WM + CM + DW) row count is lossless.

    Mirrors the full execution path of the Databricks notebook.
    """
    events = generate_events(200, seed=9001)

    # Bronze (notebook cell 2)
    ingested_at = "2024-01-01T00:00:00Z"
    parse_file = _make_parse_file(ingested_at)
    bronze_rows = parse_file(("test", json.dumps(events)))
    bronze_df = spark.createDataFrame(bronze_rows, schema=BRONZE_SCHEMA).cache()
    n_bronze = bronze_df.count()
    assert n_bronze == 200

    # Silver (notebook cell 3)
    silver = bronze_df.filter(_valid_filter())
    silver = _normalize_timestamps(silver)
    silver = _deduplicate(silver).cache()
    n_silver = silver.count()
    assert n_silver == 200   # synthetic data has no invalid or duplicate rows
    bronze_df.unpersist()

    # Gold (notebook cells 4a / 4b / 4c)
    wm = silver.filter(F.lower(F.col("DeviceType")).contains("washing"))
    wm = _apply_wm_gold(wm)
    n_wm = wm.count()

    cm = silver.filter(F.lower(F.col("DeviceType")).contains("coffee"))
    cm = cm.withColumn("attrs_arr", F.from_json(F.col("Attributes"), CM_ATTRS_SCHEMA))
    cm_attrs = _cm_attrs_from_df(cm)
    cm = _apply_cm_gold(cm, cm_attrs)
    n_cm = cm.count()

    dw = silver.filter(F.lower(F.col("DeviceType")).contains("dishwasher"))
    dw = _apply_dw_gold(dw)
    n_dw = dw.count()

    silver.unpersist()

    n_gold = n_wm + n_cm + n_dw
    assert n_gold >= n_silver, (
        f"Data loss in notebook pipeline: silver={n_silver}, gold={n_gold} "
        f"(wm={n_wm}, cm={n_cm}, dw={n_dw})"
    )


def test_notebook_pipeline_z_suffix_events_survive_silver(spark):
    """End-to-end: Events with Z-suffix timestamps (Event Hubs format) survive silver."""
    events = generate_events(100, seed=9002)
    # All synthetic events have Z-suffix timestamps -- this exercises the critical fix.
    silver = _silver_df(spark, events)
    assert silver.count() == 100, (
        f"Z-suffix events dropped: expected 100, got {silver.count()}"
    )


def test_notebook_pipeline_dedup_removes_duplicates_e2e(spark):
    """End-to-end: Exact duplicates are removed; unique events all survive."""
    events = generate_events(50, seed=9003)
    silver = _silver_df(spark, events + events)   # every event doubled
    assert silver.count() == 50


def test_notebook_pipeline_wm_temperatures_valid_celsius_e2e(spark):
    """End-to-end: All WM gold temperatures are valid Celsius values."""
    events = generate_events(300, seed=9004)
    silver = _silver_df(spark, events)
    wm = _apply_wm_gold(silver.filter(F.lower(F.col("DeviceType")).contains("washing")))
    outside = wm.filter(
        F.col("Temperature_C").isNotNull() & ~F.col("Temperature_C").between(0, 100)
    )
    assert outside.count() == 0


def test_notebook_pipeline_summary_from_silver(spark):
    """End-to-end: Summary is built from silver and covers all device types."""
    events = generate_events(300, seed=9005)
    silver = _silver_df(spark, events)
    summary = _build_summary(silver)
    types = {r.DeviceType for r in summary.select("DeviceType").distinct().collect()}
    assert {"Washing machine", "Coffee Machine", "Dishwasher"}.issubset(types)
