# IoT Case Study - Data Engineering Pipeline

## Overview

This repo has a local, runnable IoT pipeline for home appliances. You can
run it on a laptop to see how it works, and the same logic can be scaled up to
Azure Databricks when you need to handle real production volumes.

The pipeline reads JSON events from devices (washing machines, coffee machines,
dishwashers) and runs them through a three-layer medallion architecture:
bronze (raw ingest), silver (cleansing and deduplication), gold (BI-ready tables).

The authoritative reference is `notebooks/IoT case study 150326_0002.ipynb`.
All local Python / Pandas code mirrors that notebook's logic exactly.

---

## Repository Structure

- `data/sample_input.json` - sample JSON input from the case study
- `src/bronze.py` - bronze layer: raw ingest + dead-letter routing
- `src/silver.py` - silver layer: validation, timestamp normalization, deduplication
- `src/gold.py` - gold layer: per-device flattening + summary table
- `src/run_pipeline.py` - orchestrates bronze -> silver -> gold in one command
- `src/generate_synthetic.py` - generates synthetic data for load testing
- `tests/test_generate_synthetic.py` - pytest suite for the data generator
- `tests/test_layers.py` - pytest suite for bronze, silver, gold unit tests
- `tests/test_silver_ml.py` - pytest suite for silver fuzzy-dedup (recordlinkage)
- `tests/test_notebook_spark.py` - pytest suite for the Databricks notebook PySpark logic
- `tests/test_integration_e2e.py` - end-to-end integration tests (synthetic data to gold)
- `notebooks/IoT case study 150326_0002.ipynb` - full PySpark notebook (bronze to gold)
- `azure-pipelines.yml` - Azure DevOps CI/CD pipeline
- `pyproject.toml` - package metadata and entry points
- `README.md` - this file

---

## Local Setup (Laptop)

### 1) Prerequisites

- Python 3.10 or newer (3.11+ recommended)
- Git

### 2) Create a Virtual Environment

```bash
python -m venv .venv
```

On Linux / macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bat
.venv\Scripts\activate
```

### 3) Install Dependencies

```bash
pip install -e .[spark,ml,test]
```

Or without PySpark (for running just the Pandas pipeline):

```bash
pip install -e .[ml,test]
```

Or minimal, without ML or PySpark:

```bash
pip install -e .[test]
```

The `ml` extra installs recordlinkage, RapidFuzz, scikit-learn, and NumPy for
the silver layer fuzzy deduplication (see Silver ML section below). Without this
extra the fuzzy-dedup pass is silently skipped and silver still works.

Note: if you skip `pip install -e .`, pytest still works because `tests/conftest.py`
adds the repo root to the Python path automatically.

---

## Running the Pipeline

### Medallion layers (bronze -> silver -> gold)

Runs all three layers in sequence:

```bash
python -m src.run_pipeline --input data/sample_input.json --output-dir output
```

Or after `pip install -e .`:

```bash
iot-pipeline-layers --input data/sample_input.json --output-dir output
```

Output structure:

- `output/bronze/<stem>.csv` - raw records plus `_ingested_at` and `_source_file`
- `output/bronze/<stem>_dead_letter.csv` - non-serializable input records
- `output/silver/<stem>.csv` - validated, deduplicated, normalized, with `_silver_processed_at`
- `output/silver/<stem>_dead_letter.csv` - rows with missing/blank required fields
- `output/gold/gold_washing_machine.csv` - one row per FailureId (or null for non-failure)
- `output/gold/gold_coffee_machine.csv` - dynamic attribute columns
- `output/gold/gold_dishwasher.csv` - QuickPowerWashActive as boolean
- `output/gold/gold_summary.csv` - event counts, failure rates, active days per device

---

## The Medallion Layers

### Bronze

Reads the raw JSON array, adds two metadata columns, and stores `Attributes` as a
JSON string so the schema is uniform across device types (washing machine has a dict,
coffee machine has a list).

- `_ingested_at` - UTC timestamp when the file was processed
- `_source_file` - absolute path to the source JSON file

Records that are not JSON-serializable (datetime objects, circular references, custom
objects) are routed to a `_dead_letter.csv` file instead of crashing the ingest.
`ingest()` returns an `IngestResult(rows_written, dead_letter_count)` NamedTuple.

### Silver

Quality gate before anything goes downstream.

- Drops rows where any of `DeviceId`, `TimestampUTC`, `DeviceType`, `Status` is null
  or blank (whitespace-only). Invalid rows are written to a `_dead_letter.csv` with
  `_rejection_reason` and `_silver_processed_at` columns.
- Normalizes `TimestampUTC` to a canonical UTC ISO-8601 string
  (`2023-01-01T07:00:00Z`) so that events with and without the `Z` suffix
  are treated as the same event during deduplication.
- Deduplicates on `(DeviceId, TimestampUTC, DeviceType, Status)`, keeping the first
  occurrence. This removes retry duplicates that IoT devices commonly produce.
- Adds `_silver_processed_at` audit column (same UTC value for all rows in one call).
- Returns a `CleanseResult` with row counts so callers can alert when the drop rate
  is unexpectedly high.
- When the `ml` extra is installed, also runs fuzzy deduplication using recordlinkage
  and RapidFuzz (see Silver ML section below).

### Gold

Parses the `Attributes` JSON string back to typed fields and flattens each device
type into a wide table. Also writes a summary table with per-device KPIs.

#### Washing Machine

- Temperature converted to Celsius (Fahrenheit input converted automatically).
- `TwinDos` split into `TwinDos_Colour` and `TwinDos_White`.
- Failure events are **exploded**: one row per FailureId (matches PySpark `explode_outer`).
  Non-failure events have `FailureId = null`.

#### Coffee Machine

- `Attributes` is a list of `{Id, Value, Unit}` objects. Each Id discovered across
  all rows becomes a `{Id}_Value` / `{Id}_Unit` column pair.
- Reserved-column collision guard (`DeviceId`, `TimestampUTC`, etc.) with warning.

#### Dishwasher

- Temperature converted to Celsius.
- `QuickPowerWashActive` cast from raw Bit (0/1) to Python `bool` (False/True).
- Failure events exploded the same way as washing machines.

#### Summary table

Per-device KPIs: `total_events`, `failure_events`, `failure_rate_pct`,
`first_seen_utc`, `last_seen_utc`, `days_active`.

---

## Silver ML / Fuzzy Deduplication

When the `ml` extra is installed (`pip install -e .[ml]`), silver runs a fuzzy
deduplication pass after exact dedup using recordlinkage and RapidFuzz.

Approach (mirrors the notebook):
1. `recordlinkage.Index().block("DeviceId")` generates candidate pairs only within
   the same device, avoiding an O(n²) full cross-join.
2. `recordlinkage.Compare()` checks `DeviceType` and `TimestampUTC` exactly.
3. `fuzz.token_sort_ratio()` scores `Status` similarity for each candidate pair.
   Catches "Program start" vs "Program Start" casing variants.
4. Pairs where all exact fields match AND status similarity ≥ 90 are duplicates.
   The record with the higher pandas index (arrived later) is **dropped**.

`CleanseResult.dropped_fuzzy` reports how many rows were removed. Without the ml
extra, `dropped_fuzzy` is always 0.

---

## Synthetic Data Generation

Generate a synthetic JSON file for load testing:

```bash
python -m src.generate_synthetic --records 100000 --output data/synthetic.json
```

Timestamps span the last 365 days by default. Use `--days` to change the window:

```bash
python -m src.generate_synthetic --records 100000 --days 90 --output data/synthetic.json
```

Pass `--seed` for a reproducible file (same seed + same calendar day = same output):

```bash
python -m src.generate_synthetic --records 100000 --seed 42 --output data/synthetic.json
```

Device distribution: ~50% washing machine, ~30% coffee machine, ~20% dishwasher.
Device IDs use the `syn_` prefix + 13 random lowercase-alphanumeric characters so
they can be distinguished from real `initial_data` device IDs in Delta table queries.

Status weights match real fleet ratios:
- Washing machine: 40% Program Start, 45% Program End, 15% Program Failure
- Coffee machine: 35% Program Start, 55% Program End, 10% Program Failure
- Dishwasher: 38% Program Start, 50% Program End, 12% Program Failure

Run the medallion pipeline on synthetic data:

```bash
python -m src.run_pipeline --input data/synthetic.json --output-dir output/synthetic
```

---

## Running the Tests

```bash
pytest -q
```

The suite covers all device types and edge cases: temperature conversion,
TwinDos, failure records (WM and DW explode), unknown device types, bad timestamps,
column injection guards, non-string field validation, bronze dead-letter routing,
bronze metadata columns, silver validation and deduplication, silver dead-letter CSV,
gold flattening and summary (including new `first_seen_utc` / `last_seen_utc` /
`days_active` columns), end-to-end pipeline integration, silver fuzzy deduplication
(recordlinkage + RapidFuzz), and the Databricks notebook PySpark transformations
cell by cell.

To run only the fuzzy-dedup ML tests (requires the ml extra):

```bash
pytest tests/test_silver_ml.py -q
```

To run only the notebook PySpark tests (requires the spark extra):

```bash
pytest tests/test_notebook_spark.py -q
```

---

## CI/CD (Azure DevOps)

`azure-pipelines.yml` at the repo root defines a pipeline that runs on every
push to `main` and on every pull request targeting `main`.

What it does:
- Sets up Python 3.11
- Installs the package and test dependencies (`pip install -e .[test]`)
- Runs the full pytest suite
- Publishes results to the Azure DevOps test tab (visible on the PR)

To enable it: go to Azure DevOps -> Pipelines -> New pipeline -> Azure Repos Git
-> pick this repo -> "Existing Azure Pipelines YAML file" -> select
`/azure-pipelines.yml`.

The file also has a commented-out `Deploy` stage that uploads the notebook to the
Databricks workspace and triggers a job after a successful merge to main. Fill in
the `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, and `DATABRICKS_JOB_ID` pipeline
variables (as secrets in a variable group, never in the file itself) to activate it.

---

## Databricks Notebook (PySpark)

`notebooks/IoT case study 150326_0002.ipynb` has the full medallion pipeline in
PySpark. It is the authoritative reference; all local Python code mirrors its logic.

```bash
jupyter lab "notebooks/IoT case study 150326_0002.ipynb"
```

What the notebook covers:

- Bronze: `wholeTextFiles` reads JSON arrays, `safe_serialize` dead-letter routing for
  non-serializable records, `Attributes` stored as JSON string
- Silver: required-field filter, Z-suffix-safe timestamp normalization with
  `regexp_replace + to_timestamp + to_utc_timestamp`, `row_number()` window-function
  dedup, recordlinkage + RapidFuzz fuzzy dedup pass
- Data quality monitoring: configurable thresholds for invalid-field and duplicate
  drop rates; prints alerts and shows where to hook Azure Monitor / Logic App
- Gold WM: `from_json` with typed schema, `isin("f", "fahrenheit")`, `explode_outer`
  for FailureIds (one row per failure code)
- Gold CM: dynamically discovers all Ids in the dataset, produces `{Id}_Value` /
  `{Id}_Unit` column pairs for any Id found, reserved-column collision guard
- Gold DW: temperature conversion, `QuickPowerWashActive` cast from Bit to boolean
- Gold fallback: unknown device types written to `gold_unknown` with raw Attributes
- Summary: `groupBy + agg` for event counts, failure rates, first/last seen, days active
- Delta OPTIMIZE + ZORDER + VACUUM: file compaction, query performance via data
  skipping on `DeviceId`, and storage cost control
- Unity Catalog registration: `CREATE TABLE IF NOT EXISTS` so gold tables are
  queryable from Databricks SQL by name; a `devices_with_failures` view for dashboards

Production path: once the notebook logic is stable, the natural next step is to
lift it to Delta Live Tables (DLT). DLT gives built-in data quality expectations
with pass/fail/quarantine semantics, automatic retries, managed run history, and
lineage tracking. All of those are wired up manually in this notebook.

---

## Scaling to Azure

The local pipeline is designed so the same logic moves to Azure with minimal changes.

### Data flow on Azure

```
IoT Devices
  -> Azure IoT Hub / Event Hubs (streaming ingest)
  -> ADLS Gen2 bronze container (raw JSON or Event Hubs Capture)
  -> Databricks bronze job (wholeTextFiles or readStream)
  -> ADLS Gen2 silver container (Delta table)
  -> Databricks silver job (validate, normalize, dedup)
  -> ADLS Gen2 gold container (Delta tables)
  -> Power BI / Synapse Analytics (reporting)
```

### Storage

- ADLS Gen2 with three containers: bronze, silver, gold.
- Use Delta Lake format for ACID transactions and time travel.
- Paths follow the `abfss://<container>@<account>.dfs.core.windows.net/<path>` scheme.

### Useful Azure Services

- Azure IoT Hub or Event Hubs - streaming device events into the pipeline
- ADLS Gen2 - landing zone and layer storage (bronze / silver / gold containers)
- Azure Databricks - PySpark batch and streaming processing, Delta Lake, SQL warehouse
- Unity Catalog - governance, access control, lineage, and data discovery
- Delta Live Tables - production pipeline framework (built-in DQ, retries, lineage)
- Azure Data Factory - orchestration and scheduling for batch jobs
- Azure Synapse Analytics - SQL queries over Delta tables for large-scale analytics
- Azure Key Vault - secrets, storage keys, service principal credentials
- Azure Monitor / Log Analytics - pipeline health, job-failure alerts, metrics
- Azure DevOps - CI/CD, test reporting, artifact management, deployment

### Monitoring and alerting

The notebook's data quality cell checks silver drop rates against configured thresholds
and prints alerts. In production, connect this to Azure Monitor by:

1. Failing the Databricks job via `dbutils.notebook.exit("ALERT: ...")` when a
   threshold is breached. Azure Monitor can alert on job failure.
2. Sending a POST to an Azure Logic App HTTP trigger for richer routing (email,
   Teams message, PagerDuty).
3. Writing a `quality_log` Delta table and pointing a Databricks SQL alert at it.

### Performance and cost

- OPTIMIZE + ZORDER on `DeviceId` after each gold write: faster device-filtered queries
- VACUUM at 168-hour retention: prevents unbounded storage growth
- Job clusters (not all-purpose) for scheduled runs: auto-terminate when done
- Spot instances on worker nodes: significant cost reduction for batch workloads
- `spark.databricks.delta.optimizeWrite.enabled true` on the cluster: avoids
  small-file accumulation from streaming or incremental loads

### Code and Collaboration

- Keep code in a Git repo (Azure DevOps or GitHub).
- Use feature branches and pull requests.
- `azure-pipelines.yml` runs tests on every PR and can deploy on merge to main.

---

## Ideas for Later

- Schema evolution detection for new attribute keys.
- Streaming JSON parsing with `ijson` to avoid loading large files into memory.
- Delta Lake incremental processing with upserts.
- `nbstripout` pre-commit hook to keep the notebook clean in Git.
- Alerting on pipeline failures or high drop rates.

---

## Notes on the Case Study Data

- `FailureIds` for washing machines uses pool [10, 17, 23, 31, 42, 55].
- `FailureIds` for dishwashers uses pool [5, 11, 14, 22, 33].
- `TwinDos` is only on washing machines with the dosing system (~60% of events).
- Coffee machine `Attributes` is always a list; always includes Temperature.
