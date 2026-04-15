# Lockstep Benchmarks

This directory contains realistic compile/simulate workloads for measuring Lockstep frontend and simulator performance.

## Workloads

### 1) `particle_energy.lock`
* **Intent:** Physics-style particle integration pass with per-particle energy accumulation and a fold over total energy.
* **Fixture:** `fixtures/particle_energy.json` (20,000 source rows).
* **KPI:**
  * `compile_ms` — elapsed milliseconds for `compile_lockstep(...)`.
  * `simulate_ms` — elapsed milliseconds for `simulate_pipeline_entities(...)`.
  * `input_rows_per_sec` — kernel input rows processed per second.
  * `fold_rows_per_sec` — fold input values reduced per second.

### 2) `telemetry_filter_aggregation.lock`
* **Intent:** IoT telemetry stream with filtering semantics (`_keep`) and a downstream aggregation stage that folds totals/counts.
* **Fixture:** `fixtures/telemetry_filter_aggregation.json` (15,000 source rows).
* **KPI:**
  * `compile_ms`
  * `simulate_ms`
  * `input_rows_per_sec`
  * `fold_rows_per_sec`

### 3) `multi_stage_pipeline.lock`
* **Intent:** Multi-stage transform pipeline (`Normalize -> Score -> KeepActive`) with two folds to model aggregate and peak analytics.
* **Fixture:** `fixtures/multi_stage_pipeline.json` (12,000 source rows).
* **KPI:**
  * `compile_ms`
  * `simulate_ms`
  * `input_rows_per_sec`
  * `fold_rows_per_sec`

## Running benchmarks

Use the benchmark driver below, which reuses the same compiler and simulator entry points as the CLI (`compile_lockstep` and `simulate_pipeline_entities`):

```bash
python benchmarks/run_workloads.py
python benchmarks/run_workloads.py --iterations 5
python benchmarks/run_workloads.py --workload particle_energy --json
```
