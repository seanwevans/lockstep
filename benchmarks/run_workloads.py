#!/usr/bin/env python3
"""Benchmark Lockstep compile + simulation throughput across realistic workloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lockstep_compiler.compiler import compile_lockstep
from lockstep_compiler.simulator import simulate_pipeline_entities

BASE_DIR = Path(__file__).resolve().parent
WORKLOAD_DIR = BASE_DIR / "workloads"
FIXTURE_DIR = BASE_DIR / "fixtures"

WORKLOADS = {
    "particle_energy": {
        "program": "particle_energy.lock",
        "fixture": "particle_energy.json",
        "kpi": "compile_ms, simulate_ms, input_rows_per_sec, fold_rows_per_sec",
    },
    "telemetry_filter_aggregation": {
        "program": "telemetry_filter_aggregation.lock",
        "fixture": "telemetry_filter_aggregation.json",
        "kpi": "compile_ms, simulate_ms, input_rows_per_sec, fold_rows_per_sec",
    },
    "multi_stage_pipeline": {
        "program": "multi_stage_pipeline.lock",
        "fixture": "multi_stage_pipeline.json",
        "kpi": "compile_ms, simulate_ms, input_rows_per_sec, fold_rows_per_sec",
    },
}


def _load_fixture(path: Path) -> tuple[dict[str, list[object]], dict[str, list[object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    streams = payload.get("streams") or {}
    accumulators = payload.get("accumulators") or {}
    if not isinstance(streams, dict) or not isinstance(accumulators, dict):
        raise ValueError(f"Fixture {path} must contain object maps for streams/accumulators")
    return streams, accumulators


def _run_once(workload_name: str) -> dict[str, object]:
    workload = WORKLOADS[workload_name]
    source = (WORKLOAD_DIR / workload["program"]).read_text(encoding="utf-8")
    stream_inputs, accumulator_inputs = _load_fixture(FIXTURE_DIR / workload["fixture"])

    compile_start = perf_counter()
    compile_result = compile_lockstep(source, verbose=False)
    compile_ms = (perf_counter() - compile_start) * 1_000.0

    simulate_start = perf_counter()
    simulation = simulate_pipeline_entities(
        compile_result.entities,
        stream_inputs=stream_inputs,
        accumulator_inputs=accumulator_inputs,
    )
    simulate_seconds = perf_counter() - simulate_start
    simulate_ms = simulate_seconds * 1_000.0

    kernel_rows = sum(route["input_count"] for route in simulation["routes"] if route["kind"] in {"shader", "filter"})
    fold_rows = sum(route["input_count"] for route in simulation["routes"] if route["kind"] == "fold")

    return {
        "workload": workload_name,
        "kpi": workload["kpi"],
        "compile_ms": round(compile_ms, 3),
        "simulate_ms": round(simulate_ms, 3),
        "input_rows": kernel_rows,
        "fold_rows": fold_rows,
        "input_rows_per_sec": round(kernel_rows / simulate_seconds, 2) if simulate_seconds > 0 else 0.0,
        "fold_rows_per_sec": round(fold_rows / simulate_seconds, 2) if simulate_seconds > 0 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=sorted(WORKLOADS), action="append", help="Run only selected workload(s)")
    parser.add_argument("--iterations", type=int, default=1, help="Repeat each workload and report mean metrics")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of markdown table")
    args = parser.parse_args()

    selected = args.workload or sorted(WORKLOADS)
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")

    results: list[dict[str, object]] = []
    for name in selected:
        runs = [_run_once(name) for _ in range(args.iterations)]
        averaged = {
            "workload": name,
            "kpi": runs[0]["kpi"],
            "compile_ms": round(mean(float(run["compile_ms"]) for run in runs), 3),
            "simulate_ms": round(mean(float(run["simulate_ms"]) for run in runs), 3),
            "input_rows": int(mean(float(run["input_rows"]) for run in runs)),
            "fold_rows": int(mean(float(run["fold_rows"]) for run in runs)),
            "input_rows_per_sec": round(mean(float(run["input_rows_per_sec"]) for run in runs), 2),
            "fold_rows_per_sec": round(mean(float(run["fold_rows_per_sec"]) for run in runs), 2),
        }
        results.append(averaged)

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print("| workload | compile_ms | simulate_ms | input_rows/s | fold_rows/s |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for row in results:
        print(
            f"| {row['workload']} | {row['compile_ms']} | {row['simulate_ms']} "
            f"| {row['input_rows_per_sec']} | {row['fold_rows_per_sec']} |"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
