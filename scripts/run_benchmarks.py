#!/usr/bin/env python3
"""Run lightweight project benchmarks and emit JSON results."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from lockstep_compiler.compiler import compile_lockstep
from lockstep_compiler.simulator import simulate_pipeline_entities


def _read_program(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _run_timed(name: str, fn: Callable[[], None], iterations: int) -> dict[str, object]:
    samples_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        samples_ms.append(elapsed_ms)

    median_ms = statistics.median(samples_ms)
    return {
        "name": name,
        "value": round(median_ms, 3),
        "unit": "ms",
        "smaller_is_better": True,
        "samples_ms": [round(sample, 3) for sample in samples_ms],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results.json"),
        help="Path to write benchmark JSON output.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of timing iterations to capture.",
    )
    args = parser.parse_args()

    minimal_program = _read_program(Path("examples/minimal.lock"))

    compile_result = compile_lockstep(minimal_program)

    benchmarks: dict[str, dict[str, object]] = {}

    compile_bench = _run_timed(
        "compile_minimal_ms",
        lambda: compile_lockstep(minimal_program),
        iterations=args.iterations,
    )
    benchmarks["compile_minimal_ms"] = compile_bench

    simulate_bench = _run_timed(
        "simulate_minimal_ms",
        lambda: simulate_pipeline_entities(compile_result.entities),
        iterations=args.iterations,
    )
    benchmarks["simulate_minimal_ms"] = simulate_bench

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "iterations": args.iterations,
        "benchmarks": benchmarks,
        "kpis": ["compile_minimal_ms", "simulate_minimal_ms"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
