#!/usr/bin/env python3
"""Check benchmark regression against a checked-in baseline JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slowdown_fraction(
    baseline_value: float,
    current_value: float,
    smaller_is_better: bool,
) -> float:
    if baseline_value == 0:
        return 0.0

    if smaller_is_better:
        return (current_value - baseline_value) / baseline_value
    return (baseline_value - current_value) / baseline_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("benchmarks/baselines/default.json"),
        help="Checked-in benchmark baseline JSON.",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=Path("benchmark-results.json"),
        help="Current benchmark JSON output.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Maximum allowed slowdown ratio before failing (0.10 = 10%%).",
    )
    args = parser.parse_args()

    baseline = _load(args.baseline)
    current = _load(args.current)

    baseline_benchmarks = baseline.get("benchmarks", {})
    current_benchmarks = current.get("benchmarks", {})
    if not isinstance(baseline_benchmarks, dict) or not isinstance(current_benchmarks, dict):
        print("Invalid benchmark payload: expected object at 'benchmarks'.", file=sys.stderr)
        sys.exit(2)

    kpis = baseline.get("kpis")
    if isinstance(kpis, list) and kpis:
        metric_names = [name for name in kpis if isinstance(name, str)]
    else:
        metric_names = [name for name in baseline_benchmarks if isinstance(name, str)]

    regressions: list[str] = []
    checked: list[str] = []

    for name in metric_names:
        baseline_metric = baseline_benchmarks.get(name)
        current_metric = current_benchmarks.get(name)
        if not isinstance(baseline_metric, dict) or not isinstance(current_metric, dict):
            print(f"Skipping metric '{name}': missing in baseline or current results.")
            continue

        baseline_value = baseline_metric.get("value")
        current_value = current_metric.get("value")
        if not isinstance(baseline_value, (int, float)) or not isinstance(current_value, (int, float)):
            print(f"Skipping metric '{name}': non-numeric value.")
            continue

        smaller_is_better = bool(baseline_metric.get("smaller_is_better", True))
        slowdown = _slowdown_fraction(
            float(baseline_value),
            float(current_value),
            smaller_is_better,
        )
        checked.append(name)

        if slowdown > args.threshold:
            regressions.append(
                f"{name}: {slowdown * 100:.2f}% slowdown "
                f"(baseline={baseline_value}, current={current_value})"
            )

    if not checked:
        print("No comparable KPI metrics were found.", file=sys.stderr)
        sys.exit(2)

    if regressions:
        print("Benchmark regression detected:")
        for regression in regressions:
            print(f"  - {regression}")
        sys.exit(1)

    print(
        f"Benchmark regression check passed for {len(checked)} KPI(s); threshold={args.threshold * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
