#!/usr/bin/env python3
"""Gate the deterministic invariants of the native execution benchmarks.

``benchmarks/native/run_native.py`` reports two very different kinds of number:

* **Deterministic, hardware-independent** — ``rows_per_tick`` and ``arena_bytes``
  are pure ABI (fixed by the arena layout), and ``checksum`` is an exact
  reduction over an output leaf that the branchless kernel computes identically
  on any CPU. These must not change unless codegen intentionally changes; a drift
  is a real correctness/ABI regression.

* **Hardware-dependent** — ``per_tick_us``, ``mrows_per_sec`` and
  ``arena_gib_per_sec`` depend on the CPU the benchmark ran on (and, with
  ``-march=native``, on the exact microarchitecture). On shared CI runners these
  are far too noisy to gate on, so they are treated as advisory and only
  published as an artifact.

This script gates the first group and ignores the second. It compares a current
``run_native.py`` JSON report against a checked-in baseline
(``benchmarks/baselines/native.json``) and fails if the workload set changed, if
any ABI field differs, or if a checksum drifts beyond floating-point noise.

The baseline is a tracked reference, like the golden-IR snapshots: when a codegen
change intentionally moves it, regenerate with ``--update`` and review the diff.

    python scripts/check_native_benchmark.py --current native-results.json
    python scripts/check_native_benchmark.py --current native-results.json --update
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Relative tolerance for the checksum comparison. The checksum is a deterministic
# IEEE float computation, but allow a hair of slack against cross-toolchain
# floating-point noise; the ABI fields below are matched exactly.
_CHECKSUM_REL_TOL = 1e-6


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _results_by_workload(payload: dict) -> dict[str, dict]:
    results = payload.get("results", [])
    if not isinstance(results, list):
        return {}
    by_name: dict[str, dict] = {}
    for row in results:
        if isinstance(row, dict) and isinstance(row.get("workload"), str):
            by_name[row["workload"]] = row
    return by_name


def _baseline_from_results(payload: dict) -> dict:
    """Project a full run_native.py report down to its gated invariants."""
    workloads = {
        name: {
            "rows_per_tick": int(row["rows_per_tick"]),
            "arena_bytes": int(row["arena_bytes"]),
            "checksum": float(row["checksum"]),
        }
        for name, row in sorted(_results_by_workload(payload).items())
    }
    return {
        "schema_version": 1,
        "kind": "native_execution_invariants",
        "description": (
            "Deterministic, hardware-independent invariants of the native "
            "execution benchmarks (ABI + output checksums). Throughput is not "
            "gated. Regenerate with scripts/check_native_benchmark.py --update."
        ),
        "workloads": workloads,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("benchmarks/baselines/native.json"),
        help="Checked-in native invariants baseline JSON.",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=Path("native-results.json"),
        help="Current run_native.py JSON report.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Rewrite the baseline from --current instead of checking.",
    )
    args = parser.parse_args()

    current = _load(args.current)

    if args.update:
        baseline = _baseline_from_results(current)
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(
            json.dumps(baseline, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote native baseline for {len(baseline['workloads'])} workload(s) to {args.baseline}")
        return 0

    baseline = _load(args.baseline)
    baseline_workloads = baseline.get("workloads", {})
    current_workloads = _results_by_workload(current)

    if not isinstance(baseline_workloads, dict) or not baseline_workloads:
        print("Invalid baseline: expected a non-empty 'workloads' object.", file=sys.stderr)
        return 2

    failures: list[str] = []

    missing = sorted(set(baseline_workloads) - set(current_workloads))
    extra = sorted(set(current_workloads) - set(baseline_workloads))
    for name in missing:
        failures.append(f"{name}: present in baseline but missing from current results")
    for name in extra:
        failures.append(f"{name}: present in current results but not in baseline")

    for name in sorted(set(baseline_workloads) & set(current_workloads)):
        expected = baseline_workloads[name]
        actual = current_workloads[name]

        for field in ("rows_per_tick", "arena_bytes"):
            exp = int(expected[field])
            act = int(actual[field])
            if exp != act:
                failures.append(
                    f"{name}.{field}: ABI drift (baseline={exp}, current={act})"
                )

        exp_ck = float(expected["checksum"])
        act_ck = float(actual["checksum"])
        if not math.isclose(exp_ck, act_ck, rel_tol=_CHECKSUM_REL_TOL, abs_tol=1e-6):
            failures.append(
                f"{name}.checksum: correctness drift (baseline={exp_ck}, current={act_ck})"
            )

    if failures:
        print("Native benchmark invariant check FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        print(
            "\nIf this change is intentional, regenerate the baseline with\n"
            "  python scripts/check_native_benchmark.py --current <report> --update\n"
            "and review the diff.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Native benchmark invariants OK for {len(baseline_workloads)} workload(s) "
        "(ABI + checksums; throughput not gated)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
