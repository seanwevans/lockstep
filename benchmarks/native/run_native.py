#!/usr/bin/env python3
"""Native execution benchmarks for Lockstep-generated code.

Unlike ``benchmarks/run_workloads.py`` and ``scripts/run_benchmarks.py`` --
which measure the Python *frontend* (parse + in-Python simulation) -- this
harness measures the thing Lockstep actually ships: the compiled machine code.

For each workload it

1. compiles the ``.lock`` source to LLVM IR and a C header via the real
   compiler entry point (``compile_lockstep``),
2. generates a small C host driver that allocates the arena, primes the input
   streams with deterministic data, warms up, then calls ``Lockstep_Tick`` in a
   timed loop,
3. builds the IR + driver + intrinsic shim with ``clang -O3 -march=native``,
4. runs the executable and records real throughput (per-tick latency, rows/s,
   and arena bytes touched per second) plus a correctness checksum.

The kernels are straight-line/branchless by construction, so per-tick timing is
data-independent; the harness therefore fills streams to capacity with a stable
deterministic pattern rather than replaying a fixture, which keeps the row count
(and thus the throughput denominator) equal to what ``Lockstep_Tick`` processes.

Arena sizing is taken from the generated IR's ``struct.Lockstep_Arena`` type,
which is the authoritative ABI the compiled kernel was built against.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lockstep_compiler.compiler import compile_lockstep  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
WORKLOAD_DIR = REPO_ROOT / "benchmarks" / "workloads"
INTRINSICS_C = BASE_DIR / "lockstep_intrinsics.c"

# Element sizes for the primitive LLVM types that appear in the arena struct.
_LLVM_ELEMENT_BYTES = {
    "i1": 1,
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "i64": 8,
    "float": 4,
    "double": 8,
}

# Lockstep primitive -> (C type, byte size, is_integer).
_PRIMITIVE_C = {
    "bool": ("uint8_t", 1, True),
    "int": ("int32_t", 4, True),
    "uint": ("uint32_t", 4, True),
    "float": ("float", 4, False),
    "double": ("double", 8, False),
}

WORKLOADS = ("particle_energy", "telemetry_filter_aggregation", "multi_stage_pipeline")


@dataclass
class StreamField:
    """A single SoA leaf of an input stream: name, C type, offset, stride."""

    name: str
    c_type: str
    elem_bytes: int
    is_integer: bool
    offset: int


@dataclass
class WorkloadPlan:
    name: str
    ir_text: str
    arena_bytes: int
    capacity: int
    rows: int
    input_fields: list[StreamField]
    checksum_field: StreamField | None


class BenchmarkError(RuntimeError):
    pass


def _arena_bytes_from_ir(ir_text: str) -> int:
    """Sum the byte size of every field in the generated arena struct type."""

    match = re.search(
        r'%"struct\.Lockstep_Arena"\s*=\s*type\s*(?:<\s*)?\{(.+?)\}', ir_text, re.S
    )
    if match is None:
        raise BenchmarkError("could not locate struct.Lockstep_Arena in generated IR")

    body = match.group(1)
    total = 0
    matched_any = False
    for field in body.split(","):
        field = field.strip()
        if not field:
            continue
        matched_any = True
        array = re.match(r"\[(\d+)\s*x\s*(\w+)\]", field)
        if array is not None:
            count = int(array.group(1))
            elem = array.group(2)
            total += count * _LLVM_ELEMENT_BYTES.get(elem, 8)
            continue
        if field.endswith("*"):
            total += 8  # pointer leaf (opaque generic-wrapper placeholder)
            continue
        total += _LLVM_ELEMENT_BYTES.get(field, 8)  # bare scalar leaf
    if not matched_any:
        raise BenchmarkError("arena struct type was empty in generated IR")
    return total


def _offset_macros(header_text: str) -> dict[str, int]:
    macros: dict[str, int] = {}
    for name, value in re.findall(
        r"#define\s+(LOCKSTEP_OFFSET_\w+)\s+(\d+)", header_text
    ):
        macros[name] = int(value)
    return macros


def _classify_streams(entities: dict) -> tuple[set[str], list[str]]:
    """Return (input stream names, ordered output/target stream names)."""

    kernels = {k["name"]: k for k in entities.get("shaders", [])}
    kernels.update({k["name"]: k for k in entities.get("filters", [])})

    inputs: set[str] = set()
    outputs: list[str] = []
    stream_names = {s["name"] for s in entities.get("streams", [])}
    for route in entities.get("bind_routes_ir", []):
        if route.get("kind") not in {"kernel", "shader", "filter"}:
            continue
        kernel = kernels.get(route.get("kernel"))
        args = route.get("args", [])
        if kernel is not None:
            for param, arg in zip(kernel.get("params", []), args):
                if param.get("modifier") == "in" and arg in stream_names:
                    inputs.add(arg)
        target = route.get("target")
        if target in stream_names and target not in outputs:
            outputs.append(target)
    return inputs, outputs


def _build_plan(name: str, source: str) -> WorkloadPlan:
    result = compile_lockstep(source, verbose=False)
    ir_text = getattr(result, "llvm_ir", "") or ""
    header_text = getattr(result, "c_header", "") or ""
    entities = result.entities if hasattr(result, "entities") else {}
    if not ir_text:
        raise BenchmarkError(f"{name}: compiler produced no LLVM IR")

    arena_bytes = _arena_bytes_from_ir(ir_text)
    offsets = _offset_macros(header_text)

    struct_fields = {
        s["name"]: [(f["name"], f["type"]) for f in s.get("fields", [])]
        for s in entities.get("structs", [])
    }
    stream_type = {s["name"]: s["type"] for s in entities.get("streams", [])}
    capacities = {
        s["name"]: int(s.get("capacity", 0) or 0) for s in entities.get("streams", [])
    }
    input_names, output_names = _classify_streams(entities)
    if not input_names:
        # Fall back to any declared stream so the harness still primes something.
        input_names = set(stream_type)

    capacity = max((capacities.get(n, 0) for n in input_names), default=0)

    def _fields_for(stream_name: str) -> list[StreamField]:
        type_name = stream_type.get(stream_name, "")
        fields: list[StreamField] = []
        for field_name, field_type in struct_fields.get(type_name, []):
            prim = _PRIMITIVE_C.get(field_type)
            if prim is None:
                continue  # nested/opaque leaf; skip priming, kernel still runs
            c_type, elem_bytes, is_integer = prim
            macro = f"LOCKSTEP_OFFSET_STREAM_{stream_name}_{field_name}".upper()
            if macro not in offsets:
                continue
            fields.append(
                StreamField(
                    name=f"{stream_name}.{field_name}",
                    c_type=c_type,
                    elem_bytes=elem_bytes,
                    is_integer=is_integer,
                    offset=offsets[macro],
                )
            )
        return fields

    input_fields: list[StreamField] = []
    for stream_name in sorted(input_names):
        input_fields.extend(_fields_for(stream_name))

    checksum_field: StreamField | None = None
    for stream_name in output_names or sorted(input_names):
        for field in _fields_for(stream_name):
            if not field.is_integer:
                checksum_field = field
                break
        if checksum_field is not None:
            break

    return WorkloadPlan(
        name=name,
        ir_text=ir_text,
        arena_bytes=arena_bytes,
        capacity=capacity,
        rows=capacity,
        input_fields=input_fields,
        checksum_field=checksum_field,
    )


def _render_driver(plan: WorkloadPlan, iterations: int, warmup: int) -> str:
    prime_lines: list[str] = []
    for idx, field in enumerate(plan.input_fields):
        cast = f"({field.c_type}*)(base + {field.offset})"
        if field.is_integer:
            value = f"({field.c_type})((i * 2654435761u) + {idx}u)"
        else:
            # Bounded, non-trivial, deterministic float pattern per lane.
            value = (
                f"({field.c_type})(0.5f + ((float)((i + {idx}) % 4096) * 0.000244140625f))"
            )
        prime_lines.append(
            f"    {{ {field.c_type}* col = {cast}; "
            f"for (size_t i = 0; i < CAP; ++i) col[i] = {value}; }}"
        )

    if plan.checksum_field is not None:
        f = plan.checksum_field
        checksum_read = (
            f"    {{ const {f.c_type}* col = (const {f.c_type}*)(base + {f.offset}); "
            "for (size_t i = 0; i < CAP; ++i) checksum += (double)col[i]; }"
        )
    else:
        checksum_read = "    /* no float output leaf available for checksum */"

    prime_block = "\n".join(prime_lines) if prime_lines else "    /* no primeable inputs */"

    return f"""/* Auto-generated by benchmarks/native/run_native.py -- do not edit. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ARENA_BYTES ((size_t){plan.arena_bytes})
#define CAP ((size_t){plan.capacity})
#define ITERS {iterations}
#define WARMUP {warmup}

void Lockstep_Tick(void* arena);

static void prime(uint8_t* base) {{
{prime_block}
}}

int main(void) {{
    uint8_t* base = (uint8_t*)aligned_alloc(64, (ARENA_BYTES + 4095) & ~((size_t)4095));
    if (base == NULL) {{ fprintf(stderr, "alloc failed\\n"); return 2; }}
    memset(base, 0, ARENA_BYTES);
    prime(base);

    for (int w = 0; w < WARMUP; ++w) Lockstep_Tick(base);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < ITERS; ++it) Lockstep_Tick(base);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double checksum = 0.0;
{checksum_read}

    double ns = (double)(t1.tv_sec - t0.tv_sec) * 1e9
              + (double)(t1.tv_nsec - t0.tv_nsec);
    double per_tick_ns = ns / (double)ITERS;

    printf("{{\\"per_tick_ns\\": %.3f, \\"checksum\\": %.6f}}\\n",
           per_tick_ns, checksum);
    free(base);
    return 0;
}}
"""


def _run_workload(
    name: str,
    iterations: int,
    warmup: int,
    clang: str,
    keep_dir: Path | None,
) -> dict[str, object]:
    source_path = WORKLOAD_DIR / f"{name}.lock"
    if not source_path.exists():
        raise BenchmarkError(f"workload source not found: {source_path}")
    source = source_path.read_text(encoding="utf-8")

    plan = _build_plan(name, source)

    work_dir = keep_dir / name if keep_dir else Path(tempfile.mkdtemp(prefix=f"lsnat_{name}_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    ir_path = work_dir / f"{name}.ll"
    driver_path = work_dir / f"{name}_driver.c"
    exe_path = work_dir / f"{name}_bench"
    ir_path.write_text(plan.ir_text, encoding="utf-8")
    driver_path.write_text(_render_driver(plan, iterations, warmup), encoding="utf-8")

    compile_cmd = [
        clang,
        "-O3",
        "-march=native",
        "-Wno-override-module",
        str(driver_path),
        str(ir_path),
        str(INTRINSICS_C),
        "-o",
        str(exe_path),
        "-lm",
    ]
    proc = subprocess.run(compile_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise BenchmarkError(
            f"{name}: clang failed ({proc.returncode})\n{proc.stderr.strip()}"
        )

    run = subprocess.run([str(exe_path)], capture_output=True, text=True, timeout=300)
    if run.returncode != 0:
        raise BenchmarkError(
            f"{name}: benchmark executable failed ({run.returncode})\n{run.stderr.strip()}"
        )

    try:
        measured = json.loads(run.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise BenchmarkError(f"{name}: could not parse driver output: {run.stdout!r}") from exc

    per_tick_ns = float(measured["per_tick_ns"])
    per_tick_s = per_tick_ns * 1e-9
    rows = plan.rows
    rows_per_sec = rows / per_tick_s if per_tick_s > 0 else 0.0
    bytes_per_sec = plan.arena_bytes / per_tick_s if per_tick_s > 0 else 0.0

    if keep_dir is None:
        shutil.rmtree(work_dir, ignore_errors=True)

    return {
        "workload": name,
        "rows_per_tick": rows,
        "arena_bytes": plan.arena_bytes,
        "per_tick_us": round(per_tick_ns / 1000.0, 4),
        "mrows_per_sec": round(rows_per_sec / 1e6, 2),
        "arena_gib_per_sec": round(bytes_per_sec / (1024**3), 2),
        "checksum": round(float(measured["checksum"]), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        choices=sorted(WORKLOADS),
        action="append",
        help="Run only the selected workload(s); repeatable.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
        help="Timed Lockstep_Tick calls per workload (default: 2000).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Untimed warmup ticks before measurement (default: 20).",
    )
    parser.add_argument("--clang", default="clang", help="C/LLVM compiler to use.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Also write the JSON report to this path.",
    )
    parser.add_argument(
        "--keep-artifacts",
        type=Path,
        help="Keep generated IR/driver/exe under this directory for inspection.",
    )
    args = parser.parse_args()

    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")

    clang = shutil.which(args.clang)
    if clang is None:
        raise SystemExit(
            f"'{args.clang}' not found on PATH; native benchmarks require an LLVM/clang toolchain."
        )

    selected = args.workload or list(WORKLOADS)
    results: list[dict[str, object]] = []
    for name in selected:
        results.append(
            _run_workload(name, args.iterations, args.warmup, clang, args.keep_artifacts)
        )

    payload = {
        "schema_version": 1,
        "kind": "native_execution",
        "iterations": args.iterations,
        "results": results,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Native execution benchmarks ({args.iterations} ticks each, {clang})\n")
    header = (
        "| workload | rows/tick | arena | per_tick_us | Mrows/s | GiB/s |"
    )
    print(header)
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in results:
        arena_mib = float(row["arena_bytes"]) / (1024**2)
        print(
            f"| {row['workload']} | {row['rows_per_tick']} | {arena_mib:.2f} MiB "
            f"| {row['per_tick_us']} | {row['mrows_per_sec']} | {row['arena_gib_per_sec']} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
