"""Differential oracle: Python simulator vs. compiled ``Lockstep_Tick``.

``check_case`` compiles a generated program, runs one tick in both the simulator
and native code, and returns every observable disagreement:

* every row of every *sink* stream (a route target no later route reads) --
  intermediates may legitimately never be materialized once stages fuse;
* every ``fold`` uniform, read back from its published arena offset;
* the header's ``LOCKSTEP_OFFSET_*`` macros against the layout the oracle packs
  with, so the oracle can't silently drift from the ABI a host sees.

Run a single seed from the command line to reproduce a failure, or sweep a
range of seeds in parallel (``make oracle`` wraps the sweep)::

    PYTHONPATH=.:tests python -m differential.oracle 1234 [--width 4] [--keep DIR]
    PYTHONPATH=.:tests python -m differential.oracle --sweep 0:5000 --widths 4,8,16
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lockstep_compiler.compiler import compile_lockstep
from lockstep_compiler.simulator import simulate_pipeline_entities

from .native import ArenaCodec, run_tick
from .program_gen import GeneratedCase, f32, generate_case

# Fold results are reassociated (vector partial sums, fast-math reductions), so
# float folds compare with a tolerance scaled to the magnitude of the terms.
FOLD_REL_TOL = 1e-4


@dataclass
class Mismatch:
    where: str
    simulated: Any
    native: Any

    def __str__(self) -> str:
        return f"{self.where}: simulator={self.simulated!r} native={self.native!r}"


def _values_equal(sim: Any, nat: Any, type_name: str) -> bool:
    """Exact comparison: the simulator models single-precision ``float`` and
    wrapping 32-bit ``int``, so kernel outputs must agree bit for bit."""
    if sim is None:
        return False
    if type_name == "bool":
        return bool(sim) == bool(nat)
    if type_name in {"float", "double"}:
        sim_f = f32(float(sim)) if type_name == "float" else float(sim)
        if math.isnan(sim_f) and math.isnan(nat):
            return True
        return sim_f == nat
    return bool(sim == nat)


def _leaf_items(row: Any, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    if isinstance(row, dict):
        items: list[tuple[tuple[str, ...], Any]] = []
        for key, value in row.items():
            items.extend(_leaf_items(value, prefix + (key,)))
        return items
    return [(prefix, row)]


def _path_value(row: Any, path: tuple[str, ...]) -> Any:
    for part in path:
        if not isinstance(row, dict) or part not in row:
            return None
        row = row[part]
    return row


def _rows_mismatches(
    stream: str,
    sim_rows: list[Any],
    nat_rows: list[Any],
    leaf_types: dict[tuple[str, ...], str],
) -> list[Mismatch]:
    out: list[Mismatch] = []
    for index, (sim_row, nat_row) in enumerate(zip(sim_rows, nat_rows)):
        for path, nat_value in _leaf_items(nat_row):
            sim_value = _path_value(sim_row, path)
            if not _values_equal(sim_value, nat_value, leaf_types[path]):
                where = f"{stream}[{index}]" + "".join(f".{part}" for part in path)
                out.append(Mismatch(where, sim_value, nat_value))
    return out


def _fold_close(sim: Any, nat: Any, magnitude: float, type_name: str) -> bool:
    if sim is None:
        return False
    if type_name in {"float", "double"}:
        sim_f = float(sim)
        if math.isinf(sim_f) or math.isinf(nat):
            return sim_f == nat
        return abs(sim_f - nat) <= FOLD_REL_TOL * max(1.0, magnitude)
    return sim == nat


def _check_header_offsets(header: str, codec: ArenaCodec) -> list[Mismatch]:
    macros = {
        name: int(value)
        for name, value in re.findall(r"#define\s+(LOCKSTEP_OFFSET_\w+)\s+(\d+)", header)
    }
    out: list[Mismatch] = []
    for leaf in codec.layout.leaves:
        suffix = f"{leaf.kind}_{leaf.binding_name}".upper()
        if leaf.path:
            suffix += "_" + "_".join(leaf.path).upper()
        macro = f"LOCKSTEP_OFFSET_{suffix}"
        if macro in macros and macros[macro] != leaf.offset:
            out.append(Mismatch(f"header {macro}", leaf.offset, macros[macro]))
    total = re.search(r"#define\s+LOCKSTEP_ARENA_BYTES\s+(\d+)", header)
    if total is None or int(total.group(1)) != codec.layout.total_size:
        out.append(
            Mismatch(
                "header LOCKSTEP_ARENA_BYTES",
                codec.layout.total_size,
                total.group(1) if total else None,
            )
        )
    return out


def observable_outputs(entities: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return ``(sink_streams, fold_uniforms)`` for a compiled program.

    A sink is a stream whose final write is not read by any later route (those
    are the rows a host reads back; fused intermediates may never be stored).
    """
    routes = entities.get("bind_routes_ir", [])
    kernel_params = {
        kernel["name"]: kernel.get("params", [])
        for kernel in list(entities.get("shaders", [])) + list(entities.get("filters", []))
    }
    last_write: dict[str, int] = {}
    reads: dict[str, list[int]] = {}
    folds: list[str] = []
    for index, route in enumerate(routes):
        if route.get("kind") == "fold":
            folds.append(route["uniform_name"])
            continue
        params = kernel_params.get(route.get("kernel"), [])
        for param, arg in zip(params, route.get("args", [])):
            if param.get("modifier") == "in":
                reads.setdefault(arg, []).append(index)
        last_write[route["target"]] = index
    sinks = [
        stream
        for stream, written_at in last_write.items()
        if not any(read_at > written_at for read_at in reads.get(stream, []))
    ]
    return sinks, folds


def case_from_source(
    source: str,
    stream_inputs: dict[str, list[Any]],
    uniform_values: dict[str, Any] | None = None,
    *,
    capacity: int,
) -> GeneratedCase:
    """Wrap a hand-written program as a case for :func:`check_case`."""
    result = compile_lockstep(source, verbose=False)
    sinks, folds = observable_outputs(result.entities)
    return GeneratedCase(
        seed=-1,
        source=source,
        capacity=capacity,
        stream_inputs=stream_inputs,
        uniform_values=dict(uniform_values or {}),
        sink_streams=sinks,
        fold_uniforms=folds,
    )


def check_case(
    case: GeneratedCase, *, target_width: int = 8, keep_dir: Path | None = None
) -> list[Mismatch]:
    result = compile_lockstep(case.source, verbose=False, target_width=target_width)
    codec = ArenaCodec.for_entities(result.entities)
    mismatches = _check_header_offsets(result.c_header, codec)

    simulated = simulate_pipeline_entities(result, stream_inputs=case.stream_inputs)

    image = codec.new_image()
    for stream, rows in case.stream_inputs.items():
        codec.write_rows(image, stream, rows)
    for uniform, value in case.uniform_values.items():
        codec.write_scalar(image, "uniform", uniform, value)
    after = run_tick(result.llvm_ir, bytes(image), keep_dir=keep_dir)

    sink_streams, fold_uniforms = observable_outputs(result.entities)
    for stream in sink_streams:
        sim_rows = simulated["streams"].get(stream, [])
        if len(sim_rows) > case.capacity:
            mismatches.append(Mismatch(f"{stream} row count", len(sim_rows), case.capacity))
            continue
        nat_rows = codec.read_rows(after, stream, len(sim_rows))
        mismatches.extend(
            _rows_mismatches(stream, sim_rows, nat_rows, codec.leaf_types(stream))
        )

    for uniform in fold_uniforms:
        sim_value = simulated["uniforms"].get(uniform)
        nat_value = codec.read_scalar(after, "uniform", uniform)
        source = next(
            (
                route["source"]
                for route in result.entities.get("bind_routes_ir", [])
                if route.get("kind") == "fold" and route.get("uniform_name") == uniform
            ),
            None,
        )
        terms = simulated["accumulators"].get(source, []) if source else []
        magnitude = sum(abs(float(term)) for term in terms)
        uniform_type = codec.scalar_type("uniform", uniform)
        if not _fold_close(sim_value, nat_value, magnitude, uniform_type):
            mismatches.append(Mismatch(f"fold {uniform}", sim_value, nat_value))
    return mismatches


def _sweep_one(job: tuple[int, int]) -> tuple[int, int, str]:
    seed, width = job
    try:
        mismatches = check_case(generate_case(seed), target_width=width)
    except Exception as exc:  # report and keep sweeping
        return seed, width, f"error: {type(exc).__name__}: {str(exc)[:300]}"
    if mismatches:
        return seed, width, f"{len(mismatches)} mismatch(es); first: {mismatches[0]}"
    return seed, width, ""


def sweep(start: int, stop: int, widths: list[int], jobs: int) -> int:
    from concurrent.futures import ProcessPoolExecutor

    work = [(seed, width) for seed in range(start, stop) for width in widths]
    failures = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for seed, width, problem in pool.map(_sweep_one, work, chunksize=4):
            if problem:
                failures += 1
                print(f"seed {seed} width {width}: {problem}", flush=True)
    print(f"{len(work) - failures}/{len(work)} cases agree")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, nargs="?")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--keep", type=Path, default=None, help="save IR here")
    parser.add_argument("--sweep", metavar="START:STOP", default=None)
    parser.add_argument("--widths", default="4,8", help="for --sweep")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)
    if args.sweep:
        start, _, stop = args.sweep.partition(":")
        widths = [int(width) for width in args.widths.split(",")]
        return sweep(int(start), int(stop), widths, args.jobs)
    if args.seed is None:
        parser.error("give a seed or --sweep START:STOP")
    case = generate_case(args.seed)
    print(case.source)
    mismatches = check_case(case, target_width=args.width, keep_dir=args.keep)
    for mismatch in mismatches[:40]:
        print(mismatch)
    print(f"{len(mismatches)} mismatch(es)")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
