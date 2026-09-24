#!/usr/bin/env python3
"""Measure what scoped alias metadata could still buy Lockstep's codegen.

``ROADMAP.md`` lists "scoped alias metadata on arena-derived pointers" as
remaining backend work: ``noalias`` on kernel pointer parameters and/or
``!alias.scope`` per disjoint arena region, so LLVM can tell that two streams
never overlap. This probe asks whether any optimization is actually *blocked*
on that today, before anyone builds it.

Every arena access in ``Lockstep_Tick`` is ``arena + constant leaf offset +
row * stride``, and the arena is already a ``noalias`` parameter. So LLVM's
own alias analysis ought to separate distinct leaves by their constant offset
ranges without help. The probe checks that claim against LLVM's own reports.
For each program it compiles the generated IR with the benchmark flags
(``clang -O3 -march=native -ffast-math``), saves the optimization record, and
counts, for ``Lockstep_Tick`` only (the kernel functions are inlined into it):

* **runtime alias checks**: ``vector.memcheck`` blocks in the optimized IR,
  which the loop vectorizer emits when it cannot prove two pointers disjoint
  and versions the loop behind an overlap test;
* **alias-blocked vectorization**: loop-vectorize remarks whose reason is
  about memory dependences (``CantVectorizeMemory``, ``UnsafeDep``,
  ``UnsafeMemDep``, ``CantIdentifyArrayBounds``, ...);
* **alias-blocked scalar optimization**: LICM failing to hoist a
  loop-invariant load because a store in the loop "may invalidate" it, and
  GVN failing to remove a redundant load because it is "clobbered by" a
  store;
* every *other* missed loop-vectorize reason, for context. A loop that is not
  vectorized for a non-alias reason (for example, a compaction loop whose
  store index is data-dependent) would stay scalar with perfect alias
  information too.

The corpus is the shipped benchmark workloads, the golden programs, and a slice
of the differential oracle's random programs. Each is compiled twice: with
stage fusion (what ships) and with fusion disabled (the per-stage fallback,
which leans on LLVM's auto-vectorizer instead of codegen's own vector loops).

Results and their interpretation are recorded in ``benchmarks/RESULTS.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lockstep_compiler.codegen import emit_llvm_ir  # noqa: E402
from lockstep_compiler.compiler import compile_lockstep  # noqa: E402

WORKLOAD_DIR = REPO_ROOT / "benchmarks" / "workloads"
GOLDEN_DIR = REPO_ROOT / "tests" / "golden" / "programs"

CLANG_FLAGS = ["-O3", "-march=native", "-ffast-math", "-Wno-override-module"]

# loop-vectorize analysis remark names that mean "memory dependences / aliasing
# prevented vectorization".
_ALIAS_LV_REASONS = {
    "CantVectorizeMemory",
    "UnsafeDep",
    "UnsafeMemDep",
    "CantIdentifyArrayBounds",
    "UnknownArrayBounds",
    "TooManyRuntimeChecks",
    "CantCheckMemDepsAtRunTime",
}


@dataclass
class ProgramReport:
    name: str
    variant: str
    memchecks: int = 0
    vectorized: int = 0
    lv_missed: Counter = field(default_factory=Counter)
    alias_blocked: list[str] = field(default_factory=list)


def _records(yaml_text: str) -> list[dict[str, str]]:
    """Minimal parser for clang's optimization-record YAML.

    Each document is ``--- !Kind`` followed by ``Key: value`` lines and an
    ``Args`` list; we only need the kind, pass, name, function, and the joined
    argument text, so a line scan is enough (and avoids a YAML dependency).
    """
    records: list[dict[str, str]] = []
    for chunk in yaml_text.split("--- !")[1:]:
        lines = chunk.splitlines()
        record = {"Kind": lines[0].strip(), "Args": ""}
        args: list[str] = []
        for line in lines[1:]:
            match = re.match(r"^(Pass|Name|Function):\s+(.*)$", line)
            if match:
                record[match.group(1)] = match.group(2).strip().strip("'")
                continue
            match = re.match(r"^\s+- \w+:\s+(.*)$", line)
            if match:
                args.append(match.group(1).strip().strip("'"))
        record["Args"] = "".join(args)
        records.append(record)
    return records


def _analyze(name: str, variant: str, ir_text: str, clang: str, work: Path) -> ProgramReport:
    report = ProgramReport(name, variant)
    ir_path = work / f"{name}.{variant}.ll"
    opt_path = work / f"{name}.{variant}.opt.ll"
    record_path = work / f"{name}.{variant}.yaml"
    ir_path.write_text(ir_text, encoding="utf-8")
    proc = subprocess.run(
        [
            clang,
            *CLANG_FLAGS,
            "-S",
            "-emit-llvm",
            str(ir_path),
            "-o",
            str(opt_path),
            "-fsave-optimization-record",
            f"-foptimization-record-file={record_path}",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{name} ({variant}): clang failed\n{proc.stderr}")

    optimized = opt_path.read_text(encoding="utf-8")
    tick = optimized[optimized.find('@"Lockstep_Tick"') :]
    tick = tick[: tick.find("\n}\n") + 3]
    report.memchecks = len(re.findall(r"^vector\.memcheck[\w.]*:", tick, re.M))

    for record in _records(record_path.read_text(encoding="utf-8")):
        if record.get("Function") != "Lockstep_Tick":
            continue
        kind, pass_name, remark = record["Kind"], record.get("Pass"), record.get("Name")
        if pass_name == "loop-vectorize":
            if kind == "Passed" and remark == "Vectorized":
                report.vectorized += 1
            elif kind == "Analysis" and remark:
                report.lv_missed[remark] += 1
                if remark in _ALIAS_LV_REASONS:
                    report.alias_blocked.append(f"loop-vectorize: {record['Args']}")
        elif pass_name == "licm" and kind == "Missed" and "invalidate" in record["Args"]:
            report.alias_blocked.append(f"licm: {record['Args']}")
        elif pass_name == "gvn" and kind == "Missed" and "clobbered" in record["Args"]:
            report.alias_blocked.append(f"gvn: {record['Args']}")
    return report


# --- Upper-bound experiment: perfect arena alias information ---------------
#
# Every arena access in ``Lockstep_Tick`` lies inside one leaf (a stream,
# accumulator, uniform, or count column): leaves are disjoint byte ranges and
# row indices are clamped to capacity.  So tagging each access with a scope for
# its leaf, and ``!noalias`` for every other leaf, is sound -- it is the full
# form of the roadmap's "scoped alias metadata" item.  To give the tags
# something to attach to, the kernels are force-inlined first (their out and
# accum stores happen through pointer parameters otherwise), and the untagged
# baseline goes through the same inlining, so metadata is the only variable.

_LEAF_NAME = re.compile(r'^%"?((?:stream|accum|uniform|count)_.+?_byte_ptr)(?:\.\d+)?"?$')
_DEF = re.compile(r'^\s*(%[\w."$-]+|%"[^"]+")\s*=\s*(.*)$')
_PTR_OPERAND = re.compile(r'ptr (%[\w.$-]+|%"[^"]+")')


def _inline_kernels(ir_text: str, opt: str, work: Path, name: str) -> str:
    marked = re.sub(
        r'(define [^\n]*@"(?:shader|filter|pure)_[^"]*"\([^\n]*\))',
        r"\1 alwaysinline",
        ir_text,
    )
    src = work / f"{name}.inline_in.ll"
    out = work / f"{name}.inlined.ll"
    src.write_text(marked, encoding="utf-8")
    proc = subprocess.run(
        [opt, "-passes=always-inline,sroa", "-S", str(src), "-o", str(out)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{name}: opt failed\n{proc.stderr}")
    return out.read_text(encoding="utf-8")


def _tag_arena_scopes(ir_text: str) -> tuple[str, int, int]:
    """Attach per-leaf ``!alias.scope``/``!noalias`` to ``Lockstep_Tick``'s
    arena loads and stores.  Returns ``(ir, tagged, untagged)``: accesses whose
    pointer could not be traced to a leaf (stack slots, phi'd pointers) stay
    untagged, which is conservative."""
    start = ir_text.index('define void @Lockstep_Tick') if 'define void @Lockstep_Tick' in ir_text else ir_text.index('define void @"Lockstep_Tick"')
    end = ir_text.index("\n}\n", start) + 3
    body = ir_text[start:end].splitlines()
    defs: dict[str, str] = {}
    for line in body:
        match = _DEF.match(line)
        if match:
            defs[match.group(1)] = match.group(2)

    def leaf_of(pointer: str, depth: int = 0) -> str | None:
        found = _LEAF_NAME.match(pointer)
        if found:
            return found.group(1)
        if depth > 8:
            return None
        definition = defs.get(pointer, "")
        if definition.startswith(("getelementptr", "bitcast")):
            operand = _PTR_OPERAND.search(definition)
            if operand:
                return leaf_of(operand.group(1), depth + 1)
        return None

    leaves: dict[str, int] = {}
    tagged_lines: list[tuple[int, str]] = []
    untagged = 0
    for index, line in enumerate(body):
        stripped = line.strip()
        is_access = " load " in f" {stripped} " and "= load " in stripped or stripped.startswith("store ")
        if not is_access:
            continue
        operands = _PTR_OPERAND.findall(stripped)
        leaf = leaf_of(operands[-1]) if operands else None
        if leaf is None:
            untagged += 1
            continue
        leaves.setdefault(leaf, len(leaves))
        tagged_lines.append((index, leaf))

    base = 900000
    domain = base
    metadata = [f'!{domain} = distinct !{{!{domain}, !"lockstep.arena"}}']
    scope_ids = {leaf: base + 1 + i for leaf, i in leaves.items()}
    for leaf, scope in scope_ids.items():
        metadata.append(f'!{scope} = distinct !{{!{scope}, !{domain}, !"{leaf}"}}')
    list_base = base + 1 + len(leaves)
    scope_list: dict[str, int] = {}
    noalias_list: dict[str, int] = {}
    next_id = list_base
    for leaf, scope in scope_ids.items():
        scope_list[leaf] = next_id
        metadata.append(f"!{next_id} = !{{!{scope}}}")
        next_id += 1
        others = ", ".join(f"!{other}" for name, other in scope_ids.items() if name != leaf)
        noalias_list[leaf] = next_id
        metadata.append(f"!{next_id} = !{{{others}}}")
        next_id += 1
    for index, leaf in tagged_lines:
        body[index] = (
            f"{body[index]}, !alias.scope !{scope_list[leaf]}, !noalias !{noalias_list[leaf]}"
        )
    tagged_ir = ir_text[:start] + "\n".join(body) + ir_text[end - 1 :]
    return tagged_ir + "\n" + "\n".join(metadata) + "\n", len(tagged_lines), untagged


def _per_stage_ir(source: str) -> str:
    result = compile_lockstep(source, verbose=False)
    routes = [route.route for pipeline in result.ast.pipelines for route in pipeline.bind_routes]
    return emit_llvm_ir(
        result.ast,
        bind_optimization={"optimized_bind_routes": routes, "fused_groups": []},
    )


def _corpus(generated: int) -> list[tuple[str, str]]:
    programs = [(path.stem, path.read_text(encoding="utf-8")) for path in sorted(WORKLOAD_DIR.glob("*.lock"))]
    programs += [
        (f"golden_{path.stem}", path.read_text(encoding="utf-8"))
        for path in sorted(GOLDEN_DIR.glob("*.lock"))
    ]
    if generated:
        from differential.program_gen import generate_case

        programs += [(f"gen_{seed}", generate_case(seed).source) for seed in range(generated)]
    return programs


def _summarize(reports: list[ProgramReport], variant: str) -> dict[str, object]:
    subset = [r for r in reports if r.variant == variant]
    missed: Counter = Counter()
    for report in subset:
        missed.update(report.lv_missed)
    return {
        "programs": len(subset),
        "loops_vectorized_by_llvm": sum(r.vectorized for r in subset),
        "runtime_alias_checks": sum(r.memchecks for r in subset),
        "alias_blocked_remarks": sum(len(r.alias_blocked) for r in subset),
        "loop_vectorize_missed_reasons": dict(missed.most_common()),
    }


def _time_workloads(clang: str, opt: str, work: Path, iterations: int) -> list[dict[str, object]]:
    """Per-tick time of each benchmark workload: inlined control vs. the same
    IR with perfect per-leaf scopes, fused and per-stage lowerings."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_native import INTRINSICS_C, _build_plan, _render_driver

    rows: list[dict[str, object]] = []
    for path in sorted(WORKLOAD_DIR.glob("*.lock")):
        source = path.read_text(encoding="utf-8")
        plan = _build_plan(path.stem, source)
        row: dict[str, object] = {"workload": path.stem}
        for lowering, ir_text in (("fused", plan.ir_text), ("per_stage", _per_stage_ir(source))):
            inlined = _inline_kernels(ir_text, opt, work, f"{path.stem}.{lowering}.time")
            scoped, _, _ = _tag_arena_scopes(inlined)
            for label, text in (("control", inlined), ("scoped", scoped)):
                stem = work / f"{path.stem}.{lowering}.{label}"
                stem.with_suffix(".ll").write_text(text, encoding="utf-8")
                stem.with_suffix(".c").write_text(
                    _render_driver(plan, iterations, max(iterations // 20, 1)),
                    encoding="utf-8",
                )
                subprocess.run(
                    [clang, *CLANG_FLAGS, str(stem.with_suffix(".c")), str(stem.with_suffix(".ll")),
                     str(INTRINSICS_C), "-o", str(stem), "-lm"],
                    check=True, capture_output=True,
                )
            # Interleave the two binaries so neither inherits the other's cache
            # state in a fixed order; report the median of the rounds.
            samples: dict[str, list[float]] = {"control": [], "scoped": []}
            checksums: dict[str, float] = {}
            for round_index in range(7):
                order = ("control", "scoped") if round_index % 2 == 0 else ("scoped", "control")
                for label in order:
                    out = subprocess.run(
                        [str(work / f"{path.stem}.{lowering}.{label}")],
                        check=True, capture_output=True, text=True,
                    ).stdout
                    measured = json.loads(out.strip().splitlines()[-1])
                    samples[label].append(float(measured["per_tick_ns"]) / 1000.0)
                    checksums[label] = float(measured["checksum"])
            control = sorted(samples["control"])[3]
            scoped_time = sorted(samples["scoped"])[3]
            row[f"{lowering}_control_us"] = round(control, 2)
            row[f"{lowering}_scoped_us"] = round(scoped_time, 2)
            row[f"{lowering}_speedup"] = round(control / scoped_time, 3)
            row[f"{lowering}_checksums_agree"] = checksums["control"] == checksums["scoped"]
        rows.append(row)
    return rows


def run(
    generated: int, clang: str, opt: str | None, keep: Path | None, iterations: int
) -> dict[str, object]:
    work = keep or Path(tempfile.mkdtemp(prefix="lsalias_"))
    work.mkdir(parents=True, exist_ok=True)
    reports: list[ProgramReport] = []
    for name, source in _corpus(generated):
        lowerings = {
            "fused": compile_lockstep(source, verbose=False).llvm_ir,
            "per_stage": _per_stage_ir(source),
        }
        for lowering, ir_text in lowerings.items():
            reports.append(_analyze(name, lowering, ir_text, clang, work))
            if opt is None:
                continue
            inlined = _inline_kernels(ir_text, opt, work, f"{name}.{lowering}")
            scoped, _, _ = _tag_arena_scopes(inlined)
            reports.append(_analyze(name, f"{lowering}_inlined", inlined, clang, work))
            reports.append(_analyze(name, f"{lowering}_scoped", scoped, clang, work))

    variants = ["fused", "per_stage"]
    if opt is not None:
        variants += ["fused_inlined", "fused_scoped", "per_stage_inlined", "per_stage_scoped"]
    summary: dict[str, object] = {
        "census": {variant: _summarize(reports, variant) for variant in variants}
    }
    if opt is not None and iterations > 0:
        summary["timing"] = _time_workloads(clang, opt, work, iterations)
    if keep is None:
        shutil.rmtree(work, ignore_errors=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generated", type=int, default=200, help="random oracle programs to include")
    parser.add_argument("--keep", type=Path, default=None, help="keep IR and records here")
    parser.add_argument("--output", type=Path, default=None, help="write JSON summary here")
    parser.add_argument(
        "--iterations", type=int, default=2000, help="ticks per timing round (0: skip timing)"
    )
    args = parser.parse_args()
    clang = shutil.which("clang")
    if clang is None:
        print("clang not on PATH", file=sys.stderr)
        return 2
    opt = shutil.which("opt")
    if opt is None:
        print("opt not on PATH: skipping the perfect-alias-scope experiment", file=sys.stderr)
    summary = run(args.generated, clang, opt, args.keep, args.iterations)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
