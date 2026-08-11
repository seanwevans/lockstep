# Lockstep (Shader-C)
![logo.png](logo.png)

**Lockstep** is a data-oriented systems programming language for high-throughput,
deterministic compute pipelines — bridging the productivity of C and the
execution model of GPU compute shaders.

By enforcing **Straight-Line SIMD** execution and a **Static Memory Topology**,
Lockstep lets the compiler generate machine code built to saturate CPU vector
units without branch misprediction or cache contention.

## 1. Core Philosophy

* **Data-oriented by design.** Programs are modeled as physical circuits
  (pipelines), not sequences of instructions.
* **Zero branching.** `if`/`for`/`while` are banned inside kernels; branching is
  replaced by hardware-native masking and stream-splitting.
* **Predictable performance.** No `malloc`, no hidden threads, no GC. Memory is a
  static arena provided by the host.
* **Deterministic parallelism.** Race conditions are impossible by construction:
  state updates are isolated to `out` streams or linear `accumulator` types.

---

## 2. Language Architecture

### Pipeline topology

A Lockstep program is a Directed Acyclic Graph (DAG) of compute nodes:

* **`shader`** — 1-to-1: one input element produces one output element.
* **`filter`** — 1-to-0/1: conditionally passes data downstream.
* **`pure`** — a side-effect-free transform, strictly inlined.
* **`pipeline`** — the "circuit board" binding streams and uniforms to kernels.

### Memory model

Lockstep uses a **host-owned static arena**; the compiler computes every
member's byte offset at compile time.

* **SoA by default.** Structs are decomposed into parallel primitive arrays (one
  contiguous array per field) to maximize cache-line and SIMD utilization.
* **Saturated writes.** Stream indices use saturation arithmetic instead of
  bounds checks: past capacity, the final element acts as a "trash can" that
  absorbs writes without corruption or branching.

---

## 3. Syntax Guide

### Straight-line shaders

With `if`/`else` banned, conditionals use branchless intrinsics —
`step`, `mix`, `select`, `clamp`, `min`, `max`, `abs`, `sign`, `smoothstep`:

```c
shader ApplyPhysics(in Entity ent, out Entity updated, uniform float dt) {
    float fall_vy = ent.vy - (9.81 * dt);
    float bounce_vy = -ent.vy * 0.8;

    // step returns 1.0 if ent.y <= 0.0, else 0.0
    float is_grounded = step(0.0, -ent.y);

    // mix(a, b, t) acts as a hardware-level selector
    updated.vy = mix(fall_vy, bounce_vy, is_grounded);
    updated.y = max(ent.y + (updated.vy * dt), 0.0);
}
```

`select(cond, a, b)` is a branchless typed mux (a `bool` condition and matching
branch types), complementing the float-interpolating `mix`.

### Linear accumulators

Global reductions (total energy, bounds, …) use **linear types**. An accumulator
must be consumed by a `fold`, which the compiler lowers into a parallel reduction
tree:

```c
pipeline Simulation {
    stream<Entity, 10000> particles;
    accumulator<float> energy_sum;

    bind {
        particles = Calculate(particles, energy_sum);
        // fold consumes the linear type and produces a global scalar
        uniform float total_e = fold sum(energy_sum);
    }
}
```

### Type system

The semantic validator enforces a **strict type system with no implicit
coercions**.

* **Primitives:** `int`, `uint`, `float`, `double`, `bool`, `string`. `uint` has
  unsigned semantics; `double` is 64-bit float. Unknown types produce `LCK310`.
* **Composites:** struct members may be primitives, previously declared structs,
  or array suffixes (`T[4]`). Generic-wrapper spellings (`vector<float,4>`,
  nested `matrix<vector<Particle,4>,4>`) are accepted for type checking and
  arena/header layout, but are lowered as **opaque pointers** in IR — treat them
  as ABI/layout placeholders, not as kernel value types for arithmetic, field
  access, or SIMD. Type identity is name-based and exact; field chains (`a.b.c`)
  resolve only through concrete struct types.
* **Coercion:** none implicit — no widening/narrowing, no `int`⇄`float`
  promotion. Assignments, initializers, `pure` arguments/returns, and bind
  arguments require exact type equality. Mixed `int`/`float` arithmetic without a
  cast is rejected with `LCK424`. Use an explicit cast when conversion is wanted.

---

## 4. Compiler & Backend

Lockstep targets **LLVM IR** directly.

* **Single-arena ABI.** Kernels receive a `struct Lockstep_Arena*` and compute
  byte offsets into it. `Lockstep_Tick`'s arena parameter is marked
  `noalias nocapture` (it is the sole pointer parameter and every access is
  derived from it, so it is provably non-aliasing — a `restrict`-like guarantee
  at the ABI boundary). Scoped alias metadata on the individual arena-derived
  stream/accumulator pointers inside the tick is not yet emitted, so do not
  assume full intra-loop alias disambiguation (see [ROADMAP.md](ROADMAP.md)).
* **SSA locals.** Scalar and concrete-struct locals are lowered through
  SSA-friendly values where possible; arena loads/stores stay byte-addressed for
  ABI stability.
* **Manual SIMD lowering.** The fused-vector pass strip-mines contiguous stream
  elements and emits vector loads, stores, arithmetic, and reductions directly,
  rather than relying on LLVM auto-vectorization.
* **Fast-math reductions.** Reduction loops carry `fast` flags so LLVM can
  reassociate into horizontal SIMD shuffles.

---

## 5. Host Integration

The compiler emits a C-compatible header for the host (C/C++, Rust, Zig):

1. **Allocate** a `struct Lockstep_Arena` (or an aligned block of at least
   `LOCKSTEP_ARENA_BYTES`).
2. **Prime** initial data into the SoA fields at the header's byte offsets.
3. **Tick** by calling `Lockstep_Tick(arena)`. There is no separate
   `Lockstep_BindMemory` entry point.

See [`examples/minimal_host.c`](examples/) for a complete end-to-end host app.

---

## 6. CLI Usage

Install in editable mode to get the `lockstepc` entry point:

```bash
pip install -e .

lockstepc program.lock                 # compile
cat program.lock | lockstepc --dump    # read stdin, dump entities as JSON
lockstepc program.lock --format        # canonical straight-line formatting
lockstepc program.lock --emit-ir       # emit LLVM IR
lockstepc program.lock --emit-header   # emit C host header
lockstepc program.lock --simulate      # validate wiring/cardinality
lockstepc --version
```

### Pipeline simulation

`--simulate` validates pipeline wiring and cardinality before backend
generation. Provide inputs with `--simulate-input path.json`:

```json
{
  "streams": { "raw_positions": [{"id": 1}, {"id": 2, "_keep": false}] },
  "accumulators": { "energy": [0.5, 1.5] }
}
```

Output includes per-route `input_count`/`output_count`, updated stream
snapshots, accumulator contents, and folded uniforms. Folds (`sum`/`avg`) run in
deterministic pure-Python mode by default (including mixed `int`/`float`/`bool`
accumulators). An opt-in LLVM-backed reduction runs when `LOCKSTEP_SIM_USE_LLVM=1`
(or `use_llvm_runtime=True`); it executes out of process under POSIX resource
limits and reports an explicit error if `clang`/`lli` is missing rather than
silently falling back.

### Programmatic API

```python
from lockstep_compiler import LockstepCompileResult, compile_lockstep

result: LockstepCompileResult = compile_lockstep(source_code, verbose=True)
# result.parse_tree, result.entities, result.diagnostics
```

---

## 7. Diagnostics

Each diagnostic carries `severity` (`info`/`warning`/`error`), a stable `code`
(e.g. `LCK101`), a `message`, `line`, `column`, and an optional `hint`.

* **Non-fatal observations** (empty `bind` blocks, duplicate declarations,
  unreachable statements after a `return`) are returned in
  `LockstepCompileResult.diagnostics`; compilation still succeeds.
* **`pure` return rules:** `LCK413` (no `return`), `LCK414` (multiple returns),
  `LCK415` (statements after the first return), `LCK418` (return type mismatch).
* **Type mismatches:** `LCK412` (pure-arg), `LCK416` (initializer), `LCK417`
  (assignment), `LCK424` (mixed `int`/`float` without a cast).
* **Fatal parse errors** raise `LockstepCompileError` (`.errors` holds parse
  diagnostics; `.diagnostics` mirrors pre-failure context).

Generated headers expose `Lockstep_SaturatedWriteIndex(...)` and per-stream
`LOCKSTEP_CAPACITY_STREAM_<NAME>` macros. Define
`LOCKSTEP_DEBUG_SATURATED_WRITES` to log saturated writes, and override
`LOCKSTEP_SATURATED_WRITE_LOG(...)` to route them to custom telemetry.

---

## 8. Development

### Dependencies (locked + hashed)

Pinned lockfiles are generated from `pyproject.toml` with
[`uv`](https://github.com/astral-sh/uv): `requirements.lock` (runtime),
`requirements-test.lock` (+ `test`), `requirements-lsp.lock` (+ `lsp`). Install
and refresh with:

```bash
python -m pip install --require-hashes -r requirements-test.lock
make lock-deps        # regenerate after changing dependencies
```

CI enforces freshness (`make check-lock-deps`) and installs with
`--require-hashes`.

### Tests, types, and lint

```bash
make verify           # lint + tests + mypy
make test-cov         # tests with a coverage floor
```

`tests/test_golden_ir.py` pins the exact LLVM IR and C header for a curated
corpus (`tests/golden/programs/*.lock`) covering shaders, folds, filters, and
fused pipelines; any codegen change surfaces as a reviewable diff. Regenerate
intentional changes with `LOCKSTEP_UPDATE_GOLDEN=1 pytest tests/test_golden_ir.py`
(or `python tests/golden/regenerate.py`).

### Benchmarking

Measured results for every harness are published in
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md). Absolute numbers are
host-dependent — treat them as regression signals.

| Target | Measures |
| --- | --- |
| `make bench` | Frontend (parse + Python simulate) latency; writes `benchmark-results.json` |
| `make bench-check` | Frontend results vs. `benchmarks/baselines/default.json` (10% threshold, advisory) |
| `make bench-native` | Real throughput of `clang -O3` compiled code calling `Lockstep_Tick` |
| `make bench-soa` | SoA vs. AoS throughput for the same kernel |
| `make bench-fusion` | Fused vs. per-stage loop throughput |

On pull requests, CI gates the **deterministic** native invariants (arena ABI +
output `checksum`) against `benchmarks/baselines/native.json` — a drift is a real
codegen regression. Throughput is host-dependent and reported as an artifact, not
gated. See [`benchmarks/native/README.md`](benchmarks/native/README.md) for
details and baseline-update steps.

### Regenerating the parser

```bash
make generate-parser        # emits generated/parser/, committed to the repo
```

CI enforces freshness via `make check-generated-parser`.

---

## 9. Language Server (LSP)

An opt-in LSP server surfaces compiler diagnostics live and provides semantic
assistance:

```bash
pip install -e .[lsp]
lockstep-lsp
```

Capabilities: live diagnostics (`textDocument/publishDiagnostics`), go-to-
definition for struct members, hover type info (variables, fields, shader/pure
names), and bind-route / symbol autocompletion. The server speaks stdio and
works with standard editor LSP clients.
