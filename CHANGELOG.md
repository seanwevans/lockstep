# Changelog

All notable changes to Lockstep are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the API and grammar are pre-1.0, breaking changes may occur in minor
releases; see `ROADMAP.md` for the path to a frozen 1.0.0.

## [Unreleased]

### Added

- **Fusing through filters that drop rows.** A multi-stage group whose filter has
  a data-dependent `return` now fuses into one vector loop. The keep flag
  becomes a lane mask, the group's sink is written with
  `llvm.masked.compressstore` at a running write index, and fold accumulators
  add the operator identity for dropped lanes. On the new
  `telemetry_drop_unhealthy` workload this is 3.0× faster than the per-stage
  path. It is also 1.31× faster than a hand-written branchy C compaction loop,
  on AVX-512. Without AVX-512 the store is expanded lane by lane: 1.05× on AVX2.
- **Live row counts** (`LOCKSTEP_OFFSET_COUNT_<STREAM>`, `uint32_t`). They cover
  every stream whose row count is only known at run time: a filter's output,
  and any stage fed only by such streams. The tick writes the count; later
  stages loop to it, and folds reduce only the rows it covers.
- **Alias-analysis probe** (`benchmarks/native/alias_probe.py`,
  `make bench-alias`). It measures which optimizations LLVM's alias analysis
  blocks in `Lockstep_Tick`, and what sound per-leaf scoped alias metadata
  would recover. The result: perfect scopes add little once the changes below
  are in, and no benchmark workload gets measurably faster. Scoped
  metadata moves to "Deferred past v1.0.0" in `ROADMAP.md`.

- **Differential oracle** (`tests/test_differential_oracle.py`, `make oracle`).
  It generates random valid programs and checks the simulator against
  clang-compiled `Lockstep_Tick` on every sink-stream row and every folded
  uniform, at several SIMD widths. Every fix below was found by it and has a
  minimal regression test.

### Changed

- **Uniforms are loaded once per route, before its row loop.** Codegen used to
  reload a uniform on every row, and LLVM couldn't prove the row stores miss it.
  That kept per-stage shader loops scalar. Such loops are now vectorized: 1.52×
  on a 1M-row `Brighten`.
- **Row indices are clamped with a scalar `smax`/`smin`** instead of a
  `<4 x i32>` splat/select/extract idiom. Scalar evolution can analyze the new
  form, so the vectorizer can bound the loop, and instcombine removes the clamp
  when the trip count equals the capacity. That is 1.72× on a nested-struct
  fan-in stage. A loop with a run-time row count (after a filter) skips the
  clamp whenever its static row bound fits the stream, which LLVM can't infer
  on its own.

- **The simulator now uses the compiled numeric model.** `float` is IEEE single
  precision (it used to be double), and `int` wraps at 32 bits with C
  truncating `/` and `%` (it used to be Python's unbounded ints, floor `%`, and
  float `/`). An `out` row now starts from the target stream's current row. It
  used to start as a copy of the input row, which could have a different struct
  type.
- `fold` operators are now ordinary identifiers, checked by the validator
  (`LCK401`). As a result, `min`, `max`, `sum` and `avg` can be used as call
  and variable names.

### Fixed

- **Stages after a filter processed the rows the filter dropped.** A stream had
  no live row count, so the next stage ran over the filter output's full
  capacity, including the stale rows past the kept ones. Folds downstream
  counted those rows too. The differential oracle had pinned this as a known
  divergence; it now matches the simulator. When a stage reads two filtered
  streams with different counts, it runs to the longer count and reads the
  shorter stream as zeros past its end, as the simulator does. A fold over zero
  rows now gives the operator identity in the simulator too (it used to give
  `null`).

  **This changes the ABI:** `LOCKSTEP_ARENA_BYTES` grows by 4 bytes per counted
  stream. The count slots come after the uniforms, so no existing offset moves.
- **`min(...)` and `max(...)` could not be called.** They were lexer keywords,
  reserved for the `fold` operators, so the documented intrinsics failed to
  parse.
- **Folds could segfault.** The fold strip-mine loop loaded the accumulator
  with the vector type's natural alignment. The arena is packed, so an
  accumulator at an unaligned offset faulted on an aligned SSE load
  (`movaps`).
- **Fused groups dropped or corrupted in-place stages.** `s = A(..); s = B(s, s)`
  in one fused group:
  - never stored the final `s`;
  - lowered only the group's first route when `s` was its only intermediate;
  - read an in-place stage's input from an uninitialized slot in the group's
    scalar tail.
- **`bool` casts.** `(float) true` produced `-1.0`, because the cast was lowered
  as a signed conversion of an `i1`. A `float` → `bool` cast is now a
  comparison with zero.
- **`smoothstep(e, e, x)` returned 1.0 in compiled code.** The inlined scalar
  and vector lowerings divided by zero and clamped the result. It now returns
  0, like the simulator and the C reference intrinsic.
- The C reference `pure_mix` in `benchmarks/native/lockstep_intrinsics.c` now
  uses the same `a*(1-t) + b*t` operation order as the compiler and the
  simulator.

## [0.3.0] - 2026-09-24

A correctness release. Folded uniforms computed by `fold` never reached the
host in 0.2.0 — and because nothing consumed the reduction, LLVM deleted it as
dead code. That is fixed here, at the cost of a small ABI change (see below);
rebuild hosts against the regenerated header. Multi-stage pipelines with a
pass-through filter also fuse into a single vector loop now.

### Fixed

- **Folded uniforms now reach the host.** `uniform T x = fold op(y);` declares
  `x`, but that declaration produced no arena slot: `_lower_fold_route` hit its
  `uniform_name not in uniform_slots` guard and returned without storing, the
  generated header exposed no offset for `x`, and a host had no way to read the
  folded scalar. Because nothing consumed the reduction, LLVM deleted the whole
  accumulator dataflow as dead code — for a copy-shaped pipeline such as
  `telemetry_filter_aggregation` the compiled `Lockstep_Tick` collapsed to a
  column copy containing no floating-point arithmetic at all. Fold-declared
  uniforms are now registered like any other pipeline uniform, so they get an
  arena slot, a `LOCKSTEP_OFFSET_UNIFORM_*` macro, and a real store.

  **This changes the ABI:** `LOCKSTEP_ARENA_BYTES` grows by one element per
  folded uniform (stream and accumulator offsets are unchanged — the uniforms
  are appended). Rebuild hosts against the regenerated header. Output checksums
  are unchanged.

### Added

- **Pass-through filter-group fusion.** A multi-stage group whose only filter
  keeps every row unconditionally (no `return`) now fuses through that filter
  into a single vector loop instead of falling back to one strip-mined loop per
  stage. The eliminated intermediate streams stay in registers, each SoA leaf
  column is moved as a contiguous `<N x T>` vector load/store (bool columns via
  an `i8` memory type), and the fold accumulators are carried in loop-carried
  vector registers — reduced horizontally at the end, with multiple folds per
  accumulator supported — rather than an O(rows) arena buffer. This lifts the
  `multi_stage_pipeline` and `telemetry_filter_aggregation` benchmarks to ~0.9×
  and ~0.7× of hand-written C (from ~0.21× and ~0.19×), a ~3.9× native-throughput
  gain. A filter with a data-dependent `return` still uses the per-stage
  compacting fallback. The ABI (arena layout, offset macros) is unchanged.

### Changed

- The native benchmark harness (`benchmarks/native/run_native.py`) now checksums
  a pipeline's **terminal** output column instead of the first bind target, which
  may be an intermediate stream that fusion no longer materializes. This changes
  the reported `multi_stage_pipeline` checksum (13105.6 → 20969.0) with no change
  to the computed result.

## [0.2.0] - 2026-08-12

The internal representation moved to a typed `AstProgram`, the code generator
gained real vectorization and fusion, and the surrounding tooling (simulator,
LSP, formatter, CLI) was hardened. This is a large step toward the v1.0.0 gate
described in `ROADMAP.md`.

### Added

- **Fused-vector codegen.** Accumulator stages fuse into a single vectorized
  loop, fused kernels use LLVM vector loads/stores, and single-fold accumulator
  reductions fuse into the kernel loop — reaching hand-written-C parity on
  `particle_energy`.
- **First-class `uint` / `double`** with unified numeric type promotion, array
  element access in codegen, and `select` as a branchless typed mux.
- **Parameterized SIMD width** (`--target-width`, `LOCKSTEP_SIMD_WIDTH`), with
  result-equivalence pinned across widths.
- **Scoped alias metadata (first step):** `Lockstep_Tick`'s arena parameter is
  now marked `noalias nocapture`.
- **Benchmark suite:** native execution benchmarks, a Lockstep-vs-hand-written-C
  harness (`benchmarks/native/lockstep_vs_c.py`), an SoA-vs-AoS layout
  micro-benchmark, a multi-stage fusion probe, and `benchmarks/RESULTS.md`.
- **CI coverage:** ruff lint + a coverage floor, enforced `mypy --strict`,
  golden-IR snapshot tests, and native-benchmark invariants gated against a
  committed baseline.
- **Comment-preserving formatter** and improvements to LSP type analysis,
  symbol resolution, and completion.
- **Hashed, `uv`-generated universal lockfiles.**

### Changed

- **Typed AST is now the internal representation.** The semantic validator
  operates on the typed AST; legacy parse-tree validation visitors were removed.
- **Codegen split** into focused modules (`codegen_lowerer`,
  `codegen_intrinsics`, and a legacy entity-dict adapter).
- **Simulator hardened:** out-of-process execution is resource-limited (see
  `SECURITY.md`), with struct-aware defaults and stricter input validation.
- **CLI** invocation simplified; compiler keyword arguments and dependency
  diagnostics threaded through; API/CLI names exported.
- Default dependency-file limit bounded to 128; parser input-complexity limits
  (size / nesting / parse timeout) enforced.
- GitHub Actions bumped to Node 24 releases.
- `ROADMAP.md` and `README` reconciled with delivered features.

### Fixed

- C header sizing for folded array/vector leaves, SoA offsets, target width, and
  arena overflow diagnostics (with C `static_assert`); leaf arena byte
  addressing corrected.
- Compiled `filter` compaction semantics; fold routes treated as optimizer DAG
  consumers; fold accumulators sized to stream capacity and strip-mined for wide
  widths.
- Integer→bool casts preserve truthiness (`icmp ne` instead of `trunc`); true
  short-circuit lowering for `&&` / `||`; `fptoui` emitted for float→uint casts.
- Simulator multi-input trip counts, saturated-stream FIFO windowing, and
  unmapped/missing lookups (now raise instead of returning wrong values).
- Repaired the opt-in LLVM simulator reduction path.
- Formatter crash on `import` / `#include` declarations; LSP crash when indexing
  struct field types.
- Compiler crash on raw backslashes in string literals and Windows dependency
  paths.
- Dependency resolution, limit enforcement, and source-map preservation.

## [0.1.0] - 2026-03-16

Initial pre-release.

[Unreleased]: https://github.com/seanwevans/lockstep/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/seanwevans/lockstep/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/seanwevans/lockstep/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/seanwevans/lockstep/releases/tag/v0.1.0
