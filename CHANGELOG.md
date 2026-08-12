# Changelog

All notable changes to Lockstep are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the API and grammar are pre-1.0, breaking changes may occur in minor
releases; see `ROADMAP.md` for the path to a frozen 1.0.0.

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

[0.2.0]: https://github.com/seanwevans/lockstep/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/seanwevans/lockstep/releases/tag/v0.1.0
