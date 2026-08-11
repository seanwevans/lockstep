# Lockstep Roadmap

Lockstep is a single-maintainer project. This roadmap tracks the path from the
current release toward a stable v1.0.0 in two parts: what already ships, and
what remains. It is the authoritative status — earlier milestone-numbered plans
have been folded into the two lists below.

## Delivered

A number of features once planned for later milestones already ship in the tree.
Each links to the code that provides it.

| Feature | Where |
| --- | --- |
| Typed `AstProgram` as the internal representation, with an AST-based semantic validator | `ast.py`, `semantic_validator.py`, `tests/test_semantic_validator_ast_path.py` |
| `uint` / `double` as first-class declared types | `semantic_validator.py`, `codegen.py` (`_PRIMITIVE_TYPE_MAP`) |
| `select` expression (branchless typed mux) | `codegen.py`, `semantic_validator.py`, `simulator.py` |
| `import` / `#include` resolution (root-sandboxed, circular-import detection) | `compiler.py` (`_resolve_dependency_sources`) |
| SoA decomposition in the arena layout | `arena_layout.py` (`_flatten_type_leaves`, `_build_layout_from_bindings`) |
| Single-arena-pointer `Lockstep_Tick` ABI | `codegen.py`, `examples/minimal_host.c` |
| Parameterized SIMD width (`--target-width`, `LOCKSTEP_SIMD_WIDTH`) | `codegen.py`, `c_header.py`, `tests/test_target_width_execution.py` |
| Fused-vector lowering, incl. accumulator-stage fusion | `codegen.py`, `benchmarks/native/fusion_probe.py` |
| Fold-into-kernel fusion (single-fold accumulator reduced in-register, no per-row buffer; reaches hand-written-C parity on `particle_energy`) | `codegen.py` (`_lower_reduction_route`), `benchmarks/native/lockstep_vs_c.py`, `tests/test_fold_reduction_fusion.py` |
| Arena size-overflow checking + C `static_assert` (`LCK502`) | `arena_layout.py`, `c_header.py` |
| Parser input-complexity limits (size / nesting / parse timeout) | `compiler.py` (`FrontendLimits`), `cli.py` |
| Out-of-process, resource-limited simulator reduction | `simulator.py`, `SECURITY.md` |
| Comment-preserving formatter | `formatter.py`, `tests/test_formatter.py` |
| Benchmark suite (frontend, native, SoA-vs-AoS, fusion) | `benchmarks/`, `benchmarks/RESULTS.md` |
| Hashed, pinned dependency lockfiles | `requirements*.lock`, `make check-lock-deps` |
| Opt-in LSP: diagnostics, hover, go-to-definition, completion | `lsp.py` |

## Remaining toward v1.0.0

**Backend**

- **Scoped alias metadata on arena-derived pointers.** `Lockstep_Tick`'s arena
  parameter is already `noalias nocapture` (provably sound — the sole pointer
  parameter). What remains is disambiguating the individual stream/accumulator
  pointers *inside* the tick: emitting `noalias` on kernel pointer parameters
  (guarded by a whole-program check that no bind route feeds one resource to two
  pointer params) and/or `!alias.scope` metadata per disjoint arena region,
  backed by a short soundness argument in a `PROOFS.md`.
- **Filter-group fusion.** Accumulator stages already fuse; the remaining
  unfused case is a stage group containing a `filter`
  (see `benchmarks/native/README.md`).

**Language**

- **Multi-stage pipeline composition.** Let a pipeline consume another
  pipeline's output streams, with a combined arena layout and topological tick
  ordering.

**Tooling & developer experience**

- **Diagnostic catalog.** A machine-readable catalog generated from
  `SEMANTIC_DIAGNOSTIC_CODES` (code, severity, message, explanation) plus a
  `DIAGNOSTICS.md`, surfaced in LSP hovers.
- **LSP workspace features.** Multi-file diagnostics across `import`s, rename, and
  a `--visualize` DAG view.

**Verification**

- **Grammar-aware fuzzing.** A dedicated fuzz target over the existing
  property-based generators, checking for crashes, diagnostic completeness, and
  `llvm-as`-valid IR.

**Stability (the 1.0 gate)**

- **API & grammar freeze** with semantic-versioning guarantees on
  `compile_lockstep`, `LockstepCompileResult`, the diagnostic catalog, the
  grammar, and the C header ABI.
- **`MIGRATING.md`** documenting breaking changes up to 1.0.
- **Automated dependency vulnerability scanning** (Dependabot or equivalent) on
  top of the existing hashed lockfiles.

## What v1.0.0 means

1. **Correct** — every valid program's generated IR, compiled against the
   generated header, matches the simulator's observable behavior, backed by the
   `noalias` argument, fuzzing, and the benchmark suite.
2. **Stable** — the public API, diagnostic catalog, grammar, and C header ABI are
   frozen under semantic versioning.
3. **Complete** — CLI, simulator, LSP, formatter, header generator, and IR
   backend together cover authoring → validation → compilation → host
   integration.

## Non-goals for v1.0.0

- **Direct machine-code emission.** The compiler emits LLVM IR; final codegen is
  delegated to the host's LLVM toolchain.
- **GPU backends** (SPIR-V / DXIL / Metal). CPU SIMD is the initial target; GPU
  may follow in a v2.x series.
- **A package manager** for multi-file projects. `import` resolution handles
  files; there is no registry or versioning.
- **Runtime profiling / tracing** beyond the `LOCKSTEP_DEBUG_SATURATED_WRITES`
  hook.
