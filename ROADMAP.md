# Lockstep Roadmap — v0.1.0 → v1.0.0

This document describes the planned evolution of the Lockstep compiler, language, and toolchain from the current v0.1.0 release through a stable v1.0.0. Each milestone builds on the previous one, and the ordering reflects real dependency chains in the codebase: internal representation cleanup enables backend improvements, which enable runtime features, which enable the stability guarantees required for a 1.0 release.

The scope of each milestone is deliberately conservative. Lockstep is a single-maintainer project and the milestones are sized to be individually shippable without leaving the project in a half-migrated state.

---

## v0.2.0 — Internal Representation Consolidation

The central goal of v0.2.0 is to eliminate the dual AST / entity-dict representation that currently threads through the compiler pipeline. Today, `emit_llvm_ir`, `emit_c_header`, and the simulator all accept `AstProgram | dict[str, Any]`, with normalization shims (`_normalize_codegen_input`, `_normalize_structs`) bridging the gap. The typed `AstProgram` is already the primary path; v0.2.0 makes it the only path.

### Single-source-of-truth AST

The `AstProgram` frozen-dataclass tree becomes the sole internal representation passed between compiler phases. The entity dict format is retained exclusively as a serialization format for `--dump` JSON output, produced at the CLI boundary by `ast_to_entities` rather than consumed internally. The `emit_llvm_ir` and `emit_c_header` functions are refactored to accept `AstProgram` only, removing the `AstProgram | dict[str, Any]` union signatures and the associated normalization code.

### Semantic validator migration (phase 1)

The semantic validator begins its migration from operating on ANTLR parse-tree context nodes to operating on the typed AST. In v0.2.0 the scope is limited to the non-expression checks: struct declarations, kernel signatures, pipeline resource declarations, bind-route validation, and fold semantics. These checks are straightforward to rewrite against `AstStructDecl`, `AstKernelDecl`, `AstPipelineDecl`, and `AstBindRoute` because they involve simple name/type lookups rather than recursive expression traversal. The expression type resolver (`_resolve_expr_type`) remains parse-tree-based in this release.

### Debug visitor retirement

The legacy `build_debug_visitor` code path is removed from the compilation pipeline. It remains available as a standalone diagnostic tool (the `[Shader Kernel]` / `[Pipeline Topology]` pretty-printer), but the compiler no longer falls back to it when the AST builder succeeds. The `except TypeError` fallback in `_compile_lockstep_with_dependencies` is narrowed further or removed entirely once all test doubles are updated to work with the typed AST path.

### Deprecation of string-based `body` fields

The entity dict's `body` field (which contains statement text like `"new_pos.x = pos.x;"`) is marked as deprecated in the `--dump` output. The `body_ast` field containing typed `AstStatement` objects becomes the canonical representation. Downstream consumers (the simulator, the codegen) already require `body_ast`; this change makes the deprecation explicit and removes the `_statement_to_text` serialization from the hot path.

---

## v0.3.0 — Semantic Validator Completion and Expression Type System

### Expression type resolver on typed AST

The remaining half of the semantic validator migration: `_resolve_expr_type` is rewritten to operate on the `AstExpr` discriminated union (`AstExprLiteral | AstExprVar | AstExprUnary | AstExprBinary | AstExprCall | AstExprCast`) using `isinstance` dispatch. This eliminates the `hasattr(ctx, 'mulExpr')` / `hasattr(ctx, 'bitwiseOrExpr')` pattern-matching on ANTLR context shapes, roughly halving the validator's line count and making it testable without constructing fake parse-tree nodes.

### Explicit local variable declarations

Local variable declarations remain aligned with the grammar's `varDecl: typeName ID ('=' expr)? ';';` rule: every local declaration carries an explicit type annotation, and initializers are validated against that declared type. Editor tooling may still surface inferred hover information for already-declared variables, but the core language does not accept untyped local allocations.

### `uint` and `double` as first-class declared types

The primitive type set is expanded from `{int, float, bool, string}` to include `uint` and `double`. The codegen already has `_PRIMITIVE_TYPE_MAP` entries for both; the change is in the semantic validator's `_primitive_types` set and the README's type system documentation. The `uint` type maps to `i32` (unsigned semantics enforced by using `udiv`/`urem`/`icmp unsigned` in the codegen rather than the signed variants). The `double` type maps to `f64` and participates in the same strict-matching rules as `float` — no implicit `float`↔`double` promotion.

### Diagnostic catalog

A machine-readable diagnostic catalog is generated from `SEMANTIC_DIAGNOSTIC_CODES` and shipped as part of the package. Each entry includes the code, default severity, message template, and a prose explanation of what the diagnostic means and how to fix it. This catalog is referenced by the LSP server's hover provider (showing diagnostic explanations inline) and published as a section of the README or a standalone `DIAGNOSTICS.md`.

---

## v0.4.0 — Backend Hardening and Target Flexibility

### Parameterized SIMD width

The hardcoded `simd_width = 8` in `codegen.py` is replaced by a target-dependent parameter. The compiler accepts a `--target-width` CLI flag (defaulting to 8 for AVX2) and the codegen uses it for vector splat construction, reduction intrinsic selection, and fold loop trip counts. The C header gains a `LOCKSTEP_SIMD_WIDTH` macro so the host application can query the compile-time vector width.

### Sandboxed JIT execution for the simulator

The in-process MCJIT execution path in `simulator.py` is replaced by out-of-process compilation. The simulator writes LLVM IR to a temporary file, invokes `llc` + `clang` (or a bundled `lli`) as a subprocess to compile and execute the fold reduction, and reads the result back via stdout. This eliminates the highest-severity security risk identified in `SECURITY.md` (in-process native code execution with no sandboxing) and removes the process-global `@lru_cache` on the execution engine.

### Input complexity limits for the parser

The parser frontend enforces configurable limits on input size (maximum file size in bytes), expression nesting depth (maximum recursive descent depth), and parse time budget (wall-clock timeout). These limits are documented in `SECURITY.md` and configurable via CLI flags (`--max-file-size`, `--max-nesting-depth`, `--parse-timeout`). The defaults are generous enough to never trigger on real programs but prevent denial-of-service from pathological input.

### Arena size overflow checking in the C header

The `emit_c_header` arena byte-offset computation validates that the cumulative size does not overflow a 64-bit unsigned integer. If the computed arena size exceeds `SIZE_MAX` (or a configurable limit), the compiler emits a diagnostic (`LCK502`) and refuses to generate the header. The generated header also includes a `_Static_assert` (C11) or `static_assert` (C++11) that `LOCKSTEP_ARENA_BYTES` fits in `size_t` on the target platform.

---

## v0.5.0 — Runtime and Host Integration

### Generated `Lockstep_Tick` implementation in LLVM IR

The `Lockstep_Tick` function currently emits a loop skeleton that loads/stores arena fields by pointer argument. In v0.5.0, the tick function is extended to operate on the `Lockstep_Arena` struct directly (accepting a single `struct Lockstep_Arena*` parameter), with correct GEP-based field access using the arena layout computed at compile time. This makes the generated IR directly compilable to a shared library or object file that a host application can link against, completing the host integration story started in `examples/`.

### SoA decomposition in the arena layout

The C header and LLVM IR arena layout are extended to support Struct-of-Arrays decomposition. When a stream is declared as `stream<Particle, 1000>` and `Particle` has fields `{float x; float y; float z;}`, the arena allocates three contiguous `float[1000]` arrays rather than one `Particle[1000]` array. The header emits per-field offset macros (`LOCKSTEP_OFFSET_STREAM_PARTICLES_X`, etc.) and the tick function uses the decomposed layout for SIMD-friendly access patterns. This delivers on the "SoA by Default" promise in the README.

### Stream capacity as a compile-time arena dimension

Stream capacities are promoted from source-level constants to first-class arena dimensions. The generated header and tick function parameterize over capacity so that a single compiled pipeline can be instantiated with different stream sizes at host bind time. The `Lockstep_BindMemory` function signature gains a capacity table argument, and the saturated write index function uses the runtime capacity rather than the compile-time constant.

---

## v0.6.0 — Language Features for Real Workloads

### `select` expression (branchless ternary)

A `select(condition, true_value, false_value)` built-in is added as a first-class expression form. Unlike `mix` (which interpolates), `select` performs a bitwise mux: the condition must be `bool`, and the true/false branches must have matching types. The codegen lowers `select` to LLVM's `select` instruction, which is a single-cycle operation on all modern SIMD targets. This fills the gap between `step`/`mix` (which require float operands) and the need for branchless integer/struct selection.

### Multi-stage pipeline composition

Pipelines can reference other pipelines' output streams as input streams, enabling multi-stage DAG composition without manually redeclaring stream types and capacities. The compiler validates cross-pipeline type compatibility and generates a combined arena layout that encompasses all stages. The `Lockstep_Tick` function executes stages in topological order.

### `import` resolution

The `import "path/to/file.lock"` and `#include "path/to/file.lock"` declarations, which currently parse but are not resolved, are connected to a file resolver. The compiler reads the referenced file, prepends it to the compilation unit (similar to the existing `--lib` mechanism), and remaps diagnostics to the correct source file. Circular imports are detected and rejected with a dedicated diagnostic code.

---

## v0.7.0 — LSP and Developer Experience

### Comment-preserving formatter

The `--format` flag is extended to preserve comments in their original positions. The formatter maintains a parallel token stream (including `HIDDEN`-channel comment tokens) and interleaves them with the reformatted output. This removes the current behavior of returning source unchanged when comments are present, which is correct but unhelpful.

### LSP workspace-wide diagnostics

The LSP server gains multi-file awareness. When a `.lock` file is opened, the server resolves its `import` declarations and validates the transitive closure of referenced files. Diagnostics from imported files are published to the editor, and go-to-definition works across file boundaries.

### LSP rename support

The LSP server supports `textDocument/rename` for struct names, field names, shader/filter/pure function names, and pipeline resource names. The rename operation updates all references across the workspace, including bind-route arguments and type annotations.

### Interactive pipeline visualizer

A `--visualize` CLI flag emits a self-contained HTML file containing a D3.js-based interactive DAG visualization of the pipeline topology. Nodes represent shaders/filters, edges represent stream dataflow, and the visualization shows stream capacities, accumulator types, and fold reduction targets. This is a developer tool for understanding complex pipeline topologies.

---

## v0.8.0 — Performance and Correctness Verification

### Formal `noalias` correctness proof

The `noalias` annotation on all arena pointer parameters in the generated LLVM IR is formally verified to be sound. The proof demonstrates that the Lockstep memory model (static arena, no user-controlled pointers, SoA decomposition with disjoint field arrays, saturated write indices) guarantees that no two pointer parameters in `Lockstep_Tick` can alias. The proof is documented in a standalone `PROOFS.md` and referenced from `SECURITY.md`.

### Benchmark suite

A benchmark suite is added that compiles and executes representative Lockstep programs (particle simulation, signal processing, collision detection) on multiple targets (AVX2, AVX-512, ARM NEON via cross-compilation) and reports throughput in elements/second. The benchmarks run in CI on tagged releases and publish results to a tracking dashboard. This provides empirical evidence for the "mathematically guaranteed to saturate CPU vector units" claim in the README.

### Fuzz testing for the parser and codegen

The parser and codegen are fuzz-tested using a grammar-aware fuzzer that generates syntactically valid but semantically adversarial Lockstep programs. The fuzzer targets memory safety (no crashes, no undefined behavior in the Python compiler process), diagnostic completeness (every invalid program produces at least one diagnostic), and IR validity (every generated LLVM IR module passes `llvm-as` verification). Fuzz findings are triaged and fixed before v1.0.0.

---

## v0.9.0 — Stability and API Freeze

### Public API surface declaration

The `lockstep_compiler` package declares its public API surface explicitly. `__init__.py` exports only the stable symbols, and all internal modules are prefixed with `_` or documented as unstable. The `compile_lockstep` function signature, the `LockstepCompileResult` dataclass, the `LockstepDiagnostic` dataclass, and the diagnostic code catalog are frozen — no breaking changes to these interfaces after v0.9.0.

### Grammar freeze

The `Lockstep.g4` grammar is frozen. No new keywords, no new expression forms, and no changes to the precedence table after v0.9.0. The grammar is versioned independently of the compiler, and the version is embedded in the generated parser files so that the compiler can detect grammar/parser version mismatches.

### Dependency pinning and hash verification

All runtime dependencies (`antlr4-python3-runtime`, `llvmlite`) and optional dependencies (`pygls`, `lsprotocol`) are pinned to exact versions with SHA-256 hash verification in a lockfile. The CI pipeline verifies that the lockfile is fresh and that no dependency has been tampered with. Dependabot (or equivalent) is configured for automated vulnerability scanning.

### Migration guide

A `MIGRATING.md` document describes all breaking changes from v0.1.0 through v0.9.0, with before/after code examples and automated fixup scripts where feasible. The guide covers entity dict format changes, removed CLI flags, renamed diagnostic codes, and any grammar additions that changed the meaning of previously valid programs.

---

## v1.0.0 — Stable Release

The v1.0.0 release is defined by three properties.

First, the compiler is correct: every valid Lockstep program produces LLVM IR that, when compiled and linked with the generated C header and a conforming host application, executes with the same observable behavior as the pipeline simulator. The `noalias` proof, the fuzz testing results, and the benchmark suite provide evidence for this claim.

Second, the interfaces are stable: the `compile_lockstep` API, the `LockstepCompileResult` shape, the diagnostic code catalog, the grammar, the C header ABI, and the `Lockstep_Tick` calling convention are all frozen and covered by semantic versioning guarantees. Breaking changes after v1.0.0 require a major version bump.

Third, the toolchain is complete: the CLI compiler, the pipeline simulator, the LSP server, the source formatter, the C header generator, and the LLVM IR backend all work together to support the full development lifecycle of a Lockstep program, from authoring through validation through compilation through host integration.

---

## Non-goals for v1.0.0

Some features are explicitly out of scope for the 1.0 release. These may appear in future major versions but are not on the critical path.

Direct machine code emission (bypassing `llc`/`clang`) is not planned. The compiler produces LLVM IR text; final compilation to native code is delegated to the host's LLVM toolchain. This keeps the compiler simple and avoids shipping platform-specific LLVM backends.

GPU compute shader output (SPIR-V, DXIL, Metal) is not planned for v1.0.0 despite the language's GPU-inspired design. The initial target is CPU SIMD. GPU backends may follow in a v2.x series.

A package manager or build system for multi-file Lockstep projects is not planned. The `import` resolution in v0.6.0 handles file-level dependencies, but there is no lockfile, no versioning, and no repository infrastructure for shared Lockstep libraries.

Runtime profiling or tracing instrumentation is not planned. The `LOCKSTEP_DEBUG_SATURATED_WRITES` mechanism in the C header is the extent of runtime observability in v1.0.0. A more comprehensive tracing framework may follow in a future release.
