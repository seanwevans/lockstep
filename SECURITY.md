# Security policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ Active |

Lockstep is pre-1.0 software under active development. Security fixes are applied to the latest release on `main`. There are no long-term support branches at this time.

## Reporting a vulnerability

If you discover a security vulnerability in Lockstep, please report it responsibly by emailing **sean.w.evans@gmail.com** with the subject line `[lockstep security]`. Please include a description of the issue, steps to reproduce it, and any relevant source files or compiler output.

You should expect an initial acknowledgment within 72 hours. Please do not open a public GitHub issue for security vulnerabilities until a fix has been released or 90 days have passed since your initial report, whichever comes first.

## Security model

Lockstep is a compiler and language toolchain. Its security boundaries differ from those of a typical application. The following sections describe the trust model for each component and the known risks associated with it.

### Compiler frontend (parser and semantic validator)

The compiler frontend accepts Lockstep source files (`.lock`) as input and produces a typed AST, LLVM IR, and C host headers as output. The frontend is implemented in Python using an ANTLR4-generated parser.

**Trust assumption:** Source files are provided by the developer and are considered trusted input. The compiler does not sandbox or resource-limit the parsing process.

**Known risks:**

- **Denial of service via pathological input.** The ANTLR4 parser does not enforce input size limits, maximum nesting depth, or parse time budgets. A maliciously crafted `.lock` file with deeply nested expressions or extremely long identifier chains could cause excessive memory consumption or CPU time during parsing. This is not currently mitigated.
- **Generated parser code is committed to source control.** The `generated/parser/` directory contains Python files produced by the ANTLR4 tool. CI enforces that these files match the grammar via `make check-generated-parser`, but a supply-chain compromise of the ANTLR4 toolchain could inject malicious code into the generated parser. Reviewers should treat changes to `generated/` with the same scrutiny as changes to `Lockstep.g4`.

### LLVM IR code generation

The `--emit-ir` flag produces LLVM IR text output via `llvmlite`. This IR is intended to be consumed by an external LLVM toolchain (`llc`, `clang`, or an embedding host) for final compilation to machine code.

**Trust assumption:** The generated IR is a build artifact, not an executable. The compiler itself does not invoke `llc` or produce machine code directly (with one exception noted below).

**Known risks:**

- **IR correctness affects downstream safety.** If the code generator produces structurally invalid or semantically incorrect IR, the downstream LLVM toolchain may produce binaries with undefined behavior. The current generated IR uses a single arena pointer and byte-offset-derived accesses rather than blanket `noalias` annotations, so reviewers should focus on structural IR validity, correct offset calculation, and sound lowering of arena loads/stores. This is mitigated by the compiler's test suite but not formally verified.

### C host header generation

The `--emit-header` flag produces a C header file containing struct definitions, arena layout macros, byte-offset constants, and the `Lockstep_SaturatedWriteIndex()` inline function.

**Trust assumption:** The generated header is included by a host application compiled by the developer. The header contents are derived entirely from the Lockstep source program.

**Known risks:**

- **Identifier injection.** Struct names, stream names, and field names from the Lockstep source are embedded in the C header as identifiers and macro names. The `sanitize_symbol()` utility restricts identifiers to `[A-Za-z0-9_]`, but adversarial identifiers in the source could produce confusing or conflicting macro definitions (for example, a stream named `ARENA_BYTES` would collide with `LOCKSTEP_ARENA_BYTES`). Macro names are prefixed with `LOCKSTEP_` to reduce collision risk, but no namespace isolation is enforced.
- **Integer overflow in arena size calculation.** The arena byte-offset computation sums field sizes without overflow checking. A source program declaring extremely large stream capacities (for example, `stream<Entity, 4294967295>`) could produce a `LOCKSTEP_ARENA_BYTES` value that overflows the host platform's `size_t`. The host application is responsible for validating the arena size before allocation.

### Pipeline simulator and subprocess reduction execution

The `--simulate` flag executes a lightweight simulation of the pipeline's bind routes to validate wiring and cardinality. Fold reductions (`fold sum`, `fold avg`) are lowered to LLVM IR in-process, written to a temporary file, and executed via an external subprocess (`clang`-compiled executable when available, otherwise `lli`).

**Trust assumption:** The simulator operates on data derived from the developer's own source program. It is a development-time validation tool, not a production execution environment.

**Known risks:**

- **Native-code execution still occurs, but out-of-process.** The simulator no longer executes generated native code in the compiler process address space. This removes the highest-severity in-process memory-corruption risk from MCJIT execution. However, a subprocess still executes generated code and therefore still carries host-level execution risk if used on untrusted input.
- **Toolchain dependency and fallback behavior.** Subprocess execution depends on `clang` or `lli` being present at runtime. If neither is available (or subprocess compilation/execution fails), the simulator falls back to Python `sum` semantics for numeric folds to preserve simulator availability.
- **Timeout-bounded subprocesses.** Compilation and execution subprocess calls are timeout-bounded to reduce runaway execution risk, but they are not cgroup/namespace isolated by Lockstep itself.
- **Resource-limited execution subprocess (POSIX).** The subprocess that executes generated native code is launched with clamped POSIX resource limits (`RLIMIT_CPU`, `RLIMIT_AS`, `RLIMIT_FSIZE`, and `RLIMIT_CORE` set to `0`) via a `preexec_fn`. This bounds CPU time, address space, output file size, and core dumps so that a pathological or malicious generated program cannot spin, exhaust memory, or fill the disk beyond these ceilings — a stronger guarantee than the wall-clock timeout alone, which does not cap memory or output. The limits are lowered from the inherited hard limits only (never raised) and degrade gracefully where a limit is unsupported. On platforms without `rlimit` support (Windows), the subprocess falls back to the wall-clock timeout.

### LSP server

The `lockstep-lsp` command starts a Language Server Protocol server that communicates over stdio. It provides live diagnostics, hover type information, go-to-definition, and autocompletion.

**Trust assumption:** The LSP server is invoked by the developer's editor and processes files from the developer's workspace. It does not listen on a network socket.

**Known risks:**

- **File system access is scoped to the workspace.** The LSP server reads `.lock` files from paths provided by the editor client. It does not write files. However, it does not validate that file paths are within the workspace root, so a malicious LSP client could request analysis of arbitrary files readable by the user. This is a low-severity risk because the LSP protocol is a local IPC mechanism and the server only reads file contents (it does not execute them or transmit them externally).
- **No authentication.** The LSP server does not authenticate the connecting editor client. This is standard for stdio-based LSP servers and is not a meaningful risk in the intended deployment model (single-user, local editor).

### The saturated write memory model

Lockstep's memory model uses saturation arithmetic for stream indices. When a write exceeds stream capacity, the index is clamped to `capacity - 1` rather than overflowing. This means the final element of each stream acts as a "trash can" that absorbs out-of-bounds writes.

This design eliminates buffer overflows by construction in the generated code, which is a meaningful security property for the host application. However, it introduces a silent data loss mode: if a pipeline produces more output elements than the stream capacity, excess results are silently discarded. The `LOCKSTEP_DEBUG_SATURATED_WRITES` compile-time flag in the generated C header enables telemetry logging when saturation occurs, which the host application can use to detect capacity exhaustion.

## Dependency security

Lockstep depends on two runtime packages:

- **antlr4-python3-runtime** — the ANTLR4 parser runtime for Python. This is a widely used, mature package maintained by the ANTLR project.
- **llvmlite** — Python bindings for LLVM, maintained by the Numba project. This package ships platform-specific binary wheels containing compiled LLVM libraries.

Both dependencies are sourced from PyPI. Lockstep does not vendor either dependency. A supply-chain compromise of either package would directly affect Lockstep's security, particularly `llvmlite` which is used to construct LLVM IR for code generation and simulator fold reductions.

The optional `lsp` dependency group adds `pygls` and `lsprotocol`, which are used only by the LSP server and do not affect compiler output.

### Locked dependency workflow

Dependency resolution is committed to hashed lockfiles generated from `pyproject.toml`:

- `requirements.lock` (runtime)
- `requirements-test.lock` (runtime + `test`)
- `requirements-lsp.lock` (runtime + `lsp`)

Update flow:

1. Edit dependency declarations in `pyproject.toml`.
2. Regenerate lockfiles with `make lock-deps` (uses `pip-compile --generate-hashes`).
3. Commit `pyproject.toml` + all updated `requirements*.lock` files together.

Verification in CI:

- `make check-lock-deps` regenerates lockfiles and fails if the committed lockfiles are stale relative to `pyproject.toml`.
- Installation steps use `pip install --require-hashes -r <lockfile>` so tampered or missing hashes fail fast.

## Hardening roadmap

The following improvements are planned for future releases:

- **Stronger process isolation** for simulator subprocesses. POSIX resource limits (CPU, address space, file size, core dumps) are now applied to the execution subprocess; deeper isolation (namespace/cgroup sandboxing when available) remains future work.
- **Formal verification** of arena aliasing and offset correctness invariants, confirming that generated accesses are sound across all valid Lockstep programs.
- **Automated dependency vulnerability scanning** via Dependabot or similar tooling in CI.
