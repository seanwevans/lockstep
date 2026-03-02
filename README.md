# Lockstep (Shader-C)

**Lockstep** is a data-oriented systems programming language designed for high-throughput, deterministic compute pipelines. It bridges the gap between the productivity of C and the brutal execution efficiency of GPU compute shaders.

By enforcing a strict **Straight-Line SIMD** execution model and **Static Memory Topology**, Lockstep allows the compiler to generate machine code that is mathematically guaranteed to saturate CPU vector units without the overhead of branch misprediction or cache contention.

## 1. Core Philosophy

* **Data-Oriented by Design:** Logic is secondary to data flow. Programs are modeled as physical circuits (pipelines) rather than sequences of instructions.
* **Zero Branching:** Standard control flow (`if`, `for`, `while`) is banned inside compute kernels. Branching is replaced by hardware-native masking and stream-splitting.
* **Predictable Performance:** No `malloc`, no hidden threads, and no garbage collection. Memory is a static arena provided by the Host.
* **Deterministic Parallelism:** Race conditions are impossible by construction. State updates are strictly isolated to `out` streams or linear `accumulator` types.

---

## 2. Language Architecture

### The Pipeline Topology

A Lockstep program is a Directed Acyclic Graph (DAG) of compute nodes.

* **`shader`**: A 1-to-1 mapping. Processes one input element and produces one output element.
* **`filter`**: A 1-to-0/1 mapping. Conditionally passes data to downstream nodes.
* **`pure`**: A side-effect-free mathematical transform. Strictly inlined.
* **`pipeline`**: The "circuit board" that binds streams and uniforms to kernels.

### The Memory Model

Lockstep uses a **Host-Owned Static Arena**. The compiler calculates the exact byte-offset for every Struct-of-Arrays (SoA) member at compile-time.

* **SoA by Default:** Structs are automatically decomposed into parallel primitive arrays to maximize cache line utilization and SIMD width.
* **Saturated Writes:** To eliminate boundary checks, stream indices use saturation arithmetic. If a stream capacity is exceeded, the final element acts as a "trash can," absorbing further writes without memory corruption or branching.

---

## 3. Syntax Guide

### Straight-Line Shaders

Since `if/else` is banned, conditional logic is performed using branchless intrinsics like `step`, `mix`, and `clamp`.

```c
shader ApplyPhysics(in Entity ent, out Entity updated, uniform float dt) {
    // Standard math
    float fall_vy = ent.vy - (9.81 * dt);
    float bounce_vy = -ent.vy * 0.8;
    
    // Branchless Branching: step returns 1.0 if ent.y <= 0.0, else 0.0
    float is_grounded = step(0.0, -ent.y);
    
    // mix(a, b, t) acts as a hardware-level selector
    updated.vy = mix(fall_vy, bounce_vy, is_grounded);
    updated.y = max(ent.y + (updated.vy * dt), 0.0);
}

```

### Linear Accumulators

Global reductions (e.g., Total Energy, Max Bounds) are handled via **Linear Types**. Accumulators must be "consumed" by a fold operation, which the compiler lowers into a lock-free parallel reduction tree.

```c
pipeline Simulation {
    stream<Entity, 10000> particles;
    accumulator<float> energy_sum;

    bind {
        particles = Calculate(particles, energy_sum);
        // fold sum consumes the linear type and produces a global scalar
        uniform float total_e = fold sum(energy_sum);
    }
}

```

---

## 4. Compiler & Backend

Lockstep targets **LLVM IR** directly to leverage industrial-grade optimization passes.

* **`noalias` Guarantee:** Because Lockstep forbids arbitrary pointers, the compiler decorates all IR pointers with `noalias`, enabling aggressive auto-vectorization.
* **SSA Purity:** Local variables are mapped directly to SSA registers. Struct member access (`ent.pos.x`) is lowered to LLVM `extractvalue` and `insertvalue` instructions, allowing for total Scalar Replacement of Aggregates (SROA).
* **Fast-Math Reductions:** Reduction loops are emitted with `fast` math flags, permitting LLVM to reassociate floating-point operations into horizontal SIMD shuffles.

---

## 5. Host Integration

The compiler generates a C-compatible header for the Host application (C/C++, Rust, or Zig).

1. **Allocate:** Host allocates a contiguous block of size `LOCKSTEP_ARENA_BYTES`.
2. **Bind:** Host calls `Lockstep_BindMemory(ptr)`.
3. **Prime:** Host writes initial data into the SoA offsets provided by the header.
4. **Tick:** Host calls `Lockstep_Tick()` to execute the pipeline.

---

## 6. Compiler Frontend Usage

Install in editable mode to enable the packaged CLI entrypoint:

```bash
pip install -e .
lockstepc path/to/program.lock
# or read source from stdin
cat path/to/program.lock | lockstepc --dump
```

`debug_compiler.py` exposes `compile_lockstep(source_code, verbose=True)` and returns a `LockstepCompileResult` containing:

* `parse_tree`: ANTLR parse tree for the source.
* `entities`: extracted frontend entities (`structs`, `shaders`, `streams`, `accumulators`).
* `diagnostics`: first-class compiler diagnostics (`LockstepDiagnostic`) for non-fatal observations.

### Diagnostic Shape

Each diagnostic includes:

* `severity` (`"info"`, `"warning"`, or `"error"`)
* `code` (stable diagnostic identifier such as `LCK101`, `LCK201`)
* `message`
* `line`
* `column`
* optional `hint`

### Behavior

* **Non-fatal observations** (for example empty `bind` blocks, duplicate declarations, or unreachable statements after a pure-function return) are returned in `LockstepCompileResult.diagnostics` and compilation still succeeds.
* **Pure function return enforcement** is semantic and strict:
  * `LCK413` (`error`) is emitted when a `pure` function body has no `return` statement.
  * `LCK414` (`warning`) is emitted when a `pure` function body contains multiple `return` statements.
  * `LCK415` (`warning`) is emitted for statements that appear after the first `return` in a `pure` function body.
  * `LCK418` (`error`) is emitted when a pure `return` expression type does not match the declared return type.
* **Type-check mismatches** each have distinct diagnostic codes:
  * `LCK416` (`error`) is emitted for variable initializer type mismatches in `visitVarDecl`.
  * `LCK417` (`error`) is emitted for assignment type mismatches in `visitAssignStmt`.
* **Fatal parse errors** still raise `LockstepCompileError`.
  * `LockstepCompileError.errors` contains parse diagnostics.
  * `LockstepCompileError.diagnostics` mirrors available pre-failure diagnostic context when parse fails.

---

## 7. Regenerating parser

Run the project-native generator target:

```bash
make generate-parser
```

Generated Python parser files are emitted to `generated/parser/` and committed to source control. CI enforces freshness via `make check-generated-parser`, which regenerates and fails when tracked generated files are stale.

## 8. Language Server Protocol (LSP)

Lockstep now ships an opt-in LSP server so editors can surface compiler diagnostics in real time and provide semantic assistance while authoring pipelines.

```bash
pip install -e .[lsp]
lockstep-lsp
```

Current capabilities:

* **Live diagnostics:** Mirrors compiler parse/semantic diagnostics via `textDocument/publishDiagnostics`.
* **Go to Definition for struct members:** Resolves `foo.bar` member access back to the `struct` field declaration when the variable type can be inferred.
* **Bind-route autocompletion:** Suggests existing `bind` routes and callable shader/pure symbols from the current file.

The server communicates over stdio and is compatible with standard editor LSP client configuration.
