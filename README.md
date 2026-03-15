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

Since `if/else` is banned, conditional logic is performed using branchless intrinsics like `step`, `mix`, `clamp`, `min`, `max`, `abs`, `sign`, and `smoothstep`.

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

### Type System (User-Facing Rules)

Lockstep's semantic validator enforces a strict type system with **no implicit coercions**.

#### Primitive types

The currently supported primitive declared types are:

* `int`
* `float`
* `bool`
* `string`

> `uint` and `double` are **not currently supported as declared types** in source-level type annotations (for locals, params, uniforms, struct fields, etc.). Using unknown declared types produces `LCK310`.

#### Composite/struct type composition

Struct members may use:

* primitives,
* previously declared struct names,
* array suffixes (`T[4]`), and
* generic wrappers (`Ctor<T>` / `Ctor<T,4>`), including nested forms.

Examples:

* `Particle[4]`
* `vector<float,4>`
* `matrix<vector<Particle,4>,4>`

Type identity is name-based and exact. Field access chains (`a.b.c`) are valid only when each link resolves to a struct type and an existing field.

#### Type matching and coercion policy

Type checking is strict and explicit:

* No implicit widening or narrowing.
* No implicit `int`⇄`float` promotion.
* Assignment, variable initialization, pure-function arguments, pure-function returns, and bind argument/target checks all require exact type equality.
* Mixed numeric operators (`int` with `float`) without an explicit cast are rejected with `LCK424` (`implicit_numeric_widening`).

When conversion is desired, use an explicit cast.

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

See [`examples/`](examples/) for a minimal end-to-end host app in C (`examples/minimal_host.c`) that includes a generated header, allocates arena memory, primes initial data, and calls `Lockstep_Tick`.

---

## 6. Compiler Frontend Usage

Install in editable mode to enable the packaged CLI entrypoint:

```bash
pip install -e .
lockstepc path/to/program.lock
# or read source from stdin
cat path/to/program.lock | lockstepc --dump
# canonical straight-line formatting
lockstepc path/to/program.lock --format
# emit LLVM IR
lockstepc path/to/program.lock --emit-ir
# emit C host header
lockstepc path/to/program.lock --emit-header
# parser resource limits (DoS hardening)
lockstepc path/to/program.lock --max-file-size-bytes 1048576 --max-expression-nesting-depth 256 --parse-timeout-seconds 2.0
# print compiler version
lockstepc --version
```

`debug_compiler.py` exposes `compile_lockstep(source_code, verbose=True)` and returns a `LockstepCompileResult` containing:

* `parse_tree`: ANTLR parse tree for the source.
* `entities`: extracted frontend entities (`structs`, `shaders`, `streams`, `accumulators`).
* `diagnostics`: first-class compiler diagnostics (`LockstepDiagnostic`) for non-fatal observations.

### Pipeline Simulation (small datasets)

Use the CLI simulator to validate pipeline wiring/cardinality before LLVM backend generation:

```bash
lockstepc path/to/program.lock --simulate
lockstepc path/to/program.lock --simulate --simulate-input path/to/input.json
```

`--simulate-input` expects JSON with optional `streams` and `accumulators` maps, for example:

```json
{
  "streams": {
    "raw_positions": [{"id": 1}, {"id": 2, "_keep": false}]
  },
  "accumulators": {
    "energy": [0.5, 1.5]
  }
}
```

Simulation output includes per-route `input_count`/`output_count`, updated stream snapshots, accumulator contents, and folded uniform values.

Generated C headers include `Lockstep_SaturatedWriteIndex(...)` plus per-stream `LOCKSTEP_CAPACITY_STREAM_<NAME>` macros. Define `LOCKSTEP_DEBUG_SATURATED_WRITES` before including the header to log whenever a saturated write falls back to the final index. Override `LOCKSTEP_SATURATED_WRITE_LOG(...)` to integrate with custom telemetry.

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
  * `LCK412` (`error`) is emitted for pure-function argument type mismatches.
  * `LCK416` (`error`) is emitted for variable initializer type mismatches in `visitVarDecl`.
  * `LCK417` (`error`) is emitted for assignment type mismatches in `visitAssignStmt`.
  * `LCK424` (`error`) is emitted when arithmetic mixes `int` and `float` operands without an explicit cast.
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
* **Hover type info:** Shows inferred type annotations on variables, struct fields, shader names, and pure function names.
* **Bind-route autocompletion:** Suggests existing `bind` routes and callable shader/pure symbols from the current file.

The server communicates over stdio and is compatible with standard editor LSP client configuration.
