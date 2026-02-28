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

* **Non-fatal observations** (for example empty `bind` blocks or duplicate declarations) are returned in `LockstepCompileResult.diagnostics` and compilation still succeeds.
* **Fatal parse errors** still raise `LockstepCompileError`.
  * `LockstepCompileError.errors` contains parse diagnostics.
  * `LockstepCompileError.diagnostics` mirrors available pre-failure diagnostic context when parse fails.

---

## 7. Regenerating parser

Run the project-native generator target:

```bash
make generate-parser
```

The parser generator script pins ANTLR `4.13.2` to a known SHA-256 and verifies integrity before use. By default it uses `tools/antlr-4.13.2-complete.jar`, downloading the jar only when missing.

### Secure workflow

For controlled or air-gapped environments, pre-provision the jar and run in offline mode:

```bash
python scripts/generate_parser.py --offline --jar-path /path/to/antlr-4.13.2-complete.jar
```

If the jar is missing in offline mode, or the checksum does not match the pinned value, generation fails with a clear error.

CI should use:

```bash
make check-generated-parser
```

This target enforces offline generation with an explicit jar path and fails when tracked generated files are stale.
