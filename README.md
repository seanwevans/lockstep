# Lockstep (Shader-C)

**Lockstep** is a data-oriented systems programming language designed for high-throughput, deterministic compute pipelines. It bridges the gap between the productivity of C and the brutal execution efficiency of GPU compute shaders.

By enforcing a strict **Straight-Line SIMD** execution model and **Static Memory Topology**, Lockstep allows the compiler to generate machine code that is mathematically guaranteed to saturate CPU vector units without the overhead of branch misprediction or cache contention.

## Quickstart

### Required tools

* Java runtime (for the ANTLR tool)
* ANTLR4 CLI/tooling available as `antlr4`
* Python 3 with the ANTLR runtime package (`antlr4-python3-runtime`)

### Regenerate parser artifacts

```bash
./make.sh generate
```

### Run the debug compiler sample

```bash
./make.sh run-sample
```

### Expected high-level output

The sample run prints a frontend trace that shows discovered language elements in order, including:

* compiler frontend header (`=== LOCKSTEP COMPILER FRONTEND ===`)
* struct and pure function discovery
* shader kernel name + parameters
* pipeline topology, streams/accumulators, and bind routing

---

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

### The Memory Model (Option C)

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
        // fold_sum consumes the linear type and produces a global scalar
        uniform float total_e = fold_sum(energy_sum);
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
