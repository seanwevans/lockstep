# Lockstep benchmark results

Real measured results from the benchmark harnesses in this repository. Every
number below comes from running the committed harnesses unmodified — no
hand-edited figures. Reproduce any table by running the command shown above it.

Absolute throughput is host-dependent (CPU, memory bandwidth, compiler
version), so treat these as a concrete reference point and a relative/regression
signal, not a portable constant. What is portable is the *shape* of the results:
SIMD-friendly SoA layout beats AoS by an order of magnitude, and stage fusion
recovers a multiple-x throughput win over the per-stage loops codegen used to
emit. The flip side is measured and reported honestly too: against an idiomatic
single-pass **hand-written C** baseline (table 4), the single fused-kernel
workload (`particle_energy`) runs **at parity** — after codegen learned to fuse a
fold's reduction into the writing kernel loop instead of materializing a per-row
accumulator buffer. The two multi-stage filter pipelines used to trail by ~4–6×;
codegen now **fuses through their pass-through filters** into one vector pass —
loading and storing each SoA column as a contiguous vector and carrying the fold
accumulators in registers instead of an O(rows) buffer — which lifts them to
**~2× (multi-stage, near parity) and ~1.4× (telemetry)** of hand-written C, a
~4× native-throughput gain (table 1). A filter that actually *drops* rows still
falls back to the per-stage compacting path; fusing through a dropping filter is
the remaining codegen work in [`../ROADMAP.md`](../ROADMAP.md).

## Host environment

| Field | Value |
| --- | --- |
| CPU | Intel Xeon @ 2.10 GHz (AVX-512: `avx512f/dq/bw/vl/vnni/bf16/fp16`, AVX-VNNI) |
| Logical cores | 4 |
| Memory | 16 GiB |
| OS / kernel | Linux 6.18.5 (x86-64) |
| Compiler | Ubuntu clang 18.1.3 |
| Python | 3.11.15 |
| Date | 2026-08-12 |

The native harnesses build with `clang -O3 -march=native`, so codegen targets
this host's AVX-512 units. The frontend harnesses time pure CPython.

---

## 1. Native execution throughput (`make bench-native`)

Compiles each workload's generated LLVM IR, links a host driver that calls
`Lockstep_Tick` in a timed loop over a full arena, and measures the compiled
machine code — the thing Lockstep actually ships.

```bash
python benchmarks/native/run_native.py --iterations 4000
```

| workload | rows/tick | arena | per_tick_us | Mrows/s | GiB/s | checksum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| particle_energy | 32,768 | 1.62 MiB | 48.21 | 679.8 | 32.92 | 65528.0 |
| telemetry_filter_aggregation | 65,536 | 2.19 MiB | 29.58 | 2215.9 | 72.23 | 65528.0 |
| multi_stage_pipeline | 131,072 | 5.50 MiB | 91.08 | 1439.1 | 58.97 | 20969.0 |

`particle_energy` (single fused kernel) reaches ~33 GiB/s of arena traffic. The
two accumulator pipelines now run **~3.9× faster than before** (telemetry
612→2216 Mrows/s, multi-stage 371→1439 Mrows/s): codegen fuses each pipeline —
including its trailing/leading pass-through filter — into a single vector loop
that streams every SoA column as a contiguous vector and carries the fold
accumulators in registers, so it makes one trip through memory instead of one
per stage. (The `multi_stage_pipeline` checksum moved from 13105.6 to 20969.0
only because the harness now sums the pipeline's *terminal* output column rather
than an intermediate stream that fusion no longer materializes — the computed
result is unchanged and still agrees with the hand-written C reference below.)

---

## 2. SoA vs AoS layout (`make bench-soa`)

The same branchless particle kernel over identical data in Struct-of-Arrays vs
Array-of-Structs layout, swept across working-set sizes. Both layouts compute
identical results (checked), so the only variable is memory layout.

```bash
python benchmarks/native/soa_vs_aos.py
```

| n | integrate AoS Mrows/s | integrate SoA Mrows/s | **integrate speedup** | energy AoS Mrows/s | energy SoA Mrows/s | **energy speedup** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 300.9 | 16155.1 | **53.7×** | 1473.2 | 10405.8 | **7.1×** |
| 4,000 | 315.0 | 5115.1 | **16.2×** | 1339.2 | 9168.0 | **6.9×** |
| 16,000 | 311.8 | 4885.2 | **15.7×** | 1583.3 | 9334.9 | **5.9×** |
| 64,000 | 293.4 | 5120.0 | **17.5×** | 1485.6 | 9307.2 | **6.3×** |
| 256,000 | 290.4 | 1610.2 | **5.5×** | 1092.4 | 2518.6 | **2.3×** |
| 1,000,000 | 295.0 | 1646.9 | **5.6×** | 1017.6 | 2434.7 | **2.4×** |

`integrate` touches every field: SoA wins on **vectorization** (contiguous
columns feed packed SIMD; AoS's interleaved stride forces gathers/scalar code),
peaking above 50× when cache-resident. `energy` reads only velocity + mass: SoA
additionally wins on **bandwidth** by not dragging unused position fields through
cache. Both speedups narrow once the sweep goes memory-bound past ~256k rows but
stay firmly above 1× — which is the point: SoA is a win across the whole range.

---

## 3. Multi-stage fusion probe (`make bench-fusion`)

Runs the `multi_stage_pipeline` computation two ways over identical SoA data:
`unfused` (three loops writing/re-reading intermediates, as codegen emits today)
vs `fused` (one loop, intermediates in registers, as the optimizer already
*plans*). Identical results (checked), so the delta is the win fusing the whole
group recovers.

```bash
python benchmarks/native/fusion_probe.py
```

| n | unfused Mrows/s | fused Mrows/s | **fusion speedup** |
| ---: | ---: | ---: | ---: |
| 4,000 | 1907.9 | 25477.7 | **13.3×** |
| 16,000 | 1844.7 | 9352.9 | **5.1×** |
| 64,000 | 1583.4 | 9812.8 | **6.2×** |
| 256,000 | 631.9 | 6229.5 | **9.9×** |
| 1,000,000 | 650.0 | 3329.7 | **5.1×** |

Even at memory-bound sizes the fully fused form runs ~5× faster. This probe
measures the win in isolation; codegen now **realizes** it for the shipped
`multi_stage_pipeline` and `telemetry_filter_aggregation` workloads, whose
pass-through filters (`KeepActive` / the telemetry keep stage) fuse into the
group as identity copies (see table 1 and table 4). A filter that *drops* rows —
a data-dependent `return` — still has a compacting store the vector path does not
lower, so it keeps the per-stage fallback; fusing through a dropping filter is
the remaining opportunity.

---

## 4. Lockstep vs hand-written C (`make bench-vs-c`)

The three tables above compare Lockstep against *itself* and it always wins. This
one is the honest external comparison: the real shipped `Lockstep_Tick` (compiled
from each workload's LLVM IR) versus an idiomatic **single-pass C kernel** a
competent human would write for the same transform, over the same SoA arena
(identical byte offsets from the generated header). Both paths run on identically
primed arenas and must agree on an output checksum before a result is reported —
and those checksums match table 1's, cross-validating the C references. Built
with `clang -O3 -march=native -ffast-math`.

```bash
python benchmarks/native/lockstep_vs_c.py --iterations 5000
```

| workload | rows/tick | Lockstep Mrows/s | C Mrows/s | **ratio (C time / Lockstep time)** |
| --- | ---: | ---: | ---: | ---: |
| particle_energy | 32,768 | 717.9 | 737.1 | **0.97×** |
| telemetry_filter_aggregation | 65,536 | 2340.9 | 3569.3 | **0.66×** |
| multi_stage_pipeline | 131,072 | 1512.8 | 1645.5 | **0.92×** |

`ratio >= 1.0` means Lockstep matches or beats hand-written C. The C baseline is
noisy on this host (its cache-resident copy loops swing several hundred Mrows/s
run to run), while Lockstep is stable, so read each ratio as a band: across
repeats `particle_energy` spans ~0.85–1.02, `multi_stage_pipeline` ~0.81–0.93,
and `telemetry_filter_aggregation` ~0.66–0.95.

* **`particle_energy` — at parity (~0.97×, above 1.0 on some runs).** It used to
  trail C by ~20%, and the whole gap was one thing: the `accum` was lowered as a
  per-row `[rows × f32]` arena buffer that the tick **read-modify-wrote every
  row**, then a separate `fold` strip-mined it back to a scalar — while C keeps
  the sum in a register. Codegen performs **fold-into-kernel fusion**: when an
  accumulator is written by a single standalone route and consumed by exactly one
  `fold`, the reduction is carried in a loop-carried register across the route
  loop and the O(rows) buffer is never touched (the arena still *reserves* it, so
  the ABI is unchanged). The folded scalar is bit-identical to the strip-mine path
  (verified in `tests/test_fold_reduction_fusion.py`). See
  [`native/README.md`](native/README.md#fold-into-kernel-fusion).
* **`multi_stage_pipeline` (~0.92×, near parity)** and
  **`telemetry_filter_aggregation` (~0.66×)** used to trail by ~4.7× and ~5.4×.
  Codegen now fuses each pipeline **through its pass-through filter** into a
  single vector loop: the eliminated intermediate streams stay in registers, each
  SoA column moves as a contiguous vector load/store, and the fold accumulators
  are carried in loop-carried vector registers (reduced horizontally at the end)
  instead of a per-row buffer — so the tick makes **one** trip through memory,
  like the C reference. `multi_stage_pipeline` lands at near parity;
  `telemetry_filter_aggregation`'s tighter, more cache-resident copy still leaves
  the hand-written single loop a memory-bandwidth edge. A filter that actually
  drops rows keeps the per-stage compacting fallback (see the fusion probe).

Absolute Mrows/s are host-dependent; the **ratio** is the portable signal.

---

## 5. Frontend workloads (`python benchmarks/run_workloads.py`)

Times the Python frontend: `compile_lockstep` + in-Python `simulate_pipeline_entities`
over realistic fixtures. This is the toolchain/authoring path, not shipped code.

```bash
python benchmarks/run_workloads.py --json
```

| workload | compile_ms | simulate_ms | input rows | input rows/s | fold rows/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| particle_energy | 15.21 | 1567.26 | 20,000 | 12,761 | 12,762 |
| telemetry_filter_aggregation | 12.82 | 738.56 | 30,000 | 40,620 | 40,622 |
| multi_stage_pipeline | 25.49 | 1008.41 | 36,000 | 35,700 | 23,804 |

## 6. Frontend microbenchmark (`make bench`)

Median of 5 iterations over `examples/minimal.lock`, the CI regression KPI.

```bash
python scripts/run_benchmarks.py
```

| benchmark | value | unit |
| --- | ---: | --- |
| compile_minimal_ms | 2.28 | ms |
| simulate_minimal_ms | 0.009 | ms |

Both are comfortably under the advisory baseline in
`benchmarks/baselines/default.json` (6.0 ms compile, 0.01 ms simulate).

---

## Reproducing

```bash
pip install -e .
make bench-native      # native compiled-code throughput   (table 1)
make bench-soa         # SoA vs AoS layout                  (table 2)
make bench-fusion      # multi-stage fusion probe           (table 3)
make bench-vs-c        # Lockstep vs hand-written C         (table 4)
python benchmarks/run_workloads.py   # frontend workloads   (table 5)
make bench             # frontend microbenchmark KPI        (table 6)
```

The native harnesses require an LLVM/clang toolchain on `PATH`; each exits with
a clear message if `clang` is missing. See
[`benchmarks/native/README.md`](native/README.md) for methodology details.
