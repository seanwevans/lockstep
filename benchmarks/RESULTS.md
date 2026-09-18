# Lockstep benchmark results

Real measured results from the benchmark harnesses in this repository. Every
number below comes from running the committed harnesses unmodified — no
hand-edited figures. Reproduce any table by running the command shown above it.

Absolute throughput is host-dependent (CPU, memory bandwidth, compiler
version), so treat these as a concrete reference point and a relative/regression
signal, not a portable constant. What is portable is the *shape* of the results:
SIMD-friendly SoA layout beats AoS by several-x on layout alone (the order-of-
magnitude figure in table 2 is partly clang's gather/scatter lowering for the
AoS baseline, quantified there), and stage fusion recovers a multiple-x
throughput win over the per-stage loops codegen used to emit. The flip side is measured and reported honestly too: against an idiomatic
single-pass **hand-written C** baseline (table 4), the single fused-kernel
workload (`particle_energy`) runs **at parity** (1.00× median) — after codegen
learned to fuse a fold's reduction into the writing kernel loop instead of
materializing a per-row accumulator buffer. The two multi-stage filter pipelines
used to trail by ~4–6×; codegen now **fuses through their pass-through filters**
into one vector pass — loading and storing each SoA column as a contiguous
vector and carrying the fold accumulators in registers instead of an O(rows)
buffer — which brings `multi_stage_pipeline` to **0.93×** of hand-written C and
`telemetry_filter_aggregation` to **0.82×**. A filter that actually *drops* rows
still falls back to the per-stage compacting path; fusing through a dropping
filter is the remaining codegen work in [`../ROADMAP.md`](../ROADMAP.md).

> **Re-measured after the folded-uniform fix.** Every native number below was
> re-taken once folded uniforms got an arena slot. Before that, `Lockstep_Tick`
> computed each `fold` into a value it never stored, so LLVM deleted the entire
> accumulator dataflow as dead code — `telemetry_filter_aggregation`'s tick
> compiled to a column copy with no floating-point arithmetic in it at all.
> Tables 1 and 4 were therefore timing less work than the workloads describe.
>
> The host also changed between the two revisions, so these tables are not a
> clean before/after for the fix alone and every table was re-taken together.
> Measured in isolation on one host, restoring the fold made `particle_energy`
> *faster* (33.6 → 26.6 µs per tick — the dead-fold version compiled to a
> worse-scheduled store-bound loop) and cost the other two a little, as you
> would expect for work that is now actually performed:
> `telemetry_filter_aggregation` 17.5 → 19.1 µs, `multi_stage_pipeline`
> 70.2 → 71.7 µs.

## Host environment

| Field | Value |
| --- | --- |
| CPU | Intel Xeon @ 2.10 GHz (AVX-512: `avx512f/dq/bw/vl/vnni/bf16/fp16`, AVX-VNNI) |
| Logical cores | 4 |
| Memory | 16 GiB |
| OS / kernel | Linux 6.18.44 (x86-64, KVM guest) |
| Compiler | Ubuntu clang 18.1.3 |
| Python | 3.11.15 |
| Date | 2026-09-18 |

The native harnesses build with `clang -O3 -march=native`, so codegen targets
this host's AVX-512 units. The frontend harnesses time pure CPython.

This is a virtualized 4-core host and run-to-run noise is real: the coefficient
of variation on the native harnesses is 4–15%, which is the same order as some
of the effects being reported. Every figure below is a **median** over repeated
runs (counts stated per table) rather than a single sample, and where the spread
matters it is quoted alongside.

---

## 1. Native execution throughput (`make bench-native`)

Compiles each workload's generated LLVM IR, links a host driver that calls
`Lockstep_Tick` in a timed loop over a full arena, and measures the compiled
machine code — the thing Lockstep actually ships.

```bash
python benchmarks/native/run_native.py --iterations 4000
```

Median of 7 runs.

| workload | rows/tick | arena | per_tick_us | Mrows/s | GiB/s | checksum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| particle_energy | 32,768 | 1.63 MiB | 28.31 | 1157.3 | 56.05 | 65528.0 |
| telemetry_filter_aggregation | 65,536 | 2.19 MiB | 18.37 | 3567.8 | 116.30 | 65528.0 |
| multi_stage_pipeline | 131,072 | 5.50 MiB | 70.11 | 1869.6 | 76.61 | 20969.0 |

`particle_energy` (single fused kernel) reaches ~56 GiB/s of arena traffic. The
two accumulator pipelines fuse each pipeline — including its trailing/leading
pass-through filter — into a single vector loop that streams every SoA column as
a contiguous vector and carries the fold accumulators in registers, so the tick
makes one trip through memory instead of one per stage.

All three checksums are unchanged from the previous revision of this table, so
the computed results are identical; what changed is that the fold is now
actually computed. The arena grew by one element per folded uniform (4–8 bytes),
which is why `particle_energy` reads 1.63 MiB rather than 1.62 MiB.

Spread across the 7 runs, as per-tick microseconds (min–max): `particle_energy`
23.48–31.59, `telemetry_filter_aggregation` 16.77–21.91, `multi_stage_pipeline`
67.25–83.68. Read the medians with that in mind.

---

## 2. SoA vs AoS layout (`make bench-soa`)

The same branchless particle kernel over identical data in Struct-of-Arrays vs
Array-of-Structs layout, swept across working-set sizes. Both layouts compute
identical results (checked), so the only variable is memory layout.

```bash
python benchmarks/native/soa_vs_aos.py
```

Median of 3 runs.

| n | integrate AoS Mrows/s | integrate SoA Mrows/s | **integrate speedup** | energy AoS Mrows/s | energy SoA Mrows/s | **energy speedup** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 343.5 | 13850.4 | **40.3×** | 1512.6 | 10857.8 | **7.2×** |
| 4,000 | 345.6 | 5538.6 | **15.9×** | 1660.6 | 11901.2 | **7.1×** |
| 16,000 | 340.3 | 5103.2 | **15.0×** | 1597.0 | 8795.1 | **5.5×** |
| 64,000 | 319.5 | 5321.0 | **16.0×** | 1451.9 | 8921.2 | **6.3×** |
| 256,000 | 310.6 | 1516.9 | **5.0×** | 869.1 | 2332.6 | **2.7×** |
| 1,000,000 | 297.4 | 1452.1 | **4.9×** | 866.1 | 2066.9 | **2.4×** |

`integrate` touches every field: SoA wins on **vectorization** (contiguous
columns feed packed SIMD; AoS's interleaved stride forces gathers/scalar code),
peaking above 40× when cache-resident. `energy` reads only velocity + mass: SoA
additionally wins on **bandwidth** by not dragging unused position fields through
cache. Both speedups narrow once the sweep goes memory-bound past ~256k rows but
stay firmly above 1× — which is the point: SoA is a win across the whole range.

### How much of `integrate` is layout, and how much is instruction selection

The `integrate` column above overstates the layout effect on this host, and the
harness now measures by how much. Clang lowers `integrate_aos`'s 24-byte-stride
accesses to `vgatherqps`/`vscatterqps` — 10 gathers and 10 scatters per vector
iteration — and those are slow enough to dominate. That is a **codegen** cost,
not a property of Array-of-Structs.

`soa_vs_aos.py` therefore rebuilds the *same source* with `-mno-avx512f`, which
denies the vectorizer scatter (the expensive, AVX-512-only half), and reports a
second table. SoA is essentially unchanged — it never used scatter — while AoS
speeds up several-fold:

| n | integrate speedup (native) | integrate speedup (no scatter) | integrate AoS Mrows/s: native → no scatter |
| ---: | ---: | ---: | ---: |
| 1,000 | 39.3× | **7.7×** | 346 → 1,882 |
| 4,000 | 16.0× | **5.8×** | 346 → 928 |
| 16,000 | 15.0× | **3.7×** | 354 → 1,340 |
| 64,000 | 15.5× | **9.0×** | 330 → 593 |
| 256,000 | 4.7× | **3.5×** | 304 → 486 |
| 1,000,000 | 5.1× | **3.3×** | 326 → 447 |

So the honest reading of `integrate` is **~3–8×** for layout, with the rest of
the headline figure attributable to clang's instruction selection for the AoS
store on an AVX-512 host. Both costs are real and both are paid by anyone
writing AoS here — but only the first is portable to a host without AVX-512.

The `energy` column needs no such correction: it is a gather-only read with no
scatter, and its speedup barely moves between the two builds (6.9× → 7.5× at
n=1,000, 2.2× → 2.9× at n=1,000,000). That one is a genuine bandwidth win.

The AoS timings are noisy run to run, so read individual rows as approximate and
the trend as the result.

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

Median of 3 runs.

| n | unfused Mrows/s | fused Mrows/s | **fusion speedup** |
| ---: | ---: | ---: | ---: |
| 4,000 | 1875.7 | 22099.4 | **11.6×** |
| 16,000 | 1949.3 | 10635.5 | **5.4×** |
| 64,000 | 1782.7 | 10606.2 | **6.0×** |
| 256,000 | 613.9 | 5715.5 | **9.5×** |
| 1,000,000 | 650.7 | 3337.5 | **5.1×** |

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

Median of 15 runs.

| workload | rows/tick | Lockstep Mrows/s | C Mrows/s | **ratio (C time / Lockstep time)** |
| --- | ---: | ---: | ---: | ---: |
| particle_energy | 32,768 | 1209.9 | 1210.7 | **1.00×** |
| telemetry_filter_aggregation | 65,536 | 3509.1 | 4327.8 | **0.82×** |
| multi_stage_pipeline | 131,072 | 1796.4 | 1884.9 | **0.93×** |

`ratio >= 1.0` means Lockstep matches or beats hand-written C. The C baseline is
noisy on this host — its cache-resident copy loops settle into different modes
between process invocations while Lockstep is stable — so read each ratio as a
band. Across the 15 repeats:

| workload | median | interquartile | full range |
| --- | ---: | ---: | ---: |
| particle_energy | 1.00× | 0.96–1.03 | 0.81–1.32 |
| telemetry_filter_aggregation | 0.82× | 0.71–0.82 | 0.57–0.83 |
| multi_stage_pipeline | 0.93× | 0.91–0.96 | 0.87–1.04 |

The harness interleaves the two kernels in alternating rounds and reports the
median round, so these are not sensitive to which kernel is timed first — the
previous fixed-order driver was, by up to 24% on
`telemetry_filter_aggregation`. See
[`native/README.md`](native/README.md#measurement-order).

* **`particle_energy` — at parity (1.00× median, above 1.0 on roughly half the
  runs).** It used to trail C by ~20%, and the whole gap was one thing: the `accum` was lowered as a
  per-row `[rows × f32]` arena buffer that the tick **read-modify-wrote every
  row**, then a separate `fold` strip-mined it back to a scalar — while C keeps
  the sum in a register. Codegen performs **fold-into-kernel fusion**: when an
  accumulator is written by a single standalone route and consumed by exactly one
  `fold`, the reduction is carried in a loop-carried register across the route
  loop and the O(rows) buffer is never touched (the arena still *reserves* it, so
  the ABI is unchanged). The folded scalar is bit-identical to the strip-mine path
  (verified in `tests/test_fold_reduction_fusion.py`). See
  [`native/README.md`](native/README.md#fold-into-kernel-fusion).

  Worth stating plainly: the earlier 0.97× for this workload was measured while
  the fold was being dead-code-eliminated, so Lockstep was doing *less* work
  than the C reference it was compared against and still lost. Now it does the
  same work and ties.
* **`multi_stage_pipeline` (0.93×, near parity)** and
  **`telemetry_filter_aggregation` (0.82×)** used to trail by ~4.7× and ~5.4×.
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

Median of 3 runs.

| workload | compile_ms | simulate_ms | input rows | input rows/s | fold rows/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| particle_energy | 12.74 | 1324.42 | 20,000 | 15,101 | 15,102 |
| telemetry_filter_aggregation | 10.66 | 611.23 | 30,000 | 49,081 | 49,085 |
| multi_stage_pipeline | 22.27 | 848.00 | 36,000 | 42,453 | 28,307 |

## 6. Frontend microbenchmark (`make bench`)

The CI regression KPI: the harness reports the median of 5 iterations over
`examples/minimal.lock`; the figures below are the median of 5 such runs.

```bash
python scripts/run_benchmarks.py
```

| benchmark | value | unit |
| --- | ---: | --- |
| compile_minimal_ms | 2.17 | ms |
| simulate_minimal_ms | 0.008 | ms |

`compile_minimal_ms` sits well under the advisory baseline in
`benchmarks/baselines/default.json` (6.0 ms). `simulate_minimal_ms` is close to
its 0.01 ms baseline and individual runs touched it (observed 0.007–0.010 ms) —
at this scale the KPI is near timer resolution, so treat it as a smoke signal
rather than a precise measurement.

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
