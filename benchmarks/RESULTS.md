# Lockstep benchmark results

Real measured results from the benchmark harnesses in this repository. Every
number below comes from running the committed harnesses unmodified — no
hand-edited figures. Reproduce any table by running the command shown above it.

Absolute throughput is host-dependent (CPU, memory bandwidth, compiler
version), so treat these as a concrete reference point and a relative/regression
signal, not a portable constant. What is portable is the *shape* of the results:
SIMD-friendly SoA layout beats AoS by an order of magnitude, and stage fusion
recovers a multiple-x throughput win over the per-stage loops codegen emits
today.

## Host environment

| Field | Value |
| --- | --- |
| CPU | Intel Xeon @ 2.10 GHz (AVX-512: `avx512f/dq/bw/vl/vnni/bf16/fp16`, AVX-VNNI) |
| Logical cores | 4 |
| Memory | 16 GiB |
| OS / kernel | Linux 6.18.5 (x86-64) |
| Compiler | Ubuntu clang 18.1.3 |
| Python | 3.11.15 |
| Date | 2026-08-11 |

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
| particle_energy | 32,768 | 1.62 MiB | 46.62 | 702.8 | 34.04 | 65528.0 |
| telemetry_filter_aggregation | 65,536 | 2.19 MiB | 106.98 | 612.6 | 19.97 | 65528.0 |
| multi_stage_pipeline | 131,072 | 5.50 MiB | 353.43 | 370.9 | 15.20 | 13105.6 |

`particle_energy` (single fused kernel) reaches ~34 GiB/s of arena traffic. The
two accumulator pipelines sit lower per row because codegen emits one
strip-mined loop **per stage** and materializes intermediate streams: both end
in a filter, and codegen still lowers filter-containing groups per stage (the
`accum` restriction on fusion has since been lifted — see the fusion probe
below).

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

Even at memory-bound sizes the fully fused form runs ~5× faster. Codegen used to
skip this fusion whenever a stage took an `accum` parameter; that restriction has
been lifted, and a pure accumulator pipeline (no filter) now fuses — measuring
~1.9× over per-stage loops on this host. `multi_stage_pipeline` and
`telemetry_filter_aggregation` still fall back to per-stage loops because each
group ends in a **filter**, whose data-dependent compacting store the vector
path does not yet lower. Fusing through the trailing filter is the highest-
leverage remaining codegen throughput opportunity for these pipelines.

---

## 4. Frontend workloads (`python benchmarks/run_workloads.py`)

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

## 5. Frontend microbenchmark (`make bench`)

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
python benchmarks/run_workloads.py   # frontend workloads   (table 4)
make bench             # frontend microbenchmark KPI        (table 5)
```

The native harnesses require an LLVM/clang toolchain on `PATH`; each exits with
a clear message if `clang` is missing. See
[`benchmarks/native/README.md`](native/README.md) for methodology details.
