# Native execution benchmarks

These benchmarks measure the **compiled machine code** Lockstep generates, not
the Python frontend.

Real measured results from these harnesses, on a documented host, are published
in [`../RESULTS.md`](../RESULTS.md).

Lockstep's other benchmark harnesses time the toolchain:

| Harness | What it measures |
| --- | --- |
| `scripts/run_benchmarks.py` (`make bench`) | `compile_lockstep` + in-Python simulate of a minimal program |
| `benchmarks/run_workloads.py` | `compile_lockstep` + in-Python simulate of realistic workloads |
| `tests/benchmarks/` (`pytest-benchmark`) | frontend compile + Python simulator throughput |

None of them execute the generated code. That leaves the language's headline
claim — "machine code that is mathematically guaranteed to saturate CPU vector
units" — unmeasured. This harness closes that gap: it compiles each workload's
generated LLVM IR with `clang -O3 -march=native`, links a generated host driver
that calls `Lockstep_Tick` in a timed loop over a full arena, and reports real
throughput.

## What it does, per workload

1. Compile the `.lock` source to LLVM IR and a C header via `compile_lockstep`.
2. Determine the arena's true byte size from the generated
   `struct.Lockstep_Arena` type in the IR (the ABI the kernel was built for),
   and stream-leaf byte offsets from the header's `LOCKSTEP_OFFSET_*` macros.
3. Generate a C driver that allocates the arena (64-byte aligned), primes every
   input stream leaf to capacity with a deterministic pattern, warms up, then
   times `ITERS` calls to `Lockstep_Tick`.
4. Build driver + IR + `lockstep_intrinsics.c` with `clang -O3 -march=native`.
5. Run it and record per-tick latency, rows/s, arena GiB/s, and a checksum over
   an output leaf (a stable correctness signal that also blocks dead-code
   elimination of the kernel).

The kernels are branchless straight-line code, so per-tick timing is
data-independent. The harness fills streams to capacity rather than replaying a
fixture, so the row count (the throughput denominator) equals exactly what
`Lockstep_Tick` processes each tick.

## Requirements

A working LLVM/clang toolchain on `PATH` (`clang`, and the linker it drives).
The harness exits with a clear message if `clang` is missing.

## Running

```bash
# All workloads, markdown table
python benchmarks/native/run_native.py

# or via make
make bench-native

# One workload, raw JSON, more iterations
python benchmarks/native/run_native.py --workload particle_energy --iterations 5000 --json

# Persist a JSON report and keep the generated IR/driver/exe for inspection
python benchmarks/native/run_native.py --output native-results.json --keep-artifacts /tmp/lsnat
```

### Metrics

* `per_tick_us` — wall-clock microseconds for one `Lockstep_Tick` over the full arena.
* `mrows_per_sec` — stream rows processed per second (millions).
* `arena_gib_per_sec` — arena bytes touched per second (GiB), a rough memory-bandwidth proxy.
* `checksum` — deterministic sum over an output leaf; stable across runs of the same build.

Absolute numbers depend on the host CPU, so treat them as relative/regression
signals rather than portable constants.

## SoA-vs-AoS layout micro-benchmark

`soa_vs_aos.py` isolates and quantifies *why* Lockstep decomposes structs into
parallel primitive arrays (Struct-of-Arrays) instead of an Array-of-Structs. It
runs the same branchless particle kernel over the same data in both layouts,
across a sweep of working-set sizes, and reports the throughput ratio. The two
layouts compute identical results (checked), so the only variable is memory
layout. Built with `clang -O3 -march=native -ffast-math` (the `-ffast-math`
mirrors the `fast` flags Lockstep emits on reduction loops).

```bash
python benchmarks/native/soa_vs_aos.py
python benchmarks/native/soa_vs_aos.py --sizes 16000 1000000 --json
```

Two kernels model the two ways layout matters:

* **integrate** touches every field (a full physics step). SoA wins on
  *vectorization*: its contiguous columns feed packed SIMD loads/stores, while
  AoS's interleaved stride forces gathers or scalar code.
* **energy** reads only velocity + mass (a subset). SoA additionally wins on
  *bandwidth*: AoS drags the unused position fields through cache on every pass.

Expect the SoA speedup to be largest when the data is cache-resident and to
narrow (but stay well above 1×) once the sweep goes memory-bound at large sizes.

## Multi-stage fusion probe

`fusion_probe.py` quantifies throughput lost to un-fused multi-stage pipelines.
Lockstep's optimizer already plans stage fusion — for `multi_stage_pipeline` it
reports `alertsActive = FUSED[Normalize -> Score -> KeepActive]` with
`eliminated_intermediates: [eventsEnriched, alertsScored]` — and the code
generator emits that single fused loop when `_can_vectorize_fused_group` passes.

Historically that check bailed on any group whose kernel took an `accum`
parameter, so accumulator pipelines always fell back to one strip-mined loop per
stage. **That `accum` restriction has been lifted:** a group of pure shaders
that accumulates now fuses into a single vector loop that carries each
accumulator slot in a register across the fused body and writes it back per row
(the horizontal `fold` reduction runs afterwards, unchanged). A pure two-stage
accumulator pipeline measures ~1.9× faster fused than as per-stage loops on the
current host.

The remaining blocker for the shipped `multi_stage_pipeline` and
`telemetry_filter_aggregation` workloads is the **trailing filter** in each
group (`KeepActive` / the telemetry keep stage): `_lower_fused_kernel_group`
still falls back to per-stage lowering for any group containing a filter, because
a filter's compacting store has a data-dependent write index that the current
vector path does not lower. Fusing through filters is the next codegen step; the
probe below still bounds the win available once it lands.

The probe runs the multi_stage computation two ways over identical SoA data —
`unfused` (three loops writing/re-reading intermediates) and `fused` (one loop,
intermediates in registers, accumulating on the fly) — and reports the speedup.
Both produce identical results (checked), so the delta is the win fusing the
whole group recovers.

```bash
python benchmarks/native/fusion_probe.py
make bench-fusion
```

On the current host the fully fused form runs roughly **5×** faster than the
un-fused form even at memory-bound sizes, so extending fusion through the
trailing filter is the highest-leverage remaining codegen throughput opportunity
for these pipelines.
