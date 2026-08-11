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
`eliminated_intermediates: [eventsEnriched, alertsScored]` — but the code
generator only emits that single fused loop when `_can_vectorize_fused_group`
passes, and that check bails on any group whose kernel takes an `accum`
parameter:

```python
if param.modifier == "accum":
    return False
```

`multi_stage_pipeline` (its `Score` stage) and `telemetry_filter_aggregation`
(its `AggregateReadings` stage) both accumulate, so codegen falls back to one
strip-mined loop **per stage** — each streaming the whole arena and materializing
the intermediate streams the optimizer said it would eliminate. That extra memory
traffic is why those workloads sit well below the `memcpy` roofline in
`run_native.py`.

The probe runs the multi_stage computation two ways over identical SoA data —
`unfused` (three loops writing/re-reading intermediates, as codegen emits today)
and `fused` (one loop, intermediates in registers, accumulating on the fly, as
the optimizer plans) — and reports the speedup. Both produce identical results
(checked), so the delta is exactly the win a fusion-through-accumulators codegen
change would recover.

```bash
python benchmarks/native/fusion_probe.py
make bench-fusion
```

On the current host the fused form runs roughly **5×** faster than the un-fused
form even at memory-bound sizes, so lifting the `accum` restriction on stage
fusion is the highest-leverage codegen throughput opportunity for pipelines that
accumulate.
