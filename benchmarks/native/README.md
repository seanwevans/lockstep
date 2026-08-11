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

### Sweeping the SIMD width

`--target-width` selects the SIMD vector width the compiler lowers to (the same
knob as `lockstepc --target-width` and the `LOCKSTEP_SIMD_WIDTH` header macro;
default 8). Sweep it to measure how the emitted vector width affects throughput
on a given host:

```bash
for w in 4 8 16; do
  python benchmarks/native/run_native.py --workload particle_energy --target-width $w
done
```

The width is a *performance* knob only — every width computes identical results
(pinned by `tests/test_target_width_execution.py`). **Wider is not automatically
faster.** The width flows into the manual fused-vector lowering path (fused
pipelines and fold reductions); the single-kernel workloads are scalar loops
that `clang` auto-vectorizes regardless, so their timings barely move with it.
Where the width does flow through, going past the native register width tends to
*lose* throughput — on an AVX-512 host, `--target-width 16` runs a fused
accumulator pipeline slower than the default 8, because the SoA lane
gather/scatter the fused path emits grows with the width and AVX-512 carries a
frequency penalty. The default of 8 (256-bit / AVX2-shaped) is a good portable
choice; treat wider as something to measure per host, not assume.

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

## Lockstep vs hand-written C

The three harnesses above pit Lockstep against *itself* — SoA vs AoS, fused vs
per-stage — and Lockstep's model always wins. `lockstep_vs_c.py` asks the honest
external question instead: for the same computation, does the code Lockstep
actually **ships** keep up with the obvious single-pass C loop a competent human
would write by hand?

It links the real `Lockstep_Tick` (compiled from each workload's LLVM IR, exactly
as `run_native.py` builds it) against a hand-written C reference kernel that
computes the same end-to-end transform in one fused pass over the same SoA arena
— identical byte offsets, taken from the generated header. Two identical arenas
are primed with the same deterministic pattern `run_native.py` uses; both kernels
are timed in one process; and the harness refuses to report a result unless the
two paths produce the same output checksum. (Those checksums also match
`run_native.py`'s for the shared workloads, cross-validating the references.)
Built with `clang -O3 -march=native -ffast-math`.

```bash
python benchmarks/native/lockstep_vs_c.py
make bench-vs-c
python benchmarks/native/lockstep_vs_c.py --workload particle_energy --target-width 16 --json
```

`ratio = C time / Lockstep time`: `>= 1.0` means Lockstep matches or beats
hand-written C, `< 1.0` means the C baseline wins. On the current host **C wins
every workload**, and the shape of the gap is the useful part:

* `particle_energy` — a single fused kernel, the case Lockstep is supposed to
  nail. It still trails hand-written C by ~20–25% (ratio ~0.77–0.80, stable
  across `--target-width`). The gap isn't SIMD width; it's that Lockstep loads
  and stores the arena through byte-addressed pointers while the C reference
  walks typed contiguous columns and contracts its multiplies into FMAs under
  `-ffast-math`. Emitting typed column pointers (and `contract`/`fast` flags on
  kernel arithmetic, not just reductions) is the improvement this points at.
* `telemetry_filter_aggregation` and `multi_stage_pipeline` — multi-stage
  pipelines ending in a filter. Codegen lowers them to one strip-mined loop per
  stage and materializes the intermediate streams, so the single-pass C
  reference (one trip through memory instead of two or three) wins by **~5×** and
  **~4×** respectively. This is the same gap `fusion_probe.py` measures
  internally, now shown against a real external baseline: it is what
  fusing-through-filters would recover.

This is the harness to reach for when the question is "what beats us on raw
numbers." Today, hand-written C does — and the ratios say exactly where and why.
