# Native execution benchmarks

These benchmarks measure the **compiled machine code** Lockstep generates, not
the Python frontend.

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
