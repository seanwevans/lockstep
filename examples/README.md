# Examples

Two programs with C hosts, built the same way:

1. compile the Lockstep source to LLVM IR and a C header;
2. compile the IR with clang and link it with the host;
3. the host allocates the arena, primes it, and calls `Lockstep_Tick`.

`tests/test_examples.py` builds and runs both.

## `particles.lock` + `particles_host.c`: the full host contract

A small particle simulation, set up so that it touches every part of the ABI a
host deals with:

- `Integrate` updates the particles in place (gravity and a bouncing floor)
  and accumulates kinetic energy.
- `InsideBox` is a filter that keeps only the particles inside the box, so it
  keeps a data-dependent number of rows.
- `Tally` counts the kept rows. It fuses with `InsideBox` into one vector loop
  that compacts the kept rows.
- Two folds, `kineticEnergy` and `keptCount`, publish scalars.

Each frame the host calls `Lockstep_Tick`, then reads the folded uniforms. For
the filtered stream it reads `count_tallied_value`
(`LOCKSTEP_OFFSET_COUNT_TALLIED`) **before** reading its rows, because rows
past the count are unspecified. The host also writes every uniform the program
reads: the arena is host-owned, and the `= 0.02` initializers in the source
document intent but are not applied by the tick.

```bash
mkdir -p examples/build
lockstepc examples/particles.lock --emit-ir > examples/build/particles.ll
lockstepc examples/particles.lock --emit-header > examples/build/lockstep_generated.h
clang -O2 -c examples/build/particles.ll -o examples/build/particles.o
clang -std=c11 -O2 examples/particles_host.c examples/build/particles.o -Iexamples/build -o examples/build/particles_host
./examples/build/particles_host
```

Expected output:

```text
frame 1: kinetic energy 10009.488, inside 892/1000 (kept 892), mean x 50.174
frame 2: kinetic energy 10115.902, inside 890/1000 (kept 890), mean x 50.062
frame 3: kinetic energy 10298.444, inside 889/1000 (kept 889), mean x 50.006
```

`tests/test_examples.py` checks each frame against the simulator, run over the
same inputs.

## `minimal.lock` + `minimal_host.c`: the smallest possible host

One stream and one uniform, no kernels: the tick is a no-op, and the host
shows the arena layout (SoA columns as arrays, uniforms as `uniform_<name>_value`).

```bash
mkdir -p examples/build
lockstepc examples/minimal.lock --emit-ir > examples/build/minimal.ll
lockstepc examples/minimal.lock --emit-header > examples/build/lockstep_generated.h
clang -c examples/build/minimal.ll -o examples/build/minimal.o
clang -std=c11 examples/minimal_host.c examples/build/minimal.o -Iexamples/build -o examples/build/minimal_host
./examples/build/minimal_host
```

Expected output:

```text
After tick: pos=10.00 vel=2.00 dt=0.50
```
