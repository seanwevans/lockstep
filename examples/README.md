# Minimal host integration example

This example shows the full host integration loop with generated artifacts:

1. compile a Lockstep source file,
2. emit LLVM IR + C header,
3. compile/link with a C host,
4. allocate arena memory,
5. initialize data,
6. call `Lockstep_Tick`.

## Files

- `minimal.lock`: a tiny Lockstep program that declares one stream and one uniform.
- `minimal_host.c`: a C host that includes the generated header and executes one tick.

## Build and run

From the repository root:

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

This verifies the generated ABI end-to-end (header + LLVM IR + host application wiring).
