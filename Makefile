.PHONY: verify verify-parser-toolchain generate-parser check-generated-parser build test test-cov lint mypy lock-deps check-lock-deps bench bench-check bench-native bench-native-check bench-soa bench-fusion bench-vs-c

verify: lint test mypy

# Minimum overall statement+branch coverage for the shipped package. Kept a few
# points below the current level so ordinary variation never fails CI; ratchet
# it upward as coverage improves.
COV_FAIL_UNDER = 70

lint:
	ruff check lockstep_compiler

verify-parser-toolchain:
	python scripts/generate_parser.py --verify-toolchain

generate-parser:
	$(MAKE) verify-parser-toolchain
	python scripts/generate_parser.py

check-generated-parser:
	$(MAKE) verify-parser-toolchain
	python scripts/generate_parser.py
	git diff --exit-code -- generated/parser/LockstepLexer.py generated/parser/LockstepListener.py generated/parser/LockstepParser.py generated/parser/LockstepVisitor.py

# Generate universal (cross-platform, cross-version) hashed lockfiles with uv.
# `--universal` resolves for every platform/interpreter at once and records the
# environment markers, so a single lockfile installs correctly across the whole
# test matrix (Windows + Python 3.10-3.12) without per-environment phantom deps.
# `--python-version 3.10` sets the lower bound to match the oldest supported
# interpreter. uv is self-contained and does not import pip internals, so it is
# not coupled to a specific pip release the way pip-tools is.
UV_COMPILE = uv pip compile --universal --python-version 3.10 --generate-hashes

lock-deps:
	$(UV_COMPILE) --output-file requirements.lock pyproject.toml
	$(UV_COMPILE) --extra test --output-file requirements-test.lock pyproject.toml
	$(UV_COMPILE) --extra lsp --output-file requirements-lsp.lock pyproject.toml

check-lock-deps:
	$(MAKE) lock-deps
	git diff --exit-code -- pyproject.toml requirements.lock requirements-test.lock requirements-lsp.lock

build:
	python -m pip wheel . --no-deps --wheel-dir dist

test:
	pytest

test-cov:
	pytest --cov=lockstep_compiler --cov-branch --cov-report=term-missing \
		--cov-report=xml:coverage.xml --cov-fail-under=$(COV_FAIL_UNDER)

mypy:
	python -m mypy

bench:
	PYTHONPATH=. python scripts/run_benchmarks.py --output benchmark-results.json

bench-check:
	PYTHONPATH=. python scripts/check_benchmark_regression.py --baseline benchmarks/baselines/default.json --current benchmark-results.json --threshold 0.10
	pytest tests/benchmarks -q --benchmark-only

# Native execution benchmarks: compile each workload's generated code with
# clang -O3 -march=native and time Lockstep_Tick over a full arena. Requires an
# LLVM/clang toolchain on PATH.
bench-native:
	python benchmarks/native/run_native.py --output native-results.json

# Gate the deterministic native invariants (arena ABI + output checksums) against
# the checked-in baseline. Throughput is intentionally not gated -- it is
# hardware-dependent and too noisy on shared CI runners. Regenerate the baseline
# after an intentional codegen change with:
#   python scripts/check_native_benchmark.py --current native-results.json --update
bench-native-check: bench-native
	python scripts/check_native_benchmark.py \
		--baseline benchmarks/baselines/native.json --current native-results.json

# SoA-vs-AoS layout micro-benchmark: quantifies the Struct-of-Arrays throughput
# win over Array-of-Structs for the same kernel. Requires clang on PATH.
bench-soa:
	python benchmarks/native/soa_vs_aos.py --output soa-vs-aos-results.json

# Multi-stage fusion probe: measures the throughput gap between the per-stage
# loops codegen emits for accumulator pipelines and a single fused loop. Requires
# clang on PATH.
bench-fusion:
	python benchmarks/native/fusion_probe.py --output fusion-probe-results.json

# Lockstep vs hand-written C: pits the shipped Lockstep_Tick machine code against
# an idiomatic single-pass C kernel computing the same transform over the same
# arena. The honest external comparison -- shows where hand-written C still wins.
# Requires clang on PATH.
bench-vs-c:
	python benchmarks/native/lockstep_vs_c.py --output lockstep-vs-c-results.json
