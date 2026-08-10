.PHONY: verify verify-parser-toolchain generate-parser check-generated-parser build test test-cov mypy lock-deps check-lock-deps bench bench-check bench-native bench-fusion

verify: test mypy

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
	pytest --cov=. --cov-branch --cov-report=term-missing --cov-report=xml:coverage.xml

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

# Multi-stage fusion probe: measures the throughput gap between the per-stage
# loops codegen emits for accumulator pipelines and a single fused loop. Requires
# clang on PATH.
bench-fusion:
	python benchmarks/native/fusion_probe.py --output fusion-probe-results.json
