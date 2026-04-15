.PHONY: verify verify-parser-toolchain generate-parser check-generated-parser build test test-cov mypy lock-deps check-lock-deps bench

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

lock-deps:
	pip-compile --generate-hashes --strip-extras --output-file requirements.lock pyproject.toml
	pip-compile --generate-hashes --strip-extras --extra test --output-file requirements-test.lock pyproject.toml
	pip-compile --generate-hashes --strip-extras --extra lsp --output-file requirements-lsp.lock pyproject.toml

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
	pytest tests/benchmarks -q --benchmark-only
