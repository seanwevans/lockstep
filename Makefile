.PHONY: verify-parser-toolchain generate-parser check-generated-parser build test test-cov

verify-parser-toolchain:
	python scripts/generate_parser.py --verify-toolchain

generate-parser:
	$(MAKE) verify-parser-toolchain
	python scripts/generate_parser.py

check-generated-parser:
	$(MAKE) verify-parser-toolchain
	python scripts/generate_parser.py
	git diff --exit-code -- generated/parser/LockstepLexer.py generated/parser/LockstepListener.py generated/parser/LockstepParser.py generated/parser/LockstepVisitor.py

build:
	python -m pip wheel . --no-deps --wheel-dir dist

test:
	pytest

test-cov:
	pytest --cov=. --cov-branch --cov-report=term-missing --cov-report=xml:coverage.xml
