.PHONY: generate-parser check-generated-parser

generate-parser:
	python scripts/generate_parser.py

check-generated-parser:
	python scripts/generate_parser.py
	git diff --exit-code -- generated/parser/LockstepLexer.py generated/parser/LockstepListener.py generated/parser/LockstepParser.py generated/parser/LockstepVisitor.py
