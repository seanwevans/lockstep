.PHONY: generate-parser check-generated-parser

ANTLR_JAR_PATH ?= tools/antlr-4.13.2-complete.jar

generate-parser:
	python scripts/generate_parser.py --jar-path $(ANTLR_JAR_PATH)

check-generated-parser:
	python scripts/generate_parser.py --offline --jar-path $(ANTLR_JAR_PATH)
	git diff --exit-code -- generated/parser/LockstepLexer.py generated/parser/LockstepListener.py generated/parser/LockstepParser.py generated/parser/LockstepVisitor.py
