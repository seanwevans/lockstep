from typing import Any

from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener

from LockstepLexer import LockstepLexer
from LockstepParser import LockstepParser

from .diagnostics import LockstepDiagnostic


class ParseErrorCollector(ErrorListener):
    """Collects syntax errors emitted by ANTLR during lex/parse."""

    def __init__(self):
        super().__init__()
        self.errors: list[LockstepDiagnostic] = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.errors.append(
            LockstepDiagnostic(
                severity="error",
                code="LCK001",
                message=msg,
                line=line,
                column=column,
                hint="Fix syntax errors before semantic analysis can continue.",
            )
        )


def create_parse_tree(source_code: str) -> tuple[Any, list[LockstepDiagnostic]]:
    input_stream = InputStream(source_code)
    lexer = LockstepLexer(input_stream)
    error_listener = ParseErrorCollector()
    lexer.removeErrorListeners()
    lexer.addErrorListener(error_listener)
    stream = CommonTokenStream(lexer)

    parser = LockstepParser(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(error_listener)
    tree = parser.program()

    return tree, error_listener.errors
