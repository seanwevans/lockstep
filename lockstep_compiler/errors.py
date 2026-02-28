from antlr4.error.ErrorListener import ErrorListener

from .models import LockstepDiagnostic


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


class LockstepCompileError(Exception):
    """Raised when the Lockstep source contains parse errors."""

    def __init__(self, errors, diagnostics=None, *, phase: str = "parse"):
        self.errors = errors
        self.diagnostics = diagnostics or []
        self.phase = phase
        super().__init__(self._format_message())

    def _format_message(self):
        count = len(self.errors)
        suffix = "" if count == 1 else "s"
        summary = f"Compilation failed with {count} {self.phase} error{suffix}."
        details = "\n".join(
            f"line {error.line}:{error.column} {error.message}" for error in self.errors
        )
        return summary if not details else f"{summary}\n{details}"
