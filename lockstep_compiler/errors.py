from antlr4.error.ErrorListener import ErrorListener

from .models import LockstepDiagnostic


class ParseErrorCollector(ErrorListener):
    """Collects syntax errors emitted by ANTLR during lex/parse."""

    def __init__(self, source_file: str | None = None):
        super().__init__()
        self.source_file = source_file
        self.errors: list[LockstepDiagnostic] = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.errors.append(
            LockstepDiagnostic(
                severity="error",
                code="LCK001",
                message=msg,
                line=line,
                column=column,
                source_file=self.source_file,
                hint="Fix syntax errors before semantic analysis can continue.",
            )
        )


class LockstepCompileError(Exception):
    """Raised when the Lockstep source contains parse or semantic errors."""

    def __init__(
        self,
        errors,
        diagnostics=None,
        *,
        phase: str = "parse",
        source_file: str | None = None,
    ):
        self.errors = errors
        self.diagnostics = diagnostics or []
        self.phase = phase
        self.source_file = source_file
        super().__init__(self._format_message())

    def _format_message(self):
        count = len(self.errors)
        suffix = "" if count == 1 else "s"
        inferred_source_file = self.source_file or (
            self.errors[0].source_file if self.errors else None
        )
        file_context = f" in '{inferred_source_file}'" if inferred_source_file else ""
        summary = (
            f"Compilation failed with {count} {self.phase} error{suffix}{file_context}."
        )

        details = []
        for error in self.errors:
            if error.source_file:
                details.append(
                    f"{error.source_file}:{error.line}:{error.column} {error.message}"
                )
            else:
                details.append(f"line {error.line}:{error.column} {error.message}")
        detail_text = "\n".join(details)
        return summary if not detail_text else f"{summary}\n{detail_text}"
