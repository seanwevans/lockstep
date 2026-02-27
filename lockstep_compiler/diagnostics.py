from dataclasses import dataclass, field
from typing import Any

from antlr4.error.ErrorListener import ErrorListener


@dataclass
class LockstepDiagnostic:
    severity: str
    code: str
    message: str
    line: int
    column: int
    hint: str | None = None


@dataclass
class LockstepCompileResult:
    parse_tree: Any
    entities: dict[str, Any]
    diagnostics: list[LockstepDiagnostic] = field(default_factory=list)


_SEVERITY_PRIORITY = {"error": 0, "warning": 1, "info": 2}


def normalize_diagnostics(
    diagnostics: list[LockstepDiagnostic],
) -> list[LockstepDiagnostic]:
    """Deduplicate and deterministically sort diagnostics."""

    deduped: dict[tuple[str, str, int, int], LockstepDiagnostic] = {}
    for diagnostic in diagnostics:
        key = (
            diagnostic.code,
            diagnostic.message,
            diagnostic.line,
            diagnostic.column,
        )
        if key not in deduped:
            deduped[key] = diagnostic

    return sorted(
        deduped.values(),
        key=lambda diagnostic: (
            diagnostic.line,
            diagnostic.column,
            _SEVERITY_PRIORITY.get(diagnostic.severity, 99),
            diagnostic.code,
        ),
    )


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
