from dataclasses import dataclass, field
from typing import Any


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
