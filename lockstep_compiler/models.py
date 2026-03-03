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
    ast: Any | None = None
    llvm_ir: str = ""
    diagnostics: list[LockstepDiagnostic] = field(default_factory=list)


@dataclass(frozen=True)
class SemanticSymbol:
    name: str
    declared_type: str
    kind: str


@dataclass(frozen=True)
class SemanticKernelParam:
    name: str
    declared_type: str
    modifier: str


@dataclass(frozen=True)
class SemanticStructField:
    name: str
    declared_type: str


_SEVERITY_PRIORITY = {"error": 0, "warning": 1, "info": 2}


def _should_replace_diagnostic(
    existing: LockstepDiagnostic,
    candidate: LockstepDiagnostic,
) -> bool:
    existing_priority = _SEVERITY_PRIORITY.get(existing.severity, 99)
    candidate_priority = _SEVERITY_PRIORITY.get(candidate.severity, 99)
    if candidate_priority != existing_priority:
        return candidate_priority < existing_priority

    existing_hint = (existing.hint or "").strip()
    candidate_hint = (candidate.hint or "").strip()
    if bool(candidate_hint) != bool(existing_hint):
        return bool(candidate_hint)

    if candidate_hint and existing_hint:
        return candidate_hint < existing_hint

    return False


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
        if key not in deduped or _should_replace_diagnostic(deduped[key], diagnostic):
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
