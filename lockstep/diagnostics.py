from dataclasses import dataclass


@dataclass
class LockstepDiagnostic:
    severity: str
    code: str
    message: str
    line: int
    column: int
    hint: str | None = None


class LockstepCompileError(Exception):
    """Raised when the Lockstep source contains parse or semantic errors."""

    def __init__(self, errors, diagnostics=None, *, phase: str = "parse"):
        self.errors = errors
        self.diagnostics = diagnostics or []
        self.phase = phase
        super().__init__(format_compile_error_message(errors, phase=phase))


def format_compile_error_message(errors: list[LockstepDiagnostic], *, phase: str) -> str:
    count = len(errors)
    suffix = "" if count == 1 else "s"
    summary = f"Compilation failed with {count} {phase} error{suffix}."
    details = "\n".join(format_diagnostic_location(error) for error in errors)
    return summary if not details else f"{summary}\n{details}"


def format_diagnostic_location(error: LockstepDiagnostic) -> str:
    return f"line {error.line}:{error.column} {error.message}"
