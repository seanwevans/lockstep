from dataclasses import dataclass, field
from typing import Any

from .diagnostics import LockstepCompileError, LockstepDiagnostic
from .entities import LockstepDebugVisitor
from .parser import create_parse_tree
from .semantic import validate_semantics


@dataclass
class LockstepCompileResult:
    parse_tree: Any
    entities: dict[str, Any]
    diagnostics: list[LockstepDiagnostic] = field(default_factory=list)


def compile_lockstep(source_code: str, verbose: bool = True) -> LockstepCompileResult:
    tree, parse_errors = create_parse_tree(source_code)
    if parse_errors:
        raise LockstepCompileError(parse_errors, diagnostics=parse_errors)

    semantic_diagnostics = validate_semantics(tree)
    semantic_errors = [d for d in semantic_diagnostics if d.severity == "error"]
    if semantic_errors:
        raise LockstepCompileError(
            semantic_errors,
            diagnostics=semantic_diagnostics,
            phase="semantic",
        )

    visitor = LockstepDebugVisitor(verbose=verbose)
    visitor.visit(tree)
    return LockstepCompileResult(
        parse_tree=tree,
        entities={
            "structs": visitor.structs,
            "shaders": visitor.shaders,
            "filters": visitor.filters,
            "pure_functions": visitor.pure_functions,
            "streams": visitor.streams,
            "accumulators": visitor.accumulators,
            "uniforms": visitor.uniforms,
            "bind_routes": visitor.bind_routes,
        },
        diagnostics=[*semantic_diagnostics, *visitor.diagnostics],
    )
