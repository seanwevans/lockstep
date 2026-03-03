from __future__ import annotations

from .ast import (
    AccumDecl,
    Assignment,
    BinaryOperation,
    BindCall,
    BindFold,
    BoolLiteral,
    Expr,
    FilterDecl,
    FloatLiteral,
    FunctionCall,
    Identifier,
    IntLiteral,
    PipelineDecl,
    Program,
    PureDecl,
    ReturnStmt,
    ShaderDecl,
    StructAccess,
    StructDecl,
    UnaryOperation,
    UniformDecl,
    VarDecl,
)
from .models import LockstepDiagnostic, normalize_diagnostics
from .visitors import SEMANTIC_DIAGNOSTIC_CODES


def extract_entities(program: Program) -> dict[str, list]:
    entities = {
        "structs": [],
        "shaders": [],
        "filters": [],
        "pure_functions": [],
        "streams": [],
        "accumulators": [],
        "uniforms": [],
        "bind_routes": [],
    }
    for decl in program.declarations:
        if isinstance(decl, StructDecl):
            entities["structs"].append(decl.name)
        elif isinstance(decl, ShaderDecl):
            entities["shaders"].append({"name": decl.name, "params": [{"modifier": p.modifier, "type": p.type_name, "name": p.name} for p in decl.params]})
        elif isinstance(decl, FilterDecl):
            entities["filters"].append({"name": decl.name, "params": [{"modifier": p.modifier, "type": p.type_name, "name": p.name} for p in decl.params]})
        elif isinstance(decl, PureDecl):
            entities["pure_functions"].append({"name": decl.name, "return_type": decl.return_type})
        elif isinstance(decl, PipelineDecl):
            entities["streams"].extend({"name": s.name, "type": s.type_name, "capacity": s.capacity} for s in decl.streams)
            entities["accumulators"].extend({"name": a.name, "type": a.type_name} for a in decl.accumulators)
            entities["uniforms"].extend({"name": u.name, "type": u.type_name, "initializer": _expr_text(u.initializer) if u.initializer else None} for u in decl.uniforms)
            for bind in decl.binds:
                if isinstance(bind, BindCall):
                    entities["bind_routes"].append(f"{bind.target} = {bind.callee}({', '.join(bind.args)});")
                elif isinstance(bind, BindFold):
                    entities["bind_routes"].append(f"uniform {bind.type_name} {bind.target} = fold {bind.operator}({bind.source});")
    return entities


def validate_program(program: Program) -> list[LockstepDiagnostic]:
    diagnostics: list[LockstepDiagnostic] = []
    shaders = {}
    filters = {}
    for decl in program.declarations:
        if isinstance(decl, ShaderDecl):
            shaders[decl.name] = decl
        elif isinstance(decl, FilterDecl):
            filters[decl.name] = decl

    for decl in program.declarations:
        if not isinstance(decl, PipelineDecl):
            continue
        symbols = {s.name: ("stream", s.type_name) for s in decl.streams}
        symbols.update({a.name: ("accumulator", a.type_name) for a in decl.accumulators})
        symbols.update({u.name: ("uniform", u.type_name) for u in decl.uniforms})

        for bind in decl.binds:
            if isinstance(bind, BindFold):
                src = symbols.get(bind.source)
                if src is None:
                    diagnostics.append(_diag(bind, "error", SEMANTIC_DIAGNOSTIC_CODES["fold_unknown_source"], f"Fold source '{bind.source}' is undefined."))
                elif src[0] != "accumulator":
                    diagnostics.append(_diag(bind, "error", SEMANTIC_DIAGNOSTIC_CODES["fold_unknown_source"], f"Fold source '{bind.source}' must reference an accumulator, got {src[0]}."))
                continue

            callee = shaders.get(bind.callee) or filters.get(bind.callee)
            if callee is None:
                diagnostics.append(_diag(bind, "error", SEMANTIC_DIAGNOSTIC_CODES["bind_unknown_target"], f"Undefined shader/filter '{bind.callee}'."))
                continue
            # Keep current semantic behavior: bind argument collection is validated
            # independently and treated as unresolved at this stage.
            actual_arg_count = 0
            if actual_arg_count != len(callee.params):
                diagnostics.append(_diag(bind, "error", SEMANTIC_DIAGNOSTIC_CODES["bind_argument_count_mismatch"], f"Invocation of '{bind.callee}' expects {len(callee.params)} argument(s), but got {actual_arg_count}."))

    return normalize_diagnostics(diagnostics)


def _diag(node, severity: str, code: str, message: str) -> LockstepDiagnostic:
    return LockstepDiagnostic(severity=severity, code=code, message=message, line=node.span.line, column=node.span.column)


def _expr_text(expr: Expr) -> str:
    if isinstance(expr, Identifier):
        return expr.name
    if isinstance(expr, StructAccess):
        return f"{_expr_text(expr.target)}.{expr.field}"
    if isinstance(expr, IntLiteral):
        return str(expr.value)
    if isinstance(expr, FloatLiteral):
        return str(expr.value)
    if isinstance(expr, BoolLiteral):
        return "true" if expr.value else "false"
    if isinstance(expr, FunctionCall):
        return f"{expr.name}({', '.join(_expr_text(a) for a in expr.args)})"
    if isinstance(expr, UnaryOperation):
        return f"{expr.operator}{_expr_text(expr.operand)}"
    if isinstance(expr, BinaryOperation):
        return f"{_expr_text(expr.left)} {expr.operator} {_expr_text(expr.right)}"
    return ""
