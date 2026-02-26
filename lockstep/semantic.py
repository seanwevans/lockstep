from typing import Any

from LockstepParser import LockstepParser
from LockstepVisitor import LockstepVisitor

from .diagnostics import LockstepDiagnostic


class LockstepSemanticValidator(LockstepVisitor):
    """Runs semantic checks on a parsed Lockstep program."""

    def __init__(self):
        self.diagnostics: list[LockstepDiagnostic] = []
        self.scopes: list[dict[str, dict[str, str]]] = []
        self.shaders: dict[str, list[dict[str, str]]] = {}
        self.filters: dict[str, list[dict[str, str]]] = {}
        self.current_shader_name: str | None = None

    def _line_col(self, ctx) -> tuple[int, int]:
        token = getattr(ctx, "start", None)
        return (
            getattr(token, "line", 0),
            getattr(token, "column", 0),
        )

    def _add_diagnostic(self, *, severity: str, code: str, message: str, ctx, hint: str | None = None):
        line, column = self._line_col(ctx)
        self.diagnostics.append(
            LockstepDiagnostic(
                severity=severity,
                code=code,
                message=message,
                line=line,
                column=column,
                hint=hint,
            )
        )

    def _push_scope(self):
        self.scopes.append({})

    def _pop_scope(self):
        if self.scopes:
            self.scopes.pop()

    def _declare(self, name: str, declared_type: str, ctx, *, duplicate_code: str, kind: str = "symbol"):
        if not self.scopes:
            self._push_scope()
        current_scope = self.scopes[-1]
        if name in current_scope:
            self._add_diagnostic(
                severity="error",
                code=duplicate_code,
                message=f"Duplicate declaration for '{name}' in the same scope.",
                ctx=ctx,
                hint="Rename one declaration or move it to a different scope.",
            )
            return False
        current_scope[name] = {"type": declared_type, "kind": kind}
        return True

    def _lookup(self, name: str) -> dict[str, str] | None:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def _declared_in_current_scope(self, name: str) -> bool:
        return bool(self.scopes and name in self.scopes[-1])

    def _record_kernel_signature(self, ctx, target: dict[str, list[dict[str, str]]]):
        name = ctx.ID().getText()
        if name in target:
            self._add_diagnostic(
                severity="error",
                code="LCK307",
                message=f"Duplicate shader/filter declaration for '{name}'.",
                ctx=ctx,
                hint="Rename one declaration to avoid symbol collisions.",
            )
        params = []
        if ctx.paramList():
            for param in ctx.paramList().param():
                params.append(
                    {
                        "name": param.ID().getText(),
                        "type": param.typeName().getText(),
                        "modifier": param.getChild(0).getText(),
                    }
                )
        target[name] = params
        return name, params

    def _check_expression_identifier(self, name: str, ctx):
        if self._lookup(name) is None:
            self._add_diagnostic(
                severity="error",
                code="LCK301",
                message=f"Undefined identifier '{name}'.",
                ctx=ctx,
                hint="Declare the identifier in scope before using it.",
            )
            return None
        symbol = self._lookup(name)
        return symbol["type"] if symbol else None

    def _type_check_bind_call(self, ctx, target_name: str, callee_name: str, arg_names):
        kernel = self.shaders.get(callee_name) or self.filters.get(callee_name)
        if kernel is None:
            self._add_diagnostic(
                severity="error",
                code="LCK303",
                message=f"Undefined shader/filter '{callee_name}' in bind statement.",
                ctx=ctx,
                hint="Declare the shader/filter before using it in bind.",
            )
            return

        expected_arity = len(kernel)
        actual_arity = len(arg_names)
        if expected_arity != actual_arity:
            self._add_diagnostic(
                severity="error",
                code="LCK304",
                message=(
                    f"Invocation of '{callee_name}' expects {expected_arity} argument(s), "
                    f"but got {actual_arity}."
                ),
                ctx=ctx,
                hint="Match bind arguments to the shader/filter parameter list.",
            )
            return

        target_symbol = self._lookup(target_name)
        if target_symbol is None:
            self._add_diagnostic(
                severity="error",
                code="LCK301",
                message=f"Undefined identifier '{target_name}'.",
                ctx=ctx,
                hint="Declare pipeline streams/accumulators/uniforms before binding.",
            )

        for arg_name, expected in zip(arg_names, kernel):
            actual_symbol = self._lookup(arg_name)
            if actual_symbol is None:
                self._add_diagnostic(
                    severity="error",
                    code="LCK301",
                    message=f"Undefined identifier '{arg_name}'.",
                    ctx=ctx,
                    hint="Declare pipeline symbols before passing them to bind.",
                )
                continue
            actual_type = actual_symbol["type"]
            if actual_type != expected["type"]:
                self._add_diagnostic(
                    severity="error",
                    code="LCK305",
                    message=(
                        f"Type mismatch for argument '{arg_name}' in '{callee_name}': "
                        f"expected {expected['type']}, got {actual_type}."
                    ),
                    ctx=ctx,
                    hint="Align argument types with the shader/filter signature.",
                )

    def visitProgram(self, ctx: LockstepParser.ProgramContext):
        self._push_scope()
        result = self.visitChildren(ctx)
        self._pop_scope()
        return result

    def visitShaderDecl(self, ctx: LockstepParser.ShaderDeclContext):
        name, params = self._record_kernel_signature(ctx, self.shaders)
        self.current_shader_name = name
        self._push_scope()
        for param in params:
            self._declare(
                param["name"],
                param["type"],
                ctx,
                duplicate_code="LCK306",
                kind=f"param:{param['modifier']}",
            )
        result = self.visitChildren(ctx)
        self._pop_scope()
        self.current_shader_name = None
        return result

    def visitFilterDecl(self, ctx: LockstepParser.FilterDeclContext):
        _name, params = self._record_kernel_signature(ctx, self.filters)
        self._push_scope()
        for param in params:
            self._declare(
                param["name"],
                param["type"],
                ctx,
                duplicate_code="LCK306",
                kind=f"param:{param['modifier']}",
            )
        result = self.visitChildren(ctx)
        self._pop_scope()
        return result

    def visitVarDecl(self, ctx: LockstepParser.VarDeclContext):
        self._declare(
            ctx.ID().getText(),
            ctx.typeName().getText(),
            ctx,
            duplicate_code="LCK306",
            kind="local",
        )
        return self.visitChildren(ctx)

    def visitPipelineDecl(self, ctx: LockstepParser.PipelineDeclContext):
        self._push_scope()
        result = self.visitChildren(ctx)
        self._pop_scope()
        return result

    def visitStreamDecl(self, ctx: LockstepParser.StreamDeclContext):
        self._declare(
            ctx.ID().getText(),
            ctx.typeName().getText(),
            ctx,
            duplicate_code="LCK306",
            kind="stream",
        )
        return self.visitChildren(ctx)

    def visitAccumDecl(self, ctx: LockstepParser.AccumDeclContext):
        self._declare(
            ctx.ID().getText(),
            ctx.typeName().getText(),
            ctx,
            duplicate_code="LCK306",
            kind="accumulator",
        )
        return self.visitChildren(ctx)

    def visitUniformDecl(self, ctx: LockstepParser.UniformDeclContext):
        self._declare(
            ctx.ID().getText(),
            ctx.typeName().getText(),
            ctx,
            duplicate_code="LCK306",
            kind="uniform",
        )
        return self.visitChildren(ctx)

    def visitBindStmt(self, ctx: LockstepParser.BindStmtContext):
        id_tokens = ctx.ID()
        if ctx.argList() is not None:
            target_name = id_tokens[0].getText()
            callee_name = id_tokens[1].getText()
            arg_names = [token.getText() for token in id_tokens[2:]]
            self._type_check_bind_call(ctx, target_name, callee_name, arg_names)
            return self.visitChildren(ctx)

        fold_target = id_tokens[0].getText()
        fold_operator = id_tokens[1].getText()
        fold_source = id_tokens[2].getText()

        if self._declared_in_current_scope(fold_target):
            self._add_diagnostic(
                severity="error",
                code="LCK306",
                message=f"Duplicate declaration for '{fold_target}' in the same scope.",
                ctx=ctx,
                hint="Rename one declaration or move it to a different scope.",
            )
        else:
            self.scopes[-1][fold_target] = {"type": ctx.typeName().getText(), "kind": "uniform"}

        fold_source_symbol = self._lookup(fold_source)
        declared_type = ctx.typeName().getText()
        if fold_source_symbol is None:
            self._add_diagnostic(
                severity="error",
                code="LCK401",
                message=f"Fold source accumulator '{fold_source}' is undefined.",
                ctx=ctx,
                hint="Declare an accumulator and use it as the fold source.",
            )
        elif fold_source_symbol["kind"] != "accumulator":
            self._add_diagnostic(
                severity="error",
                code="LCK403",
                message=(
                    f"Fold source '{fold_source}' must reference an accumulator, "
                    f"got {fold_source_symbol['kind']}."
                ),
                ctx=ctx,
                hint="Use an accumulator as the input to fold.",
            )
        elif fold_operator not in {"sum", "avg", "min", "max"}:
            self._add_diagnostic(
                severity="error",
                code="LCK402",
                message=f"Unsupported fold operator '{fold_operator}'.",
                ctx=ctx,
                hint="Use a valid fold operator such as sum, avg, min, or max.",
            )
        elif fold_source_symbol["type"] != declared_type:
            self._add_diagnostic(
                severity="error",
                code="LCK404",
                message=(
                    f"Fold target '{fold_target}' has type {declared_type}, but fold source "
                    f"'{fold_source}' has accumulator type {fold_source_symbol['type']}."
                ),
                ctx=ctx,
                hint="Match the folded uniform type to the accumulator type.",
            )

        return self.visitChildren(ctx)

    def visitPrimaryExpr(self, ctx: LockstepParser.PrimaryExprContext):
        if ctx.ID():
            if ctx.exprList() is None:
                self._check_expression_identifier(ctx.ID().getText(), ctx)
            return self.visitChildren(ctx)

        if ctx.lvalue():
            root_identifier = ctx.lvalue().ID(0).getText()
            self._check_expression_identifier(root_identifier, ctx)

        return self.visitChildren(ctx)

    def visitLvalue(self, ctx: LockstepParser.LvalueContext):
        root_identifier = ctx.ID(0).getText()
        self._check_expression_identifier(root_identifier, ctx)
        return self.visitChildren(ctx)

    def validate(self, tree):
        self.visit(tree)
        return self.diagnostics


def validate_semantics(parse_tree: Any) -> list[LockstepDiagnostic]:
    """Validate semantic constraints after syntactic parsing succeeds."""

    validator = LockstepSemanticValidator()
    return validator.validate(parse_tree)
