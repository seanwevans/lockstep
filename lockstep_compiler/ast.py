from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AstLocation:
    line: int = 0
    column: int = 0


@dataclass(frozen=True)
class AstExprLiteral:
    kind: str
    value: str


@dataclass(frozen=True)
class AstExprVar:
    path: tuple[str, ...]


@dataclass(frozen=True)
class AstExprUnary:
    op: str
    operand: "AstExpr"


@dataclass(frozen=True)
class AstExprBinary:
    op: str
    left: "AstExpr"
    right: "AstExpr"


@dataclass(frozen=True)
class AstExprCall:
    name: str
    args: tuple["AstExpr", ...]


AstExpr = AstExprLiteral | AstExprVar | AstExprUnary | AstExprBinary | AstExprCall


@dataclass(frozen=True)
class AstVarDeclStmt:
    declared_type: str | None
    name: str
    initializer: AstExpr | None


@dataclass(frozen=True)
class AstAssignStmt:
    target: tuple[str, ...]
    value: AstExpr


@dataclass(frozen=True)
class AstReturnStmt:
    value: AstExpr


AstStatement = AstVarDeclStmt | AstAssignStmt | AstReturnStmt


@dataclass(frozen=True)
class AstKernelParam:
    modifier: str
    declared_type: str
    name: str


@dataclass(frozen=True)
class AstStructField:
    declared_type: str
    name: str


@dataclass(frozen=True)
class AstStructDecl:
    name: str
    fields: tuple[AstStructField, ...] = ()
    location: AstLocation = AstLocation()


@dataclass(frozen=True)
class AstPureDecl:
    name: str
    return_type: str
    params: tuple[AstKernelParam, ...] = ()
    body: tuple[AstStatement, ...] = ()
    location: AstLocation = AstLocation()


@dataclass(frozen=True)
class AstKernelDecl:
    name: str
    params: tuple[AstKernelParam, ...] = ()
    body: tuple[AstStatement, ...] = ()
    location: AstLocation = AstLocation()


@dataclass(frozen=True)
class AstStreamDecl:
    name: str
    declared_type: str
    capacity: str
    location: AstLocation = AstLocation()


@dataclass(frozen=True)
class AstAccumulatorDecl:
    name: str
    declared_type: str
    location: AstLocation = AstLocation()


@dataclass(frozen=True)
class AstUniformDecl:
    name: str
    declared_type: str
    initializer: str | None = None
    location: AstLocation = AstLocation()


@dataclass(frozen=True)
class AstKernelBindRoute:
    target: str
    kernel: str
    args: tuple[str, ...]
    route: str


@dataclass(frozen=True)
class AstFoldBindRoute:
    uniform_type: str
    uniform_name: str
    operator: str
    source: str
    route: str


AstBindRoute = AstKernelBindRoute | AstFoldBindRoute


@dataclass(frozen=True)
class AstPipelineDecl:
    name: str
    streams: tuple[AstStreamDecl, ...] = ()
    accumulators: tuple[AstAccumulatorDecl, ...] = ()
    uniforms: tuple[AstUniformDecl, ...] = ()
    bind_routes: tuple[AstBindRoute, ...] = ()
    location: AstLocation = AstLocation()


@dataclass(frozen=True)
class AstProgram:
    structs: tuple[AstStructDecl, ...] = ()
    shaders: tuple[AstKernelDecl, ...] = ()
    filters: tuple[AstKernelDecl, ...] = ()
    pure_functions: tuple[AstPureDecl, ...] = ()
    pipelines: tuple[AstPipelineDecl, ...] = ()


class _AstBuilderMixin:
    @staticmethod
    def _location(ctx: Any) -> AstLocation:
        token = getattr(ctx, "start", None)
        return AstLocation(line=getattr(token, "line", 0), column=getattr(token, "column", 0))

    @staticmethod
    def _call(ctx: Any, method_name: str, default: Any = None) -> Any:
        method = getattr(ctx, method_name, None)
        if callable(method):
            return method()
        return default


class AstBuilder(_AstBuilderMixin):
    """Parse-tree visitor independent AST constructor.

    This intentionally stores only stable scalar values and omits parser ctx objects.
    """

    def __init__(self):
        self._structs: list[AstStructDecl] = []
        self._shaders: list[AstKernelDecl] = []
        self._filters: list[AstKernelDecl] = []
        self._pure_functions: list[AstPureDecl] = []
        self._pipelines: list[AstPipelineDecl] = []

        self._active_pipeline_name: str | None = None
        self._active_pipeline_location: AstLocation | None = None
        self._active_streams: list[AstStreamDecl] = []
        self._active_accumulators: list[AstAccumulatorDecl] = []
        self._active_uniforms: list[AstUniformDecl] = []
        self._active_bind_routes: list[AstBindRoute] = []

    def _parse_params(self, param_list_ctx: Any) -> tuple[AstKernelParam, ...]:
        if param_list_ctx is None:
            return ()
        params_ctx = self._call(param_list_ctx, "param", []) or []
        params: list[AstKernelParam] = []
        for param in params_ctx:
            modifier = param.getChild(0).getText()
            declared_type = self._call(param, "typeName").getText()
            name = self._call(param, "ID").getText()
            params.append(AstKernelParam(modifier=modifier, declared_type=declared_type, name=name))
        return tuple(params)

    def _parse_lvalue(self, lvalue_ctx: Any) -> tuple[str, ...]:
        return tuple(token.getText() for token in self._call(lvalue_ctx, "ID", []) or [])

    def _parse_left_associative(self, ctx: Any, parts: list[Any] | tuple[Any, ...]):
        if not parts:
            raise ValueError("malformed expression tree")
        node = self._parse_expr(parts[0])
        for index, part in enumerate(parts[1:], start=1):
            op = ctx.getChild(index * 2 - 1).getText()
            node = AstExprBinary(op=op, left=node, right=self._parse_expr(part))
        return node

    def _parse_expr(self, expr_ctx: Any):
        parsed = self.visit(expr_ctx)
        if isinstance(parsed, (AstExprLiteral, AstExprVar, AstExprUnary, AstExprBinary, AstExprCall)):
            return parsed
        raise ValueError(f"unsupported expression node: {expr_ctx.__class__.__name__}")

    def visitExpr(self, ctx: Any):
        return self._parse_expr(ctx.logicalExpr())

    def visitLogicalExpr(self, ctx: Any):
        return self._parse_expr(ctx.logicalOrExpr())

    def visitLogicalOrExpr(self, ctx: Any):
        return self._parse_left_associative(ctx, ctx.logicalAndExpr())

    def visitLogicalAndExpr(self, ctx: Any):
        return self._parse_left_associative(ctx, ctx.equalityExpr())

    def visitEqualityExpr(self, ctx: Any):
        return self._parse_left_associative(ctx, ctx.relExpr())

    def visitRelExpr(self, ctx: Any):
        return self._parse_left_associative(ctx, ctx.addExpr())

    def visitAddExpr(self, ctx: Any):
        return self._parse_left_associative(ctx, ctx.mulExpr())

    def visitMulExpr(self, ctx: Any):
        return self._parse_left_associative(ctx, ctx.unaryExpr())

    def visitUnaryExpr(self, ctx: Any):
        nested = ctx.unaryExpr()
        if nested is not None:
            return AstExprUnary(op=ctx.getChild(0).getText(), operand=self._parse_expr(nested))
        return self._parse_expr(ctx.primaryExpr())

    def visitPrimaryExpr(self, ctx: Any):
        inner = ctx.expr()
        if inner is not None:
            return self._parse_expr(inner)

        expr_list = ctx.exprList()
        id_token = ctx.ID()
        if id_token is not None and ctx.getChildCount() >= 3 and ctx.getChild(1).getText() == "(":
            args = ()
            if expr_list is not None:
                args = tuple(self._parse_expr(child) for child in expr_list.expr())
            return AstExprCall(name=id_token.getText(), args=args)

        lvalue = ctx.lvalue()
        if lvalue is not None:
            return AstExprVar(path=self._parse_lvalue(lvalue))

        int_token = ctx.INT()
        if int_token is not None:
            return AstExprLiteral(kind="int", value=int_token.getText())

        float_token = ctx.FLOAT()
        if float_token is not None:
            return AstExprLiteral(kind="float", value=float_token.getText())

        bool_token = ctx.BOOL()
        if bool_token is not None:
            return AstExprLiteral(kind="bool", value=bool_token.getText())

        raise ValueError(f"unsupported primary expression node: {ctx.getText()}")

    def _parse_statement_text(self, ctx: Any) -> tuple[AstStatement, ...]:
        statements = self._call(ctx, "statement", []) or []
        parsed: list[AstStatement] = []
        for statement in statements:
            var_decl = self._call(statement, "varDecl")
            if var_decl is not None:
                declared_type = self._call(var_decl, "typeName")
                initializer_ctx = self._call(var_decl, "expr")
                parsed.append(
                    AstVarDeclStmt(
                        declared_type=declared_type.getText() if declared_type else None,
                        name=self._call(var_decl, "ID").getText(),
                        initializer=self._parse_expr(initializer_ctx) if initializer_ctx else None,
                    )
                )
                continue

            assign_stmt = self._call(statement, "assignStmt")
            if assign_stmt is not None:
                parsed.append(
                    AstAssignStmt(
                        target=self._parse_lvalue(self._call(assign_stmt, "lvalue")),
                        value=self._parse_expr(self._call(assign_stmt, "expr")),
                    )
                )
                continue

            return_stmt = self._call(statement, "returnStmt")
            if return_stmt is not None:
                parsed.append(AstReturnStmt(value=self._parse_expr(self._call(return_stmt, "expr"))))
        return tuple(parsed)

    def visitProgram(self, ctx: Any):
        return self.visitChildren(ctx)

    def visitStructDecl(self, ctx: Any):
        members = self._call(ctx, "structMember", []) or []
        fields = tuple(
            AstStructField(declared_type=self._call(member, "typeName").getText(), name=self._call(member, "ID").getText())
            for member in members
        )
        self._structs.append(
            AstStructDecl(name=self._call(ctx, "ID").getText(), fields=fields, location=self._location(ctx))
        )
        return self.visitChildren(ctx)

    def visitPureDecl(self, ctx: Any):
        params: list[AstKernelParam] = []
        pure_param_list = self._call(ctx, "pureParamList")
        if pure_param_list:
            for declared_type, name in zip(self._call(pure_param_list, "typeName", []), self._call(pure_param_list, "ID", [])):
                params.append(AstKernelParam(modifier="in", declared_type=declared_type.getText(), name=name.getText()))
        self._pure_functions.append(
            AstPureDecl(
                name=self._call(ctx, "ID").getText(),
                return_type=self._call(ctx, "typeName").getText(),
                params=tuple(params),
                body=self._parse_statement_text(ctx),
                location=self._location(ctx),
            )
        )
        return self.visitChildren(ctx)

    def visitShaderDecl(self, ctx: Any):
        self._shaders.append(
            AstKernelDecl(
                name=self._call(ctx, "ID").getText(),
                params=self._parse_params(self._call(ctx, "paramList")),
                body=self._parse_statement_text(ctx),
                location=self._location(ctx),
            )
        )
        return self.visitChildren(ctx)

    def visitFilterDecl(self, ctx: Any):
        self._filters.append(
            AstKernelDecl(
                name=self._call(ctx, "ID").getText(),
                params=self._parse_params(self._call(ctx, "paramList")),
                body=self._parse_statement_text(ctx),
                location=self._location(ctx),
            )
        )
        return self.visitChildren(ctx)

    def visitPipelineDecl(self, ctx: Any):
        self._active_pipeline_name = self._call(ctx, "ID").getText()
        self._active_pipeline_location = self._location(ctx)
        self._active_streams = []
        self._active_accumulators = []
        self._active_uniforms = []
        self._active_bind_routes = []
        self.visitChildren(ctx)
        self._pipelines.append(
            AstPipelineDecl(
                name=self._active_pipeline_name,
                streams=tuple(self._active_streams),
                accumulators=tuple(self._active_accumulators),
                uniforms=tuple(self._active_uniforms),
                bind_routes=tuple(self._active_bind_routes),
                location=self._active_pipeline_location or AstLocation(),
            )
        )
        return None

    def visitStreamDecl(self, ctx: Any):
        self._active_streams.append(
            AstStreamDecl(
                name=self._call(ctx, "ID").getText(),
                declared_type=self._call(ctx, "typeName").getText(),
                capacity=self._call(ctx, "INT").getText(),
                location=self._location(ctx),
            )
        )
        return self.visitChildren(ctx)

    def visitAccumDecl(self, ctx: Any):
        self._active_accumulators.append(
            AstAccumulatorDecl(
                name=self._call(ctx, "ID").getText(),
                declared_type=self._call(ctx, "typeName").getText(),
                location=self._location(ctx),
            )
        )
        return self.visitChildren(ctx)

    def visitUniformDecl(self, ctx: Any):
        expr_ctx = self._call(ctx, "expr")
        self._active_uniforms.append(
            AstUniformDecl(
                name=self._call(ctx, "ID").getText(),
                declared_type=self._call(ctx, "typeName").getText(),
                initializer=expr_ctx.getText() if expr_ctx else None,
                location=self._location(ctx),
            )
        )
        return self.visitChildren(ctx)

    def visitBindBlock(self, ctx: Any):
        for bind_stmt in self._call(ctx, "bindStmt", []) or []:
            id_tokens = self._call(bind_stmt, "ID", []) or []
            fold_operator = self._call(bind_stmt, "foldOperator")
            route_text = bind_stmt.getText()
            if fold_operator is not None and len(id_tokens) >= 2:
                self._active_bind_routes.append(
                    AstFoldBindRoute(
                        uniform_type=self._call(bind_stmt, "typeName").getText() if self._call(bind_stmt, "typeName") else "",
                        uniform_name=id_tokens[0].getText(),
                        operator=fold_operator.getText(),
                        source=id_tokens[1].getText(),
                        route=route_text,
                    )
                )
                continue

            if len(id_tokens) >= 2:
                self._active_bind_routes.append(
                    AstKernelBindRoute(
                        target=id_tokens[0].getText(),
                        kernel=id_tokens[1].getText(),
                        args=tuple(token.getText() for token in id_tokens[2:]),
                        route=route_text,
                    )
                )
        return self.visitChildren(ctx)

    def build(self) -> AstProgram:
        return AstProgram(
            structs=tuple(self._structs),
            shaders=tuple(self._shaders),
            filters=tuple(self._filters),
            pure_functions=tuple(self._pure_functions),
            pipelines=tuple(self._pipelines),
        )


def build_program_ast(parse_tree: Any, visitor_cls: type) -> AstProgram:
    """Build a typed AST using the parser's generated visitor base class."""

    class _AstBuilderVisitor(AstBuilder, visitor_cls):
        def __init__(self):
            AstBuilder.__init__(self)

    visitor = _AstBuilderVisitor()
    visitor.visit(parse_tree)
    return visitor.build()


def ast_to_entities(program: AstProgram) -> dict[str, Any]:
    def _expr_to_text(expr: AstExpr) -> str:
        if isinstance(expr, AstExprLiteral):
            return expr.value
        if isinstance(expr, AstExprVar):
            return ".".join(expr.path)
        if isinstance(expr, AstExprUnary):
            return f"{expr.op}{_expr_to_text(expr.operand)}"
        if isinstance(expr, AstExprBinary):
            return f"({_expr_to_text(expr.left)} {expr.op} {_expr_to_text(expr.right)})"
        return f"{expr.name}({', '.join(_expr_to_text(arg) for arg in expr.args)})"

    def _statement_to_text(statement: AstStatement) -> str:
        if isinstance(statement, AstVarDeclStmt):
            prefix = f"{statement.declared_type} " if statement.declared_type else ""
            if statement.initializer is None:
                return f"{prefix}{statement.name};"
            return f"{prefix}{statement.name} = {_expr_to_text(statement.initializer)};"
        if isinstance(statement, AstAssignStmt):
            return f"{'.'.join(statement.target)} = {_expr_to_text(statement.value)};"
        return f"return {_expr_to_text(statement.value)};"

    streams = []
    accumulators = []
    uniforms = []
    bind_routes = []
    bind_routes_ir = []
    for pipeline in program.pipelines:
        streams.extend(
            {"name": stream.name, "type": stream.declared_type, "capacity": stream.capacity}
            for stream in pipeline.streams
        )
        accumulators.extend(
            {"name": accum.name, "type": accum.declared_type}
            for accum in pipeline.accumulators
        )
        uniforms.extend(
            {
                "name": uniform.name,
                "type": uniform.declared_type,
                "initializer": uniform.initializer,
            }
            for uniform in pipeline.uniforms
        )
        for route in pipeline.bind_routes:
            bind_routes.append(route.route)
            if isinstance(route, AstKernelBindRoute):
                bind_routes_ir.append(
                    {
                        "kind": "kernel",
                        "target": route.target,
                        "kernel": route.kernel,
                        "args": list(route.args),
                        "route": route.route,
                    }
                )
            else:
                bind_routes_ir.append(
                    {
                        "kind": "fold",
                        "uniform_type": route.uniform_type,
                        "uniform_name": route.uniform_name,
                        "operator": route.operator,
                        "source": route.source,
                        "route": route.route,
                    }
                )

    return {
        "structs": [
            {
                "name": decl.name,
                "fields": [
                    {"type": field.declared_type, "name": field.name}
                    for field in decl.fields
                ],
            }
            for decl in program.structs
        ],
        "shaders": [
            {
                "name": shader.name,
                "params": [
                    {"modifier": param.modifier, "type": param.declared_type, "name": param.name}
                    for param in shader.params
                ],
                "body": [_statement_to_text(statement) for statement in shader.body],
                "body_ast": list(shader.body),
            }
            for shader in program.shaders
        ],
        "filters": [
            {
                "name": flt.name,
                "params": [
                    {"modifier": param.modifier, "type": param.declared_type, "name": param.name}
                    for param in flt.params
                ],
                "body": [_statement_to_text(statement) for statement in flt.body],
                "body_ast": list(flt.body),
            }
            for flt in program.filters
        ],
        "pure_functions": [
            {
                "name": pure.name,
                "return_type": pure.return_type,
                "params": [
                    {"type": param.declared_type, "name": param.name}
                    for param in pure.params
                ],
                "body": [_statement_to_text(statement) for statement in pure.body],
                "body_ast": list(pure.body),
            }
            for pure in program.pure_functions
        ],
        "streams": streams,
        "accumulators": accumulators,
        "uniforms": uniforms,
        "bind_routes": bind_routes,
        "bind_routes_ir": bind_routes_ir,
    }
