from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AstLocation:
    line: int = 0
    column: int = 0


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
    initializer: AstExpr | None = None
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


@dataclass(frozen=True)
class AstLValue:
    parts: tuple[str, ...]
    location: AstLocation = AstLocation()


@dataclass(frozen=True)
class AstIntExpr:
    value: int
    location: AstLocation = AstLocation()


@dataclass(frozen=True)
class AstFloatExpr:
    value: float
    location: AstLocation = AstLocation()


@dataclass(frozen=True)
class AstBoolExpr:
    value: bool
    location: AstLocation = AstLocation()


@dataclass(frozen=True)
class AstLValueExpr:
    lvalue: AstLValue
    location: AstLocation = AstLocation()


@dataclass(frozen=True)
class AstUnaryExpr:
    operator: str
    operand: AstExpr
    location: AstLocation = AstLocation()


@dataclass(frozen=True)
class AstBinaryExpr:
    operator: str
    left: AstExpr
    right: AstExpr
    location: AstLocation = AstLocation()


@dataclass(frozen=True)
class AstCallExpr:
    name: str
    args: tuple[AstExpr, ...] = ()
    location: AstLocation = AstLocation()


AstExpr = AstIntExpr | AstFloatExpr | AstBoolExpr | AstLValueExpr | AstUnaryExpr | AstBinaryExpr | AstCallExpr


@dataclass(frozen=True)
class AstVarDeclStmt:
    name: str
    declared_type: str | None = None
    initializer: AstExpr | None = None
    location: AstLocation = AstLocation()


@dataclass(frozen=True)
class AstAssignStmt:
    target: AstLValue = AstLValue(())
    value: AstExpr = AstIntExpr(0)
    location: AstLocation = AstLocation()


@dataclass(frozen=True)
class AstReturnStmt:
    value: AstExpr = AstIntExpr(0)
    location: AstLocation = AstLocation()


AstStatement = AstVarDeclStmt | AstAssignStmt | AstReturnStmt


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

    def _parse_lvalue(self, ctx: Any) -> AstLValue:
        tokens = self._call(ctx, "ID", []) or []
        return AstLValue(parts=tuple(token.getText() for token in tokens), location=self._location(ctx))

    def _parse_left_associative(self, ctx: Any, child_attr: str, child_parser) -> AstExpr:
        operands_ctx = self._call(ctx, child_attr, []) or []
        node = child_parser(operands_ctx[0])
        children = getattr(ctx, "children", []) or []
        operators = [children[index].getText() for index in range(1, len(children), 2)]
        for operator, operand_ctx in zip(operators, operands_ctx[1:]):
            node = AstBinaryExpr(operator=operator, left=node, right=child_parser(operand_ctx), location=self._location(ctx))
        return node

    def _parse_expr(self, ctx: Any) -> AstExpr:
        return self._parse_logical_or_expr(self._call(self._call(ctx, "logicalExpr"), "logicalOrExpr"))

    def _parse_logical_or_expr(self, ctx: Any) -> AstExpr:
        return self._parse_left_associative(ctx, "logicalAndExpr", self._parse_logical_and_expr)

    def _parse_logical_and_expr(self, ctx: Any) -> AstExpr:
        return self._parse_left_associative(ctx, "equalityExpr", self._parse_equality_expr)

    def _parse_equality_expr(self, ctx: Any) -> AstExpr:
        return self._parse_left_associative(ctx, "relExpr", self._parse_rel_expr)

    def _parse_rel_expr(self, ctx: Any) -> AstExpr:
        return self._parse_left_associative(ctx, "addExpr", self._parse_add_expr)

    def _parse_add_expr(self, ctx: Any) -> AstExpr:
        return self._parse_left_associative(ctx, "mulExpr", self._parse_mul_expr)

    def _parse_mul_expr(self, ctx: Any) -> AstExpr:
        return self._parse_left_associative(ctx, "unaryExpr", self._parse_unary_expr)

    def _parse_unary_expr(self, ctx: Any) -> AstExpr:
        unary_ctx = self._call(ctx, "unaryExpr")
        if unary_ctx is not None:
            return AstUnaryExpr(operator=ctx.getChild(0).getText(), operand=self._parse_unary_expr(unary_ctx), location=self._location(ctx))
        return self._parse_primary_expr(self._call(ctx, "primaryExpr"))

    def _parse_primary_expr(self, ctx: Any) -> AstExpr:
        int_token = self._call(ctx, "INT")
        if int_token is not None:
            return AstIntExpr(value=int(int_token.getText()), location=self._location(ctx))
        float_token = self._call(ctx, "FLOAT")
        if float_token is not None:
            return AstFloatExpr(value=float(float_token.getText()), location=self._location(ctx))
        bool_token = self._call(ctx, "BOOL")
        if bool_token is not None:
            return AstBoolExpr(value=bool_token.getText() == "true", location=self._location(ctx))

        lvalue_ctx = self._call(ctx, "lvalue")
        if lvalue_ctx is not None:
            return AstLValueExpr(lvalue=self._parse_lvalue(lvalue_ctx), location=self._location(ctx))

        call_id = self._call(ctx, "ID")
        if call_id is not None:
            expr_list = self._call(ctx, "exprList")
            args = ()
            if expr_list is not None:
                args = tuple(self._parse_expr(expr_ctx) for expr_ctx in self._call(expr_list, "expr", []) or [])
            return AstCallExpr(name=call_id.getText(), args=args, location=self._location(ctx))

        grouped = self._call(ctx, "expr")
        if grouped is not None:
            return self._parse_expr(grouped)

        return AstIntExpr(value=0, location=self._location(ctx))

    def _parse_statement(self, ctx: Any) -> AstStatement:
        var_decl = self._call(ctx, "varDecl")
        if var_decl is not None:
            declared_type_ctx = self._call(var_decl, "typeName")
            initializer_ctx = self._call(var_decl, "expr")
            return AstVarDeclStmt(
                name=self._call(var_decl, "ID").getText(),
                declared_type=declared_type_ctx.getText() if declared_type_ctx is not None else None,
                initializer=self._parse_expr(initializer_ctx) if initializer_ctx is not None else None,
                location=self._location(var_decl),
            )

        assign_stmt = self._call(ctx, "assignStmt")
        if assign_stmt is not None:
            return AstAssignStmt(
                target=self._parse_lvalue(self._call(assign_stmt, "lvalue")),
                value=self._parse_expr(self._call(assign_stmt, "expr")),
                location=self._location(assign_stmt),
            )

        return_stmt = self._call(ctx, "returnStmt")
        if return_stmt is not None:
            return AstReturnStmt(
                value=self._parse_expr(self._call(return_stmt, "expr")),
                location=self._location(return_stmt),
            )

        return AstVarDeclStmt(name="", location=self._location(ctx))

    def _parse_statements(self, ctx: Any) -> tuple[AstStatement, ...]:
        statements = self._call(ctx, "statement", []) or []
        return tuple(self._parse_statement(statement) for statement in statements)

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
                body=self._parse_statements(ctx),
                location=self._location(ctx),
            )
        )
        return self.visitChildren(ctx)

    def visitShaderDecl(self, ctx: Any):
        self._shaders.append(
            AstKernelDecl(
                name=self._call(ctx, "ID").getText(),
                params=self._parse_params(self._call(ctx, "paramList")),
                body=self._parse_statements(ctx),
                location=self._location(ctx),
            )
        )
        return self.visitChildren(ctx)

    def visitFilterDecl(self, ctx: Any):
        self._filters.append(
            AstKernelDecl(
                name=self._call(ctx, "ID").getText(),
                params=self._parse_params(self._call(ctx, "paramList")),
                body=self._parse_statements(ctx),
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
                initializer=self._parse_expr(expr_ctx) if expr_ctx else None,
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
        if isinstance(expr, AstIntExpr):
            return str(expr.value)
        if isinstance(expr, AstFloatExpr):
            return str(expr.value)
        if isinstance(expr, AstBoolExpr):
            return "true" if expr.value else "false"
        if isinstance(expr, AstLValueExpr):
            return ".".join(expr.lvalue.parts)
        if isinstance(expr, AstUnaryExpr):
            return f"{expr.operator}{_expr_to_text(expr.operand)}"
        if isinstance(expr, AstBinaryExpr):
            return f"({_expr_to_text(expr.left)} {expr.operator} {_expr_to_text(expr.right)})"
        if isinstance(expr, AstCallExpr):
            return f"{expr.name}({', '.join(_expr_to_text(arg) for arg in expr.args)})"
        return "0"

    def _statement_to_text(statement: AstStatement) -> str:
        if isinstance(statement, AstVarDeclStmt):
            prefix = f"{statement.declared_type} " if statement.declared_type else ""
            if statement.initializer is None:
                return f"{prefix}{statement.name};"
            return f"{prefix}{statement.name} = {_expr_to_text(statement.initializer)};"
        if isinstance(statement, AstAssignStmt):
            return f"{'.'.join(statement.target.parts)} = {_expr_to_text(statement.value)};"
        if isinstance(statement, AstReturnStmt):
            return f"return {_expr_to_text(statement.value)};"
        return ";"

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
                "initializer": _expr_to_text(uniform.initializer) if uniform.initializer is not None else None,
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
            }
            for pure in program.pure_functions
        ],
        "streams": streams,
        "accumulators": accumulators,
        "uniforms": uniforms,
        "bind_routes": bind_routes,
        "bind_routes_ir": bind_routes_ir,
    }
