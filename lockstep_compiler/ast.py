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
    body: tuple[str, ...] = ()
    location: AstLocation = AstLocation()


@dataclass(frozen=True)
class AstKernelDecl:
    name: str
    params: tuple[AstKernelParam, ...] = ()
    body: tuple[str, ...] = ()
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

    def _parse_statement_text(self, ctx: Any) -> tuple[str, ...]:
        statements = self._call(ctx, "statement", []) or []
        return tuple(statement.getText() for statement in statements)

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

    class _AstBuilderVisitor(visitor_cls, AstBuilder):
        def __init__(self):
            AstBuilder.__init__(self)

    visitor = _AstBuilderVisitor()
    visitor.visit(parse_tree)
    return visitor.build()


def ast_to_entities(program: AstProgram) -> dict[str, Any]:
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
        "structs": [decl.name for decl in program.structs],
        "shaders": [
            {
                "name": shader.name,
                "params": [
                    {"modifier": param.modifier, "type": param.declared_type, "name": param.name}
                    for param in shader.params
                ],
                "body": list(shader.body),
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
                "body": list(flt.body),
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
                "body": list(pure.body),
            }
            for pure in program.pure_functions
        ],
        "streams": streams,
        "accumulators": accumulators,
        "uniforms": uniforms,
        "bind_routes": bind_routes,
        "bind_routes_ir": bind_routes_ir,
    }
