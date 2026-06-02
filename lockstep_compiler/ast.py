from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AstTypeSuffix:
    kind: str
    size: str | None = None
    type_arg: "AstType | None" = None

    def __str__(self) -> str:
        if self.kind == "array":
            return f"[{self.size}]"
        if self.kind == "template":
            size_segment = f",{self.size}" if self.size is not None else ""
            return f"<{self.type_arg}{size_segment}>"
        return ""


@dataclass(frozen=True)
class AstType:
    name: str
    kind: str = "named"
    suffixes: tuple[AstTypeSuffix, ...] = ()

    def __str__(self) -> str:
        return f"{self.name}{''.join(str(suffix) for suffix in self.suffixes)}"


def _parse_type_text(type_text: str) -> AstType:
    text = type_text.strip()
    index = 0
    while index < len(text) and (text[index].isalnum() or text[index] == "_"):
        index += 1
    base_name = text[:index] or text
    suffixes: list[AstTypeSuffix] = []

    def _find_matching(start: int, opener: str, closer: str) -> int:
        depth = 0
        for cursor in range(start, len(text)):
            char = text[cursor]
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return cursor
        return len(text) - 1

    while index < len(text):
        char = text[index]
        if char == "[":
            end = text.find("]", index + 1)
            if end < 0:
                break
            suffixes.append(AstTypeSuffix(kind="array", size=text[index + 1 : end]))
            index = end + 1
            continue
        if char == "<":
            end = _find_matching(index, "<", ">")
            inner = text[index + 1 : end]
            comma = _split_template_suffix(inner)
            if comma is None:
                suffixes.append(
                    AstTypeSuffix(kind="template", type_arg=_parse_type_text(inner))
                )
            else:
                suffixes.append(
                    AstTypeSuffix(
                        kind="template",
                        type_arg=_parse_type_text(inner[:comma]),
                        size=inner[comma + 1 :].strip(),
                    )
                )
            index = end + 1
            continue
        index += 1

    kind = (
        "primitive"
        if not suffixes
        and base_name in {"int", "float", "bool", "uint", "double", "string"}
        else "named"
    )
    return AstType(name=base_name, kind=kind, suffixes=tuple(suffixes))


def _split_template_suffix(inner: str) -> int | None:
    angle_depth = 0
    bracket_depth = 0
    for index, char in enumerate(inner):
        if char == "<":
            angle_depth += 1
        elif char == ">":
            angle_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif char == "," and angle_depth == 0 and bracket_depth == 0:
            return index
    return None


def _normalize_type(value: AstType | str) -> AstType:
    if isinstance(value, AstType):
        return value
    return _parse_type_text(value)


def _type_name(value: AstType | str | None) -> str | None:
    if value is None:
        return None
    return str(value) if isinstance(value, AstType) else value


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


@dataclass(frozen=True)
class AstExprCast:
    target_type: AstType
    value: "AstExpr"

    def __post_init__(self):
        object.__setattr__(self, "target_type", _normalize_type(self.target_type))


AstExpr = (
    AstExprLiteral
    | AstExprVar
    | AstExprUnary
    | AstExprBinary
    | AstExprCall
    | AstExprCast
)


@dataclass(frozen=True)
class AstVarDeclStmt:
    declared_type: AstType | None
    name: str
    initializer: AstExpr | None

    def __post_init__(self):
        if self.declared_type is not None:
            object.__setattr__(
                self, "declared_type", _normalize_type(self.declared_type)
            )


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
    declared_type: AstType
    name: str

    def __post_init__(self):
        object.__setattr__(self, "declared_type", _normalize_type(self.declared_type))


@dataclass(frozen=True)
class AstStructField:
    declared_type: AstType
    name: str
    location: AstLocation = AstLocation()

    def __post_init__(self):
        object.__setattr__(self, "declared_type", _normalize_type(self.declared_type))


@dataclass(frozen=True)
class AstStructDecl:
    name: str
    fields: tuple[AstStructField, ...] = ()
    location: AstLocation = AstLocation()
    identifier_location: AstLocation = AstLocation()


@dataclass(frozen=True)
class AstPureDecl:
    name: str
    return_type: AstType
    params: tuple[AstKernelParam, ...] = ()
    body: tuple[AstStatement, ...] = ()
    intrinsic: bool = False
    location: AstLocation = AstLocation()
    identifier_location: AstLocation = AstLocation()

    def __post_init__(self):
        object.__setattr__(self, "return_type", _normalize_type(self.return_type))


@dataclass(frozen=True)
class AstKernelDecl:
    name: str
    params: tuple[AstKernelParam, ...] = ()
    body: tuple[AstStatement, ...] = ()
    location: AstLocation = AstLocation()
    identifier_location: AstLocation = AstLocation()


@dataclass(frozen=True)
class AstStreamDecl:
    name: str
    declared_type: AstType
    capacity: str
    location: AstLocation = AstLocation()

    def __post_init__(self):
        object.__setattr__(self, "declared_type", _normalize_type(self.declared_type))


@dataclass(frozen=True)
class AstAccumulatorDecl:
    name: str
    declared_type: AstType
    location: AstLocation = AstLocation()

    def __post_init__(self):
        object.__setattr__(self, "declared_type", _normalize_type(self.declared_type))


@dataclass(frozen=True)
class AstUniformDecl:
    name: str
    declared_type: AstType
    initializer: str | None = None
    location: AstLocation = AstLocation()

    def __post_init__(self):
        object.__setattr__(self, "declared_type", _normalize_type(self.declared_type))


@dataclass(frozen=True)
class AstKernelBindRoute:
    target: str
    kernel: str
    args: tuple[str, ...]
    route: str
    location: AstLocation = AstLocation()


@dataclass(frozen=True)
class AstFoldBindRoute:
    uniform_type: AstType
    uniform_name: str
    operator: str
    source: str
    route: str
    location: AstLocation = AstLocation()

    def __post_init__(self):
        object.__setattr__(self, "uniform_type", _normalize_type(self.uniform_type))


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
        return AstLocation(
            line=getattr(token, "line", 0), column=getattr(token, "column", 0)
        )

    @staticmethod
    def _call(ctx: Any, method_name: str, default: Any = None) -> Any:
        method = getattr(ctx, method_name, None)
        if callable(method):
            return method()
        return default

    @staticmethod
    def _token_location(token: Any) -> AstLocation:
        symbol = getattr(token, "symbol", token)
        return AstLocation(
            line=getattr(symbol, "line", 0), column=getattr(symbol, "column", 0)
        )


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
        self._types: dict[str, AstType] = {}

    def _resolve_type(self, type_name: AstType | str) -> AstType:
        ast_type = _normalize_type(type_name)
        cache_key = str(ast_type)
        if cache_key not in self._types:
            self._types[cache_key] = ast_type
        return self._types[cache_key]

    def _resolve_type_ctx(self, type_ctx: Any) -> AstType:
        if type_ctx is None:
            return self._resolve_type("float")

        id_token = self._call(type_ctx, "ID")
        suffix_ctxs = self._call(type_ctx, "typeSuffix", []) or []
        if id_token is None or not isinstance(suffix_ctxs, list):
            return self._resolve_type(type_ctx.getText())

        suffixes: list[AstTypeSuffix] = []
        for suffix_ctx in suffix_ctxs:
            text = suffix_ctx.getText()
            if text.startswith("["):
                int_token = self._call(suffix_ctx, "INT")
                suffixes.append(
                    AstTypeSuffix(
                        kind="array",
                        size=(
                            int_token.getText() if int_token is not None else text[1:-1]
                        ),
                    )
                )
                continue

            nested_types = self._call(suffix_ctx, "typeName", []) or []
            if not isinstance(nested_types, list):
                nested_types = [nested_types]
            nested_type = (
                self._resolve_type_ctx(nested_types[0])
                if nested_types
                else self._resolve_type("void")
            )
            generic_width = self._call(suffix_ctx, "genericWidth")
            int_token = (
                self._call(generic_width, "INT")
                if generic_width is not None
                else self._call(suffix_ctx, "INT")
            )
            if isinstance(int_token, list):
                int_token = int_token[0] if int_token else None
            suffixes.append(
                AstTypeSuffix(
                    kind="template",
                    type_arg=nested_type,
                    size=int_token.getText() if int_token is not None else None,
                )
            )

        return self._resolve_type(
            AstType(
                name=id_token.getText(),
                kind=(
                    "primitive"
                    if not suffixes
                    and id_token.getText()
                    in {"int", "float", "bool", "uint", "double", "string"}
                    else "named"
                ),
                suffixes=tuple(suffixes),
            )
        )

    def _parse_params(self, param_list_ctx: Any) -> tuple[AstKernelParam, ...]:
        if param_list_ctx is None:
            return ()
        params_ctx = self._call(param_list_ctx, "param", []) or []
        params: list[AstKernelParam] = []
        for param in params_ctx:
            modifier = param.getChild(0).getText()
            declared_type = self._resolve_type_ctx(self._call(param, "typeName"))
            name = self._call(param, "ID").getText()
            params.append(
                AstKernelParam(
                    modifier=modifier, declared_type=declared_type, name=name
                )
            )
        return tuple(params)

    def _parse_lvalue(self, lvalue_ctx: Any) -> tuple[str, ...]:
        return tuple(
            token.getText() for token in self._call(lvalue_ctx, "ID", []) or []
        )

    def _parse_left_associative(self, ctx: Any, parts: list[Any]):
        if not parts:
            raise ValueError("malformed expression tree")
        node = self.visit(parts[0])
        for index, part in enumerate(parts[1:], start=1):
            op = ctx.getChild(index * 2 - 1).getText()
            node = AstExprBinary(op=op, left=node, right=self.visit(part))
        return node

    def _coerce_expr_literal_for_type(
        self, declared_type: AstType | None, expr: AstExpr | None
    ) -> AstExpr | None:
        if declared_type is None or not isinstance(expr, AstExprLiteral):
            return expr
        if declared_type.suffixes:
            return expr
        if declared_type.name in {"uint", "double"} and (
            expr.kind == "int"
            or (expr.kind == "float" and declared_type.name == "double")
        ):
            return AstExprLiteral(kind=declared_type.name, value=expr.value)
        return expr

    def _parse_expr(self, expr_ctx: Any):
        return self.visit(expr_ctx)

    def visitExpr(self, ctx: Any):
        return self.visit(self._call(ctx, "logicalExpr"))

    def visitLogicalExpr(self, ctx: Any):
        return self.visit(self._call(ctx, "logicalOrExpr"))

    def visitLogicalOrExpr(self, ctx: Any):
        return self._parse_left_associative(
            ctx, self._call(ctx, "logicalAndExpr", []) or []
        )

    def visitLogicalAndExpr(self, ctx: Any):
        return self._parse_left_associative(
            ctx, self._call(ctx, "bitwiseOrExpr", []) or []
        )

    def visitBitwiseOrExpr(self, ctx: Any):
        return self._parse_left_associative(
            ctx, self._call(ctx, "bitwiseXorExpr", []) or []
        )

    def visitBitwiseXorExpr(self, ctx: Any):
        return self._parse_left_associative(
            ctx, self._call(ctx, "bitwiseAndExpr", []) or []
        )

    def visitBitwiseAndExpr(self, ctx: Any):
        return self._parse_left_associative(
            ctx, self._call(ctx, "equalityExpr", []) or []
        )

    def visitEqualityExpr(self, ctx: Any):
        return self._parse_left_associative(ctx, self._call(ctx, "relExpr", []) or [])

    def visitRelExpr(self, ctx: Any):
        return self._parse_left_associative(ctx, self._call(ctx, "shiftExpr", []) or [])

    def visitShiftExpr(self, ctx: Any):
        return self._parse_left_associative(ctx, self._call(ctx, "addExpr", []) or [])

    def visitAddExpr(self, ctx: Any):
        return self._parse_left_associative(ctx, self._call(ctx, "mulExpr", []) or [])

    def visitMulExpr(self, ctx: Any):
        return self._parse_left_associative(ctx, self._call(ctx, "unaryExpr", []) or [])

    def visitUnaryExpr(self, ctx: Any):
        nested = self._call(ctx, "unaryExpr")
        if nested is not None:
            if (
                ctx.getChildCount() >= 4
                and ctx.getChild(0).getText() == "("
                and ctx.getChild(2).getText() == ")"
            ):
                return AstExprCast(
                    target_type=self._resolve_type_ctx(self._call(ctx, "typeName")),
                    value=self.visit(nested),
                )
            return AstExprUnary(
                op=ctx.getChild(0).getText(), operand=self.visit(nested)
            )
        return self.visit(self._call(ctx, "primaryExpr"))

    def visitPrimaryExpr(self, ctx: Any):
        inner = self._call(ctx, "expr")
        if inner is not None:
            return self.visit(inner)

        expr_list = self._call(ctx, "exprList")
        id_token = self._call(ctx, "ID")
        if (
            id_token is not None
            and ctx.getChildCount() >= 3
            and ctx.getChild(1).getText() == "("
        ):
            args = ()
            if expr_list is not None:
                args = tuple(
                    self.visit(child)
                    for child in self._call(expr_list, "expr", []) or []
                )
            callee_name = id_token.getText()
            if callee_name in {"int", "float", "double", "uint", "bool"}:
                if len(args) != 1:
                    raise ValueError(
                        f"cast '{callee_name}(...)' expects exactly one argument"
                    )
                return AstExprCast(
                    target_type=self._resolve_type(callee_name), value=args[0]
                )
            return AstExprCall(name=callee_name, args=args)

        lvalue = self._call(ctx, "lvalue")
        if lvalue is not None:
            return AstExprVar(path=self._parse_lvalue(lvalue))

        int_token = self._call(ctx, "INT")
        if int_token is not None:
            return AstExprLiteral(kind="int", value=int_token.getText())

        float_token = self._call(ctx, "FLOAT")
        if float_token is not None:
            return AstExprLiteral(kind="float", value=float_token.getText())

        bool_token = self._call(ctx, "BOOL")
        if bool_token is not None:
            return AstExprLiteral(kind="bool", value=bool_token.getText())

        string_token = self._call(ctx, "STRING")
        if string_token is not None:
            return AstExprLiteral(kind="string", value=string_token.getText())

        raise ValueError(f"unsupported expression node: {ctx.__class__.__name__}")

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
                        declared_type=(
                            self._resolve_type_ctx(declared_type)
                            if declared_type
                            else None
                        ),
                        name=self._call(var_decl, "ID").getText(),
                        initializer=self._coerce_expr_literal_for_type(
                            self._resolve_type_ctx(declared_type)
                            if declared_type
                            else None,
                            self._parse_expr(initializer_ctx)
                            if initializer_ctx
                            else None,
                        ),
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
                parsed.append(
                    AstReturnStmt(
                        value=self._parse_expr(self._call(return_stmt, "expr"))
                    )
                )
        return tuple(parsed)

    def visitProgram(self, ctx: Any):
        return self.visitChildren(ctx)

    def visitStructDecl(self, ctx: Any):
        members = self._call(ctx, "structMember", []) or []
        fields = tuple(
            AstStructField(
                declared_type=self._resolve_type_ctx(self._call(member, "typeName")),
                name=self._call(member, "ID").getText(),
                location=AstLocation(
                    line=getattr(
                        getattr(self._call(member, "ID"), "symbol", None), "line", 0
                    ),
                    column=getattr(
                        getattr(self._call(member, "ID"), "symbol", None), "column", 0
                    ),
                ),
            )
            for member in members
        )
        id_token = self._call(ctx, "ID")
        self._structs.append(
            AstStructDecl(
                name=id_token.getText(),
                fields=fields,
                location=self._location(ctx),
                identifier_location=self._token_location(id_token),
            )
        )
        return self.visitChildren(ctx)

    def visitPureDecl(self, ctx: Any):
        id_token = self._call(ctx, "ID")
        params: list[AstKernelParam] = []
        pure_param_list = self._call(ctx, "pureParamList")
        if pure_param_list:
            for declared_type, name in zip(
                self._call(pure_param_list, "typeName", []),
                self._call(pure_param_list, "ID", []),
            ):
                params.append(
                    AstKernelParam(
                        modifier="in",
                        declared_type=self._resolve_type_ctx(declared_type),
                        name=name.getText(),
                    )
                )
        self._pure_functions.append(
            AstPureDecl(
                name=id_token.getText(),
                return_type=self._resolve_type_ctx(self._call(ctx, "typeName")),
                params=tuple(params),
                body=self._parse_statement_text(ctx),
                location=self._location(ctx),
                identifier_location=self._token_location(id_token),
            )
        )
        return self.visitChildren(ctx)

    def visitShaderDecl(self, ctx: Any):
        id_token = self._call(ctx, "ID")
        self._shaders.append(
            AstKernelDecl(
                name=id_token.getText(),
                params=self._parse_params(self._call(ctx, "paramList")),
                body=self._parse_statement_text(ctx),
                location=self._location(ctx),
                identifier_location=self._token_location(id_token),
            )
        )
        return self.visitChildren(ctx)

    def visitFilterDecl(self, ctx: Any):
        id_token = self._call(ctx, "ID")
        self._filters.append(
            AstKernelDecl(
                name=id_token.getText(),
                params=self._parse_params(self._call(ctx, "paramList")),
                body=self._parse_statement_text(ctx),
                location=self._location(ctx),
                identifier_location=self._token_location(id_token),
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
                declared_type=self._resolve_type_ctx(self._call(ctx, "typeName")),
                capacity=self._call(ctx, "INT").getText(),
                location=self._location(ctx),
            )
        )
        return self.visitChildren(ctx)

    def visitAccumDecl(self, ctx: Any):
        self._active_accumulators.append(
            AstAccumulatorDecl(
                name=self._call(ctx, "ID").getText(),
                declared_type=self._resolve_type_ctx(self._call(ctx, "typeName")),
                location=self._location(ctx),
            )
        )
        return self.visitChildren(ctx)

    def visitUniformDecl(self, ctx: Any):
        expr_ctx = self._call(ctx, "expr")
        self._active_uniforms.append(
            AstUniformDecl(
                name=self._call(ctx, "ID").getText(),
                declared_type=self._resolve_type_ctx(self._call(ctx, "typeName")),
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
                        uniform_type=(
                            self._resolve_type_ctx(self._call(bind_stmt, "typeName"))
                            if self._call(bind_stmt, "typeName")
                            else self._resolve_type("float")
                        ),
                        uniform_name=id_tokens[0].getText(),
                        operator=fold_operator.getText(),
                        source=id_tokens[1].getText(),
                        route=route_text,
                        location=self._token_location(id_tokens[0]),
                    )
                )
                continue

            if len(id_tokens) >= 2:
                arg_tokens = (
                    self._call(self._call(bind_stmt, "argList"), "ID", []) or []
                )
                if not isinstance(arg_tokens, list):
                    arg_tokens = [arg_tokens]
                self._active_bind_routes.append(
                    AstKernelBindRoute(
                        target=id_tokens[0].getText(),
                        kernel=id_tokens[1].getText(),
                        args=tuple(token.getText() for token in arg_tokens),
                        route=route_text,
                        location=self._token_location(id_tokens[0]),
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


def ast_to_entities(program: AstProgram | dict[str, Any]) -> dict[str, Any]:
    if isinstance(program, dict):
        return program

    def _expr_to_text(expr: AstExpr) -> str:
        if isinstance(expr, AstExprLiteral):
            return expr.value
        if isinstance(expr, AstExprVar):
            return ".".join(expr.path)
        if isinstance(expr, AstExprUnary):
            return f"{expr.op}{_expr_to_text(expr.operand)}"
        if isinstance(expr, AstExprBinary):
            return f"({_expr_to_text(expr.left)} {expr.op} {_expr_to_text(expr.right)})"
        if isinstance(expr, AstExprCast):
            return f"({_type_name(expr.target_type)}){_expr_to_text(expr.value)}"
        return f"{expr.name}({', '.join(_expr_to_text(arg) for arg in expr.args)})"

    def _statement_to_text(statement: AstStatement) -> str:
        if isinstance(statement, AstVarDeclStmt):
            prefix = (
                f"{_type_name(statement.declared_type)} "
                if statement.declared_type
                else ""
            )
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
    bind_routes_meta = []
    bind_routes_ir = []
    for pipeline in program.pipelines:
        streams.extend(
            {
                "name": stream.name,
                "type": _type_name(stream.declared_type),
                "capacity": stream.capacity,
            }
            for stream in pipeline.streams
        )
        accumulators.extend(
            {"name": accum.name, "type": _type_name(accum.declared_type)}
            for accum in pipeline.accumulators
        )
        uniforms.extend(
            {
                "name": uniform.name,
                "type": _type_name(uniform.declared_type),
                "initializer": uniform.initializer,
            }
            for uniform in pipeline.uniforms
        )
        for route in pipeline.bind_routes:
            bind_routes.append(route.route)
            bind_routes_meta.append(
                {
                    "route": route.route,
                    "line": route.location.line,
                    "column": route.location.column,
                }
            )
            if isinstance(route, AstKernelBindRoute):
                bind_routes_ir.append(
                    {
                        "kind": "kernel",
                        "target": route.target,
                        "kernel": route.kernel,
                        "args": list(route.args),
                        "route": route.route,
                        "line": route.location.line,
                        "column": route.location.column,
                    }
                )
            else:
                bind_routes_ir.append(
                    {
                        "kind": "fold",
                        "uniform_type": _type_name(route.uniform_type),
                        "uniform_name": route.uniform_name,
                        "operator": route.operator,
                        "source": route.source,
                        "route": route.route,
                        "line": route.location.line,
                        "column": route.location.column,
                    }
                )

    return {
        "structs": [
            {
                "name": decl.name,
                "line": decl.identifier_location.line,
                "column": decl.identifier_location.column,
                "fields": [
                    {
                        "type": _type_name(field.declared_type),
                        "name": field.name,
                        "line": field.location.line,
                        "column": field.location.column,
                    }
                    for field in decl.fields
                ],
            }
            for decl in program.structs
        ],
        "shaders": [
            {
                "name": shader.name,
                "params": [
                    {
                        "modifier": param.modifier,
                        "type": _type_name(param.declared_type),
                        "name": param.name,
                    }
                    for param in shader.params
                ],
                "body": [_statement_to_text(statement) for statement in shader.body],
                "body_ast": list(shader.body),
                "line": shader.identifier_location.line,
                "column": shader.identifier_location.column,
            }
            for shader in program.shaders
        ],
        "filters": [
            {
                "name": flt.name,
                "params": [
                    {
                        "modifier": param.modifier,
                        "type": _type_name(param.declared_type),
                        "name": param.name,
                    }
                    for param in flt.params
                ],
                "body": [_statement_to_text(statement) for statement in flt.body],
                "body_ast": list(flt.body),
                "line": flt.identifier_location.line,
                "column": flt.identifier_location.column,
            }
            for flt in program.filters
        ],
        "pure_functions": [
            {
                "name": pure.name,
                "return_type": _type_name(pure.return_type),
                "params": [
                    {"type": _type_name(param.declared_type), "name": param.name}
                    for param in pure.params
                ],
                "body": [_statement_to_text(statement) for statement in pure.body],
                "body_ast": list(pure.body),
                "intrinsic": pure.intrinsic,
                "line": pure.identifier_location.line,
                "column": pure.identifier_location.column,
            }
            for pure in program.pure_functions
        ],
        "streams": streams,
        "accumulators": accumulators,
        "uniforms": uniforms,
        "bind_routes": bind_routes,
        "bind_routes_meta": bind_routes_meta,
        "bind_routes_ir": bind_routes_ir,
    }
