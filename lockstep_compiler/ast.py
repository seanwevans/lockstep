from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceSpan:
    line: int
    column: int


@dataclass(frozen=True)
class AstNode:
    span: SourceSpan
    raw_ctx: Any = None


@dataclass(frozen=True)
class TypeRef(AstNode):
    name: str = ""


@dataclass(frozen=True)
class Program(AstNode):
    declarations: list[Declaration] = field(default_factory=list)


class Declaration(AstNode):
    pass


@dataclass(frozen=True)
class StructField(AstNode):
    type_name: str = ""
    name: str = ""


@dataclass(frozen=True)
class StructDecl(Declaration):
    name: str = ""
    fields: list[StructField] = field(default_factory=list)


@dataclass(frozen=True)
class Parameter(AstNode):
    modifier: str = "value"
    type_name: str = ""
    name: str = ""


@dataclass(frozen=True)
class ShaderDecl(Declaration):
    name: str = ""
    params: list[Parameter] = field(default_factory=list)


@dataclass(frozen=True)
class FilterDecl(Declaration):
    name: str = ""
    params: list[Parameter] = field(default_factory=list)


class Statement(AstNode):
    pass


class Expr(AstNode):
    pass


@dataclass(frozen=True)
class Identifier(Expr):
    name: str = ""


@dataclass(frozen=True)
class IntLiteral(Expr):
    value: int = 0


@dataclass(frozen=True)
class FloatLiteral(Expr):
    value: float = 0.0


@dataclass(frozen=True)
class BoolLiteral(Expr):
    value: bool = False


@dataclass(frozen=True)
class UnaryOperation(Expr):
    operator: str = ""
    operand: Expr | None = None


@dataclass(frozen=True)
class BinaryOperation(Expr):
    operator: str = ""
    left: Expr | None = None
    right: Expr | None = None


@dataclass(frozen=True)
class StructAccess(Expr):
    target: Expr | None = None
    field: str = ""


@dataclass(frozen=True)
class FunctionCall(Expr):
    name: str = ""
    args: list[Expr] = field(default_factory=list)


@dataclass(frozen=True)
class VarDecl(Statement):
    type_name: str | None = None
    name: str = ""
    initializer: Expr | None = None


@dataclass(frozen=True)
class Assignment(Statement):
    target: Expr | None = None
    value: Expr | None = None


@dataclass(frozen=True)
class ReturnStmt(Statement):
    value: Expr | None = None


@dataclass(frozen=True)
class PureDecl(Declaration):
    return_type: str = ""
    name: str = ""
    params: list[Parameter] = field(default_factory=list)
    statements: list[Statement] = field(default_factory=list)


@dataclass(frozen=True)
class StreamDecl(AstNode):
    type_name: str = ""
    capacity: str = ""
    name: str = ""


@dataclass(frozen=True)
class AccumDecl(AstNode):
    type_name: str = ""
    name: str = ""


@dataclass(frozen=True)
class UniformDecl(AstNode):
    type_name: str = ""
    name: str = ""
    initializer: Expr | None = None


class BindStmt(AstNode):
    pass


@dataclass(frozen=True)
class BindCall(BindStmt):
    target: str = ""
    callee: str = ""
    args: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BindFold(BindStmt):
    type_name: str = ""
    target: str = ""
    operator: str = ""
    source: str = ""


@dataclass(frozen=True)
class PipelineDecl(Declaration):
    name: str = ""
    streams: list[StreamDecl] = field(default_factory=list)
    accumulators: list[AccumDecl] = field(default_factory=list)
    uniforms: list[UniformDecl] = field(default_factory=list)
    binds: list[BindStmt] = field(default_factory=list)
