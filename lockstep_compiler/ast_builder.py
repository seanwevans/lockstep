from __future__ import annotations

from .ast import (
    AccumDecl,
    Assignment,
    BinaryOperation,
    BindCall,
    BindFold,
    BoolLiteral,
    FilterDecl,
    FloatLiteral,
    FunctionCall,
    Identifier,
    IntLiteral,
    Parameter,
    PipelineDecl,
    Program,
    PureDecl,
    ReturnStmt,
    ShaderDecl,
    SourceSpan,
    StreamDecl,
    StructAccess,
    StructDecl,
    StructField,
    UnaryOperation,
    UniformDecl,
    VarDecl,
)


def _span(ctx) -> SourceSpan:
    token = getattr(ctx, "start", None)
    return SourceSpan(getattr(token, "line", 0), getattr(token, "column", 0))


def build_ast(parse_tree) -> Program:
    declarations = []
    for decl in parse_tree.declaration() or []:
        if decl.structDecl() is not None:
            ctx = decl.structDecl()
            declarations.append(
                StructDecl(
                    span=_span(ctx),
                    raw_ctx=ctx,
                    name=ctx.ID().getText(),
                    fields=[
                        StructField(
                            span=_span(member),
                            raw_ctx=member,
                            type_name=member.typeName().getText(),
                            name=member.ID().getText(),
                        )
                        for member in (ctx.structMember() or [])
                    ],
                )
            )
        elif decl.shaderDecl() is not None:
            declarations.append(_kernel_decl(decl.shaderDecl(), ShaderDecl))
        elif decl.filterDecl() is not None:
            declarations.append(_kernel_decl(decl.filterDecl(), FilterDecl))
        elif decl.pureDecl() is not None:
            declarations.append(_pure_decl(decl.pureDecl()))
        elif decl.pipelineDecl() is not None:
            declarations.append(_pipeline_decl(decl.pipelineDecl()))

    return Program(span=_span(parse_tree), raw_ctx=parse_tree, declarations=declarations)


def _kernel_decl(ctx, cls):
    params = []
    if ctx.paramList() is not None:
        for param in ctx.paramList().param() or []:
            params.append(
                Parameter(
                    span=_span(param),
                    raw_ctx=param,
                    modifier=param.getChild(0).getText(),
                    type_name=param.typeName().getText(),
                    name=param.ID().getText(),
                )
            )
    return cls(span=_span(ctx), raw_ctx=ctx, name=ctx.ID().getText(), params=params)


def _pure_decl(ctx) -> PureDecl:
    params = []
    if ctx.pureParamList() is not None:
        pctx = ctx.pureParamList()
        for type_ctx, name_ctx in zip(pctx.typeName(), pctx.ID()):
            params.append(
                Parameter(
                    span=_span(type_ctx),
                    raw_ctx=type_ctx,
                    modifier="value",
                    type_name=type_ctx.getText(),
                    name=name_ctx.getText(),
                )
            )
    statements = [_statement(stmt) for stmt in (ctx.statement() or [])]
    return PureDecl(
        span=_span(ctx),
        raw_ctx=ctx,
        return_type=ctx.typeName().getText(),
        name=ctx.ID().getText(),
        params=params,
        statements=statements,
    )


def _pipeline_decl(ctx) -> PipelineDecl:
    streams = []
    accums = []
    uniforms = []
    for member in ctx.pipelineMember() or []:
        if member.streamDecl() is not None:
            m = member.streamDecl()
            streams.append(StreamDecl(span=_span(m), raw_ctx=m, type_name=m.typeName().getText(), capacity=m.INT().getText(), name=m.ID().getText()))
        elif member.accumDecl() is not None:
            m = member.accumDecl()
            accums.append(AccumDecl(span=_span(m), raw_ctx=m, type_name=m.typeName().getText(), name=m.ID().getText()))
        elif member.uniformDecl() is not None:
            m = member.uniformDecl()
            uniforms.append(UniformDecl(span=_span(m), raw_ctx=m, type_name=m.typeName().getText(), name=m.ID().getText(), initializer=_expr(m.expr()) if m.expr() else None))

    binds = []
    bind_block = ctx.bindBlock()
    if bind_block is not None:
        for stmt in bind_block.bindStmt() or []:
            if stmt.argList() is not None:
                binds.append(BindCall(span=_span(stmt), raw_ctx=stmt, target=stmt.ID(0).getText(), callee=stmt.ID(1).getText(), args=[tok.getText() for tok in stmt.argList().ID()]))
            else:
                binds.append(BindFold(span=_span(stmt), raw_ctx=stmt, type_name=stmt.typeName().getText(), target=stmt.ID(0).getText(), operator=stmt.foldOperator().getText(), source=stmt.ID(1).getText()))

    return PipelineDecl(span=_span(ctx), raw_ctx=ctx, name=ctx.ID().getText(), streams=streams, accumulators=accums, uniforms=uniforms, binds=binds)


def _statement(ctx):
    if ctx.varDecl() is not None:
        v = ctx.varDecl()
        return VarDecl(span=_span(v), raw_ctx=v, type_name=v.typeName().getText() if v.typeName() else None, name=v.ID().getText(), initializer=_expr(v.expr()) if v.expr() else None)
    if ctx.assignStmt() is not None:
        a = ctx.assignStmt()
        return Assignment(span=_span(a), raw_ctx=a, target=_lvalue(a.lvalue()), value=_expr(a.expr()))
    r = ctx.returnStmt()
    return ReturnStmt(span=_span(r), raw_ctx=r, value=_expr(r.expr()))


def _lvalue(ctx):
    parts = [tok.getText() for tok in ctx.ID()]
    node = Identifier(span=_span(ctx), raw_ctx=ctx, name=parts[0])
    for field in parts[1:]:
        node = StructAccess(span=_span(ctx), raw_ctx=ctx, target=node, field=field)
    return node


def _expr(ctx):
    return _logical_or(ctx.logicalExpr().logicalOrExpr())


def _logical_or(ctx):
    nodes = [_logical_and(n) for n in ctx.logicalAndExpr()]
    return _chain(ctx, nodes, "||")


def _logical_and(ctx):
    nodes = [_equality(n) for n in ctx.equalityExpr()]
    return _chain(ctx, nodes, "&&")


def _equality(ctx):
    rel_nodes = [_rel(n) for n in ctx.relExpr()]
    ops = [ctx.getChild(i).getText() for i in range(1, ctx.getChildCount(), 2)]
    return _chain_ops(ctx, rel_nodes, ops)


def _rel(ctx):
    add_nodes = [_add(n) for n in ctx.addExpr()]
    ops = [ctx.getChild(i).getText() for i in range(1, ctx.getChildCount(), 2)]
    return _chain_ops(ctx, add_nodes, ops)


def _add(ctx):
    mul_nodes = [_mul(n) for n in ctx.mulExpr()]
    ops = [ctx.getChild(i).getText() for i in range(1, ctx.getChildCount(), 2)]
    return _chain_ops(ctx, mul_nodes, ops)


def _mul(ctx):
    unary_nodes = [_unary(n) for n in ctx.unaryExpr()]
    ops = [ctx.getChild(i).getText() for i in range(1, ctx.getChildCount(), 2)]
    return _chain_ops(ctx, unary_nodes, ops)


def _chain(ctx, nodes, op):
    if not nodes:
        return Identifier(span=_span(ctx), raw_ctx=ctx, name="")
    cur = nodes[0]
    for node in nodes[1:]:
        cur = BinaryOperation(span=_span(ctx), raw_ctx=ctx, operator=op, left=cur, right=node)
    return cur


def _chain_ops(ctx, nodes, ops):
    if not nodes:
        return Identifier(span=_span(ctx), raw_ctx=ctx, name="")
    cur = nodes[0]
    for op, node in zip(ops, nodes[1:]):
        cur = BinaryOperation(span=_span(ctx), raw_ctx=ctx, operator=op, left=cur, right=node)
    return cur


def _unary(ctx):
    if ctx.primaryExpr() is not None:
        return _primary(ctx.primaryExpr())
    return UnaryOperation(span=_span(ctx), raw_ctx=ctx, operator=ctx.getChild(0).getText(), operand=_unary(ctx.unaryExpr()))


def _primary(ctx):
    if ctx.expr() is not None:
        return _expr(ctx.expr())
    if ctx.lvalue() is not None:
        return _lvalue(ctx.lvalue())
    if ctx.INT() is not None:
        return IntLiteral(span=_span(ctx), raw_ctx=ctx, value=int(ctx.INT().getText()))
    if ctx.FLOAT() is not None:
        return FloatLiteral(span=_span(ctx), raw_ctx=ctx, value=float(ctx.FLOAT().getText()))
    if ctx.BOOL() is not None:
        return BoolLiteral(span=_span(ctx), raw_ctx=ctx, value=ctx.BOOL().getText() == "true")
    if ctx.ID() is not None and ctx.getChildCount() >= 3 and ctx.getChild(1).getText() == "(":
        args = []
        if ctx.exprList() is not None:
            args = [_expr(e) for e in ctx.exprList().expr()]
        return FunctionCall(span=_span(ctx), raw_ctx=ctx, name=ctx.ID().getText(), args=args)
    return Identifier(span=_span(ctx), raw_ctx=ctx, name="")
