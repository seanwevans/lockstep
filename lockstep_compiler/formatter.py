from __future__ import annotations

import sys
from pathlib import Path

from antlr4 import CommonTokenStream, InputStream

try:
    from generated.parser.LockstepLexer import LockstepLexer
    from generated.parser.LockstepParser import LockstepParser
    from generated.parser.LockstepVisitor import LockstepVisitor
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from generated.parser.LockstepLexer import LockstepLexer
    from generated.parser.LockstepParser import LockstepParser
    from generated.parser.LockstepVisitor import LockstepVisitor

from .errors import ParseErrorCollector


def _lex_tokens(source: str) -> list[str]:
    lexer = LockstepLexer(InputStream(source))
    stream = CommonTokenStream(lexer)
    stream.fill()
    return [
        token.text
        for token in stream.tokens
        if token.type != -1 and token.channel == LockstepLexer.DEFAULT_TOKEN_CHANNEL
    ]


def _needs_space(previous: str, current: str) -> bool:
    if not previous:
        return False
    if current in {")", "]", "}", ",", ";", "(", "<", ">", ".", "="}:
        return False
    if previous in {"(", "[", "{", ",", "<", ".", "="}:
        return False
    if current[0] in {'"', "'"} and (previous[-1].isalnum() or previous[-1] == "_"):
        return True
    if previous[-1] == ">" and (current[0].isalnum() or current[0] == "_"):
        return True
    return (previous[-1].isalnum() or previous[-1] == "_") and (
        current[0].isalnum() or current[0] == "_"
    )


def _format_token_stream(source, *, indent="    "):
    tokens = _lex_tokens(source)
    lines = []
    current = ""
    depth = 0

    def append_token(token: str):
        nonlocal current
        if _needs_space(current, token):
            current = f"{current} {token}"
        else:
            current = f"{current}{token}"

    def flush_current():
        nonlocal current
        if current.strip():
            lines.append(f"{indent * depth}{current.strip()}")
        current = ""

    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token == "{":
            if current.strip():
                lines.append(f"{indent * depth}{current.strip()} {{")
            else:
                lines.append(f"{indent * depth}{{")
            current = ""
            depth += 1
        elif token == "}":
            flush_current()
            depth = max(depth - 1, 0)
            closing = "}"
            if index + 1 < len(tokens) and tokens[index + 1] == ";":
                closing += ";"
                index += 1
            lines.append(f"{indent * depth}{closing}")
        elif token == ";":
            append_token(token)
            flush_current()
        else:
            append_token(token)

        index += 1

    flush_current()
    return "\n".join(lines) + "\n"


class _FormattingVisitor(LockstepVisitor):
    def __init__(self, *, indent: str = "    "):
        super().__init__()
        self._indent = indent
        self._depth = 0
        self._lines: list[str] = []

    def _append(self, text: str):
        self._lines.append(f"{self._indent * self._depth}{text}")

    def _open_block(self, prefix: str):
        self._append(f"{prefix} {{")
        self._depth += 1

    def _close_block(self, suffix: str = ""):
        self._depth = max(self._depth - 1, 0)
        self._append(f"}}{suffix}")

    def _format_context_tokens(self, ctx) -> str:
        stream = ctx.parser.getTokenStream()
        stream.fill()
        if ctx.start is None or ctx.stop is None:
            return ""

        text = ""
        for token in stream.tokens[ctx.start.tokenIndex : ctx.stop.tokenIndex + 1]:
            if token.type == -1 or token.channel != LockstepLexer.DEFAULT_TOKEN_CHANNEL:
                continue
            if _needs_space(text, token.text):
                text = f"{text} {token.text}"
            else:
                text = f"{text}{token.text}"
        return text

    def _format_params(self, param_list_ctx):
        if param_list_ctx is None:
            return ""
        return ",".join(self.visit(param) for param in param_list_ctx.param())

    def visitProgram(self, ctx: LockstepParser.ProgramContext):
        for declaration in ctx.declaration():
            self.visit(declaration)
        return "\n".join(self._lines) + "\n"

    def visitDeclaration(self, ctx: LockstepParser.DeclarationContext):
        return self.visitChildren(ctx)

    def visitDependencyDecl(self, ctx: LockstepParser.DependencyDeclContext):
        return self.visitChildren(ctx)

    def visitImportDecl(self, ctx: LockstepParser.ImportDeclContext):
        self._append_dependency(ctx)

    def visitIncludeDecl(self, ctx: LockstepParser.IncludeDeclContext):
        self._append_dependency(ctx)

    def _append_dependency(self, ctx):
        keyword = ctx.getChild(0).getText()
        path = ctx.STRING().getText()
        self._append(f"{keyword} {path};")

    def visitStructDecl(self, ctx: LockstepParser.StructDeclContext):
        self._open_block(f"struct {ctx.ID().getText()}")
        for member in ctx.structMember():
            self.visit(member)
        self._close_block(";")

    def visitStructMember(self, ctx: LockstepParser.StructMemberContext):
        self._append(f"{self.visit(ctx.typeName())} {ctx.ID().getText()};")

    def visitPureDecl(self, ctx: LockstepParser.PureDeclContext):
        params = ""
        if ctx.pureParamList() is not None:
            params = self.visit(ctx.pureParamList())
        self._open_block(
            f"pure {self.visit(ctx.typeName())} {ctx.ID().getText()}({params})"
        )
        for statement in ctx.statement():
            self.visit(statement)
        self._close_block()

    def visitPureParamList(self, ctx: LockstepParser.PureParamListContext):
        params = []
        type_names = ctx.typeName()
        ids = ctx.ID()
        for index, type_name in enumerate(type_names):
            params.append(f"{self.visit(type_name)} {ids[index].getText()}")
        return ",".join(params)

    def visitShaderDecl(self, ctx: LockstepParser.ShaderDeclContext):
        params = self._format_params(ctx.paramList())
        self._open_block(f"shader {ctx.ID().getText()}({params})")
        for statement in ctx.statement():
            self.visit(statement)
        self._close_block()

    def visitFilterDecl(self, ctx: LockstepParser.FilterDeclContext):
        params = self._format_params(ctx.paramList())
        self._open_block(f"filter {ctx.ID().getText()}({params})")
        for statement in ctx.statement():
            self.visit(statement)
        self._close_block()

    def visitParam(self, ctx: LockstepParser.ParamContext):
        qualifier = ctx.getChild(0).getText()
        return f"{qualifier} {self.visit(ctx.typeName())} {ctx.ID().getText()}"

    def visitPipelineDecl(self, ctx: LockstepParser.PipelineDeclContext):
        self._open_block(f"pipeline {ctx.ID().getText()}")
        for member in ctx.pipelineMember():
            self.visit(member)
        self.visit(ctx.bindBlock())
        self._close_block()

    def visitPipelineMember(self, ctx: LockstepParser.PipelineMemberContext):
        return self.visitChildren(ctx)

    def visitStreamDecl(self, ctx: LockstepParser.StreamDeclContext):
        self._append(
            f"stream<{self.visit(ctx.typeName())},{ctx.INT().getText()}> {ctx.ID().getText()};"
        )

    def visitAccumDecl(self, ctx: LockstepParser.AccumDeclContext):
        self._append(f"accumulator<{self.visit(ctx.typeName())}> {ctx.ID().getText()};")

    def visitUniformDecl(self, ctx: LockstepParser.UniformDeclContext):
        prefix = f"uniform {self.visit(ctx.typeName())} {ctx.ID().getText()}"
        if ctx.expr() is not None:
            prefix = f"{prefix}={self.visit(ctx.expr())}"
        self._append(f"{prefix};")

    def visitBindBlock(self, ctx: LockstepParser.BindBlockContext):
        self._open_block("bind")
        for bind_stmt in ctx.bindStmt():
            self.visit(bind_stmt)
        self._close_block()

    def visitBindStmt(self, ctx: LockstepParser.BindStmtContext):
        if ctx.foldOperator() is not None:
            self._append(
                "uniform "
                f"{self.visit(ctx.typeName())} {ctx.ID(0).getText()}="
                f"fold {self.visit(ctx.foldOperator())}({ctx.ID(1).getText()});"
            )
            return

        self._append(
            f"{ctx.ID(0).getText()}={ctx.ID(1).getText()}({self.visit(ctx.argList())});"
        )

    def visitFoldOperator(self, ctx: LockstepParser.FoldOperatorContext):
        return ctx.getText()

    def visitArgList(self, ctx: LockstepParser.ArgListContext):
        return ",".join(token.getText() for token in ctx.ID())

    def visitStatement(self, ctx: LockstepParser.StatementContext):
        return self.visitChildren(ctx)

    def visitVarDecl(self, ctx: LockstepParser.VarDeclContext):
        segments = []
        if ctx.typeName() is not None:
            segments.append(self.visit(ctx.typeName()))
        segments.append(ctx.ID().getText())
        body = " ".join(segments)
        if ctx.expr() is not None:
            body = f"{body}={self.visit(ctx.expr())}"
        self._append(f"{body};")

    def visitAssignStmt(self, ctx: LockstepParser.AssignStmtContext):
        self._append(f"{self.visit(ctx.lvalue())}={self.visit(ctx.expr())};")

    def visitReturnStmt(self, ctx: LockstepParser.ReturnStmtContext):
        self._append(f"return {self.visit(ctx.expr())};")

    def visitExpr(self, ctx: LockstepParser.ExprContext):
        return self.visit(ctx.logicalExpr())

    def visitLogicalExpr(self, ctx: LockstepParser.LogicalExprContext):
        return self.visit(ctx.logicalOrExpr())

    def visitLogicalOrExpr(self, ctx: LockstepParser.LogicalOrExprContext):
        return self._format_context_tokens(ctx)

    def visitLogicalAndExpr(self, ctx: LockstepParser.LogicalAndExprContext):
        return self._format_context_tokens(ctx)

    def visitBitwiseOrExpr(self, ctx: LockstepParser.BitwiseOrExprContext):
        return self._format_context_tokens(ctx)

    def visitBitwiseXorExpr(self, ctx: LockstepParser.BitwiseXorExprContext):
        return self._format_context_tokens(ctx)

    def visitBitwiseAndExpr(self, ctx: LockstepParser.BitwiseAndExprContext):
        return self._format_context_tokens(ctx)

    def visitEqualityExpr(self, ctx: LockstepParser.EqualityExprContext):
        return self._format_context_tokens(ctx)

    def visitRelExpr(self, ctx: LockstepParser.RelExprContext):
        return self._format_context_tokens(ctx)

    def visitShiftExpr(self, ctx: LockstepParser.ShiftExprContext):
        return self._format_context_tokens(ctx)

    def visitAddExpr(self, ctx: LockstepParser.AddExprContext):
        return self._format_context_tokens(ctx)

    def visitMulExpr(self, ctx: LockstepParser.MulExprContext):
        return self._format_context_tokens(ctx)

    def visitUnaryExpr(self, ctx: LockstepParser.UnaryExprContext):
        return self._format_context_tokens(ctx)

    def visitPrimaryExpr(self, ctx: LockstepParser.PrimaryExprContext):
        return self._format_context_tokens(ctx)

    def visitExprList(self, ctx: LockstepParser.ExprListContext):
        return ",".join(self.visit(expr) for expr in ctx.expr())

    def visitLvalue(self, ctx: LockstepParser.LvalueContext):
        return self._format_context_tokens(ctx)

    def visitTypeName(self, ctx: LockstepParser.TypeNameContext):
        return self._format_context_tokens(ctx)

    def visitTypeSuffix(self, ctx: LockstepParser.TypeSuffixContext):
        return self._format_context_tokens(ctx)


def format_lockstep_source(source, *, indent="    "):
    input_stream = InputStream(source)
    lexer = LockstepLexer(input_stream)
    error_listener = ParseErrorCollector()
    lexer.removeErrorListeners()
    lexer.addErrorListener(error_listener)

    stream = CommonTokenStream(lexer)
    stream.fill()
    if any(token.type == LockstepLexer.COMMENT for token in stream.tokens):
        return source

    parser = LockstepParser(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(error_listener)
    tree = parser.program()

    if error_listener.errors:
        return _format_token_stream(source, indent=indent)

    return _FormattingVisitor(indent=indent).visit(tree)
