from antlr4 import CommonTokenStream, InputStream

from generated.parser.LockstepLexer import LockstepLexer
from generated.parser.LockstepParser import LockstepParser
from generated.parser.LockstepVisitor import LockstepVisitor

from .errors import ParseErrorCollector

def _tokenize(source):
    tokens = []
    current = []

    for char in source:
        if char in {"{", "}", ";", "\n"}:
            if current:
                token = "".join(current).strip()
                if token:
                    tokens.append(token)
                current = []
            tokens.append(char)
        else:
            current.append(char)

    if current:
        token = "".join(current).strip()
        if token:
            tokens.append(token)

    return tokens


def _format_token_stream(source, *, indent):
    tokens = _tokenize(source)
    lines = []
    current = ""
    depth = 0

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
            current = f"{current.strip()};"
            flush_current()
        elif token == "\n":
            flush_current()
        else:
            if current:
                current = f"{current} {token}"
            else:
                current = token

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
        return ctx.getText()

    def visitLogicalAndExpr(self, ctx: LockstepParser.LogicalAndExprContext):
        return ctx.getText()

    def visitEqualityExpr(self, ctx: LockstepParser.EqualityExprContext):
        return ctx.getText()

    def visitRelExpr(self, ctx: LockstepParser.RelExprContext):
        return ctx.getText()

    def visitAddExpr(self, ctx: LockstepParser.AddExprContext):
        return ctx.getText()

    def visitMulExpr(self, ctx: LockstepParser.MulExprContext):
        return ctx.getText()

    def visitUnaryExpr(self, ctx: LockstepParser.UnaryExprContext):
        return ctx.getText()

    def visitPrimaryExpr(self, ctx: LockstepParser.PrimaryExprContext):
        return ctx.getText()

    def visitExprList(self, ctx: LockstepParser.ExprListContext):
        return ",".join(self.visit(expr) for expr in ctx.expr())

    def visitLvalue(self, ctx: LockstepParser.LvalueContext):
        return ctx.getText()

    def visitTypeName(self, ctx: LockstepParser.TypeNameContext):
        return ctx.getText()


def format_lockstep_source(source, *, indent="    "):
    input_stream = InputStream(source)
    lexer = LockstepLexer(input_stream)
    error_listener = ParseErrorCollector()
    lexer.removeErrorListeners()
    lexer.addErrorListener(error_listener)

    stream = CommonTokenStream(lexer)
    parser = LockstepParser(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(error_listener)
    tree = parser.program()

    if error_listener.errors:
        return _format_token_stream(source, indent=indent)

    return _FormattingVisitor(indent=indent).visit(tree)
