from __future__ import annotations

import bisect
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _Comment:
    """A single line comment recovered from the HIDDEN token channel."""

    line: int  # 1-based source line the comment appears on
    text: str  # comment text including the leading `//`, trailing space stripped
    trailing: bool  # True if code precedes it on the same source line


def _extract_comments(stream: CommonTokenStream) -> list[_Comment]:
    """Recover line comments (dropped to the HIDDEN channel) in source order.

    A comment is classified as *trailing* when a real (default-channel) token
    occurs earlier on the same line, and *own-line* otherwise. This is enough to
    place `//` comments correctly because they always run to end-of-line, so a
    comment is either the whole line's content or the tail of a code line.
    """
    comments: list[_Comment] = []
    last_default_line: int | None = None
    for token in stream.tokens:
        if token.type == LockstepLexer.COMMENT:
            comments.append(
                _Comment(
                    line=token.line,
                    text=token.text.rstrip(),
                    trailing=last_default_line == token.line,
                )
            )
        elif (
            token.type != -1
            and token.channel == LockstepLexer.DEFAULT_TOKEN_CHANNEL
        ):
            last_default_line = token.line
    return comments


@dataclass
class _OutputLine:
    """One rendered line of code plus the source line it originated from, so
    comments can be re-attached to the right place after reformatting."""

    depth: int
    src_line: int
    text: str
    is_close: bool = False


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
        self._lines: list[_OutputLine] = []

    def _append(self, text: str, ctx):
        self._lines.append(_OutputLine(self._depth, ctx.start.line, text))

    def _open_block(self, prefix: str, ctx):
        self._lines.append(_OutputLine(self._depth, ctx.start.line, f"{prefix} {{"))
        self._depth += 1

    def _close_block(self, ctx, suffix: str = ""):
        self._depth = max(self._depth - 1, 0)
        self._lines.append(
            _OutputLine(self._depth, ctx.stop.line, f"}}{suffix}", is_close=True)
        )

    def render(self, comments: list[_Comment]) -> str:
        """Render the recorded lines, re-interleaving `comments` by source line.

        Own-line comments are emitted as their own line, indented to match the
        code that follows them; trailing comments are appended to the code line
        they sat on. When there are no comments this is identical to a plain
        indent-and-join of the recorded lines, so comment-free output is
        unchanged.
        """
        own_line = [comment for comment in comments if not comment.trailing]
        src_lines = [line.src_line for line in self._lines]

        # Attach each trailing comment to the last code line at or before it.
        trailing: dict[int, list[str]] = {}
        unplaced: list[str] = []
        for comment in comments:
            if not comment.trailing:
                continue
            index = bisect.bisect_right(src_lines, comment.line) - 1
            if index >= 0:
                trailing.setdefault(index, []).append(comment.text)
            else:
                unplaced.append(comment.text)

        out: list[str] = []
        own_index = 0
        for record_index, record in enumerate(self._lines):
            while own_index < len(own_line) and own_line[own_index].line < record.src_line:
                # A comment sitting just before a closing brace belongs inside
                # the block that is about to close, so indent one level deeper.
                comment_depth = record.depth + 1 if record.is_close else record.depth
                out.append(f"{self._indent * comment_depth}{own_line[own_index].text}")
                own_index += 1
            line_text = f"{self._indent * record.depth}{record.text}"
            if record_index in trailing:
                line_text = f"{line_text}  {' '.join(trailing[record_index])}"
            out.append(line_text)

        # Comments after the final construct (or an all-comments file).
        for comment in own_line[own_index:]:
            out.append(comment.text)
        out.extend(unplaced)
        return "\n".join(out) + "\n"

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
        return None

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
        self._append(f"{keyword} {path};", ctx)

    def visitStructDecl(self, ctx: LockstepParser.StructDeclContext):
        self._open_block(f"struct {ctx.ID().getText()}", ctx)
        for member in ctx.structMember():
            self.visit(member)
        self._close_block(ctx, ";")

    def visitStructMember(self, ctx: LockstepParser.StructMemberContext):
        self._append(f"{self.visit(ctx.typeName())} {ctx.ID().getText()};", ctx)

    def visitPureDecl(self, ctx: LockstepParser.PureDeclContext):
        params = ""
        if ctx.pureParamList() is not None:
            params = self.visit(ctx.pureParamList())
        self._open_block(
            f"pure {self.visit(ctx.typeName())} {ctx.ID().getText()}({params})", ctx
        )
        for statement in ctx.statement():
            self.visit(statement)
        self._close_block(ctx)

    def visitPureParamList(self, ctx: LockstepParser.PureParamListContext):
        params = []
        type_names = ctx.typeName()
        ids = ctx.ID()
        for index, type_name in enumerate(type_names):
            params.append(f"{self.visit(type_name)} {ids[index].getText()}")
        return ",".join(params)

    def visitShaderDecl(self, ctx: LockstepParser.ShaderDeclContext):
        params = self._format_params(ctx.paramList())
        self._open_block(f"shader {ctx.ID().getText()}({params})", ctx)
        for statement in ctx.statement():
            self.visit(statement)
        self._close_block(ctx)

    def visitFilterDecl(self, ctx: LockstepParser.FilterDeclContext):
        params = self._format_params(ctx.paramList())
        self._open_block(f"filter {ctx.ID().getText()}({params})", ctx)
        for statement in ctx.statement():
            self.visit(statement)
        self._close_block(ctx)

    def visitParam(self, ctx: LockstepParser.ParamContext):
        qualifier = ctx.getChild(0).getText()
        return f"{qualifier} {self.visit(ctx.typeName())} {ctx.ID().getText()}"

    def visitPipelineDecl(self, ctx: LockstepParser.PipelineDeclContext):
        self._open_block(f"pipeline {ctx.ID().getText()}", ctx)
        for member in ctx.pipelineMember():
            self.visit(member)
        self.visit(ctx.bindBlock())
        self._close_block(ctx)

    def visitPipelineMember(self, ctx: LockstepParser.PipelineMemberContext):
        return self.visitChildren(ctx)

    def visitStreamDecl(self, ctx: LockstepParser.StreamDeclContext):
        self._append(
            f"stream<{self.visit(ctx.typeName())},{ctx.INT().getText()}> {ctx.ID().getText()};",
            ctx,
        )

    def visitAccumDecl(self, ctx: LockstepParser.AccumDeclContext):
        self._append(
            f"accumulator<{self.visit(ctx.typeName())}> {ctx.ID().getText()};", ctx
        )

    def visitUniformDecl(self, ctx: LockstepParser.UniformDeclContext):
        prefix = f"uniform {self.visit(ctx.typeName())} {ctx.ID().getText()}"
        if ctx.expr() is not None:
            prefix = f"{prefix}={self.visit(ctx.expr())}"
        self._append(f"{prefix};", ctx)

    def visitBindBlock(self, ctx: LockstepParser.BindBlockContext):
        self._open_block("bind", ctx)
        for bind_stmt in ctx.bindStmt():
            self.visit(bind_stmt)
        self._close_block(ctx)

    def visitBindStmt(self, ctx: LockstepParser.BindStmtContext):
        if ctx.foldOperator() is not None:
            self._append(
                "uniform "
                f"{self.visit(ctx.typeName())} {ctx.ID(0).getText()}="
                f"fold {self.visit(ctx.foldOperator())}({ctx.ID(1).getText()});",
                ctx,
            )
            return

        self._append(
            f"{ctx.ID(0).getText()}={ctx.ID(1).getText()}({self.visit(ctx.argList())});",
            ctx,
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
        self._append(f"{body};", ctx)

    def visitAssignStmt(self, ctx: LockstepParser.AssignStmtContext):
        self._append(f"{self.visit(ctx.lvalue())}={self.visit(ctx.expr())};", ctx)

    def visitReturnStmt(self, ctx: LockstepParser.ReturnStmtContext):
        self._append(f"return {self.visit(ctx.expr())};", ctx)

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
    comments = _extract_comments(stream)

    parser = LockstepParser(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(error_listener)
    tree = parser.program()

    if error_listener.errors:
        # The token-stream fallback cannot safely re-place comments around a
        # broken parse, so leave a comment-bearing source untouched rather than
        # dropping its comments; comment-free sources still get best-effort
        # reformatting.
        if comments:
            return source
        return _format_token_stream(source, indent=indent)

    visitor = _FormattingVisitor(indent=indent)
    visitor.visit(tree)
    return visitor.render(comments)
