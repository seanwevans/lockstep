import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from antlr4 import InputStream, CommonTokenStream
from antlr4.error.ErrorListener import ErrorListener

# antlr4 -Dlanguage=Python3 -visitor Lockstep.g4
from LockstepLexer import LockstepLexer
from LockstepParser import LockstepParser
from LockstepVisitor import LockstepVisitor


class LockstepDebugVisitor(LockstepVisitor):
    """Walks the Parse Tree and extracts the pipeline architecture."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.structs = []
        self.shaders = []
        self.streams = []
        self.accumulators = []

    def _print(self, message: str):
        if self.verbose:
            print(message)

    def visitProgram(self, ctx: LockstepParser.ProgramContext):
        self._print("=== LOCKSTEP COMPILER FRONTEND ===")
        self._print("Parsing program...\n")
        return self.visitChildren(ctx)

    def visitStructDecl(self, ctx: LockstepParser.StructDeclContext):
        name = ctx.ID().getText()
        self.structs.append(name)
        self._print(f"[Struct] Discovered: {name}")
        return self.visitChildren(ctx)

    def visitPureDecl(self, ctx: LockstepParser.PureDeclContext):
        name = ctx.ID().getText()
        ret_type = ctx.typeName().getText()
        self._print(f"[Pure Function] {name} -> {ret_type}")
        return self.visitChildren(ctx)

    def visitShaderDecl(self, ctx: LockstepParser.ShaderDeclContext):
        name = ctx.ID().getText()
        params = []
        self._print(f"\n[Shader Kernel] {name}")
        if ctx.paramList():
            for param in ctx.paramList().param():
                modifier = param.getChild(0).getText()
                p_type = param.typeName().getText()
                p_name = param.ID().getText()
                params.append({"modifier": modifier, "type": p_type, "name": p_name})
                self._print(f"  └─ Param: ({modifier}) {p_type} {p_name}")
        self.shaders.append({"name": name, "params": params})
        return self.visitChildren(ctx)

    def visitPipelineDecl(self, ctx: LockstepParser.PipelineDeclContext):
        name = ctx.ID().getText()
        self._print(f"\n[Pipeline Topology] {name}")
        return self.visitChildren(ctx)

    def visitStreamDecl(self, ctx: LockstepParser.StreamDeclContext):
        s_type = ctx.typeName().getText()
        capacity = ctx.INT().getText()
        name = ctx.ID().getText()
        self.streams.append({"name": name, "type": s_type, "capacity": capacity})
        self._print(f"  └─ Stream: {name} <{s_type}, {capacity}>")
        return self.visitChildren(ctx)

    def visitAccumDecl(self, ctx: LockstepParser.AccumDeclContext):
        a_type = ctx.typeName().getText()
        name = ctx.ID().getText()
        self.accumulators.append({"name": name, "type": a_type})
        self._print(f"  └─ Accumulator: {name} <{a_type}>")
        return self.visitChildren(ctx)

    def visitBindBlock(self, ctx: LockstepParser.BindBlockContext):
        self._print("  └─ Routing:")
        for stmt in ctx.bindStmt():
            self._print(f"       {stmt.getText()}")
        return self.visitChildren(ctx)


@dataclass
class LockstepCompileResult:
    parse_tree: Any
    entities: dict[str, Any]
    diagnostics: list["SemanticDiagnostic"] = field(default_factory=list)


@dataclass(frozen=True)
class SemanticDiagnostic:
    line: int
    column: int
    severity: str
    message: str


@dataclass(frozen=True)
class CallableParam:
    modifier: str
    type_name: str
    name: str


@dataclass(frozen=True)
class CallableSymbol:
    kind: str
    name: str
    params: list[CallableParam]
    return_type: str | None = None


@dataclass
class PipelineScope:
    name: str
    streams: dict[str, str] = field(default_factory=dict)
    accumulators: dict[str, str] = field(default_factory=dict)
    uniforms: dict[str, str] = field(default_factory=dict)

    def resolve(self, ident: str) -> tuple[str, str] | None:
        if ident in self.streams:
            return ("stream", self.streams[ident])
        if ident in self.accumulators:
            return ("accumulator", self.accumulators[ident])
        if ident in self.uniforms:
            return ("uniform", self.uniforms[ident])
        return None


@dataclass
class SymbolTable:
    structs: list[str] = field(default_factory=list)
    pure_functions: dict[str, CallableSymbol] = field(default_factory=dict)
    shaders: dict[str, CallableSymbol] = field(default_factory=dict)
    filters: dict[str, CallableSymbol] = field(default_factory=dict)
    pipelines: dict[str, PipelineScope] = field(default_factory=dict)

    @property
    def callables(self) -> dict[str, CallableSymbol]:
        merged = {}
        merged.update(self.pure_functions)
        merged.update(self.shaders)
        merged.update(self.filters)
        return merged


class LockstepSemanticAnalyzer:
    """Builds a symbol table and validates semantic correctness."""

    def __init__(self):
        self.symbols = SymbolTable()
        self.diagnostics: list[SemanticDiagnostic] = []

    def analyze(self, tree: Any) -> tuple[SymbolTable, list[SemanticDiagnostic]]:
        self.symbols = SymbolTable()
        self.diagnostics: list[SemanticDiagnostic] = []

        declarations = getattr(tree, "declaration", None)
        if callable(declarations):
            for decl in declarations():
                self._visit_declaration(decl)

        return self.symbols, self.diagnostics

    def _visit_declaration(self, decl: Any):
        for method_name, handler in (
            ("structDecl", self._record_struct),
            ("pureDecl", self._record_pure),
            ("shaderDecl", self._record_shader),
            ("filterDecl", self._record_filter),
            ("pipelineDecl", self._record_pipeline),
        ):
            getter = getattr(decl, method_name, None)
            if callable(getter):
                ctx = getter()
                if ctx is not None:
                    handler(ctx)
                    return

    def _record_struct(self, ctx: Any):
        self.symbols.structs.append(ctx.ID().getText())

    def _record_pure(self, ctx: Any):
        name = ctx.ID().getText()
        params = []
        plist = getattr(ctx, "pureParamList", lambda: None)()
        if plist is not None:
            children = plist.getChildren()
            chunks = [node.getText() for node in children if node.getText() != ","]
            for i in range(0, len(chunks), 2):
                params.append(CallableParam("in", chunks[i], chunks[i + 1]))
        self.symbols.pure_functions[name] = CallableSymbol(
            kind="pure", name=name, params=params, return_type=ctx.typeName().getText()
        )

    def _record_shader(self, ctx: Any):
        self.symbols.shaders[ctx.ID().getText()] = self._callable_from_param_list(
            kind="shader", name=ctx.ID().getText(), param_list_ctx=ctx.paramList()
        )

    def _record_filter(self, ctx: Any):
        self.symbols.filters[ctx.ID().getText()] = self._callable_from_param_list(
            kind="filter", name=ctx.ID().getText(), param_list_ctx=ctx.paramList()
        )

    def _callable_from_param_list(self, kind: str, name: str, param_list_ctx: Any):
        params = []
        if param_list_ctx is not None:
            for param in param_list_ctx.param():
                params.append(
                    CallableParam(
                        modifier=param.getChild(0).getText(),
                        type_name=param.typeName().getText(),
                        name=param.ID().getText(),
                    )
                )
        return CallableSymbol(kind=kind, name=name, params=params)

    def _record_pipeline(self, ctx: Any):
        scope = PipelineScope(name=ctx.ID().getText())
        for member in ctx.pipelineMember():
            if member.streamDecl() is not None:
                decl = member.streamDecl()
                scope.streams[decl.ID().getText()] = decl.typeName().getText()
            elif member.accumDecl() is not None:
                decl = member.accumDecl()
                scope.accumulators[decl.ID().getText()] = decl.typeName().getText()
            elif member.uniformDecl() is not None:
                decl = member.uniformDecl()
                scope.uniforms[decl.ID().getText()] = decl.typeName().getText()

        self.symbols.pipelines[scope.name] = scope
        self._validate_bind_block(ctx.bindBlock(), scope)

    def _validate_bind_block(self, bind_ctx: Any, scope: PipelineScope):
        for stmt in bind_ctx.bindStmt():
            ids = stmt.ID()
            if len(ids) == 3:
                self._validate_fold(stmt, scope)
            else:
                self._validate_bind_call(stmt, scope)

    def _validate_bind_call(self, stmt: Any, scope: PipelineScope):
        target_id, kernel_id = stmt.ID()[0], stmt.ID()[1]
        target_name, kernel_name = target_id.getText(), kernel_id.getText()
        target_symbol = scope.resolve(target_name)
        if target_symbol is None or target_symbol[0] != "stream":
            self._error(stmt, f"bind target '{target_name}' must be a declared stream")

        callable_symbol = self.symbols.callables.get(kernel_name)
        if callable_symbol is None:
            self._error(stmt, f"callable '{kernel_name}' is not declared")
            return

        args = stmt.argList().ID() if stmt.argList() is not None else []
        if len(args) != len(callable_symbol.params):
            self._error(
                stmt,
                f"call to '{kernel_name}' expects {len(callable_symbol.params)} args, got {len(args)}",
            )
            return

        for arg_token, param in zip(args, callable_symbol.params):
            resolved = scope.resolve(arg_token.getText())
            if resolved is None:
                self._error_from_token(
                    arg_token,
                    f"argument '{arg_token.getText()}' is not declared in pipeline scope",
                )
                continue

            arg_kind, arg_type = resolved
            if not self._modifier_compatible(param.modifier, arg_kind):
                self._error_from_token(
                    arg_token,
                    f"argument '{arg_token.getText()}' ({arg_kind}) is invalid for '{param.modifier}' parameter",
                )
            if arg_type != param.type_name:
                self._error_from_token(
                    arg_token,
                    f"argument '{arg_token.getText()}' type '{arg_type}' does not match expected '{param.type_name}'",
                )

    def _validate_fold(self, stmt: Any, scope: PipelineScope):
        source_id = stmt.ID()[2]
        source = source_id.getText()
        resolved = scope.resolve(source)
        if resolved is None:
            self._error_from_token(
                source_id,
                f"fold source '{source}' is not declared in pipeline scope",
            )
            return
        if resolved[0] != "accumulator":
            self._error_from_token(source_id, f"fold source '{source}' must be an accumulator")

    def _modifier_compatible(self, modifier: str, symbol_kind: str):
        compat = {
            "in": {"stream", "uniform", "accumulator"},
            "out": {"stream"},
            "uniform": {"uniform"},
            "accum": {"accumulator"},
        }
        return symbol_kind in compat.get(modifier, set())

    def _location(self, ctx_or_token: Any) -> tuple[int, int]:
        token = getattr(ctx_or_token, "start", None)
        if token is None:
            token = ctx_or_token
        line = getattr(token, "line", 0) or 0
        column = getattr(token, "column", 0) or 0
        return line, column

    def _error(self, ctx: Any, message: str):
        line, column = self._location(ctx)
        self.diagnostics.append(SemanticDiagnostic(line, column, "error", message))

    def _error_from_token(self, token: Any, message: str):
        line, column = self._location(token)
        self.diagnostics.append(SemanticDiagnostic(line, column, "error", message))


class ParseErrorCollector(ErrorListener):
    """Collects syntax errors emitted by ANTLR during lex/parse."""

    def __init__(self):
        super().__init__()
        self.errors = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.errors.append((line, column, msg))


class LockstepCompileError(Exception):
    """Raised when the Lockstep source contains parse errors."""

    def __init__(self, errors):
        self.errors = errors
        super().__init__(self._format_message())

    def _format_message(self):
        count = len(self.errors)
        suffix = "" if count == 1 else "s"
        summary = f"Compilation failed with {count} parse error{suffix}."
        details = "\n".join(
            f"line {line}:{column} {message}" for line, column, message in self.errors
        )
        return summary if not details else f"{summary}\n{details}"


def compile_lockstep(source_code: str, verbose: bool = True) -> LockstepCompileResult:
    input_stream = InputStream(source_code)
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
        raise LockstepCompileError(error_listener.errors)

    visitor = LockstepDebugVisitor(verbose=verbose)
    visitor.visit(tree)
    semantic_symbols, semantic_diagnostics = LockstepSemanticAnalyzer().analyze(tree)
    return LockstepCompileResult(
        parse_tree=tree,
        entities={
            "structs": visitor.structs,
            "shaders": visitor.shaders,
            "streams": visitor.streams,
            "accumulators": visitor.accumulators,
            "symbol_table": {
                "structs": semantic_symbols.structs,
                "pure_functions": list(semantic_symbols.pure_functions),
                "shaders": list(semantic_symbols.shaders),
                "filters": list(semantic_symbols.filters),
                "pipelines": {
                    name: {
                        "streams": scope.streams,
                        "accumulators": scope.accumulators,
                        "uniforms": scope.uniforms,
                    }
                    for name, scope in semantic_symbols.pipelines.items()
                },
            },
        },
        diagnostics=semantic_diagnostics,
    )


TEST_SOURCE = """
struct Vec3 { float x; float y; float z; };

pure Vec3 add(Vec3 a, Vec3 b) {
    Vec3 r; 
    r.x = a.x + b.x; 
    return r;
}

shader ApplyGravity(in Vec3 pos, out Vec3 new_pos, accum float energy, uniform float dt) {
    new_pos.x = pos.x;
    new_pos.y = pos.y - (9.8 * dt);
    energy = new_pos.y; 
}

pipeline Physics {
    stream<Vec3, 1000> raw_positions;
    stream<Vec3, 1000> final_positions;
    accumulator<float> total_energy;
    uniform float dt = 0.016;

    bind {
        final_positions = ApplyGravity(raw_positions, final_positions, total_energy, dt);
        uniform float sys_energy = fold sum(total_energy);
    }
}
"""


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Debug parser for Lockstep source files."
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Optional path to a Lockstep source file. Reads from stdin when omitted.",
    )
    return parser


def run_cli(argv=None, *, stdin=None, stderr=None, compiler=compile_lockstep):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    stdin = sys.stdin if stdin is None else stdin
    stderr = sys.stderr if stderr is None else stderr

    if args.path:
        source = Path(args.path).read_text(encoding="utf-8")
    else:
        source = stdin.read()

    try:
        compiler(source)
    except LockstepCompileError as err:
        print(str(err), file=stderr)
        for line, column, message in err.errors:
            print(f"  line {line}:{column} {message}", file=stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
