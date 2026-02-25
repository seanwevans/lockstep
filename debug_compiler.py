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
    diagnostics: list[tuple[int, int, str]] = field(default_factory=list)


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
    return LockstepCompileResult(
        parse_tree=tree,
        entities={
            "structs": visitor.structs,
            "shaders": visitor.shaders,
            "streams": visitor.streams,
            "accumulators": visitor.accumulators,
        },
        diagnostics=[],
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
        source_path = Path(args.path)
        try:
            source = source_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"Unable to read '{source_path}': file not found.", file=stderr)
            return 1
        except PermissionError as err:
            reason = err.strerror or "permission denied"
            print(f"Unable to read '{source_path}': {reason}.", file=stderr)
            return 1
        except UnicodeDecodeError as err:
            print(
                f"Unable to read '{source_path}': invalid UTF-8 ({err.reason}).",
                file=stderr,
            )
            return 1
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
