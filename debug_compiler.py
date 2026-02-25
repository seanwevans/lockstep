import argparse
import sys
from pathlib import Path
from antlr4 import InputStream, CommonTokenStream
from antlr4.error.ErrorListener import ErrorListener

# antlr4 -Dlanguage=Python3 -visitor Lockstep.g4
from LockstepLexer import LockstepLexer
from LockstepParser import LockstepParser
from LockstepVisitor import LockstepVisitor


class LockstepDebugVisitor(LockstepVisitor):
    """Walks the Parse Tree and extracts the pipeline architecture."""

    def visitProgram(self, ctx: LockstepParser.ProgramContext):
        print("=== LOCKSTEP COMPILER FRONTEND ===")
        print("Parsing program...\n")
        return self.visitChildren(ctx)

    def visitStructDecl(self, ctx: LockstepParser.StructDeclContext):
        name = ctx.ID().getText()
        print(f"[Struct] Discovered: {name}")
        return self.visitChildren(ctx)

    def visitPureDecl(self, ctx: LockstepParser.PureDeclContext):
        name = ctx.ID().getText()
        ret_type = ctx.typeName().getText()
        print(f"[Pure Function] {name} -> {ret_type}")
        return self.visitChildren(ctx)

    def visitShaderDecl(self, ctx: LockstepParser.ShaderDeclContext):
        name = ctx.ID().getText()
        print(f"\n[Shader Kernel] {name}")
        if ctx.paramList():
            for param in ctx.paramList().param():
                modifier = param.getChild(0).getText()
                p_type = param.typeName().getText()
                p_name = param.ID().getText()
                print(f"  └─ Param: ({modifier}) {p_type} {p_name}")
        return self.visitChildren(ctx)

    def visitPipelineDecl(self, ctx: LockstepParser.PipelineDeclContext):
        name = ctx.ID().getText()
        print(f"\n[Pipeline Topology] {name}")
        return self.visitChildren(ctx)

    def visitStreamDecl(self, ctx: LockstepParser.StreamDeclContext):
        s_type = ctx.typeName().getText()
        capacity = ctx.INT().getText()
        name = ctx.ID().getText()
        print(f"  └─ Stream: {name} <{s_type}, {capacity}>")
        return self.visitChildren(ctx)

    def visitAccumDecl(self, ctx: LockstepParser.AccumDeclContext):
        a_type = ctx.typeName().getText()
        name = ctx.ID().getText()
        print(f"  └─ Accumulator: {name} <{a_type}>")
        return self.visitChildren(ctx)

    def visitBindBlock(self, ctx: LockstepParser.BindBlockContext):
        print("  └─ Routing:")
        for stmt in ctx.bindStmt():
            print(f"       {stmt.getText()}")
        return self.visitChildren(ctx)


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
        return f"Compilation failed with {count} parse error{suffix}."


def compile_lockstep(source_code: str):
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

    visitor = LockstepDebugVisitor()
    visitor.visit(tree)


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
    parser = argparse.ArgumentParser(description="Debug parser for Lockstep source files.")
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
