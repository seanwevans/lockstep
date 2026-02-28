from antlr4 import CommonTokenStream
import sys
from pathlib import Path

from lockstep_compiler import (
    LockstepCompileError,
    LockstepDiagnostic,
    ParseErrorCollector,
    build_arg_parser,
    normalize_diagnostics,
    run_cli,
)
from lockstep_compiler.compiler import _compile_lockstep_with_dependencies as _compile_lockstep
from lockstep_compiler.models import LockstepCompileResult
from lockstep_compiler.visitors import (
    build_debug_visitor,
    build_semantic_validator,
    validate_semantics as _validate_semantics,
)

PARSER_DIR = Path(__file__).parent / "generated" / "parser"
if str(PARSER_DIR) not in sys.path:
    sys.path.insert(0, str(PARSER_DIR))

from LockstepLexer import LockstepLexer
from LockstepParser import LockstepParser
from LockstepVisitor import LockstepVisitor

LockstepDebugVisitor = build_debug_visitor(LockstepVisitor)
LockstepSemanticValidator = build_semantic_validator(LockstepVisitor)


def validate_semantics(parse_tree):
    return _validate_semantics(parse_tree, LockstepVisitor)


def compile_lockstep(source_code: str, verbose: bool = True) -> LockstepCompileResult:
    return _compile_lockstep(
        source_code,
        verbose=verbose,
        lexer_cls=LockstepLexer,
        parser_cls=LockstepParser,
        visitor_cls=LockstepVisitor,
        semantic_validator=validate_semantics,
        token_stream_cls=CommonTokenStream,
        debug_visitor_cls=LockstepDebugVisitor,
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


if __name__ == "__main__":
    sys.exit(run_cli(compiler=compile_lockstep))
