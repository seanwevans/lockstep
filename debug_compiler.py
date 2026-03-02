import sys

from lockstep_compiler import (
    LockstepCompileError,
    LockstepCompileResult,
    LockstepDiagnostic,
    ParseErrorCollector,
    build_debug_visitor,
    build_semantic_validator,
    compile_lockstep,
    load_default_parser_classes,
    normalize_diagnostics,
    run_cli,
    validate_semantics,
)


_LockstepLexer, _LockstepParser, _LockstepVisitor = load_default_parser_classes()
LockstepLexer = _LockstepLexer
LockstepParser = _LockstepParser
LockstepVisitor = _LockstepVisitor

class _CompatibilityVisitor(LockstepVisitor):
    """Visitor shim that keeps debug_compiler test doubles lightweight."""

    def visitChildren(self, node):
        get_child_count = getattr(node, "getChildCount", None)
        if callable(get_child_count):
            return super().visitChildren(node)
        return node


LockstepDebugVisitor = build_debug_visitor(_CompatibilityVisitor)
LockstepSemanticValidator = build_semantic_validator(_CompatibilityVisitor)


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
