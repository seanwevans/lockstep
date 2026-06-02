import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lockstep_compiler import compile_lockstep
from lockstep_compiler.errors import LockstepCompileError


def _semantic_error_codes(source: str) -> list[str]:
    with pytest.raises(LockstepCompileError) as exc_info:
        compile_lockstep(source, verbose=False)
    assert exc_info.value.phase == "semantic"
    return [diag.code for diag in exc_info.value.errors]


def test_ast_validator_rejects_assignment_type_mismatch():
    codes = _semantic_error_codes("""
shader BadAssign(in float value, out int result) {
    result = value;
}
""")

    assert "LCK417" in codes


def test_ast_validator_rejects_pure_return_type_mismatch():
    codes = _semantic_error_codes("""
pure float BadReturn(int value) {
    return value;
}
""")

    assert "LCK418" in codes


def test_ast_validator_checks_pipeline_bind_routes():
    codes = _semantic_error_codes("""
shader Route(in float input, out float output, accum float total) {
    output = input;
}

pipeline BadBind {
    stream<int, 4> source;
    stream<float, 4> result;
    uniform float not_accum;

    bind {
        result = Route(source, result, not_accum);
    }
}
""")

    assert "LCK305" in codes
    assert "LCK308" in codes


def test_ast_validator_rejects_invalid_return_expression_operands():
    codes = _semantic_error_codes("""
pure int BadOperands(int value, float scale) {
    return value + scale;
}
""")

    assert "LCK424" in codes
