import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lockstep_compiler import LockstepCompileError, compile_lockstep


def test_compile_accepts_array_and_generic_declared_types():
    source = """
struct Particle {
    float[4] weights;
    vector<float, 4> velocity;
    matrix<float, 4>[2] transforms;
};

shader Copy(in vector<float,4> src, out vector<float,4> dst) {
    dst = src;
}
"""

    result = compile_lockstep(source, verbose=False)

    particle_struct = result.entities["structs"][0]
    assert particle_struct["fields"][0]["type"] == "float[4]"
    assert particle_struct["fields"][1]["type"] == "vector<float,4>"
    assert particle_struct["fields"][2]["type"] == "matrix<float,4>[2]"


def test_compile_rejects_unknown_nested_declared_type_inside_generic():
    source = """
shader Bad(in vector<flaot,4> src, out vector<float,4> dst) {
    dst = dst;
}
"""

    with pytest.raises(LockstepCompileError) as exc_info:
        compile_lockstep(source, verbose=False)

    assert exc_info.value.phase == "semantic"
    assert [diag.code for diag in exc_info.value.errors] == ["LCK310"]
    assert exc_info.value.errors[0].message == "Unknown declared type 'vector<flaot,4>'."
