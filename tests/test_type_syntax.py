import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lockstep_compiler import compile_lockstep
from lockstep_compiler.errors import LockstepCompileError


def test_composite_type_syntax_supports_arrays_and_generic_wrappers():
    source = """
struct Particle {
    float mass;
};

struct Payload {
    Particle[4] neighbors;
    vector<float,4> weights;
    matrix<vector<Particle,4>,4> transforms;
};

pure vector<Particle,4> passthrough(vector<Particle,4> value) {
    return value;
}

pipeline Stage {
    uniform matrix<vector<Particle,4>,4> frame;
    bind { }
}
"""

    result = compile_lockstep(source, verbose=False)

    assert all(diag.severity != "error" for diag in result.diagnostics)


def test_composite_type_syntax_rejects_unknown_nested_type_reference():
    source = """
pipeline Stage {
    uniform vector<MissingType,4> frame;
    bind { }
}
"""

    with pytest.raises(LockstepCompileError) as exc_info:
        compile_lockstep(source, verbose=False)

    assert exc_info.value.phase == "semantic"
    assert [diag.code for diag in exc_info.value.errors] == ["LCK310"]
    assert (
        "Unknown declared type 'MissingType' in 'vector<MissingType,4>'"
        in exc_info.value.errors[0].message
    )


def test_local_var_declaration_requires_explicit_type_annotation():
    source = """
shader Stage() {
    localValue;
}

pipeline Main {
    bind { }
}
"""

    with pytest.raises(LockstepCompileError) as exc_info:
        compile_lockstep(source, verbose=False)

    assert exc_info.value.phase == "parse"
    assert [diag.code for diag in exc_info.value.errors] == ["LCK001"]
    assert "no viable alternative" in exc_info.value.errors[0].message


def test_dependency_declarations_and_string_literals_compile_successfully():
    source = """
import "core/math.lock";
#include "runtime/platform.lock";

pure string asset_label(string src) {
    string local = "asset://textures/noise";
    return local;
}

pipeline Main {
    uniform string title = "frame begin";
    bind { }
}
"""

    result = compile_lockstep(source, verbose=False)

    assert all(diag.severity != "error" for diag in result.diagnostics)
    assert any(uniform["type"] == "string" for uniform in result.entities["uniforms"])
    assert "private unnamed_addr constant" in result.llvm_ir
    assert "c\"asset://textures/noise\\00\"" in result.llvm_ir


def test_uint_and_double_are_supported_as_primitive_declared_types():
    source = """
pure uint advance(uint value) {
    return value + uint(1);
}

pure double amplify(double value) {
    return value * double(2.0);
}

pipeline Main {
    uniform uint frame = uint(0);
    uniform double exposure = double(1.0);
    bind { }
}
"""

    result = compile_lockstep(source, verbose=False)

    assert all(diag.severity != "error" for diag in result.diagnostics)
    assert any(uniform["type"] == "uint" for uniform in result.entities["uniforms"])
    assert any(uniform["type"] == "double" for uniform in result.entities["uniforms"])
