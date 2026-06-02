import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lockstep_compiler import compile_lockstep
from lockstep_compiler.errors import LockstepCompileError


def test_complex_pipeline_bind_failures_accumulate_across_multi_node_bind_block():
    source = """
struct Vec3 {
    float x;
};

shader Scatter(in Vec3 inp, out Vec3 outp) {
    outp = inp;
}

shader Integrate(in Vec3 inp, accum float energy) {
    Vec3 local = inp;
}

pipeline StressTopology {
    stream<Vec3, 32> source;
    stream<Vec3, 32> stage;
    accumulator<float> energy;
    uniform int dt;

    bind {
        stage = Scatter(source, stage);
        stage = MissingKernel(stage);
        source = Integrate(stage, dt);
        uniform float folded = fold sum(stage);
    }
}
"""

    with pytest.raises(LockstepCompileError) as exc_info:
        compile_lockstep(source, verbose=False)

    assert exc_info.value.phase == "semantic"
    assert [diag.code for diag in exc_info.value.errors] == [
        "LCK304",
        "LCK305",
        "LCK308",
        "LCK309",
        "LCK403",
    ]
    assert "Undefined shader/filter 'MissingKernel'" in exc_info.value.errors[0].message
    assert (
        "Type mismatch for argument 'dt' in 'Integrate'"
        in exc_info.value.errors[1].message
    )
    assert "parameter 'energy' requires accum" in exc_info.value.errors[2].message
    assert "kernel has no out parameter" in exc_info.value.errors[3].message
    assert (
        "must reference an accumulator, got stream" in exc_info.value.errors[4].message
    )


def test_multi_pipeline_topology_reports_failures_without_short_circuiting_after_first_error():
    source = """
struct Vec3 {
    float x;
};

shader Blend(in Vec3 lhs, in Vec3 rhs, out Vec3 outp) {
    outp = lhs;
}

pipeline StageA {
    stream<Vec3, 8> a0;
    stream<Vec3, 8> a1;

    bind {
        a1 = Blend(a0, a0, a1);
    }
}

pipeline StageB {
    stream<Vec3, 8> b0;
    uniform float dt;

    bind {
        b0 = Blend(a0, b0, b0);
        uniform int energy_total = fold sum(dt);
    }
}
"""

    with pytest.raises(LockstepCompileError) as exc_info:
        compile_lockstep(source, verbose=False)

    assert exc_info.value.phase == "semantic"
    assert [diag.code for diag in exc_info.value.errors] == [
        "LCK301",
        "LCK403",
    ]
    assert "Undefined identifier 'a0'" in exc_info.value.errors[0].message
    assert (
        "must reference an accumulator, got uniform" in exc_info.value.errors[1].message
    )
