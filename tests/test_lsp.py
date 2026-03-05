from lockstep_compiler.lsp import (
    build_struct_member_index,
    compile_for_lsp,
    find_member_definition,
    provide_bind_completion_items,
)
from lockstep_compiler.errors import LockstepCompileError
from lockstep_compiler.models import LockstepDiagnostic


SOURCE = """
struct Vec3 { float x; float y; float z; };

shader Integrate(in Vec3 pos, out Vec3 out_pos, uniform float dt) {
    out_pos.x = pos.x + dt;
}

pipeline P {
    stream<Vec3, 32> src;
    stream<Vec3, 32> dst;

    bind {
        dst = Integrate(src, dst, 0.1);
    }
}
"""


def test_build_struct_member_index_tracks_fields():
    index = build_struct_member_index(SOURCE)

    assert "Vec3" in index
    assert set(index["Vec3"]) == {"x", "y", "z"}


def test_find_member_definition_resolves_struct_field():
    target_line = 4
    target_column = 11  # out_pos.x

    definition = find_member_definition(SOURCE, target_line, target_column)

    assert definition is not None
    assert definition.struct_name == "Vec3"
    assert definition.field_name == "x"


def test_provide_bind_completion_items_includes_routes_and_kernels():
    items = provide_bind_completion_items(SOURCE)

    assert "dst=Integrate(src,dst,0.1);" in items
    assert "Integrate(...)" in items


def test_compile_for_lsp_falls_back_to_errors_when_diagnostics_missing(monkeypatch):
    parse_errors = [
        LockstepDiagnostic(
            severity="error",
            code="LCK001",
            message="missing ';'",
            line=2,
            column=7,
            hint="Terminate the statement with a semicolon.",
        )
    ]

    def _raise_compile_error(*_args, **_kwargs):
        raise LockstepCompileError(parse_errors, diagnostics=[])

    monkeypatch.setattr("lockstep_compiler.lsp.compile_lockstep", _raise_compile_error)

    entities, diagnostics = compile_for_lsp("shader Broken(")

    assert entities == {}
    assert diagnostics == [
        {
            "severity": "error",
            "code": "LCK001",
            "message": "missing ';'",
            "line": 2,
            "column": 7,
            "hint": "Terminate the statement with a semicolon.",
        }
    ]


def test_compile_for_lsp_prefers_diagnostics_over_errors(monkeypatch):
    parse_errors = [
        LockstepDiagnostic(
            severity="error",
            code="LCK001",
            message="parser message",
            line=2,
            column=7,
            hint="parse hint",
        )
    ]
    semantic_diagnostics = [
        LockstepDiagnostic(
            severity="error",
            code="LCK210",
            message="semantic message",
            line=4,
            column=3,
            hint="semantic hint",
        )
    ]

    def _raise_compile_error(*_args, **_kwargs):
        raise LockstepCompileError(parse_errors, diagnostics=semantic_diagnostics)

    monkeypatch.setattr("lockstep_compiler.lsp.compile_lockstep", _raise_compile_error)

    _, diagnostics = compile_for_lsp("shader Broken(")

    assert diagnostics == [
        {
            "severity": "error",
            "code": "LCK210",
            "message": "semantic message",
            "line": 4,
            "column": 3,
            "hint": "semantic hint",
        }
    ]
