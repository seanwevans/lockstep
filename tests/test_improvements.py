"""Tests for new intrinsics, CLI flags, and LSP hover support."""

import io
import pathlib
import sys
import tempfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lockstep_compiler.cli import build_arg_parser, run_cli
from lockstep_compiler.ast import AstExprCall, AstExprVar, AstReturnStmt
from lockstep_compiler.codegen import emit_llvm_ir
from lockstep_compiler.lsp import provide_hover_info
from lockstep_compiler import compiler as compiler_module
from lockstep_compiler.compiler import FrontendLimits, compile_lockstep
from lockstep_compiler.errors import LockstepCompileError
from lockstep_compiler.prelude import load_intrinsics


# ---------------------------------------------------------------------------
# Prelude intrinsic declarations
# ---------------------------------------------------------------------------


def test_prelude_declares_min():
    intrinsics = load_intrinsics()
    assert "min" in intrinsics
    assert intrinsics["min"].return_type == "float"
    assert len(intrinsics["min"].params) == 2


def test_prelude_declares_abs():
    intrinsics = load_intrinsics()
    assert "abs" in intrinsics
    assert intrinsics["abs"].return_type == "float"
    assert len(intrinsics["abs"].params) == 1


def test_prelude_declares_sign():
    intrinsics = load_intrinsics()
    assert "sign" in intrinsics
    assert intrinsics["sign"].return_type == "float"
    assert len(intrinsics["sign"].params) == 1


def test_prelude_declares_smoothstep():
    intrinsics = load_intrinsics()
    assert "smoothstep" in intrinsics
    assert intrinsics["smoothstep"].return_type == "float"
    assert len(intrinsics["smoothstep"].params) == 3


# ---------------------------------------------------------------------------
# Codegen intrinsic lowering
# ---------------------------------------------------------------------------


def _intrinsic_defs():
    """Return intrinsic pure function declarations needed by codegen."""
    intrinsics = load_intrinsics()
    return [
        {
            "name": decl.name,
            "return_type": decl.return_type,
            "params": [
                {"type": param.type_name, "name": param.name} for param in decl.params
            ],
            "body": [],
            "intrinsic": True,
        }
        for decl in intrinsics.values()
    ]


_CODEGEN_ENTITIES_BASE = {
    "structs": [],
    "shaders": [],
    "filters": [],
    "streams": [],
    "accumulators": [],
    "uniforms": [],
    "bind_routes": [],
    "bind_routes_ir": [],
}


def _entities_with_pure(name, params, body, return_type="float"):
    return {
        **_CODEGEN_ENTITIES_BASE,
        "pure_functions": [
            {
                "name": name,
                "return_type": return_type,
                "params": params,
                "body_ast": body,
                "intrinsic": False,
            },
            *_intrinsic_defs(),
        ],
    }


def test_codegen_min_intrinsic():
    entities = _entities_with_pure(
        "test_min",
        [{"type": "float", "name": "a"}, {"type": "float", "name": "b"}],
        [
            AstReturnStmt(
                value=AstExprCall(
                    name="min", args=(AstExprVar(path=("a",)), AstExprVar(path=("b",)))
                )
            )
        ],
    )
    ir_text = emit_llvm_ir(entities)
    assert "llvm.minnum.f32" in ir_text


def test_codegen_max_intrinsic():
    entities = _entities_with_pure(
        "test_max",
        [{"type": "float", "name": "a"}, {"type": "float", "name": "b"}],
        [
            AstReturnStmt(
                value=AstExprCall(
                    name="max", args=(AstExprVar(path=("a",)), AstExprVar(path=("b",)))
                )
            )
        ],
    )
    ir_text = emit_llvm_ir(entities)
    assert "llvm.maxnum.f32" in ir_text


def test_codegen_abs_intrinsic():
    entities = _entities_with_pure(
        "test_abs",
        [{"type": "float", "name": "x"}],
        [AstReturnStmt(value=AstExprCall(name="abs", args=(AstExprVar(path=("x",)),)))],
    )
    ir_text = emit_llvm_ir(entities)
    assert "llvm.fabs.f32" in ir_text


def test_codegen_sign_intrinsic():
    entities = _entities_with_pure(
        "test_sign",
        [{"type": "float", "name": "x"}],
        [
            AstReturnStmt(
                value=AstExprCall(name="sign", args=(AstExprVar(path=("x",)),))
            )
        ],
    )
    ir_text = emit_llvm_ir(entities)
    assert "sign_pos" in ir_text or "sign" in ir_text


def test_codegen_smoothstep_intrinsic():
    entities = _entities_with_pure(
        "test_smoothstep",
        [
            {"type": "float", "name": "e0"},
            {"type": "float", "name": "e1"},
            {"type": "float", "name": "x"},
        ],
        [
            AstReturnStmt(
                value=AstExprCall(
                    name="smoothstep",
                    args=(
                        AstExprVar(path=("e0",)),
                        AstExprVar(path=("e1",)),
                        AstExprVar(path=("x",)),
                    ),
                )
            )
        ],
    )
    ir_text = emit_llvm_ir(entities)
    assert "smoothstep" in ir_text


# ---------------------------------------------------------------------------
# CLI: --version flag
# ---------------------------------------------------------------------------


def test_cli_version_flag():
    stdout = io.StringIO()
    code = run_cli(["--version"], stdout=stdout, stderr=io.StringIO())
    assert code == 0
    output = stdout.getvalue()
    assert "lockstepc" in output


# ---------------------------------------------------------------------------
# CLI: --emit-ir flag
# ---------------------------------------------------------------------------


_TEST_SOURCE = """\
struct Vec3 { float x; float y; float z; };

pure float magnitude(float a, float b) {
    return a + b;
}
"""


def test_cli_emit_ir():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lock", delete=False) as f:
        f.write(_TEST_SOURCE)
        f.flush()
        stdout = io.StringIO()
        code = run_cli([f.name, "--emit-ir"], stdout=stdout, stderr=io.StringIO())
    assert code == 0
    output = stdout.getvalue()
    assert "define" in output or "module" in output.lower()


def test_cli_emit_header():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lock", delete=False) as f:
        f.write(_TEST_SOURCE)
        f.flush()
        stdout = io.StringIO()
        code = run_cli([f.name, "--emit-header"], stdout=stdout, stderr=io.StringIO())
    assert code == 0
    output = stdout.getvalue()
    assert "LOCKSTEP_GENERATED_H" in output


# ---------------------------------------------------------------------------
# CLI: arg parser has new flags
# ---------------------------------------------------------------------------


def test_arg_parser_has_emit_ir():
    parser = build_arg_parser()
    args = parser.parse_args(["file.lock", "--emit-ir"])
    assert args.emit_ir is True


def test_arg_parser_has_emit_header():
    parser = build_arg_parser()
    args = parser.parse_args(["file.lock", "--emit-header"])
    assert args.emit_header is True


def test_arg_parser_has_version():
    parser = build_arg_parser()
    args = parser.parse_args(["--version"])
    assert args.version is True


def test_arg_parser_rejects_multiple_output_modes():
    parser = build_arg_parser()
    try:
        parser.parse_args(["--dump", "--simulate"])
    except SystemExit as err:
        assert err.code == 2
    else:
        raise AssertionError(
            "Expected parser to reject multiple output-producing flags"
        )


def test_arg_parser_has_report():
    parser = build_arg_parser()
    args = parser.parse_args(["--report"])
    assert args.report is True


# ---------------------------------------------------------------------------
# LSP hover
# ---------------------------------------------------------------------------


_HOVER_SOURCE = """\
struct Vec3 { float x; float y; float z; };

shader Integrate(in Vec3 pos, out Vec3 out_pos, uniform float dt) {
    out_pos.x = pos.x + dt;
}

pipeline P {
    stream<Vec3, 32> src;
    stream<Vec3, 32> dst;
    uniform float dt = 0.1;

    bind {
        dst = Integrate(src, dst, dt);
    }
}
"""


def test_hover_variable_shows_type():
    # "pos" on line 3, first occurrence
    result = provide_hover_info(_HOVER_SOURCE, 3, 17)
    assert result is not None
    assert "Vec3" in result


def test_hover_struct_name():
    # "Vec3" on line 0
    result = provide_hover_info(_HOVER_SOURCE, 0, 7)
    assert result is not None
    assert "struct" in result
    assert "Vec3" in result


def test_hover_member_access():
    # "out_pos.x" on line 3
    result = provide_hover_info(_HOVER_SOURCE, 3, 8)
    assert result is not None
    assert "field" in result
    assert "Vec3" in result


def test_hover_shader_name():
    # "Integrate" on line 2
    result = provide_hover_info(_HOVER_SOURCE, 2, 10)
    assert result is not None
    assert "shader" in result


def test_hover_returns_none_for_empty_line():
    source = "\n\n\n"
    result = provide_hover_info(source, 0, 0)
    assert result is None


# ---------------------------------------------------------------------------
# Frontend limits validation and CLI parsing
# ---------------------------------------------------------------------------

_LIMIT_SOURCE = "pipeline P { bind { } }"
_NESTED_EXPR_SOURCE = """pure float nested(float v) {
    return (((((((v)))))));
}

pipeline P {
    bind {
    }
}
"""



@pytest.mark.parametrize(
    ("limit_name", "value"),
    [
        ("max_source_bytes", -1),
        ("parse_timeout_ms", -1),
        ("max_expression_nesting", -1),
    ],
)
def test_compile_lockstep_rejects_negative_frontend_limits(limit_name, value):
    with pytest.raises(LockstepCompileError) as exc_info:
        compile_lockstep(_LIMIT_SOURCE, frontend_limits=FrontendLimits(**{limit_name: value}))

    assert exc_info.value.errors[0].code == "LCK006"
    assert limit_name in exc_info.value.errors[0].message


def test_compile_lockstep_zero_disables_source_size_limit():
    compile_lockstep(
        _LIMIT_SOURCE,
        frontend_limits=FrontendLimits(max_source_bytes=0),
    )


def test_compile_lockstep_zero_disables_expression_nesting_limit():
    compile_lockstep(
        _NESTED_EXPR_SOURCE,
        frontend_limits=FrontendLimits(max_expression_nesting=0),
    )


def test_compile_lockstep_zero_disables_parse_timeout(monkeypatch):
    class StubLexer:
        def __init__(self, input_stream):
            self.input_stream = input_stream

        def removeErrorListeners(self):
            pass

        def addErrorListener(self, listener):
            pass

    class StubParser:
        def __init__(self, stream):
            self._stream = stream
            self._listeners = []

        def removeErrorListeners(self):
            self._listeners = []

        def addErrorListener(self, listener):
            self._listeners.append(listener)

        def program(self):
            return "TREE"

    class StubVisitor:
        def visit(self, _tree):
            return None

    monkeypatch.setattr(
        compiler_module,
        "_load_default_parser_classes",
        lambda: (StubLexer, StubParser, StubVisitor),
    )

    compile_lockstep(
        _LIMIT_SOURCE,
        frontend_limits=FrontendLimits(parse_timeout_ms=0),
    )


@pytest.mark.parametrize(
    ("frontend_limits", "error_code"),
    [
        (FrontendLimits(max_source_bytes=8), "LCK003"),
        (FrontendLimits(max_expression_nesting=2), "LCK005"),
    ],
)
def test_compile_lockstep_positive_frontend_limits_are_enforced(frontend_limits, error_code):
    source = _NESTED_EXPR_SOURCE if error_code == "LCK005" else _LIMIT_SOURCE
    with pytest.raises(LockstepCompileError) as exc_info:
        compile_lockstep(source, frontend_limits=frontend_limits)

    assert exc_info.value.errors[0].code == error_code


def test_compile_lockstep_positive_parse_timeout_is_enforced(monkeypatch):
    class StubLexer:
        def __init__(self, input_stream):
            self.input_stream = input_stream

        def removeErrorListeners(self):
            pass

        def addErrorListener(self, listener):
            pass

    class StubParser:
        def __init__(self, stream):
            self._stream = stream
            self._listeners = []

        def removeErrorListeners(self):
            self._listeners = []

        def addErrorListener(self, listener):
            self._listeners.append(listener)

        def program(self):
            self._stream.LT(1)
            return "TREE"

    class StubVisitor:
        def visit(self, _tree):
            return None

    monkeypatch.setattr(
        compiler_module,
        "_load_default_parser_classes",
        lambda: (StubLexer, StubParser, StubVisitor),
    )
    monotonic_values = iter([0.0, 0.002])
    monkeypatch.setattr(compiler_module.time, "monotonic", lambda: next(monotonic_values))

    with pytest.raises(LockstepCompileError) as exc_info:
        compile_lockstep(
            _LIMIT_SOURCE,
            frontend_limits=FrontendLimits(parse_timeout_ms=1),
        )

    assert exc_info.value.errors[0].code == "LCK004"


@pytest.mark.parametrize(
    "flag",
    ["--max-source-bytes", "--parse-timeout-ms", "--max-expr-nesting"],
)
def test_cli_rejects_negative_limit_values(flag, capsys):
    with pytest.raises(SystemExit) as exc_info:
        run_cli(["--simulate", flag, "-1"], stdin=io.StringIO(_LIMIT_SOURCE))

    assert exc_info.value.code == 2
    assert "must be >= 0 (0 disables the limit)" in capsys.readouterr().err
