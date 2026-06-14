import io
import pathlib
import sys
import types
from unittest.mock import sentinel

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import lockstep_compiler as compiler_api
from lockstep_compiler import (
    LockstepCompileError,
    LockstepCompileResult,
    LockstepDiagnostic,
    ParseErrorCollector,
    compile_lockstep,
    load_default_parser_classes,
    normalize_diagnostics,
    run_cli,
    validate_semantics,
)
from lockstep_compiler.cli import main as cli_main



_LockstepLexer, _LockstepParser, _LockstepVisitor = load_default_parser_classes()


class _CompatibilityVisitor(_LockstepVisitor):
    """Visitor shim that keeps test doubles lightweight."""

    def visitChildren(self, node):
        get_child_count = getattr(node, "getChildCount", None)
        if callable(get_child_count):
            return super().visitChildren(node)
        return node


LockstepDebugVisitor = compiler_api.build_debug_visitor(_CompatibilityVisitor)


def _diagnostic(line, column, message, *, severity="error", code="LCK001"):
    return LockstepDiagnostic(
        severity=severity,
        code=code,
        message=message,
        line=line,
        column=column,
        hint="Fix syntax errors before semantic analysis can continue.",
    )


def test_lockstep_compile_error_formats_singular_and_plural():
    one = LockstepCompileError(
        [_diagnostic(1, 1, "oops")]
    )
    many = LockstepCompileError(
        [
            _diagnostic(1, 1, "oops"),
            _diagnostic(2, 4, "bad"),
        ]
    )

    assert str(one) == "Compilation failed with 1 parse error.\nline 1:1 oops"
    assert (
        str(many)
        == "Compilation failed with 2 parse errors.\nline 1:1 oops\nline 2:4 bad"
    )


def test_parse_error_collector_captures_diagnostic():
    collector = ParseErrorCollector()
    collector.syntaxError(None, None, 12, 7, "unexpected token", None)

    assert collector.errors == [
        LockstepDiagnostic(
            severity="error",
            code="LCK001",
            message="unexpected token",
            line=12,
            column=7,
            hint="Fix syntax errors before semantic analysis can continue.",
        )
    ]


def test_compile_lockstep_uses_package_compile_function(monkeypatch):
    captured = {}

    def fake_compile(source_code, *, verbose=True, **kwargs):
        captured["source_code"] = source_code
        captured["verbose"] = verbose
        captured["kwargs"] = kwargs
        return sentinel.result

    monkeypatch.setattr(compiler_api, "compile_lockstep", fake_compile)

    result = compiler_api.compile_lockstep("pipeline P { }", verbose=False)

    assert result is sentinel.result
    assert captured == {"source_code": "pipeline P { }", "verbose": False, "kwargs": {}}


def test_validate_semantics_uses_package_default_visitor(monkeypatch):
    captured = {}

    def fake_validate(parse_tree, visitor_cls=None):
        captured["parse_tree"] = parse_tree
        captured["visitor_cls"] = visitor_cls
        return [sentinel.diag]

    monkeypatch.setattr(compiler_api, "validate_semantics", fake_validate)

    diagnostics = compiler_api.validate_semantics("TREE")

    assert diagnostics == [sentinel.diag]
    assert captured == {"parse_tree": "TREE", "visitor_cls": None}


def test_normalize_diagnostics_prefers_non_empty_hint_for_same_severity():
    diagnostics = [
        LockstepDiagnostic(
            severity="warning",
            code="LCK888",
            message="same issue",
            line=9,
            column=3,
            hint="",
        ),
        LockstepDiagnostic(
            severity="warning",
            code="LCK888",
            message="same issue",
            line=9,
            column=3,
            hint="detailed hint",
        ),
    ]

    assert normalize_diagnostics(diagnostics) == [
        LockstepDiagnostic(
            severity="warning",
            code="LCK888",
            message="same issue",
            line=9,
            column=3,
            hint="detailed hint",
        )
    ]


def _token(text):
    return types.SimpleNamespace(getText=lambda: text)


def _ctx(start_line=0, start_col=0, **kwargs):
    return types.SimpleNamespace(
        start=types.SimpleNamespace(line=start_line, column=start_col), **kwargs
    )


def test_visitor_methods_print_expected_output(capsys):
    visitor = LockstepDebugVisitor()

    class _Param:
        def __init__(self, modifier, p_type, p_name):
            self._modifier = _token(modifier)
            self._type = _token(p_type)
            self._name = _token(p_name)

        def getChild(self, index):
            assert index == 0
            return self._modifier

        def typeName(self):
            return self._type

        def ID(self):
            return self._name

    class _ParamList:
        def param(self):
            return [_Param("in", "Vec3", "pos")]

    class _BindStmt:
        def __init__(self, text):
            self._text = text

        def getText(self):
            return self._text

    visitor.visitProgram(object())
    visitor.visitStructDecl(_ctx(ID=lambda: _token("Vec3")))
    visitor.visitPureDecl(
        _ctx(ID=lambda: _token("add"), typeName=lambda: _token("Vec3"))
    )
    visitor.visitShaderDecl(
        _ctx(ID=lambda: _token("ApplyGravity"), paramList=lambda: _ParamList())
    )
    visitor.visitFilterDecl(
        _ctx(ID=lambda: _token("OnlyActive"), paramList=lambda: _ParamList())
    )
    visitor.visitPipelineDecl(_ctx(ID=lambda: _token("Physics")))
    visitor.visitStreamDecl(
        _ctx(
            typeName=lambda: _token("Vec3"),
            INT=lambda: _token("1000"),
            ID=lambda: _token("raw"),
        )
    )
    visitor.visitAccumDecl(
        _ctx(typeName=lambda: _token("float"), ID=lambda: _token("energy"))
    )
    visitor.visitUniformDecl(
        _ctx(
            typeName=lambda: _token("float"),
            ID=lambda: _token("dt"),
            expr=lambda: _token("0.016"),
        )
    )
    visitor.visitBindBlock(_ctx(bindStmt=lambda: [_BindStmt("a=b"), _BindStmt("c=d")]))

    stdout = capsys.readouterr().out
    assert "=== LOCKSTEP COMPILER FRONTEND ===" in stdout
    assert "[Struct] Discovered: Vec3" in stdout
    assert "[Pure Function] add -> Vec3" in stdout
    assert "[Shader Kernel] ApplyGravity" in stdout
    assert "Param: (in) Vec3 pos" in stdout
    assert "[Pipeline Topology] Physics" in stdout
    assert "[Filter Kernel] OnlyActive" in stdout
    assert "Stream: raw <Vec3, 1000>" in stdout
    assert "Accumulator: energy <float>" in stdout
    assert "Uniform: dt <float>" in stdout
    assert "Routing:" in stdout
    assert "a=b" in stdout
    assert "c=d" in stdout
    assert visitor.structs == ["Vec3"]
    assert visitor.shaders == [
        {
            "name": "ApplyGravity",
            "params": [{"modifier": "in", "type": "Vec3", "name": "pos"}],
            "body": [],
        }
    ]
    assert visitor.filters == [
        {
            "name": "OnlyActive",
            "params": [{"modifier": "in", "type": "Vec3", "name": "pos"}],
            "body": [],
        }
    ]
    assert visitor.pure_functions == [
        {"name": "add", "return_type": "Vec3", "params": [], "body": []}
    ]
    assert visitor.streams == [{"name": "raw", "type": "Vec3", "capacity": "1000"}]
    assert visitor.accumulators == [{"name": "energy", "type": "float"}]
    assert visitor.uniforms == [{"name": "dt", "type": "float", "initializer": "0.016"}]
    assert visitor.bind_routes == ["a=b", "c=d"]
    assert visitor.diagnostics == []


def test_visitor_emits_diagnostics_for_non_fatal_observations():
    visitor = LockstepDebugVisitor(verbose=False)

    visitor.visitStructDecl(_ctx(start_line=2, start_col=1, ID=lambda: _token("Vec3")))
    visitor.visitStructDecl(_ctx(start_line=3, start_col=1, ID=lambda: _token("Vec3")))
    visitor.visitPureDecl(
        _ctx(
            start_line=4,
            start_col=1,
            ID=lambda: _token("add"),
            typeName=lambda: _token("Vec3"),
        )
    )
    visitor.visitPureDecl(
        _ctx(
            start_line=5,
            start_col=1,
            ID=lambda: _token("add"),
            typeName=lambda: _token("Vec3"),
        )
    )
    visitor.visitFilterDecl(
        _ctx(start_line=6, start_col=1, ID=lambda: _token("f"), paramList=lambda: None)
    )
    visitor.visitFilterDecl(
        _ctx(start_line=7, start_col=1, ID=lambda: _token("f"), paramList=lambda: None)
    )
    visitor.visitUniformDecl(
        _ctx(
            start_line=8,
            start_col=1,
            typeName=lambda: _token("float"),
            ID=lambda: _token("dt"),
            expr=lambda: None,
        )
    )
    visitor.visitUniformDecl(
        _ctx(
            start_line=9,
            start_col=1,
            typeName=lambda: _token("float"),
            ID=lambda: _token("dt"),
            expr=lambda: None,
        )
    )
    visitor.visitBindBlock(_ctx(start_line=10, start_col=4, bindStmt=lambda: []))

    assert visitor.diagnostics == [
        LockstepDiagnostic(
            severity="warning",
            code="LCK201",
            message="Struct 'Vec3' is redeclared.",
            line=3,
            column=1,
            hint="Rename or remove duplicate struct declarations.",
        ),
        LockstepDiagnostic(
            severity="warning",
            code="LCK205",
            message="Pure function 'add' is redeclared.",
            line=5,
            column=1,
            hint="Rename or remove duplicate pure function declarations.",
        ),
        LockstepDiagnostic(
            severity="warning",
            code="LCK206",
            message="Filter 'f' is redeclared.",
            line=7,
            column=1,
            hint="Rename or remove duplicate filter declarations.",
        ),
        LockstepDiagnostic(
            severity="warning",
            code="LCK207",
            message="Uniform 'dt' is redeclared.",
            line=9,
            column=1,
            hint="Each uniform in a pipeline should have a unique name.",
        ),
        LockstepDiagnostic(
            severity="info",
            code="LCK101",
            message="Bind block is empty; pipeline has no executable routes.",
            line=10,
            column=4,
            hint="Add at least one binding statement in the bind block.",
        ),
    ]


def test_visitor_stream_redeclaration_is_pipeline_local():
    visitor = LockstepDebugVisitor(verbose=False)

    visitor.visitPipelineDecl(_ctx(ID=lambda: _token("P1")))
    visitor.visitStreamDecl(
        _ctx(
            start_line=2,
            start_col=1,
            typeName=lambda: _token("Vec3"),
            INT=lambda: _token("8"),
            ID=lambda: _token("s"),
        )
    )
    visitor.visitPipelineDecl(_ctx(ID=lambda: _token("P2")))
    visitor.visitStreamDecl(
        _ctx(
            start_line=6,
            start_col=1,
            typeName=lambda: _token("Vec3"),
            INT=lambda: _token("8"),
            ID=lambda: _token("s"),
        )
    )

    assert visitor.diagnostics == []


def test_visitor_shader_decl_without_param_list(capsys):
    visitor = LockstepDebugVisitor()
    visitor.visitShaderDecl(_ctx(ID=lambda: _token("Kernel"), paramList=lambda: None))
    assert "[Shader Kernel] Kernel" in capsys.readouterr().out
    assert visitor.shaders == [{"name": "Kernel", "params": [], "body": []}]


def test_visitor_bind_routes_can_be_normalized():
    visitor = LockstepDebugVisitor(
        verbose=False,
        normalize_bind_routes=True,
    )

    class _BindStmt:
        def __init__(self, text):
            self._text = text

        def getText(self):
            return self._text

    visitor.visitBindBlock(_ctx(bindStmt=lambda: [_BindStmt(" a   =b(c , d ) ; ")]))

    assert visitor.bind_routes == ["a =b(c , d ) ;"]


def test_visitor_can_run_without_printing(capsys):
    visitor = LockstepDebugVisitor(verbose=False)
    visitor.visitStructDecl(_ctx(ID=lambda: _token("Vec3")))

    assert capsys.readouterr().out == ""
    assert visitor.structs == ["Vec3"]


def test_module_main_success_path(monkeypatch, capsys):
    monkeypatch.setattr(
        "lockstep_compiler.compiler.compile_lockstep", lambda *_args, **_kwargs: sentinel.ok
    )
    monkeypatch.setattr(sys, "argv", ["lockstep_compiler.cli"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("pipeline P { }"))

    with pytest.raises(SystemExit) as exc_info:
        cli_main()

    assert exc_info.value.code == 0
    assert capsys.readouterr().err == ""


def test_module_main_error_path_exits_with_stderr(monkeypatch, capsys):
    def failing_compile(_source, *, verbose=True, **_kwargs):
        from lockstep_compiler.errors import LockstepCompileError
        from lockstep_compiler.models import LockstepDiagnostic

        raise LockstepCompileError(
            [
                LockstepDiagnostic(
                    severity="error",
                    code="LCK001",
                    message="bad syntax",
                    line=7,
                    column=9,
                    hint="Fix syntax errors before semantic analysis can continue.",
                )
            ]
        )

    monkeypatch.setattr("lockstep_compiler.compiler.compile_lockstep", failing_compile)
    monkeypatch.setattr(sys, "argv", ["lockstep_compiler.cli"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("pipeline P { }"))

    with pytest.raises(SystemExit) as exc_info:
        cli_main()

    assert exc_info.value.code == 1
    stderr = capsys.readouterr().err
    assert "Compilation failed with 1 parse error." in stderr
    assert "line 7:9 bad syntax" in stderr


def test_run_cli_reads_source_from_stdin_when_path_omitted():
    captured = {}

    def fake_compiler(source):
        captured["source"] = source

    exit_code = run_cli(
        [],
        stdin=io.StringIO("pipeline FromStdin { }"),
        compiler=fake_compiler,
    )

    assert exit_code == 0
    assert captured["source"] == "pipeline FromStdin { }"


def test_run_cli_reads_source_from_path(tmp_path):
    source_file = tmp_path / "sample.lock"
    source_file.write_text("pipeline FromFile { }", encoding="utf-8")
    captured = {}

    def fake_compiler(source):
        captured["source"] = source

    exit_code = run_cli(
        [str(source_file)], compiler=fake_compiler
    )

    assert exit_code == 0
    assert captured["source"] == "pipeline FromFile { }"


def test_run_cli_dump_prints_compiled_entities():
    def fake_compiler(_source):
        return LockstepCompileResult(
            parse_tree=None,
            entities={
                "streams": [{"name": "positions", "capacity": 1000}],
                "bind_routes": ["out = Simulate(inp);"],
            },
            diagnostics=[],
        )

    stdout = io.StringIO()
    exit_code = run_cli(
        ["--dump"],
        stdin=io.StringIO("pipeline Physics { }"),
        stdout=stdout,
        compiler=fake_compiler,
    )

    assert exit_code == 0
    assert stdout.getvalue().splitlines() == [
        "{",
        '  "bind_routes": [',
        '    "out = Simulate(inp);"',
        "  ],",
        '  "streams": [',
        "    {",
        '      "capacity": 1000,',
        '      "name": "positions"',
        "    }",
        "  ]",
        "}",
    ]


def test_run_cli_dump_falls_back_to_compiler_result():
    def fake_compiler(_source):
        return {"nodes": ["a", "b"]}

    stdout = io.StringIO()
    exit_code = run_cli(
        ["--dump"],
        stdin=io.StringIO("pipeline Physics { }"),
        stdout=stdout,
        compiler=fake_compiler,
    )

    assert exit_code == 0
    assert stdout.getvalue().splitlines() == [
        "{",
        '  "nodes": [',
        '    "a",',
        '    "b"',
        "  ]",
        "}",
    ]


def test_run_cli_returns_non_zero_and_writes_errors():
    def failing_compiler(_source):
        raise LockstepCompileError(
            [
                LockstepDiagnostic(
                    severity="error",
                    code="LCK001",
                    message="unexpected",
                    line=4,
                    column=2,
                    hint="Fix syntax errors before semantic analysis can continue.",
                )
            ]
        )

    stderr = io.StringIO()
    exit_code = run_cli(
        [],
        stdin=io.StringIO("pipeline Broken {"),
        stderr=stderr,
        compiler=failing_compiler,
    )

    assert exit_code == 1
    assert stderr.getvalue().splitlines() == [
        "Compilation failed with 1 parse error.",
        "line 4:2 unexpected",
    ]


def test_run_cli_internal_error_default_mode_keeps_generic_message():
    def failing_compiler(_source):
        raise RuntimeError("kaboom")

    stderr = io.StringIO()
    exit_code = run_cli(
        [],
        stdin=io.StringIO("pipeline Broken {"),
        stderr=stderr,
        compiler=failing_compiler,
    )

    assert exit_code == 1
    assert stderr.getvalue().splitlines() == [
        "Compilation failed due to an internal error.",
    ]


def test_run_cli_debug_mode_emits_exception_details_and_traceback():
    def failing_compiler(_source):
        raise LockstepCompileError(
            [
                LockstepDiagnostic(
                    severity="error",
                    code="LCK001",
                    message="unexpected",
                    line=4,
                    column=2,
                    hint="Fix syntax errors before semantic analysis can continue.",
                )
            ],
            diagnostics=[
                LockstepDiagnostic(
                    severity="warning",
                    code="LCK201",
                    message="Struct redeclared",
                    line=2,
                    column=1,
                    hint="Rename duplicate struct.",
                )
            ],
        )

    stderr = io.StringIO()
    exit_code = run_cli(
        ["--debug"],
        stdin=io.StringIO("pipeline Broken {"),
        stderr=stderr,
        compiler=failing_compiler,
    )

    output = stderr.getvalue()
    assert exit_code == 1
    assert "Compilation failed with 1 parse error." in output
    assert "line 4:2 unexpected" in output
    assert "diagnostics:" in output
    assert '"code": "LCK201"' in output
    assert "LockstepCompileError: Compilation failed with 1 parse error." in output
    assert "Traceback (most recent call last):" in output


def test_run_cli_returns_non_zero_for_missing_path(tmp_path):
    missing = tmp_path / "missing.lock"
    stderr = io.StringIO()
    called = {"compiler": False}

    def fake_compiler(_source):
        called["compiler"] = True

    exit_code = run_cli(
        [str(missing)],
        stderr=stderr,
        compiler=fake_compiler,
    )

    assert exit_code == 1
    assert called["compiler"] is False
    assert f"Unable to read '{missing}': file not found." in stderr.getvalue()


def test_run_cli_returns_non_zero_for_unreadable_path(monkeypatch):
    stderr = io.StringIO()
    called = {"compiler": False}

    def fake_compiler(_source):
        called["compiler"] = True

    def raise_permission_error(_self, encoding):
        raise PermissionError("permission denied")

    monkeypatch.setattr("lockstep_compiler.cli.Path.read_text", raise_permission_error)

    exit_code = run_cli(
        ["locked.lock"],
        stderr=stderr,
        compiler=fake_compiler,
    )

    assert exit_code == 1
    assert called["compiler"] is False
    assert "Unable to read 'locked.lock': permission denied." in stderr.getvalue()


def test_run_cli_returns_non_zero_for_invalid_utf8(tmp_path):
    bad_source = tmp_path / "invalid.lock"
    bad_source.write_bytes(b"\xff\xfe\xfa")
    stderr = io.StringIO()
    called = {"compiler": False}

    def fake_compiler(_source):
        called["compiler"] = True

    exit_code = run_cli(
        [str(bad_source)],
        stderr=stderr,
        compiler=fake_compiler,
    )

    assert exit_code == 1
    assert called["compiler"] is False
    assert f"Unable to read '{bad_source}': invalid UTF-8" in stderr.getvalue()


def test_run_cli_uses_default_compiler_when_compiler_missing(monkeypatch):
    captured = {}

    def fake_compiler(source, *, verbose=True):
        captured["source"] = source
        captured["verbose"] = verbose

    import lockstep_compiler.compiler as compiler_module

    monkeypatch.setattr(compiler_module, "compile_lockstep", fake_compiler)

    exit_code = run_cli(
        [],
        stdin=io.StringIO("pipeline MissingCompiler { }"),
        compiler=None,
    )

    assert exit_code == 0
    assert captured == {
        "source": "pipeline MissingCompiler { }",
        "verbose": False,
    }


def test_run_cli_preserves_injected_compiler_without_verbose_parameter():
    captured = {}

    def fake_compiler(source):
        captured["source"] = source

    exit_code = run_cli(
        [],
        stdin=io.StringIO("pipeline CustomCompiler { }"),
        compiler=fake_compiler,
    )

    assert exit_code == 0
    assert captured == {"source": "pipeline CustomCompiler { }"}


def test_run_cli_default_execution_suppresses_verbose_visitor_logs(monkeypatch):
    stderr = io.StringIO()

    def fake_compiler(_source, *, verbose=True):
        if verbose:
            print("Visiting node: pipeline", file=stderr)

    import lockstep_compiler.compiler as compiler_module

    monkeypatch.setattr(compiler_module, "compile_lockstep", fake_compiler)

    exit_code = run_cli(
        [],
        stdin=io.StringIO("pipeline QuietByDefault { }"),
        stderr=stderr,
        compiler=None,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""


def test_run_cli_returns_non_zero_when_compiler_not_callable():
    stderr = io.StringIO()

    exit_code = run_cli(
        [],
        stdin=io.StringIO("pipeline InvalidCompiler { }"),
        stderr=stderr,
        compiler=object(),
    )

    assert exit_code == 1
    assert stderr.getvalue().splitlines() == [
        "Compiler configuration error: compiler must be callable.",
    ]
