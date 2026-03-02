import importlib
import io
import pathlib
import runpy
import sys
import types
from unittest.mock import sentinel

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from lockstep_compiler.models import SemanticKernelParam, SemanticStructField, SemanticSymbol

@pytest.fixture
def debug_compiler_module(monkeypatch):
    sys.modules.pop("debug_compiler", None)
    return importlib.import_module("debug_compiler")



def _diagnostic(module, line, column, message, *, severity="error", code="LCK001"):
    return module.LockstepDiagnostic(
        severity=severity,
        code=code,
        message=message,
        line=line,
        column=column,
        hint="Fix syntax errors before semantic analysis can continue.",
    )


def test_lockstep_compile_error_formats_singular_and_plural(debug_compiler_module):
    one = debug_compiler_module.LockstepCompileError(
        [_diagnostic(debug_compiler_module, 1, 1, "oops")]
    )
    many = debug_compiler_module.LockstepCompileError(
        [
            _diagnostic(debug_compiler_module, 1, 1, "oops"),
            _diagnostic(debug_compiler_module, 2, 4, "bad"),
        ]
    )

    assert str(one) == "Compilation failed with 1 parse error.\nline 1:1 oops"
    assert (
        str(many)
        == "Compilation failed with 2 parse errors.\nline 1:1 oops\nline 2:4 bad"
    )


def test_parse_error_collector_captures_diagnostic(debug_compiler_module):
    collector = debug_compiler_module.ParseErrorCollector()
    collector.syntaxError(None, None, 12, 7, "unexpected token", None)

    assert collector.errors == [
        debug_compiler_module.LockstepDiagnostic(
            severity="error",
            code="LCK001",
            message="unexpected token",
            line=12,
            column=7,
            hint="Fix syntax errors before semantic analysis can continue.",
        )
    ]


def test_compile_lockstep_uses_package_compile_function(debug_compiler_module, monkeypatch):
    captured = {}

    def fake_compile(source_code, *, verbose=True, **kwargs):
        captured["source_code"] = source_code
        captured["verbose"] = verbose
        captured["kwargs"] = kwargs
        return sentinel.result

    monkeypatch.setattr(debug_compiler_module, "compile_lockstep", fake_compile)

    result = debug_compiler_module.compile_lockstep("pipeline P { }", verbose=False)

    assert result is sentinel.result
    assert captured == {"source_code": "pipeline P { }", "verbose": False, "kwargs": {}}


def test_validate_semantics_uses_package_default_visitor(debug_compiler_module, monkeypatch):
    captured = {}

    def fake_validate(parse_tree, visitor_cls=None):
        captured["parse_tree"] = parse_tree
        captured["visitor_cls"] = visitor_cls
        return [sentinel.diag]

    monkeypatch.setattr(debug_compiler_module, "validate_semantics", fake_validate)

    diagnostics = debug_compiler_module.validate_semantics("TREE")

    assert diagnostics == [sentinel.diag]
    assert captured == {"parse_tree": "TREE", "visitor_cls": None}


def test_normalize_diagnostics_prefers_non_empty_hint_for_same_severity(
    debug_compiler_module,
):
    diagnostics = [
        debug_compiler_module.LockstepDiagnostic(
            severity="warning",
            code="LCK888",
            message="same issue",
            line=9,
            column=3,
            hint="",
        ),
        debug_compiler_module.LockstepDiagnostic(
            severity="warning",
            code="LCK888",
            message="same issue",
            line=9,
            column=3,
            hint="detailed hint",
        ),
    ]

    assert debug_compiler_module.normalize_diagnostics(diagnostics) == [
        debug_compiler_module.LockstepDiagnostic(
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
    return types.SimpleNamespace(start=types.SimpleNamespace(line=start_line, column=start_col), **kwargs)


def test_visitor_methods_print_expected_output(debug_compiler_module, capsys):
    visitor = debug_compiler_module.LockstepDebugVisitor()

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
    visitor.visitPureDecl(_ctx(ID=lambda: _token("add"), typeName=lambda: _token("Vec3")))
    visitor.visitShaderDecl(_ctx(ID=lambda: _token("ApplyGravity"), paramList=lambda: _ParamList()))
    visitor.visitFilterDecl(_ctx(ID=lambda: _token("OnlyActive"), paramList=lambda: _ParamList()))
    visitor.visitPipelineDecl(_ctx(ID=lambda: _token("Physics")))
    visitor.visitStreamDecl(
        _ctx(
            typeName=lambda: _token("Vec3"),
            INT=lambda: _token("1000"),
            ID=lambda: _token("raw"),
        )
    )
    visitor.visitAccumDecl(_ctx(typeName=lambda: _token("float"), ID=lambda: _token("energy")))
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
        }
    ]
    assert visitor.filters == [
        {
            "name": "OnlyActive",
            "params": [{"modifier": "in", "type": "Vec3", "name": "pos"}],
        }
    ]
    assert visitor.pure_functions == [{"name": "add", "return_type": "Vec3"}]
    assert visitor.streams == [{"name": "raw", "type": "Vec3", "capacity": "1000"}]
    assert visitor.accumulators == [{"name": "energy", "type": "float"}]
    assert visitor.uniforms == [{"name": "dt", "type": "float", "initializer": "0.016"}]
    assert visitor.bind_routes == ["a=b", "c=d"]
    assert visitor.diagnostics == []


def test_visitor_emits_diagnostics_for_non_fatal_observations(debug_compiler_module):
    visitor = debug_compiler_module.LockstepDebugVisitor(verbose=False)

    visitor.visitStructDecl(_ctx(start_line=2, start_col=1, ID=lambda: _token("Vec3")))
    visitor.visitStructDecl(_ctx(start_line=3, start_col=1, ID=lambda: _token("Vec3")))
    visitor.visitPureDecl(
        _ctx(start_line=4, start_col=1, ID=lambda: _token("add"), typeName=lambda: _token("Vec3"))
    )
    visitor.visitPureDecl(
        _ctx(start_line=5, start_col=1, ID=lambda: _token("add"), typeName=lambda: _token("Vec3"))
    )
    visitor.visitFilterDecl(_ctx(start_line=6, start_col=1, ID=lambda: _token("f"), paramList=lambda: None))
    visitor.visitFilterDecl(_ctx(start_line=7, start_col=1, ID=lambda: _token("f"), paramList=lambda: None))
    visitor.visitUniformDecl(
        _ctx(start_line=8, start_col=1, typeName=lambda: _token("float"), ID=lambda: _token("dt"), expr=lambda: None)
    )
    visitor.visitUniformDecl(
        _ctx(start_line=9, start_col=1, typeName=lambda: _token("float"), ID=lambda: _token("dt"), expr=lambda: None)
    )
    visitor.visitBindBlock(_ctx(start_line=10, start_col=4, bindStmt=lambda: []))

    assert visitor.diagnostics == [
        debug_compiler_module.LockstepDiagnostic(
            severity="warning",
            code="LCK201",
            message="Struct 'Vec3' is redeclared.",
            line=3,
            column=1,
            hint="Rename or remove duplicate struct declarations.",
        ),
        debug_compiler_module.LockstepDiagnostic(
            severity="warning",
            code="LCK205",
            message="Pure function 'add' is redeclared.",
            line=5,
            column=1,
            hint="Rename or remove duplicate pure function declarations.",
        ),
        debug_compiler_module.LockstepDiagnostic(
            severity="warning",
            code="LCK206",
            message="Filter 'f' is redeclared.",
            line=7,
            column=1,
            hint="Rename or remove duplicate filter declarations.",
        ),
        debug_compiler_module.LockstepDiagnostic(
            severity="warning",
            code="LCK207",
            message="Uniform 'dt' is redeclared.",
            line=9,
            column=1,
            hint="Each uniform in a pipeline should have a unique name.",
        ),
        debug_compiler_module.LockstepDiagnostic(
            severity="info",
            code="LCK101",
            message="Bind block is empty; pipeline has no executable routes.",
            line=10,
            column=4,
            hint="Add at least one binding statement in the bind block.",
        ),
    ]


def test_visitor_stream_redeclaration_is_pipeline_local(debug_compiler_module):
    visitor = debug_compiler_module.LockstepDebugVisitor(verbose=False)

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


def test_visitor_shader_decl_without_param_list(debug_compiler_module, capsys):
    visitor = debug_compiler_module.LockstepDebugVisitor()
    visitor.visitShaderDecl(_ctx(ID=lambda: _token("Kernel"), paramList=lambda: None))
    assert "[Shader Kernel] Kernel" in capsys.readouterr().out
    assert visitor.shaders == [{"name": "Kernel", "params": []}]


def test_visitor_bind_routes_can_be_normalized(debug_compiler_module):
    visitor = debug_compiler_module.LockstepDebugVisitor(
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


def test_visitor_can_run_without_printing(debug_compiler_module, capsys):
    visitor = debug_compiler_module.LockstepDebugVisitor(verbose=False)
    visitor.visitStructDecl(_ctx(ID=lambda: _token("Vec3")))

    assert capsys.readouterr().out == ""
    assert visitor.structs == ["Vec3"]


def test_module_main_success_path(monkeypatch, capsys):
    monkeypatch.setattr("lockstep_compiler.compile_lockstep", lambda *_args, **_kwargs: sentinel.ok)
    monkeypatch.setattr(sys, "argv", ["debug_compiler.py"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("pipeline P { }"))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("debug_compiler", run_name="__main__")

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

    monkeypatch.setattr("lockstep_compiler.compile_lockstep", failing_compile)
    monkeypatch.setattr(sys, "argv", ["debug_compiler.py"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("pipeline P { }"))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("debug_compiler", run_name="__main__")

    assert exc_info.value.code == 1
    stderr = capsys.readouterr().err
    assert "Compilation failed with 1 parse error." in stderr
    assert "line 7:9 bad syntax" in stderr


def test_run_cli_reads_source_from_stdin_when_path_omitted(debug_compiler_module):
    captured = {}

    def fake_compiler(source):
        captured["source"] = source

    exit_code = debug_compiler_module.run_cli(
        [],
        stdin=io.StringIO("pipeline FromStdin { }"),
        compiler=fake_compiler,
    )

    assert exit_code == 0
    assert captured["source"] == "pipeline FromStdin { }"


def test_run_cli_reads_source_from_path(debug_compiler_module, tmp_path):
    source_file = tmp_path / "sample.lock"
    source_file.write_text("pipeline FromFile { }", encoding="utf-8")
    captured = {}

    def fake_compiler(source):
        captured["source"] = source

    exit_code = debug_compiler_module.run_cli(
        [str(source_file)], compiler=fake_compiler
    )

    assert exit_code == 0
    assert captured["source"] == "pipeline FromFile { }"


def test_run_cli_dump_prints_compiled_entities(debug_compiler_module):
    def fake_compiler(_source):
        return debug_compiler_module.LockstepCompileResult(
            parse_tree=None,
            entities={
                "streams": [{"name": "positions", "capacity": 1000}],
                "bind_routes": ["out = Simulate(inp);"],
            },
            diagnostics=[],
        )

    stdout = io.StringIO()
    exit_code = debug_compiler_module.run_cli(
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


def test_run_cli_dump_falls_back_to_compiler_result(debug_compiler_module):
    def fake_compiler(_source):
        return {"nodes": ["a", "b"]}

    stdout = io.StringIO()
    exit_code = debug_compiler_module.run_cli(
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


def test_run_cli_returns_non_zero_and_writes_errors(debug_compiler_module):
    def failing_compiler(_source):
        raise debug_compiler_module.LockstepCompileError(
            [
                debug_compiler_module.LockstepDiagnostic(
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
    exit_code = debug_compiler_module.run_cli(
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


def test_run_cli_internal_error_default_mode_keeps_generic_message(debug_compiler_module):
    def failing_compiler(_source):
        raise RuntimeError("kaboom")

    stderr = io.StringIO()
    exit_code = debug_compiler_module.run_cli(
        [],
        stdin=io.StringIO("pipeline Broken {"),
        stderr=stderr,
        compiler=failing_compiler,
    )

    assert exit_code == 1
    assert stderr.getvalue().splitlines() == [
        "Compilation failed due to an internal error.",
    ]


def test_run_cli_debug_mode_emits_exception_details_and_traceback(debug_compiler_module):
    def failing_compiler(_source):
        raise debug_compiler_module.LockstepCompileError(
            [
                debug_compiler_module.LockstepDiagnostic(
                    severity="error",
                    code="LCK001",
                    message="unexpected",
                    line=4,
                    column=2,
                    hint="Fix syntax errors before semantic analysis can continue.",
                )
            ],
            diagnostics=[
                debug_compiler_module.LockstepDiagnostic(
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
    exit_code = debug_compiler_module.run_cli(
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


def test_run_cli_returns_non_zero_for_missing_path(debug_compiler_module, tmp_path):
    missing = tmp_path / "missing.lock"
    stderr = io.StringIO()
    called = {"compiler": False}

    def fake_compiler(_source):
        called["compiler"] = True

    exit_code = debug_compiler_module.run_cli(
        [str(missing)],
        stderr=stderr,
        compiler=fake_compiler,
    )

    assert exit_code == 1
    assert called["compiler"] is False
    assert f"Unable to read '{missing}': file not found." in stderr.getvalue()


def test_run_cli_returns_non_zero_for_unreadable_path(debug_compiler_module, monkeypatch):
    stderr = io.StringIO()
    called = {"compiler": False}

    def fake_compiler(_source):
        called["compiler"] = True

    def raise_permission_error(_self, encoding):
        raise PermissionError("permission denied")

    monkeypatch.setattr("lockstep_compiler.cli.Path.read_text", raise_permission_error)

    exit_code = debug_compiler_module.run_cli(
        ["locked.lock"],
        stderr=stderr,
        compiler=fake_compiler,
    )

    assert exit_code == 1
    assert called["compiler"] is False
    assert "Unable to read 'locked.lock': permission denied." in stderr.getvalue()


def test_run_cli_returns_non_zero_for_invalid_utf8(debug_compiler_module, tmp_path):
    bad_source = tmp_path / "invalid.lock"
    bad_source.write_bytes(b"\xff\xfe\xfa")
    stderr = io.StringIO()
    called = {"compiler": False}

    def fake_compiler(_source):
        called["compiler"] = True

    exit_code = debug_compiler_module.run_cli(
        [str(bad_source)],
        stderr=stderr,
        compiler=fake_compiler,
    )

    assert exit_code == 1
    assert called["compiler"] is False
    assert f"Unable to read '{bad_source}': invalid UTF-8" in stderr.getvalue()


def test_run_cli_returns_non_zero_when_compiler_missing(debug_compiler_module):
    stderr = io.StringIO()

    exit_code = debug_compiler_module.run_cli(
        [],
        stdin=io.StringIO("pipeline MissingCompiler { }"),
        stderr=stderr,
        compiler=None,
    )

    assert exit_code == 1
    assert stderr.getvalue().splitlines() == [
        "Compiler configuration error: no compiler callable was provided.",
    ]


def test_semantic_validator_reports_undefined_identifier_in_bind(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()
    validator.shaders = {
        "Apply": [
            SemanticKernelParam(name="inp", declared_type="Vec3", modifier="in"),
            SemanticKernelParam(name="outp", declared_type="Vec3", modifier="out"),
        ]
    }
    validator._push_scope()
    validator._declare("out_stream", "Vec3", _ctx(), duplicate_code="LCK306", kind="stream")

    bind_ctx = _ctx(
        start_line=12,
        start_col=3,
        ID=lambda: [_token("out_stream"), _token("Apply"), _token("missing_stream"), _token("out_stream")],
        argList=lambda: object(),
        typeName=lambda: _token("float"),
    )

    validator.visitBindStmt(bind_ctx)

    assert validator.diagnostics == [
        debug_compiler_module.LockstepDiagnostic(
            severity="error",
            code="LCK301",
            message="Undefined identifier 'missing_stream'.",
            line=12,
            column=3,
            hint="Declare pipeline symbols before passing them to bind.",
        )
    ]


def test_semantic_validator_reports_bind_arity_and_type_errors(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()
    validator.shaders = {
        "Apply": [
            SemanticKernelParam(name="inp", declared_type="Vec3", modifier="in"),
            SemanticKernelParam(name="energy", declared_type="float", modifier="accum"),
        ]
    }
    validator._push_scope()
    validator._declare("s0", "Vec3", _ctx(), duplicate_code="LCK306", kind="stream")
    validator._declare("acc", "int", _ctx(), duplicate_code="LCK306", kind="accumulator")

    arity_ctx = _ctx(
        start_line=20,
        start_col=2,
        ID=lambda: [_token("s0"), _token("Apply"), _token("s0")],
        argList=lambda: object(),
        typeName=lambda: _token("float"),
    )
    validator.visitBindStmt(arity_ctx)

    type_ctx = _ctx(
        start_line=21,
        start_col=2,
        ID=lambda: [_token("s0"), _token("Apply"), _token("s0"), _token("acc")],
        argList=lambda: object(),
        typeName=lambda: _token("float"),
    )
    validator.visitBindStmt(type_ctx)

    assert validator.diagnostics[0].code == "LCK304"
    assert "expects 2 argument(s), but got 1" in validator.diagnostics[0].message
    assert validator.diagnostics[1].code == "LCK309"
    assert "kernel has no out parameter" in validator.diagnostics[1].message
    assert validator.diagnostics[2].code == "LCK305"
    assert "expected float, got int" in validator.diagnostics[2].message


def test_semantic_validator_reports_bind_modifier_kind_mismatches(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()
    validator.shaders = {
        "Apply": [
            SemanticKernelParam(name="inp", declared_type="Vec3", modifier="in"),
            SemanticKernelParam(name="u_dt", declared_type="float", modifier="uniform"),
            SemanticKernelParam(name="energy", declared_type="float", modifier="accum"),
        ]
    }
    validator._push_scope()
    validator._declare("s0", "Vec3", _ctx(), duplicate_code="LCK306", kind="stream")
    validator._declare("dt_stream", "float", _ctx(), duplicate_code="LCK306", kind="stream")
    validator._declare("dt_uniform", "float", _ctx(), duplicate_code="LCK306", kind="uniform")

    mismatch_ctx = _ctx(
        start_line=24,
        start_col=2,
        ID=lambda: [_token("s0"), _token("Apply"), _token("s0"), _token("dt_stream"), _token("dt_uniform")],
        argList=lambda: object(),
        typeName=lambda: _token("float"),
    )
    validator.visitBindStmt(mismatch_ctx)

    assert [diag.code for diag in validator.diagnostics] == ["LCK309", "LCK308", "LCK308"]
    assert "kernel has no out parameter" in validator.diagnostics[0].message
    assert "requires uniform" in validator.diagnostics[1].message
    assert "requires accum" in validator.diagnostics[2].message


def test_semantic_validator_reports_bind_target_output_semantics(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()
    validator.shaders = {
        "Apply": [
            SemanticKernelParam(name="inp", declared_type="Vec3", modifier="in"),
            SemanticKernelParam(name="outp", declared_type="Vec3", modifier="out"),
        ]
    }
    validator._push_scope()
    validator._declare("inp_stream", "Vec3", _ctx(), duplicate_code="LCK306", kind="stream")
    validator._declare("out_stream", "Vec3", _ctx(), duplicate_code="LCK306", kind="stream")
    validator._declare("target_uniform", "Vec3", _ctx(), duplicate_code="LCK306", kind="uniform")

    mismatch_ctx = _ctx(
        start_line=26,
        start_col=2,
        ID=lambda: [_token("target_uniform"), _token("Apply"), _token("inp_stream"), _token("out_stream")],
        argList=lambda: object(),
        typeName=lambda: _token("float"),
    )
    validator.visitBindStmt(mismatch_ctx)

    assert [diag.code for diag in validator.diagnostics] == ["LCK312", "LCK309"]
    assert "must match out argument" in validator.diagnostics[0].message
    assert "must be a stream" in validator.diagnostics[1].message


def test_semantic_validator_allows_bind_when_target_matches_out_argument(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()
    validator.shaders = {
        "Apply": [
            SemanticKernelParam(name="inp", declared_type="Vec3", modifier="in"),
            SemanticKernelParam(name="outp", declared_type="Vec3", modifier="out"),
        ]
    }
    validator._push_scope()
    validator._declare("inp_stream", "Vec3", _ctx(), duplicate_code="LCK306", kind="stream")
    validator._declare("out_stream", "Vec3", _ctx(), duplicate_code="LCK306", kind="stream")

    bind_ctx = _ctx(
        start_line=28,
        start_col=2,
        ID=lambda: [_token("out_stream"), _token("Apply"), _token("inp_stream"), _token("out_stream")],
        argList=lambda: object(),
        typeName=lambda: _token("float"),
    )
    validator.visitBindStmt(bind_ctx)

    assert validator.diagnostics == []


def test_semantic_validator_reports_mismatched_target_and_out_argument(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()
    validator.shaders = {
        "Apply": [
            SemanticKernelParam(name="inp", declared_type="Vec3", modifier="in"),
            SemanticKernelParam(name="outp", declared_type="Vec3", modifier="out"),
        ]
    }
    validator._push_scope()
    validator._declare("inp_stream", "Vec3", _ctx(), duplicate_code="LCK306", kind="stream")
    validator._declare("out_stream", "Vec3", _ctx(), duplicate_code="LCK306", kind="stream")
    validator._declare("other_stream", "Vec3", _ctx(), duplicate_code="LCK306", kind="stream")

    bind_ctx = _ctx(
        start_line=29,
        start_col=2,
        ID=lambda: [_token("out_stream"), _token("Apply"), _token("inp_stream"), _token("other_stream")],
        argList=lambda: object(),
        typeName=lambda: _token("float"),
    )
    validator.visitBindStmt(bind_ctx)

    assert [diag.code for diag in validator.diagnostics] == ["LCK312"]
    assert "must match out argument 'other_stream'" in validator.diagnostics[0].message


def test_semantic_validator_reports_bind_missing_out_parameter(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()
    validator.shaders = {
        "Apply": [
            SemanticKernelParam(name="inp", declared_type="Vec3", modifier="in"),
            SemanticKernelParam(name="energy", declared_type="float", modifier="accum"),
        ]
    }
    validator._push_scope()
    validator._declare("s0", "Vec3", _ctx(), duplicate_code="LCK306", kind="stream")
    validator._declare("acc", "float", _ctx(), duplicate_code="LCK306", kind="accumulator")

    bind_ctx = _ctx(
        start_line=30,
        start_col=2,
        ID=lambda: [_token("s0"), _token("Apply"), _token("s0"), _token("acc")],
        argList=lambda: object(),
        typeName=lambda: _token("float"),
    )
    validator.visitBindStmt(bind_ctx)

    assert [diag.code for diag in validator.diagnostics] == ["LCK309"]
    assert "kernel has no out parameter" in validator.diagnostics[0].message


def test_semantic_validator_reports_duplicate_pipeline_symbols(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()
    validator._push_scope()

    duplicate_ctx = _ctx(start_line=7, start_col=1, ID=lambda: _token("energy"), typeName=lambda: _token("float"))
    validator.visitAccumDecl(duplicate_ctx)
    validator.visitUniformDecl(duplicate_ctx)

    assert validator.diagnostics == [
        debug_compiler_module.LockstepDiagnostic(
            severity="error",
            code="LCK306",
            message="Duplicate declaration for 'energy' in the same scope.",
            line=7,
            column=1,
            hint="Rename one declaration or move it to a different scope.",
        )
    ]


def test_semantic_validator_reports_fold_reference_errors(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()
    validator._push_scope()
    validator._declare("not_acc", "float", _ctx(), duplicate_code="LCK306", kind="uniform")
    validator._declare("acc_energy", "float", _ctx(), duplicate_code="LCK306", kind="accumulator")

    non_acc_fold_ctx = _ctx(
        start_line=30,
        start_col=6,
        ID=lambda: [_token("u0"), _token("not_acc")],
        foldOperator=lambda: _token("sum"),
        argList=lambda: None,
        typeName=lambda: _token("float"),
    )
    validator.visitBindStmt(non_acc_fold_ctx)

    mismatched_type_ctx = _ctx(
        start_line=31,
        start_col=6,
        ID=lambda: [_token("u1"), _token("acc_energy")],
        foldOperator=lambda: _token("sum"),
        argList=lambda: None,
        typeName=lambda: _token("int"),
    )
    validator.visitBindStmt(mismatched_type_ctx)

    assert validator.diagnostics[0].code == "LCK403"
    assert "must reference an accumulator" in validator.diagnostics[0].message
    assert validator.diagnostics[1].code == "LCK404"
    assert "has type int" in validator.diagnostics[1].message


def test_semantic_validator_reports_duplicate_fold_target(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()
    validator._push_scope()
    validator._declare("acc_energy", "float", _ctx(), duplicate_code="LCK306", kind="accumulator")
    validator._declare("u0", "float", _ctx(), duplicate_code="LCK306", kind="uniform")

    duplicate_fold_ctx = _ctx(
        start_line=33,
        start_col=4,
        ID=lambda: [_token("u0"), _token("acc_energy")],
        foldOperator=lambda: _token("sum"),
        argList=lambda: None,
        typeName=lambda: _token("float"),
    )

    validator.visitBindStmt(duplicate_fold_ctx)

    assert validator.diagnostics == [
        debug_compiler_module.LockstepDiagnostic(
            severity="error",
            code="LCK306",
            message="Duplicate declaration for 'u0' in the same scope.",
            line=33,
            column=4,
            hint="Rename one declaration or move it to a different scope.",
        )
    ]


def test_semantic_validator_fold_declares_target_uniform_without_scope(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()
    validator._declare("acc_energy", "float", _ctx(), duplicate_code="LCK306", kind="accumulator")

    fold_ctx = _ctx(
        start_line=34,
        start_col=5,
        ID=lambda: [_token("u1"), _token("acc_energy")],
        foldOperator=lambda: _token("sum"),
        argList=lambda: None,
        typeName=lambda: _token("float"),
    )

    validator.visitBindStmt(fold_ctx)

    assert validator.diagnostics == []
    assert validator.scopes
    assert validator.scopes[-1]["u1"] == SemanticSymbol(name="u1", declared_type="float", kind="uniform")


def test_semantic_validator_records_pure_signature_from_declaration(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()

    pure_param_list_ctx = _ctx(
        typeName=lambda: [_token("float"), _token("int")],
        ID=lambda: [_token("left"), _token("right")],
    )
    pure_decl_ctx = _ctx(
        ID=lambda: _token("blend"),
        typeName=lambda: _token("float"),
        pureParamList=lambda: pure_param_list_ctx,
    )

    validator.visitPureDecl(pure_decl_ctx)

    assert validator.pure_functions == {
        "blend": {
            "return_type": "float",
            "params": [
                SemanticKernelParam(name="left", declared_type="float", modifier="value"),
                SemanticKernelParam(name="right", declared_type="int", modifier="value"),
            ],
        }
    }


def test_semantic_validator_reports_error_for_pure_function_without_return(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()

    non_return_stmt = _ctx(start_line=40, start_col=2, returnStmt=lambda: None)
    pure_decl_ctx = _ctx(
        start_line=39,
        start_col=0,
        ID=lambda: _token("blend"),
        typeName=lambda: _token("float"),
        pureParamList=lambda: None,
        statement=lambda: [non_return_stmt],
    )

    validator.visitPureDecl(pure_decl_ctx)

    assert validator.diagnostics == [
        debug_compiler_module.LockstepDiagnostic(
            severity="error",
            code="LCK413",
            message="Pure function 'blend' must include a return statement.",
            line=39,
            column=0,
            hint="Add a return statement that produces a value matching the declared return type.",
        )
    ]


def test_semantic_validator_accepts_pure_function_with_single_return(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()

    return_stmt = _ctx(start_line=42, start_col=4)
    pure_decl_ctx = _ctx(
        start_line=41,
        start_col=0,
        ID=lambda: _token("blend"),
        typeName=lambda: _token("float"),
        pureParamList=lambda: None,
        statement=lambda: [_ctx(start_line=42, start_col=4, returnStmt=lambda: return_stmt)],
    )

    validator.visitPureDecl(pure_decl_ctx)

    assert validator.diagnostics == []


def test_semantic_validator_optional_multi_return_policy_warns_for_dead_code(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()

    first_return_stmt = _ctx(start_line=44, start_col=4)
    second_return_stmt = _ctx(start_line=45, start_col=4)
    unreachable_stmt = _ctx(start_line=46, start_col=4, returnStmt=lambda: None)
    pure_decl_ctx = _ctx(
        start_line=43,
        start_col=0,
        ID=lambda: _token("blend"),
        typeName=lambda: _token("float"),
        pureParamList=lambda: None,
        statement=lambda: [
            _ctx(start_line=44, start_col=4, returnStmt=lambda: first_return_stmt),
            _ctx(start_line=45, start_col=4, returnStmt=lambda: second_return_stmt),
            unreachable_stmt,
        ],
    )

    validator.visitPureDecl(pure_decl_ctx)

    assert validator.diagnostics == [
        debug_compiler_module.LockstepDiagnostic(
            severity="warning",
            code="LCK414",
            message=(
                "Pure function 'blend' contains multiple return statements; "
                "only the first return is reachable in straight-line semantics."
            ),
            line=45,
            column=4,
            hint="Keep a single terminal return to avoid dead code and ambiguous intent.",
        ),
        debug_compiler_module.LockstepDiagnostic(
            severity="warning",
            code="LCK415",
            message="Unreachable statement in pure function 'blend' after return statement.",
            line=45,
            column=4,
            hint="Remove or move statements before the return.",
        ),
        debug_compiler_module.LockstepDiagnostic(
            severity="warning",
            code="LCK415",
            message="Unreachable statement in pure function 'blend' after return statement.",
            line=46,
            column=4,
            hint="Remove or move statements before the return.",
        ),
    ]


def test_semantic_validator_reports_undefined_pure_function_call(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()

    call_ctx = _ctx(
        start_line=36,
        start_col=3,
        ID=lambda: _token("missing_pure"),
        exprList=lambda: _ctx(expr=lambda: []),
    )

    validator.visitPrimaryExpr(call_ctx)

    assert validator.diagnostics == [
        debug_compiler_module.LockstepDiagnostic(
            severity="error",
            code="LCK410",
            message="Undefined pure function 'missing_pure'.",
            line=36,
            column=3,
            hint="Declare the pure function before calling it.",
        )
    ]


def test_semantic_validator_reports_pure_call_arity_mismatch(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()
    validator.pure_functions = {
        "mix": {
            "return_type": "float",
            "params": [
                SemanticKernelParam(name="a", declared_type="float", modifier="value"),
                SemanticKernelParam(name="b", declared_type="float", modifier="value"),
            ],
        }
    }

    call_ctx = _ctx(
        start_line=37,
        start_col=3,
        ID=lambda: _token("mix"),
        exprList=lambda: _ctx(expr=lambda: [_ctx(declared_type="float")]),
    )

    validator.visitPrimaryExpr(call_ctx)

    assert validator.diagnostics == [
        debug_compiler_module.LockstepDiagnostic(
            severity="error",
            code="LCK411",
            message="Pure function 'mix' expects 2 argument(s), but got 1.",
            line=37,
            column=3,
            hint="Pass the exact number of arguments declared in the pure function signature.",
        )
    ]


def test_semantic_validator_reports_pure_call_argument_type_mismatch(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()
    validator.pure_functions = {
        "negate": {
            "return_type": "float",
            "params": [SemanticKernelParam(name="value", declared_type="float", modifier="value")],
        }
    }

    call_ctx = _ctx(
        start_line=38,
        start_col=3,
        ID=lambda: _token("negate"),
        exprList=lambda: _ctx(expr=lambda: [_ctx(declared_type="int")]),
    )

    validator.visitPrimaryExpr(call_ctx)

    assert validator.diagnostics == [
        debug_compiler_module.LockstepDiagnostic(
            severity="error",
            code="LCK412",
            message="Type mismatch for argument 1 in pure call 'negate': expected float, got int.",
            line=38,
            column=3,
            hint="Ensure each argument type matches the pure function parameter type.",
        )
    ]


def test_semantic_validator_accepts_valid_pure_call(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()
    validator.pure_functions = {
        "dot": {
            "return_type": "float",
            "params": [
                SemanticKernelParam(name="x", declared_type="float", modifier="value"),
                SemanticKernelParam(name="y", declared_type="float", modifier="value"),
            ],
        }
    }

    call_ctx = _ctx(
        start_line=39,
        start_col=3,
        ID=lambda: _token("dot"),
        exprList=lambda: _ctx(expr=lambda: [_ctx(declared_type="float"), _ctx(declared_type="float")]),
    )

    validator.visitPrimaryExpr(call_ctx)

    assert validator.diagnostics == []
def test_semantic_validator_accepts_struct_types_in_declarations(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()

    class _Param:
        def __init__(self, modifier, declared_type, name):
            self._modifier = _token(modifier)
            self._declared_type = _token(declared_type)
            self._name = _token(name)

        def getChild(self, index):
            assert index == 0
            return self._modifier

        def typeName(self):
            return self._declared_type

        def ID(self):
            return self._name

    class _ParamList:
        def __init__(self, params):
            self._params = params

        def param(self):
            return self._params

    validator.visitStructDecl(_ctx(ID=lambda: _token("Particle"), structMember=lambda: []))
    validator.visitVarDecl(_ctx(ID=lambda: _token("entity"), typeName=lambda: _token("Particle")))
    validator.visitStreamDecl(
        _ctx(ID=lambda: _token("particles"), typeName=lambda: _token("Particle"))
    )
    validator.visitAccumDecl(_ctx(ID=lambda: _token("acc_particles"), typeName=lambda: _token("Particle")))
    validator.visitUniformDecl(_ctx(ID=lambda: _token("u_enabled"), typeName=lambda: _token("bool")))

    validator.visitShaderDecl(
        _ctx(
            ID=lambda: _token("Apply"),
            paramList=lambda: _ParamList([_Param("in", "Particle", "inp"), _Param("out", "Particle", "outp")]),
        )
    )
    validator.visitFilterDecl(
        _ctx(
            ID=lambda: _token("OnlyEnabled"),
            paramList=lambda: _ParamList([_Param("uniform", "bool", "enabled")]),
        )
    )

    validator._declare("acc_fold", "Particle", _ctx(), duplicate_code="LCK306", kind="accumulator")
    validator.visitBindStmt(
        _ctx(
            ID=lambda: [_token("u_particle"), _token("acc_fold")],
            foldOperator=lambda: _token("sum"),
            argList=lambda: None,
            typeName=lambda: _token("Particle"),
        )
    )

    assert validator.diagnostics == []


def test_semantic_validator_reports_unknown_declared_type_with_hint(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()

    class _Param:
        def __init__(self, modifier, declared_type, name):
            self._modifier = _token(modifier)
            self._declared_type = _token(declared_type)
            self._name = _token(name)

        def getChild(self, index):
            assert index == 0
            return self._modifier

        def typeName(self):
            return self._declared_type

        def ID(self):
            return self._name

    class _ParamList:
        def __init__(self, params):
            self._params = params

        def param(self):
            return self._params

    validator._push_scope()
    validator.visitVarDecl(_ctx(start_line=60, start_col=1, ID=lambda: _token("v0"), typeName=lambda: _token("flaot")))
    validator.visitStreamDecl(
        _ctx(start_line=61, start_col=1, ID=lambda: _token("s0"), typeName=lambda: _token("flaot"))
    )
    validator.visitAccumDecl(
        _ctx(start_line=62, start_col=1, ID=lambda: _token("a0"), typeName=lambda: _token("flaot"))
    )
    validator.visitUniformDecl(
        _ctx(start_line=63, start_col=1, ID=lambda: _token("u0"), typeName=lambda: _token("flaot"))
    )
    validator.visitShaderDecl(
        _ctx(
            start_line=64,
            start_col=1,
            ID=lambda: _token("S"),
            paramList=lambda: _ParamList([_Param("in", "flaot", "inp")]),
        )
    )
    validator.visitFilterDecl(
        _ctx(
            start_line=65,
            start_col=1,
            ID=lambda: _token("F"),
            paramList=lambda: _ParamList([_Param("uniform", "flaot", "u")]),
        )
    )
    validator._declare("acc_src", "flaot", _ctx(), duplicate_code="LCK306", kind="accumulator")
    validator.visitBindStmt(
        _ctx(
            start_line=66,
            start_col=1,
            ID=lambda: [_token("u_fold"), _token("acc_src")],
            foldOperator=lambda: _token("sum"),
            argList=lambda: None,
            typeName=lambda: _token("flaot"),
        )
    )

    type_errors = [diag for diag in validator.diagnostics if diag.code == "LCK310"]
    assert len(type_errors) == 7
    assert all(diag.message == "Unknown declared type 'flaot'." for diag in type_errors)
    assert any(diag.hint is not None and "Did you mean float?" in diag.hint for diag in type_errors)


def test_semantic_validator_nested_lvalue_reports_single_undefined_identifier(
    debug_compiler_module,
):
    validator = debug_compiler_module.LockstepSemanticValidator()
    validator._push_scope()

    lvalue_ctx = _ctx(
        start_line=40,
        start_col=8,
        ID=lambda index=0: [_token("missing")][index],
    )

    primary_ctx = _ctx(
        start_line=40,
        start_col=8,
        ID=lambda: None,
        lvalue=lambda: lvalue_ctx,
    )

    original_visit_children = validator.visitChildren

    def _visit_children(ctx):
        if hasattr(ctx, "lvalue") and callable(ctx.lvalue) and ctx.lvalue() is not None:
            validator.visitLvalue(ctx.lvalue())
        return original_visit_children(ctx)

    validator.visitChildren = _visit_children
    validator.visitPrimaryExpr(primary_ctx)

    assert validator.diagnostics == [
        debug_compiler_module.LockstepDiagnostic(
            severity="error",
            code="LCK301",
            message="Undefined identifier 'missing'.",
            line=40,
            column=8,
            hint="Declare the identifier in scope before using it.",
        )
    ]


def test_semantic_validator_lvalue_validates_struct_field_chain(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()

    struct_ctx = _ctx(
        ID=lambda: _token("Particle"),
        structMember=lambda: [
            _ctx(typeName=lambda: _token("Vec3"), ID=lambda: _token("position")),
        ],
    )
    nested_struct_ctx = _ctx(
        ID=lambda: _token("Vec3"),
        structMember=lambda: [
            _ctx(typeName=lambda: _token("float"), ID=lambda: _token("x")),
        ],
    )
    validator.visitStructDecl(struct_ctx)
    validator.visitStructDecl(nested_struct_ctx)

    validator._push_scope()
    validator._declare("entity", "Particle", _ctx(), duplicate_code="LCK306", kind="local")

    lvalue_ctx = _ctx(
        start_line=50,
        start_col=2,
        ID=lambda: [_token("entity"), _token("position"), _token("x")],
    )

    validator.visitLvalue(lvalue_ctx)

    assert validator.diagnostics == []


def test_semantic_validator_struct_decl_reports_duplicate_member(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()

    struct_ctx = _ctx(
        ID=lambda: _token("Particle"),
        structMember=lambda: [
            _ctx(start_line=60, start_col=6, typeName=lambda: _token("Vec3"), ID=lambda: _token("position")),
            _ctx(start_line=61, start_col=8, typeName=lambda: _token("float"), ID=lambda: _token("position")),
        ],
    )

    validator.visitStructDecl(struct_ctx)

    assert validator.diagnostics == [
        debug_compiler_module.LockstepDiagnostic(
            severity="error",
            code="LCK311",
            message="Struct 'Particle' has duplicate field declaration 'position'.",
            line=61,
            column=8,
            hint="Rename or remove duplicate struct member declarations.",
        )
    ]


def test_semantic_validator_struct_decl_keeps_first_duplicate_member(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()

    struct_ctx = _ctx(
        ID=lambda: _token("Particle"),
        structMember=lambda: [
            _ctx(typeName=lambda: _token("Vec3"), ID=lambda: _token("position")),
            _ctx(typeName=lambda: _token("float"), ID=lambda: _token("position")),
        ],
    )

    validator.visitStructDecl(struct_ctx)

    assert validator.structs["Particle"] == {
        "position": SemanticStructField(name="position", declared_type="Vec3")
    }


def test_semantic_validator_lvalue_reports_unknown_struct_field(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()

    struct_ctx = _ctx(
        ID=lambda: _token("Particle"),
        structMember=lambda: [
            _ctx(typeName=lambda: _token("Vec3"), ID=lambda: _token("position")),
        ],
    )
    validator.visitStructDecl(struct_ctx)

    validator._push_scope()
    validator._declare("entity", "Particle", _ctx(), duplicate_code="LCK306", kind="local")

    lvalue_ctx = _ctx(
        start_line=52,
        start_col=4,
        ID=lambda: [_token("entity"), _token("velocity")],
    )

    validator.visitLvalue(lvalue_ctx)

    assert validator.diagnostics == [
        debug_compiler_module.LockstepDiagnostic(
            severity="error",
            code="LCK302",
            message="Struct 'Particle' has no field 'velocity'.",
            line=52,
            column=4,
            hint="Use one of the fields declared on this struct.",
        )
    ]


def test_semantic_validator_reports_assignment_type_mismatch(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()
    validator._push_scope()
    validator._declare("x", "int", _ctx(), duplicate_code="LCK306", kind="local")

    assign_ctx = _ctx(
        start_line=70,
        start_col=3,
        lvalue=lambda: _ctx(ID=lambda: [_token("x")]),
        expr=lambda: _ctx(declared_type="float"),
    )

    validator.visitAssignStmt(assign_ctx)

    assert validator.diagnostics == [
        debug_compiler_module.LockstepDiagnostic(
            severity="error",
            code="LCK413",
            message="Type mismatch in assignment: left-hand side expects int, got float.",
            line=70,
            column=3,
            hint="Assign expressions whose type matches the lvalue declaration.",
        )
    ]


def test_semantic_validator_reports_var_initializer_type_mismatch(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()

    var_decl_ctx = _ctx(
        start_line=72,
        start_col=1,
        ID=lambda: _token("count"),
        typeName=lambda: _token("int"),
        expr=lambda: _ctx(declared_type="float"),
    )

    validator.visitVarDecl(var_decl_ctx)

    assert validator.diagnostics == [
        debug_compiler_module.LockstepDiagnostic(
            severity="error",
            code="LCK414",
            message="Type mismatch in initializer for 'count': expected int, got float.",
            line=72,
            column=1,
            hint="Use an initializer expression with the same type as the declared variable.",
        )
    ]


def test_semantic_validator_reports_pure_return_type_mismatch(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()

    return_ctx = _ctx(start_line=74, start_col=5, expr=lambda: _ctx(declared_type="float"))
    pure_ctx = _ctx(
        ID=lambda: _token("compute"),
        typeName=lambda: _token("int"),
        pureParamList=lambda: None,
        returnStmt=lambda: return_ctx,
    )

    original_visit_children = validator.visitChildren

    def _visit_children(ctx):
        if hasattr(ctx, "returnStmt") and callable(ctx.returnStmt) and ctx.returnStmt() is not None:
            validator.visitReturnStmt(ctx.returnStmt())
        return original_visit_children(ctx)

    validator.visitChildren = _visit_children
    validator.visitPureDecl(pure_ctx)

    assert validator.diagnostics == [
        debug_compiler_module.LockstepDiagnostic(
            severity="error",
            code="LCK415",
            message="Return type mismatch in pure function 'compute': expected int, got float.",
            line=74,
            column=5,
            hint="Return an expression whose type matches the pure function return type.",
        )
    ]


def test_semantic_validator_accepts_matching_assignment_initializer_and_return(debug_compiler_module):
    validator = debug_compiler_module.LockstepSemanticValidator()
    validator._push_scope()
    validator._declare("x", "int", _ctx(), duplicate_code="LCK306", kind="local")

    assign_ctx = _ctx(
        lvalue=lambda: _ctx(ID=lambda: [_token("x")]),
        expr=lambda: _ctx(declared_type="int"),
    )
    validator.visitAssignStmt(assign_ctx)

    var_decl_ctx = _ctx(
        ID=lambda: _token("count"),
        typeName=lambda: _token("int"),
        expr=lambda: _ctx(declared_type="int"),
    )
    validator.visitVarDecl(var_decl_ctx)

    return_ctx = _ctx(expr=lambda: _ctx(declared_type="int"))
    pure_ctx = _ctx(
        ID=lambda: _token("compute"),
        typeName=lambda: _token("int"),
        pureParamList=lambda: None,
        returnStmt=lambda: return_ctx,
    )

    original_visit_children = validator.visitChildren

    def _visit_children(ctx):
        if hasattr(ctx, "returnStmt") and callable(ctx.returnStmt) and ctx.returnStmt() is not None:
            validator.visitReturnStmt(ctx.returnStmt())
        return original_visit_children(ctx)

    validator.visitChildren = _visit_children
    validator.visitPureDecl(pure_ctx)

    assert validator.diagnostics == []
