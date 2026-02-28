import importlib
import io
import pathlib
import runpy
import sys
import types

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_fake_generated_modules(monkeypatch):
    """Install minimal ANTLR-generated modules so debug_compiler can import."""

    lexer_module = types.ModuleType("LockstepLexer")

    class StubLexer:
        def __init__(self, input_stream):
            self.input_stream = input_stream

        def removeErrorListeners(self):
            pass

        def addErrorListener(self, listener):
            pass

    lexer_module.LockstepLexer = StubLexer

    parser_module = types.ModuleType("LockstepParser")

    class StubParser:
        error_to_emit = None

        class ProgramContext:
            pass

        class StructDeclContext:
            pass

        class PureDeclContext:
            pass

        class ShaderDeclContext:
            pass

        class FilterDeclContext:
            pass

        class PipelineDeclContext:
            pass

        class StreamDeclContext:
            pass

        class AccumDeclContext:
            pass

        class UniformDeclContext:
            pass

        class BindBlockContext:
            pass

        class BindStmtContext:
            pass

        class FilterDeclContext:
            pass

        class UniformDeclContext:
            pass

        class VarDeclContext:
            pass

        class PrimaryExprContext:
            pass

        class LvalueContext:
            pass

        def __init__(self, stream):
            self.stream = stream
            self._listeners = []

        def removeErrorListeners(self):
            self._listeners = []

        def addErrorListener(self, listener):
            self._listeners.append(listener)

        def program(self):
            if self.error_to_emit is not None:
                line, column, msg = self.error_to_emit
                for listener in self._listeners:
                    listener.syntaxError(None, None, line, column, msg, None)
            return object()

    parser_module.LockstepParser = StubParser

    visitor_module = types.ModuleType("LockstepVisitor")

    class StubVisitor:
        def visit(self, tree):
            return tree

        def visitChildren(self, ctx):
            return ctx

    visitor_module.LockstepVisitor = StubVisitor

    monkeypatch.setitem(sys.modules, "LockstepLexer", lexer_module)
    monkeypatch.setitem(sys.modules, "LockstepParser", parser_module)
    monkeypatch.setitem(sys.modules, "LockstepVisitor", visitor_module)


@pytest.fixture
def debug_compiler_module(monkeypatch):
    _install_fake_generated_modules(monkeypatch)
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


def test_compile_lockstep_raises_when_parser_reports_errors(
    debug_compiler_module, monkeypatch
):
    class FailingParser:
        def __init__(self, stream):
            self._listeners = []

        def removeErrorListeners(self):
            self._listeners = []

        def addErrorListener(self, listener):
            self._listeners.append(listener)

        def program(self):
            for listener in self._listeners:
                listener.syntaxError(None, None, 3, 5, "mismatched input", None)
            return object()

    monkeypatch.setattr(
        debug_compiler_module, "CommonTokenStream", lambda lexer: object()
    )
    monkeypatch.setattr(debug_compiler_module, "LockstepParser", FailingParser)

    with pytest.raises(debug_compiler_module.LockstepCompileError) as exc_info:
        debug_compiler_module.compile_lockstep("pipeline P { }")

    assert exc_info.value.errors == [
        debug_compiler_module.LockstepDiagnostic(
            severity="error",
            code="LCK001",
            message="mismatched input",
            line=3,
            column=5,
            hint="Fix syntax errors before semantic analysis can continue.",
        )
    ]
    assert exc_info.value.diagnostics == exc_info.value.errors


def test_compile_lockstep_visits_tree_on_success(debug_compiler_module, monkeypatch):
    class SuccessParser:
        def __init__(self, stream):
            self._listeners = []

        def removeErrorListeners(self):
            self._listeners = []

        def addErrorListener(self, listener):
            self._listeners.append(listener)

        def program(self):
            return "TREE"

    visited = {"tree": None}

    class SpyVisitor:
        def __init__(self, verbose=True):
            self.verbose = verbose
            self.structs = ["Vec3"]
            self.shaders = [{"name": "ApplyGravity", "params": []}]
            self.filters = [{"name": "OnlyActive", "params": []}]
            self.pure_functions = [{"name": "add", "return_type": "Vec3"}]
            self.streams = [{"name": "raw", "type": "Vec3", "capacity": "1000"}]
            self.accumulators = [{"name": "energy", "type": "float"}]
            self.uniforms = [{"name": "dt", "type": "float", "initializer": "0.016"}]
            self.bind_routes = ["final=ApplyGravity(raw,final,energy,dt);"]
            self.diagnostics = [
                debug_compiler_module.LockstepDiagnostic(
                    severity="warning",
                    code="LCK203",
                    message="Stream 'raw' is redeclared.",
                    line=8,
                    column=4,
                    hint="Each stream in a pipeline should have a unique name.",
                )
            ]

        def visit(self, tree):
            visited["tree"] = tree

    monkeypatch.setattr(
        debug_compiler_module, "CommonTokenStream", lambda lexer: object()
    )
    monkeypatch.setattr(debug_compiler_module, "LockstepParser", SuccessParser)
    monkeypatch.setattr(debug_compiler_module, "LockstepDebugVisitor", SpyVisitor)

    result = debug_compiler_module.compile_lockstep("pipeline P { }", verbose=False)

    assert visited["tree"] == "TREE"
    assert result.parse_tree == "TREE"
    assert result.entities == {
        "structs": ["Vec3"],
        "shaders": [{"name": "ApplyGravity", "params": []}],
        "filters": [{"name": "OnlyActive", "params": []}],
        "pure_functions": [{"name": "add", "return_type": "Vec3"}],
        "streams": [{"name": "raw", "type": "Vec3", "capacity": "1000"}],
        "accumulators": [{"name": "energy", "type": "float"}],
        "uniforms": [{"name": "dt", "type": "float", "initializer": "0.016"}],
        "bind_routes": ["final=ApplyGravity(raw,final,energy,dt);"],
    }
    assert result.diagnostics == [
        debug_compiler_module.LockstepDiagnostic(
            severity="warning",
            code="LCK203",
            message="Stream 'raw' is redeclared.",
            line=8,
            column=4,
            hint="Each stream in a pipeline should have a unique name.",
        )
    ]


def test_compile_lockstep_runs_semantic_validation_phase(
    debug_compiler_module, monkeypatch
):
    class SuccessParser:
        def __init__(self, stream):
            self._listeners = []

        def removeErrorListeners(self):
            self._listeners = []

        def addErrorListener(self, listener):
            self._listeners.append(listener)

        def program(self):
            return "TREE"

    monkeypatch.setattr(
        debug_compiler_module, "CommonTokenStream", lambda lexer: object()
    )
    monkeypatch.setattr(debug_compiler_module, "LockstepParser", SuccessParser)
    monkeypatch.setattr(
        debug_compiler_module,
        "validate_semantics",
        lambda parse_tree: [
            debug_compiler_module.LockstepDiagnostic(
                severity="info",
                code="LCK301",
                message=f"validated {parse_tree}",
                line=1,
                column=0,
                hint="semantic phase ran",
            )
        ],
    )

    result = debug_compiler_module.compile_lockstep("pipeline P { }", verbose=False)

    assert result.diagnostics == [
        debug_compiler_module.LockstepDiagnostic(
            severity="info",
            code="LCK301",
            message="validated TREE",
            line=1,
            column=0,
            hint="semantic phase ran",
        )
    ]


def test_compile_lockstep_raises_for_semantic_errors(
    debug_compiler_module, monkeypatch
):
    class SuccessParser:
        def __init__(self, stream):
            self._listeners = []

        def removeErrorListeners(self):
            self._listeners = []

        def addErrorListener(self, listener):
            self._listeners.append(listener)

        def program(self):
            return "TREE"

    monkeypatch.setattr(
        debug_compiler_module, "CommonTokenStream", lambda lexer: object()
    )
    monkeypatch.setattr(debug_compiler_module, "LockstepParser", SuccessParser)
    monkeypatch.setattr(
        debug_compiler_module,
        "validate_semantics",
        lambda _parse_tree: [
            debug_compiler_module.LockstepDiagnostic(
                severity="error",
                code="LCK401",
                message="semantic problem",
                line=5,
                column=2,
                hint="fix semantic issue",
            )
        ],
    )

    with pytest.raises(debug_compiler_module.LockstepCompileError) as exc_info:
        debug_compiler_module.compile_lockstep("pipeline P { }", verbose=False)

    assert str(exc_info.value) == (
        "Compilation failed with 1 semantic error.\nline 5:2 semantic problem"
    )


def test_compile_lockstep_normalizes_diagnostics_order_and_duplicates(
    debug_compiler_module, monkeypatch
):
    class SuccessParser:
        def __init__(self, stream):
            self._listeners = []

        def removeErrorListeners(self):
            self._listeners = []

        def addErrorListener(self, listener):
            self._listeners.append(listener)

        def program(self):
            return "TREE"

    class SpyVisitor:
        def __init__(self, verbose=True):
            self.verbose = verbose
            self.structs = []
            self.shaders = []
            self.filters = []
            self.pure_functions = []
            self.streams = []
            self.accumulators = []
            self.uniforms = []
            self.bind_routes = []
            self.diagnostics = [
                debug_compiler_module.LockstepDiagnostic(
                    severity="warning",
                    code="LCK099",
                    message="same issue",
                    line=3,
                    column=1,
                ),
                debug_compiler_module.LockstepDiagnostic(
                    severity="error",
                    code="LCK200",
                    message="later diagnostic",
                    line=10,
                    column=4,
                ),
            ]

        def visit(self, tree):
            return tree

    monkeypatch.setattr(
        debug_compiler_module, "CommonTokenStream", lambda lexer: object()
    )
    monkeypatch.setattr(debug_compiler_module, "LockstepParser", SuccessParser)
    monkeypatch.setattr(debug_compiler_module, "LockstepDebugVisitor", SpyVisitor)
    monkeypatch.setattr(
        debug_compiler_module,
        "validate_semantics",
        lambda _parse_tree: [
            debug_compiler_module.LockstepDiagnostic(
                severity="warning",
                code="LCK099",
                message="same issue",
                line=3,
                column=1,
            ),
            debug_compiler_module.LockstepDiagnostic(
                severity="info",
                code="LCK050",
                message="same location info",
                line=3,
                column=1,
            ),
            debug_compiler_module.LockstepDiagnostic(
                severity="info",
                code="LCK010",
                message="same location info from semantics",
                line=3,
                column=1,
            ),
        ],
    )

    result = debug_compiler_module.compile_lockstep("pipeline P { }", verbose=False)

    assert [
        (d.line, d.column, d.severity, d.code, d.message) for d in result.diagnostics
    ] == [
        (3, 1, "warning", "LCK099", "same issue"),
        (3, 1, "info", "LCK010", "same location info from semantics"),
        (3, 1, "info", "LCK050", "same location info"),
        (10, 4, "error", "LCK200", "later diagnostic"),
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
    _install_fake_generated_modules(monkeypatch)
    sys.modules.pop("debug_compiler", None)
    monkeypatch.setattr(sys, "argv", ["debug_compiler.py"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("pipeline P { }"))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("debug_compiler", run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().err == ""


def test_module_main_error_path_exits_with_stderr(monkeypatch, capsys):
    _install_fake_generated_modules(monkeypatch)
    sys.modules["LockstepParser"].LockstepParser.error_to_emit = (7, 9, "bad syntax")
    sys.modules.pop("debug_compiler", None)
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

    monkeypatch.setattr(debug_compiler_module.Path, "read_text", raise_permission_error)

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
            {"name": "inp", "type": "Vec3", "modifier": "in"},
            {"name": "outp", "type": "Vec3", "modifier": "out"},
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
            {"name": "inp", "type": "Vec3", "modifier": "in"},
            {"name": "energy", "type": "float", "modifier": "accum"},
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
            {"name": "inp", "type": "Vec3", "modifier": "in"},
            {"name": "u_dt", "type": "float", "modifier": "uniform"},
            {"name": "energy", "type": "float", "modifier": "accum"},
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
            {"name": "inp", "type": "Vec3", "modifier": "in"},
            {"name": "outp", "type": "Vec3", "modifier": "out"},
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

    assert [diag.code for diag in validator.diagnostics] == ["LCK309"]
    assert "must be a stream" in validator.diagnostics[0].message


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
        ID=lambda: [_token("u0"), _token("sum"), _token("not_acc")],
        argList=lambda: None,
        typeName=lambda: _token("float"),
    )
    validator.visitBindStmt(non_acc_fold_ctx)

    mismatched_type_ctx = _ctx(
        start_line=31,
        start_col=6,
        ID=lambda: [_token("u1"), _token("sum"), _token("acc_energy")],
        argList=lambda: None,
        typeName=lambda: _token("int"),
    )
    validator.visitBindStmt(mismatched_type_ctx)

    assert validator.diagnostics[0].code == "LCK403"
    assert "must reference an accumulator" in validator.diagnostics[0].message
    assert validator.diagnostics[1].code == "LCK404"
    assert "has type int" in validator.diagnostics[1].message


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
