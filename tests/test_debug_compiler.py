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

        class PipelineDeclContext:
            pass

        class StreamDeclContext:
            pass

        class AccumDeclContext:
            pass

        class BindBlockContext:
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
            self.streams = [{"name": "raw", "type": "Vec3", "capacity": "1000"}]
            self.accumulators = [{"name": "energy", "type": "float"}]
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
        "streams": [{"name": "raw", "type": "Vec3", "capacity": "1000"}],
        "accumulators": [{"name": "energy", "type": "float"}],
        "symbol_table": {
            "structs": [],
            "pure_functions": [],
            "shaders": [],
            "filters": [],
            "pipelines": {},
        },
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


def test_semantic_bind_stmt_validation_reports_errors(debug_compiler_module):
    analyzer = debug_compiler_module.LockstepSemanticAnalyzer()
    analyzer.symbols.shaders["ApplyGravity"] = debug_compiler_module.CallableSymbol(
        kind="shader",
        name="ApplyGravity",
        params=[
            debug_compiler_module.CallableParam("in", "Vec3", "pos"),
            debug_compiler_module.CallableParam("out", "Vec3", "new_pos"),
            debug_compiler_module.CallableParam("accum", "float", "energy"),
            debug_compiler_module.CallableParam("uniform", "float", "dt"),
        ],
    )
    scope = debug_compiler_module.PipelineScope(
        name="Physics",
        streams={"raw_positions": "Vec3", "final_positions": "Vec3"},
        accumulators={"total_energy": "float"},
        uniforms={"dt": "float"},
    )

    def _id_token(text, line=1, column=0):
        return types.SimpleNamespace(getText=lambda: text, line=line, column=column)

    bad_call = types.SimpleNamespace(
        ID=lambda: [_id_token("missing_stream"), _id_token("MissingKernel")],
        argList=lambda: types.SimpleNamespace(ID=lambda: [_id_token("raw_positions")]),
        start=types.SimpleNamespace(line=4, column=2),
    )
    analyzer._validate_bind_call(bad_call, scope)

    arity_call = types.SimpleNamespace(
        ID=lambda: [_id_token("final_positions"), _id_token("ApplyGravity")],
        argList=lambda: types.SimpleNamespace(ID=lambda: [_id_token("raw_positions")]),
        start=types.SimpleNamespace(line=5, column=2),
    )
    analyzer._validate_bind_call(arity_call, scope)

    type_call = types.SimpleNamespace(
        ID=lambda: [_id_token("final_positions"), _id_token("ApplyGravity")],
        argList=lambda: types.SimpleNamespace(
            ID=lambda: [
                _id_token("raw_positions", 6, 10),
                _id_token("final_positions", 6, 20),
                _id_token("raw_positions", 6, 30),
                _id_token("total_energy", 6, 40),
            ]
        ),
        start=types.SimpleNamespace(line=6, column=2),
    )
    analyzer._validate_bind_call(type_call, scope)

    fold_stmt = types.SimpleNamespace(
        ID=lambda: [_id_token("sys_energy"), _id_token("sum"), _id_token("final_positions", 7, 12)],
        start=types.SimpleNamespace(line=7, column=2),
    )
    analyzer._validate_fold(fold_stmt, scope)

    diagnostics = analyzer.diagnostics
    assert any("must be a declared stream" in d.message for d in diagnostics)
    assert any("is not declared" in d.message for d in diagnostics)
    assert any("expects 4 args, got 1" in d.message for d in diagnostics)
    assert any("invalid for 'accum'" in d.message for d in diagnostics)
    assert any("raw_positions' type 'Vec3' does not match expected 'float'" in d.message for d in diagnostics)
    assert any("must be an accumulator" in d.message for d in diagnostics)
    assert diagnostics[0].line == 4
    assert diagnostics[0].column == 2


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
    visitor.visitPipelineDecl(_ctx(ID=lambda: _token("Physics")))
    visitor.visitStreamDecl(
        _ctx(
            typeName=lambda: _token("Vec3"),
            INT=lambda: _token("1000"),
            ID=lambda: _token("raw"),
        )
    )
    visitor.visitAccumDecl(_ctx(typeName=lambda: _token("float"), ID=lambda: _token("energy")))
    visitor.visitBindBlock(_ctx(bindStmt=lambda: [_BindStmt("a=b"), _BindStmt("c=d")]))

    stdout = capsys.readouterr().out
    assert "=== LOCKSTEP COMPILER FRONTEND ===" in stdout
    assert "[Struct] Discovered: Vec3" in stdout
    assert "[Pure Function] add -> Vec3" in stdout
    assert "[Shader Kernel] ApplyGravity" in stdout
    assert "Param: (in) Vec3 pos" in stdout
    assert "[Pipeline Topology] Physics" in stdout
    assert "Stream: raw <Vec3, 1000>" in stdout
    assert "Accumulator: energy <float>" in stdout
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
    assert visitor.streams == [{"name": "raw", "type": "Vec3", "capacity": "1000"}]
    assert visitor.accumulators == [{"name": "energy", "type": "float"}]
    assert visitor.diagnostics == []


def test_visitor_emits_diagnostics_for_non_fatal_observations(debug_compiler_module):
    visitor = debug_compiler_module.LockstepDebugVisitor(verbose=False)

    visitor.visitStructDecl(_ctx(start_line=2, start_col=1, ID=lambda: _token("Vec3")))
    visitor.visitStructDecl(_ctx(start_line=3, start_col=1, ID=lambda: _token("Vec3")))
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
    assert "Compilation failed with 1 parse error." in stderr.getvalue()
    assert "line 4:2 unexpected" in stderr.getvalue()


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
