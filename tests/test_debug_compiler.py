import importlib
import runpy
import sys
import types

import pytest


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


def test_lockstep_compile_error_formats_singular_and_plural(debug_compiler_module):
    one = debug_compiler_module.LockstepCompileError([(1, 1, "oops")])
    many = debug_compiler_module.LockstepCompileError([(1, 1, "oops"), (2, 4, "bad")])

    assert str(one) == "Compilation failed with 1 parse error."
    assert str(many) == "Compilation failed with 2 parse errors."


def test_parse_error_collector_captures_location_and_message(debug_compiler_module):
    collector = debug_compiler_module.ParseErrorCollector()
    collector.syntaxError(None, None, 12, 7, "unexpected token", None)

    assert collector.errors == [(12, 7, "unexpected token")]


def test_compile_lockstep_raises_when_parser_reports_errors(debug_compiler_module, monkeypatch):
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

    monkeypatch.setattr(debug_compiler_module, "CommonTokenStream", lambda lexer: object())
    monkeypatch.setattr(debug_compiler_module, "LockstepParser", FailingParser)

    with pytest.raises(debug_compiler_module.LockstepCompileError) as exc_info:
        debug_compiler_module.compile_lockstep("pipeline P { }")

    assert exc_info.value.errors == [(3, 5, "mismatched input")]


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

        def visit(self, tree):
            visited["tree"] = tree

    monkeypatch.setattr(debug_compiler_module, "CommonTokenStream", lambda lexer: object())
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
    }
    assert result.diagnostics == []


def _token(text):
    return types.SimpleNamespace(getText=lambda: text)


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
    visitor.visitStructDecl(types.SimpleNamespace(ID=lambda: _token("Vec3")))
    visitor.visitPureDecl(
        types.SimpleNamespace(ID=lambda: _token("add"), typeName=lambda: _token("Vec3"))
    )
    visitor.visitShaderDecl(
        types.SimpleNamespace(ID=lambda: _token("ApplyGravity"), paramList=lambda: _ParamList())
    )
    visitor.visitPipelineDecl(types.SimpleNamespace(ID=lambda: _token("Physics")))
    visitor.visitStreamDecl(
        types.SimpleNamespace(
            typeName=lambda: _token("Vec3"), INT=lambda: _token("1000"), ID=lambda: _token("raw")
        )
    )
    visitor.visitAccumDecl(
        types.SimpleNamespace(typeName=lambda: _token("float"), ID=lambda: _token("energy"))
    )
    visitor.visitBindBlock(types.SimpleNamespace(bindStmt=lambda: [_BindStmt("a=b"), _BindStmt("c=d")]))

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
    assert visitor.shaders == [{"name": "ApplyGravity", "params": [{"modifier": "in", "type": "Vec3", "name": "pos"}]}]
    assert visitor.streams == [{"name": "raw", "type": "Vec3", "capacity": "1000"}]
    assert visitor.accumulators == [{"name": "energy", "type": "float"}]


def test_visitor_shader_decl_without_param_list(debug_compiler_module, capsys):
    visitor = debug_compiler_module.LockstepDebugVisitor()
    visitor.visitShaderDecl(types.SimpleNamespace(ID=lambda: _token("Kernel"), paramList=lambda: None))
    assert "[Shader Kernel] Kernel" in capsys.readouterr().out
    assert visitor.shaders == [{"name": "Kernel", "params": []}]


def test_visitor_can_run_without_printing(debug_compiler_module, capsys):
    visitor = debug_compiler_module.LockstepDebugVisitor(verbose=False)
    visitor.visitStructDecl(types.SimpleNamespace(ID=lambda: _token("Vec3")))

    assert capsys.readouterr().out == ""
    assert visitor.structs == ["Vec3"]


def test_module_main_success_path(monkeypatch, capsys):
    _install_fake_generated_modules(monkeypatch)
    sys.modules.pop("debug_compiler", None)
    runpy.run_module("debug_compiler", run_name="__main__")
    assert capsys.readouterr().err == ""


def test_module_main_error_path_exits_with_stderr(monkeypatch, capsys):
    _install_fake_generated_modules(monkeypatch)
    sys.modules["LockstepParser"].LockstepParser.error_to_emit = (7, 9, "bad syntax")
    sys.modules.pop("debug_compiler", None)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("debug_compiler", run_name="__main__")

    assert exc_info.value.code == 1
    stderr = capsys.readouterr().err
    assert "Compilation failed with 1 parse error." in stderr
    assert "line 7:9 bad syntax" in stderr
