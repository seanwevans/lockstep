import importlib
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

        def removeErrorListeners(self):
            pass

        def addErrorListener(self, listener):
            pass

        def program(self):
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
        def visit(self, tree):
            visited["tree"] = tree

    monkeypatch.setattr(debug_compiler_module, "CommonTokenStream", lambda lexer: object())
    monkeypatch.setattr(debug_compiler_module, "LockstepParser", SuccessParser)
    monkeypatch.setattr(debug_compiler_module, "LockstepDebugVisitor", SpyVisitor)

    debug_compiler_module.compile_lockstep("pipeline P { }")

    assert visited["tree"] == "TREE"
