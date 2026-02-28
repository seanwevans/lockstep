import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lockstep_compiler
import lockstep_compiler.compiler as compiler_module


def test_package_exports_user_friendly_compile_function():
    assert lockstep_compiler.compile_lockstep is compiler_module.compile_lockstep


def test_compile_lockstep_works_without_passing_parser_classes(monkeypatch):
    class StubLexer:
        def __init__(self, input_stream):
            self.input_stream = input_stream

        def removeErrorListeners(self):
            pass

        def addErrorListener(self, listener):
            pass

    class StubParser:
        def __init__(self, stream):
            self._listeners = []

        def removeErrorListeners(self):
            self._listeners = []

        def addErrorListener(self, listener):
            self._listeners.append(listener)

        def program(self):
            return "TREE"

    class StubVisitor:
        pass

    class StubDebugVisitor:
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
            self.diagnostics = []
            self.tree = None

        def visit(self, tree):
            self.tree = tree

    monkeypatch.setattr(
        compiler_module,
        "_load_default_parser_classes",
        lambda: (StubLexer, StubParser, StubVisitor),
    )

    result = lockstep_compiler.compile_lockstep(
        "pipeline P { }",
        verbose=False,
        semantic_validator=lambda _tree: [],
        token_stream_cls=lambda lexer: object(),
        debug_visitor_cls=StubDebugVisitor,
    )

    assert result.parse_tree == "TREE"
    assert result.entities == {
        "structs": [],
        "shaders": [],
        "filters": [],
        "pure_functions": [],
        "streams": [],
        "accumulators": [],
        "uniforms": [],
        "bind_routes": [],
    }
