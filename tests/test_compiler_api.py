import pathlib
import sys
import builtins

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lockstep_compiler
import lockstep_compiler.compiler as compiler_module
from lockstep_compiler.codegen import emit_llvm_ir


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
    assert result.entities["structs"] == []
    assert result.entities["shaders"] == []
    assert result.entities["filters"] == []
    assert {fn["name"] for fn in result.entities["pure_functions"]} == {"step", "mix", "clamp"}
    assert result.entities["streams"] == []
    assert result.entities["accumulators"] == []
    assert result.entities["uniforms"] == []
    assert result.entities["bind_routes"] == []
    assert result.entities["bind_routes_ir"] == []
    assert result.entities["optimized_bind_routes"] == []
    assert result.entities["fused_bind_groups"] == []

    assert result.llvm_ir.startswith('; ModuleID = "lockstep"\n')
    assert "define void @\"Lockstep_Tick\"()" in result.llvm_ir


def test_emit_llvm_ir_generates_expected_declarations():
    llvm_ir = emit_llvm_ir(
        {
            "structs": [
                {
                    "name": "Vec3",
                    "fields": [
                        {"type": "float", "name": "x"},
                        {"type": "float", "name": "y"},
                        {"type": "float", "name": "z"},
                    ],
                }
            ],
            "shaders": [{"name": "ApplyGravity"}],
            "filters": [{"name": "Cull"}],
            "pure_functions": [{"name": "mix", "return_type": "float", "params": [], "body": ["returnmix(0.0,1.0,step(0.5,1.0));"]}],
            "streams": [{"name": "raw_positions", "type": "Vec3"}],
            "accumulators": [{"name": "energy", "type": "float"}],
            "uniforms": [{"name": "dt", "type": "float", "initializer": "0.016"}],
            "bind_routes": ["out = ApplyGravity(inp, out, energy, dt);"],
        }
    )

    assert "%\"struct.Vec3\" = type {float, float, float}" in llvm_ir
    assert "define float @\"pure_mix\"()" in llvm_ir
    assert "define void @\"shader_ApplyGravity\"()" in llvm_ir
    assert "define void @\"filter_Cull\"()" in llvm_ir
    assert '@"stream_raw_positions" = external global %"struct.Vec3"' in llvm_ir
    assert '@"accum_energy" = external global float' in llvm_ir
    assert '@"uniform_dt" = external global float' in llvm_ir
    assert '; bind: out = ApplyGravity(inp, out, energy, dt);' in llvm_ir
    assert "fmul float" in llvm_ir
    assert "uitofp i1" in llvm_ir


def test_emit_llvm_ir_lowers_struct_member_extract_and_insert():
    llvm_ir = emit_llvm_ir(
        {
            "structs": [
                {
                    "name": "Vec3",
                    "fields": [
                        {"type": "float", "name": "x"},
                        {"type": "float", "name": "y"},
                        {"type": "float", "name": "z"},
                    ],
                }
            ],
            "pure_functions": [
                {
                    "name": "set_and_get_x",
                    "return_type": "float",
                    "params": [],
                    "body": ["Vec3 v;", "v.x = 1.0;", "return v.x;"],
                }
            ],
            "shaders": [],
            "filters": [],
            "streams": [],
            "accumulators": [],
            "uniforms": [],
            "bind_routes": [],
        }
    )

    assert "insertvalue %\"struct.Vec3\"" in llvm_ir
    assert "extractvalue %\"struct.Vec3\"" in llvm_ir


def test_cli_main_wires_default_compiler(monkeypatch):
    import lockstep_compiler.cli as cli_module

    sentinel_compiler = object()

    def fake_run_cli(argv, *, compiler):
        assert argv == ["--dump"]
        assert compiler is sentinel_compiler
        return 7

    monkeypatch.setattr(cli_module, "run_cli", fake_run_cli)
    monkeypatch.setattr(compiler_module, "compile_lockstep", sentinel_compiler)

    assert cli_module.main(["--dump"]) == 7


def test_load_default_parser_classes_is_cached(monkeypatch):
    compiler_module.load_default_parser_classes.cache_clear()
    import_count = 0
    original_import = builtins.__import__

    def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        nonlocal import_count
        if name.startswith("generated.parser"):
            import_count += 1
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", tracking_import)

    first = compiler_module.load_default_parser_classes()
    first_import_count = import_count
    second = compiler_module.load_default_parser_classes()

    assert first == second
    assert import_count == first_import_count


def test_compile_lockstep_accepts_library_sources(monkeypatch):
    captured = {}

    def fake_compile(source_code, **kwargs):
        captured["source_code"] = source_code
        captured["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(compiler_module, "_compile_lockstep_with_dependencies", fake_compile)
    monkeypatch.setattr(
        compiler_module,
        "_load_default_parser_classes",
        lambda: ("Lexer", "Parser", "Visitor"),
    )

    result = compiler_module.compile_lockstep(
        "pipeline Main { bind { } }",
        library_sources=["struct Vec { float x; };", "pure float id(float v) { return v; }"],
    )

    assert result == "ok"
    assert captured["source_code"].startswith("struct Vec { float x; };\n\npure float id(float v) { return v; }")
    assert captured["source_code"].endswith("pipeline Main { bind { } }")


def test_emit_llvm_ir_accepts_ast_program_input():
    from lockstep_compiler.ast import (
        AstKernelBindRoute,
        AstKernelDecl,
        AstKernelParam,
        AstPipelineDecl,
        AstProgram,
        AstStreamDecl,
    )

    llvm_ir = emit_llvm_ir(
        AstProgram(
            shaders=(
                AstKernelDecl(
                    name="ApplyGravity",
                    params=(
                        AstKernelParam(modifier="in", declared_type="float", name="inp"),
                        AstKernelParam(modifier="out", declared_type="float", name="out"),
                    ),
                ),
            ),
            pipelines=(
                AstPipelineDecl(
                    name="Main",
                    streams=(
                        AstStreamDecl(name="inp", declared_type="float", capacity="2"),
                        AstStreamDecl(name="out", declared_type="float", capacity="2"),
                    ),
                    bind_routes=(
                        AstKernelBindRoute(
                            target="out",
                            kernel="ApplyGravity",
                            args=("inp", "out"),
                            route="out = ApplyGravity(inp, out);",
                        ),
                    ),
                ),
            ),
        )
    )

    assert 'route_ApplyGravity_cond' in llvm_ir
    assert 'icmp slt i32 %"idx", 2' in llvm_ir


def test_emit_llvm_ir_lowers_kernel_bind_routes_into_counted_loops():
    llvm_ir = emit_llvm_ir(
        {
            "structs": [],
            "shaders": [
                {
                    "name": "ApplyGravity",
                    "params": [
                        {"modifier": "in", "type": "float", "name": "inp"},
                        {"modifier": "out", "type": "float", "name": "out"},
                        {"modifier": "uniform", "type": "float", "name": "dt"},
                    ],
                    "body": [],
                }
            ],
            "filters": [],
            "pure_functions": [],
            "streams": [
                {"name": "inp", "type": "float", "capacity": 4},
                {"name": "out", "type": "float", "capacity": 4},
            ],
            "accumulators": [],
            "uniforms": [{"name": "dt", "type": "float"}],
            "bind_routes": ["out = ApplyGravity(inp, out, dt);"],
            "bind_routes_ir": [
                {
                    "kind": "kernel",
                    "target": "out",
                    "kernel": "ApplyGravity",
                    "args": ["inp", "out", "dt"],
                    "route": "out = ApplyGravity(inp, out, dt);",
                }
            ],
        }
    )

    assert 'route_ApplyGravity_cond' in llvm_ir
    assert 'icmp slt i32 %"idx", 4' in llvm_ir
    assert 'call void @"shader_ApplyGravity"(float' in llvm_ir


def test_emit_llvm_ir_keeps_integer_math_in_integer_domain():
    llvm_ir = emit_llvm_ir(
        {
            "structs": [],
            "shaders": [],
            "filters": [],
            "pure_functions": [
                {
                    "name": "sum_and_scale",
                    "return_type": "int",
                    "params": [
                        {"name": "a", "type": "int"},
                        {"name": "b", "type": "int"},
                    ],
                    "body": ["return (a + b) * 2;"],
                }
            ],
            "streams": [],
            "accumulators": [],
            "uniforms": [],
            "bind_routes": [],
        }
    )

    assert 'define i32 @"pure_sum_and_scale"(i32 %"a", i32 %"b")' in llvm_ir
    assert 'add i32' in llvm_ir
    assert 'mul i32' in llvm_ir
    assert 'fadd float' not in llvm_ir
    assert 'fmul float' not in llvm_ir
