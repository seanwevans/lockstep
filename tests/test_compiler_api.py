import pathlib
import sys
import builtins

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lockstep_compiler
import lockstep_compiler.compiler as compiler_module
from lockstep_compiler.c_header import emit_c_header
from lockstep_compiler.codegen import emit_llvm_ir
import pytest


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
    assert {fn["name"] for fn in result.entities["pure_functions"]} == {"step", "mix", "clamp", "max", "min", "abs", "sign", "smoothstep"}
    assert result.entities["streams"] == []
    assert result.entities["accumulators"] == []
    assert result.entities["uniforms"] == []
    assert result.entities["bind_routes"] == []
    assert result.entities["bind_routes_ir"] == []
    assert result.entities["optimized_bind_routes"] == []
    assert result.entities["fused_bind_groups"] == []

    assert result.llvm_ir.startswith('; ModuleID = "lockstep"\n')
    assert 'define void @"Lockstep_Tick"()' in result.llvm_ir


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
            "pure_functions": [
                {"name": "step", "return_type": "float", "params": [{"name": "edge", "type": "float"}, {"name": "x", "type": "float"}], "intrinsic": True},
                {"name": "mix", "return_type": "float", "params": [{"name": "a", "type": "float"}, {"name": "b", "type": "float"}, {"name": "t", "type": "float"}], "intrinsic": True},
                {"name": "clamp", "return_type": "float", "params": [{"name": "x", "type": "float"}, {"name": "min_value", "type": "float"}, {"name": "max_value", "type": "float"}], "intrinsic": True},
                {"name": "max", "return_type": "float", "params": [{"name": "x", "type": "float"}, {"name": "y", "type": "float"}], "intrinsic": True},
                {"name": "demo", "return_type": "float", "params": [], "body": ["return clamp(max(mix(0.0, 1.0, step(0.5, 1.0)), 0.25), 0.0, 1.0);"]},
            ],
            "streams": [{"name": "raw_positions", "type": "Vec3"}],
            "accumulators": [{"name": "energy", "type": "float"}],
            "uniforms": [{"name": "dt", "type": "float", "initializer": "0.016"}],
            "bind_routes": ["out = ApplyGravity(inp, out, energy, dt);"],
        }
    )

    assert "%\"struct.Vec3\" = type {float, float, float}" in llvm_ir
    assert "define float @\"pure_demo\"()" in llvm_ir
    assert "define void @\"shader_ApplyGravity\"()" in llvm_ir
    assert "define void @\"filter_Cull\"()" in llvm_ir
    assert 'define void @"Lockstep_Tick"(%"struct.Vec3"* noalias %"stream_raw_positions", float* noalias %"accum_energy", float* %"uniform_dt")' in llvm_ir
    assert '; bind: out = ApplyGravity(inp, out, energy, dt);' in llvm_ir
    assert 'declare float @"llvm.maxnum.f32"(float %".1", float %".2")' in llvm_ir
    assert 'declare float @"llvm.minnum.f32"(float %".1", float %".2")' in llvm_ir
    assert 'call float @"llvm.maxnum.f32"' in llvm_ir
    assert 'call float @"llvm.minnum.f32"' in llvm_ir
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

    import pytest

    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(["--dump"])
    assert exc_info.value.code == 7


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


def test_emit_llvm_ir_keeps_integer_arithmetic_in_integer_domain():
    llvm_ir = emit_llvm_ir(
        {
            "structs": [],
            "pure_functions": [
                {
                    "name": "sum_ints",
                    "return_type": "int",
                    "params": [{"type": "int", "name": "v"}],
                    "body": ["return v + 1;"],
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

    assert "add i32" in llvm_ir
    assert "fadd float" not in llvm_ir


def test_emit_llvm_ir_lowers_fold_routes_to_vector_reduce_intrinsics():
    llvm_ir = emit_llvm_ir(
        {
            "structs": [],
            "pure_functions": [],
            "shaders": [],
            "filters": [],
            "streams": [],
            "accumulators": [{"name": "energy", "type": "float"}],
            "uniforms": [{"name": "total", "type": "float"}],
            "bind_routes": ["uniform float total = fold sum(energy);"],
            "bind_routes_ir": [
                {
                    "kind": "fold",
                    "uniform_type": "float",
                    "uniform_name": "total",
                    "operator": "sum",
                    "source": "energy",
                    "route": "uniform float total = fold sum(energy);",
                }
            ],
        }
    )

    assert 'call fast float @"llvm.vector.reduce.fadd.v8f32"' in llvm_ir
    assert 'store float %"fold_reduce", float* %"arena_slot_1"' in llvm_ir


def test_emit_llvm_ir_does_not_raise_on_mixed_int_float_expression():
    llvm_ir = emit_llvm_ir(
        {
            "structs": [],
            "pure_functions": [
                {
                    "name": "bad_mix",
                    "return_type": "float",
                    "params": [],
                    "body": ["return 1 + 1.0;"],
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

    assert 'define float @"pure_bad_mix"()' in llvm_ir


def test_compile_lockstep_builds_structured_statement_ast():
    from lockstep_compiler.ast import AstExprBinary, AstExprVar, AstReturnStmt

    result = compiler_module.compile_lockstep(
        """
        pure float f(float x, float y) {
            return x + y * x;
        }
        pipeline Main {
            bind { }
        }
        """,
        verbose=False,
    )

    body = result.ast.pure_functions[0].body
    assert isinstance(body[0], AstReturnStmt)
    assert isinstance(body[0].value, AstExprBinary)
    assert body[0].value.op == "+"
    assert isinstance(body[0].value.left, AstExprVar)
    assert isinstance(body[0].value.right, AstExprBinary)


def test_ast_dataclasses_normalize_declared_types_to_ast_type():
    from lockstep_compiler.ast import AstKernelParam, AstType

    param = AstKernelParam(modifier="in", declared_type="float", name="value")

    assert isinstance(param.declared_type, AstType)
    assert param.declared_type.name == "float"
    assert param.declared_type.kind == "primitive"


def test_emit_c_header_generates_structs_offsets_and_tick_signature():
    header = emit_c_header(
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
            "streams": [{"name": "raw_positions", "type": "Vec3"}],
            "accumulators": [{"name": "energy", "type": "float"}],
            "uniforms": [{"name": "dt", "type": "float"}],
        }
    )

    assert "#ifndef LOCKSTEP_GENERATED_H" in header
    assert "struct Lockstep_Vec3" in header
    assert "struct Lockstep_Arena" in header
    assert "#define LOCKSTEP_ARENA_BYTES 20" in header
    assert "#define LOCKSTEP_OFFSET_STREAM_RAW_POSITIONS 0" in header
    assert "#define LOCKSTEP_OFFSET_ACCUM_ENERGY 12" in header
    assert "#define LOCKSTEP_OFFSET_UNIFORM_DT 16" in header
    assert "void Lockstep_Tick(struct Lockstep_Arena* arena);" in header


def test_compile_result_includes_c_header(monkeypatch):
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
            self.streams = [{"name": "s", "type": "float"}]
            self.accumulators = []
            self.uniforms = []
            self.bind_routes = []
            self.bind_routes_ir = []
            self.diagnostics = []

        def visit(self, tree):
            return None

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

    assert "#define LOCKSTEP_ARENA_BYTES 4" in result.c_header
    assert "void Lockstep_Tick(struct Lockstep_Arena* arena);" in result.c_header
