import pathlib
import sys
import builtins

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lockstep_compiler
import lockstep_compiler.c_header as c_header_module
import lockstep_compiler.compiler as compiler_module
from lockstep_compiler.c_header import emit_c_header
from lockstep_compiler.arena_layout import build_arena_layout
from lockstep_compiler.codegen import CodegenError, emit_llvm_ir
from lockstep_compiler.models import LockstepDiagnostic
from lockstep_compiler.ast import (
    AstAssignStmt,
    AstProgram,
    AstPureDecl,
    AstKernelParam,
    AstType,
    AstExprBinary,
    AstExprCall,
    AstExprCast,
    AstExprLiteral,
    AstExprVar,
    AstReturnStmt,
    AstVarDeclStmt,
)
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
        def visit(self, _tree):
            return None

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
        semantic_validator=lambda _tree, **_kwargs: [],
        token_stream_cls=lambda lexer: object(),
        debug_visitor_cls=StubDebugVisitor,
    )

    assert result.parse_tree == "TREE"
    assert result.entities["structs"] == []
    assert result.entities["shaders"] == []
    assert result.entities["filters"] == []
    assert {fn["name"] for fn in result.entities["pure_functions"]} == {
        "step",
        "mix",
        "clamp",
        "max",
        "min",
        "abs",
        "sign",
        "smoothstep",
    }
    assert result.entities["streams"] == []
    assert result.entities["accumulators"] == []
    assert result.entities["uniforms"] == []
    assert result.entities["bind_routes"] == []
    assert result.entities["bind_routes_ir"] == []
    assert result.entities["optimized_bind_routes"] == []
    assert result.entities["fused_bind_groups"] == []

    assert result.llvm_ir.startswith('; ModuleID = "lockstep"\n')
    assert (
        'define void @"Lockstep_Tick"(%"struct.Lockstep_Arena"* %"arena")'
        in result.llvm_ir
    )


def test_codegen_logical_and_short_circuits_rhs_division():
    program = AstProgram(
        pure_functions=(
            AstPureDecl(
                name="guarded",
                return_type=AstType("bool"),
                params=(
                    AstKernelParam(modifier="", declared_type=AstType("int"), name="x"),
                ),
                body=(
                    AstReturnStmt(
                        value=AstExprBinary(
                            op="&&",
                            left=AstExprBinary(
                                op="!=",
                                left=AstExprVar(path=("x",)),
                                right=AstExprLiteral(kind="int", value="0"),
                            ),
                            right=AstExprBinary(
                                op=">",
                                left=AstExprBinary(
                                    op="/",
                                    left=AstExprLiteral(kind="int", value="100"),
                                    right=AstExprVar(path=("x",)),
                                ),
                                right=AstExprLiteral(kind="int", value="5"),
                            ),
                        )
                    ),
                ),
            ),
        )
    )

    ir_text = emit_llvm_ir(program)

    assert 'br i1 %".4", label %"logic_and_rhs", label %"logic_and_merge"' in ir_text
    assert "logic_and_rhs:" in ir_text
    assert "sdiv i32 100" in ir_text
    assert "logic_and_merge:" in ir_text
    assert '[0, %"entry"], [%".7", %"logic_and_rhs"]' in ir_text


def test_codegen_logical_or_short_circuits_rhs_call():
    program = AstProgram(
        pure_functions=(
            AstPureDecl(
                name="expensive",
                return_type=AstType("bool"),
                body=(AstReturnStmt(value=AstExprLiteral(kind="bool", value="true")),),
            ),
            AstPureDecl(
                name="guarded",
                return_type=AstType("bool"),
                params=(
                    AstKernelParam(
                        modifier="", declared_type=AstType("bool"), name="ready"
                    ),
                ),
                body=(
                    AstReturnStmt(
                        value=AstExprBinary(
                            op="||",
                            left=AstExprVar(path=("ready",)),
                            right=AstExprCall(name="expensive", args=()),
                        )
                    ),
                ),
            ),
        )
    )

    ir_text = emit_llvm_ir(program)

    assert (
        'br i1 %"ready_val", label %"logic_or_merge", label %"logic_or_rhs"' in ir_text
    )
    assert "logic_or_rhs:" in ir_text
    assert 'call i1 @"pure_expensive"()' in ir_text
    assert "logic_or_merge:" in ir_text
    assert '[1, %"entry"], [%"call_expensive", %"logic_or_rhs"]' in ir_text


def test_compile_lockstep_surfaces_typed_ast_type_error_without_legacy_fallback(
    monkeypatch,
):
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
        def visit(self, _tree):
            return None

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
            self.bind_routes_ir = []
            self.diagnostics = []

        def visit(self, tree):
            return None

    monkeypatch.setattr(
        compiler_module,
        "_load_default_parser_classes",
        lambda: (StubLexer, StubParser, StubVisitor),
    )
    monkeypatch.setattr(
        compiler_module,
        "build_program_ast",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            TypeError("stub incompatibility")
        ),
    )

    with pytest.raises(TypeError, match="stub incompatibility"):
        lockstep_compiler.compile_lockstep(
            "pipeline P { }",
            verbose=False,
            semantic_validator=lambda _tree, **_kwargs: [],
            token_stream_cls=lambda lexer: object(),
            debug_visitor_cls=StubDebugVisitor,
        )


def test_compile_lockstep_surfaces_typed_ast_builder_bugs(monkeypatch):
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
        def visit(self, _tree):
            return None

    monkeypatch.setattr(
        compiler_module,
        "_load_default_parser_classes",
        lambda: (StubLexer, StubParser, StubVisitor),
    )
    monkeypatch.setattr(
        compiler_module,
        "build_program_ast",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyError("missing key")),
    )

    with pytest.raises(KeyError, match="missing key"):
        lockstep_compiler.compile_lockstep(
            "pipeline P { }",
            verbose=False,
            semantic_validator=lambda _tree, **_kwargs: [],
            token_stream_cls=lambda lexer: object(),
        )


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
            "shaders": [{"name": "ApplyGravity", "body_ast": []}],
            "filters": [{"name": "Cull", "body_ast": []}],
            "pure_functions": [
                {
                    "name": "step",
                    "return_type": "float",
                    "params": [
                        {"name": "edge", "type": "float"},
                        {"name": "x", "type": "float"},
                    ],
                    "intrinsic": True,
                },
                {
                    "name": "mix",
                    "return_type": "float",
                    "params": [
                        {"name": "a", "type": "float"},
                        {"name": "b", "type": "float"},
                        {"name": "t", "type": "float"},
                    ],
                    "intrinsic": True,
                },
                {
                    "name": "clamp",
                    "return_type": "float",
                    "params": [
                        {"name": "x", "type": "float"},
                        {"name": "min_value", "type": "float"},
                        {"name": "max_value", "type": "float"},
                    ],
                    "intrinsic": True,
                },
                {
                    "name": "max",
                    "return_type": "float",
                    "params": [
                        {"name": "x", "type": "float"},
                        {"name": "y", "type": "float"},
                    ],
                    "intrinsic": True,
                },
                {
                    "name": "demo",
                    "return_type": "float",
                    "params": [],
                    "body_ast": [
                        AstReturnStmt(
                            value=AstExprCall(
                                name="clamp",
                                args=(
                                    AstExprCall(
                                        name="max",
                                        args=(
                                            AstExprCall(
                                                name="mix",
                                                args=(
                                                    AstExprLiteral(
                                                        kind="float", value="0.0"
                                                    ),
                                                    AstExprLiteral(
                                                        kind="float", value="1.0"
                                                    ),
                                                    AstExprCall(
                                                        name="step",
                                                        args=(
                                                            AstExprLiteral(
                                                                kind="float",
                                                                value="0.5",
                                                            ),
                                                            AstExprLiteral(
                                                                kind="float",
                                                                value="1.0",
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                            AstExprLiteral(kind="float", value="0.25"),
                                        ),
                                    ),
                                    AstExprLiteral(kind="float", value="0.0"),
                                    AstExprLiteral(kind="float", value="1.0"),
                                ),
                            )
                        )
                    ],
                },
            ],
            "streams": [{"name": "raw_positions", "type": "Vec3"}],
            "accumulators": [{"name": "energy", "type": "float"}],
            "uniforms": [{"name": "dt", "type": "float", "initializer": "0.016"}],
            "bind_routes": ["out = ApplyGravity(inp, out, energy, dt);"],
        }
    )

    assert '%"struct.Vec3" = type {float, float, float}' in llvm_ir
    assert 'define float @"pure_demo"()' in llvm_ir
    assert 'define void @"shader_ApplyGravity"()' in llvm_ir
    assert 'define i1 @"filter_Cull"()' in llvm_ir
    assert 'define void @"Lockstep_Tick"(%"struct.Lockstep_Arena"* %"arena")' in llvm_ir
    assert "; bind: out = ApplyGravity(inp, out, energy, dt);" in llvm_ir
    assert 'declare float @"llvm.maxnum.f32"(float %".1", float %".2")' in llvm_ir
    assert 'declare float @"llvm.minnum.f32"(float %".1", float %".2")' in llvm_ir
    assert 'call float @"llvm.maxnum.f32"' in llvm_ir
    assert 'call float @"llvm.minnum.f32"' in llvm_ir
    assert "uitofp i1" in llvm_ir


def test_emit_llvm_ir_lowers_array_type_suffixes_for_local_allocas():
    llvm_ir = emit_llvm_ir(
        AstProgram(
            pure_functions=(
                AstPureDecl(
                    name="array_local",
                    return_type="float",
                    body=(
                        AstVarDeclStmt(
                            declared_type="float[2][3]",
                            name="matrix",
                            initializer=None,
                        ),
                        AstReturnStmt(value=AstExprLiteral(kind="float", value="1.0")),
                    ),
                ),
            ),
        )
    )

    assert "alloca [2 x [3 x float]]" in llvm_ir


def test_compile_lockstep_preserves_array_type_suffixes_in_codegen():
    result = lockstep_compiler.compile_lockstep(
        "pure float array_local(){ float[2][3] matrix; return 1.0; }",
        verbose=False,
    )

    assert "alloca [2 x [3 x float]]" in result.llvm_ir


def test_emit_llvm_ir_rejects_assignment_to_undeclared_local():
    program = AstProgram(
        pure_functions=(
            AstPureDecl(
                name="typo_assignment",
                return_type="float",
                body=(
                    AstVarDeclStmt(
                        declared_type="float",
                        name="position",
                        initializer=AstExprLiteral(kind="float", value="1.0"),
                    ),
                    AstAssignStmt(
                        target=("positiom",),
                        value=AstExprLiteral(kind="float", value="2.0"),
                    ),
                    AstReturnStmt(value=AstExprVar(path=("position",))),
                ),
            ),
        ),
    )

    with pytest.raises(
        CodegenError, match="undefined variable 'positiom' in assignment"
    ):
        emit_llvm_ir(program)


def test_emit_llvm_ir_lowers_array_element_extract_and_insert():
    llvm_ir = emit_llvm_ir(
        AstProgram(
            pure_functions=(
                AstPureDecl(
                    name="set_and_get_array_element",
                    return_type="float",
                    body=(
                        AstVarDeclStmt(
                            declared_type="float[2]", name="values", initializer=None
                        ),
                        AstAssignStmt(
                            target=("values", "0"),
                            value=AstExprLiteral(kind="float", value="1.0"),
                        ),
                        AstAssignStmt(
                            target=("values", "1"),
                            value=AstExprLiteral(kind="float", value="2.0"),
                        ),
                        AstReturnStmt(value=AstExprVar(path=("values", "1"))),
                    ),
                ),
            ),
        )
    )

    assert "insertvalue [2 x float]" in llvm_ir
    assert "extractvalue [2 x float]" in llvm_ir


def test_emit_llvm_ir_lowers_nested_array_element_extract_and_insert():
    llvm_ir = emit_llvm_ir(
        AstProgram(
            pure_functions=(
                AstPureDecl(
                    name="set_and_get_nested_array_element",
                    return_type="float",
                    body=(
                        AstVarDeclStmt(
                            declared_type="float[2][3]",
                            name="matrix",
                            initializer=None,
                        ),
                        AstAssignStmt(
                            target=("matrix", "1", "2"),
                            value=AstExprLiteral(kind="float", value="4.0"),
                        ),
                        AstReturnStmt(value=AstExprVar(path=("matrix", "1", "2"))),
                    ),
                ),
            ),
        )
    )

    assert "insertvalue [3 x float]" in llvm_ir
    assert "insertvalue [2 x [3 x float]]" in llvm_ir
    assert "extractvalue [3 x float]" in llvm_ir


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
                    "body_ast": [
                        AstVarDeclStmt(
                            declared_type="Vec3", name="v", initializer=None
                        ),
                        AstAssignStmt(
                            target=("v", "x"),
                            value=AstExprLiteral(kind="float", value="1.0"),
                        ),
                        AstReturnStmt(value=AstExprVar(path=("v", "x"))),
                    ],
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

    assert 'insertvalue %"struct.Vec3"' in llvm_ir
    assert 'extractvalue %"struct.Vec3"' in llvm_ir


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

    monkeypatch.setattr(
        compiler_module, "_compile_lockstep_with_dependencies", fake_compile
    )
    monkeypatch.setattr(
        compiler_module,
        "_load_default_parser_classes",
        lambda: ("Lexer", "Parser", "Visitor"),
    )

    result = compiler_module.compile_lockstep(
        "pipeline Main { bind { } }",
        library_sources=[
            "struct Vec { float x; };",
            "pure float id(float v) { return v; }",
        ],
    )

    assert result == "ok"
    assert captured["source_code"].startswith(
        "struct Vec { float x; };\n\npure float id(float v) { return v; }"
    )
    assert captured["source_code"].endswith("pipeline Main { bind { } }")


def test_compile_lockstep_maps_parse_diagnostics_to_library_source_file():
    library_source = "@"

    with pytest.raises(lockstep_compiler.LockstepCompileError) as exc_info:
        lockstep_compiler.compile_lockstep(
            "pipeline Main { bind { } }",
            source_file="main.lock",
            library_sources=[library_source],
            library_source_files=["lib/math.lock"],
            verbose=False,
        )

    assert exc_info.value.errors[0].source_file == "lib/math.lock"
    assert exc_info.value.errors[0].line == 1


def test_compile_lockstep_maps_semantic_diagnostics_to_primary_source_file():
    with pytest.raises(lockstep_compiler.LockstepCompileError) as exc_info:
        lockstep_compiler.compile_lockstep(
            "pipeline Main {\n    stream<MissingType, 1> values;\n    bind { }\n}",
            source_file="main.lock",
            library_sources=["struct Vec { float x; };"],
            library_source_files=["lib/math.lock"],
            verbose=False,
        )

    assert exc_info.value.errors[0].source_file == "main.lock"
    assert exc_info.value.errors[0].line == 2


def test_compile_lockstep_wraps_codegen_errors_with_compile_error(monkeypatch):
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
        def visit(self, _tree):
            return None

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
            self.bind_routes_ir = []
            self.diagnostics = [
                LockstepDiagnostic(
                    severity="warning",
                    code="LCK421",
                    message="unused symbol 'x'",
                    line=2,
                    column=4,
                )
            ]

        def visit(self, tree):
            return None

    monkeypatch.setattr(
        compiler_module,
        "_load_default_parser_classes",
        lambda: (StubLexer, StubParser, StubVisitor),
    )
    monkeypatch.setattr(
        compiler_module,
        "emit_llvm_ir",
        lambda _program: (_ for _ in ()).throw(
            CodegenError("undefined variable 'missing'")
        ),
    )

    with pytest.raises(lockstep_compiler.LockstepCompileError) as exc_info:
        lockstep_compiler.compile_lockstep(
            "pipeline Main { bind { } }",
            source_file="main.lock",
            verbose=False,
            semantic_validator=lambda _tree, **_kwargs: [],
            token_stream_cls=lambda lexer: object(),
            debug_visitor_cls=StubDebugVisitor,
        )

    error = exc_info.value
    assert error.phase == "codegen"
    assert [diag.code for diag in error.errors] == ["LCK501"]
    assert error.errors[0].message == "undefined variable 'missing'"
    assert error.errors[0].source_file == "main.lock"
    assert {diag.code for diag in error.diagnostics} == {"LCK501"}


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
                        AstKernelParam(
                            modifier="in", declared_type="float", name="inp"
                        ),
                        AstKernelParam(
                            modifier="out", declared_type="float", name="out"
                        ),
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

    assert "route_ApplyGravity_cond" in llvm_ir
    assert 'icmp slt i32 %"idx", 2' in llvm_ir
    assert (
        '%"stream_inp_arena_bytes" = bitcast %"struct.Lockstep_Arena"* %"arena" to i8*'
        in llvm_ir
    )
    assert (
        '%"stream_inp_value_byte_ptr" = getelementptr i8, i8* %"stream_inp_arena_bytes", i32 %"stream_inp_byte_offset"'
        in llvm_ir
    )
    assert 'bitcast i8* %"stream_inp_value_byte_ptr" to float*' in llvm_ir
    assert '%"stream_out_byte_offset" = add i32 8, %"stream_out_byte_index"' in llvm_ir
    assert (
        '%"stream_out_value_byte_ptr" = getelementptr i8, i8* %"stream_out_arena_bytes", i32 %"stream_out_byte_offset"'
        in llvm_ir
    )
    assert 'bitcast i8* %"stream_out_value_byte_ptr" to float*' in llvm_ir
    assert (
        'getelementptr %"struct.Lockstep_Arena", %"struct.Lockstep_Arena"* %"arena", i32 0, i32 0'
        not in llvm_ir
    )
    assert (
        'getelementptr %"struct.Lockstep_Arena", %"struct.Lockstep_Arena"* %"arena", i32 0, i32 1'
        not in llvm_ir
    )


def test_emit_llvm_ir_uses_array_field_for_single_capacity_streams():
    from lockstep_compiler.ast import (
        AstPipelineDecl,
        AstProgram,
        AstStreamDecl,
        AstUniformDecl,
    )

    llvm_ir = emit_llvm_ir(
        AstProgram(
            pipelines=(
                AstPipelineDecl(
                    name="Main",
                    streams=(
                        AstStreamDecl(name="inp", declared_type="float", capacity="1"),
                    ),
                    uniforms=(AstUniformDecl(name="dt", declared_type="float"),),
                ),
            ),
        )
    )

    assert '%"struct.Lockstep_Arena" = type {[1 x float], float}' in llvm_ir


def test_emit_llvm_ir_reports_undefined_kernel_bind_route():
    from lockstep_compiler.ast import AstKernelBindRoute, AstPipelineDecl, AstProgram

    program = AstProgram(
        pipelines=(
            AstPipelineDecl(
                name="Main",
                bind_routes=(
                    AstKernelBindRoute(
                        target="out",
                        kernel="MissingKernel",
                        args=(),
                        route="out = MissingKernel();",
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(
        CodegenError,
        match="undefined shader/filter 'MissingKernel' in bind route",
    ):
        emit_llvm_ir(program)


def test_emit_llvm_ir_reports_undefined_kernel_in_fused_trip_count():
    from lockstep_compiler.ast import (
        AstKernelBindRoute,
        AstKernelDecl,
        AstKernelParam,
        AstPipelineDecl,
        AstProgram,
        AstStreamDecl,
    )

    first_route = AstKernelBindRoute(
        target="tmp",
        kernel="KnownKernel",
        args=("inp", "tmp"),
        route="tmp = KnownKernel(inp, tmp);",
    )
    missing_route = AstKernelBindRoute(
        target="out",
        kernel="MissingKernel",
        args=("tmp", "out"),
        route="out = MissingKernel(tmp, out);",
    )
    program = AstProgram(
        shaders=(
            AstKernelDecl(
                name="KnownKernel",
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
                    AstStreamDecl(name="inp", declared_type="float", capacity="4"),
                    AstStreamDecl(name="out", declared_type="float", capacity="4"),
                ),
                bind_routes=(first_route, missing_route),
            ),
        ),
    )

    with pytest.raises(
        CodegenError,
        match=r"undefined shader/filter 'MissingKernel' in bind route: out = MissingKernel\(tmp, out\);",
    ):
        emit_llvm_ir(
            program,
            bind_optimization={
                "optimized_bind_routes": [first_route.route, missing_route.route],
                "fused_groups": [
                    {"source_routes": [first_route.route, missing_route.route]}
                ],
            },
        )


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
                    "body_ast": [],
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

    assert "route_ApplyGravity_cond" in llvm_ir
    assert 'icmp slt i32 %"idx", 4' in llvm_ir
    assert 'call void @"shader_ApplyGravity"(float' in llvm_ir
    assert (
        '%"struct.Lockstep_Arena" = type {[4 x float], [4 x float], float}' in llvm_ir
    )
    assert (
        'getelementptr %"struct.Lockstep_Arena", %"struct.Lockstep_Arena"* %"arena", i32 0, i32 0, i32 %"route_i32_lane0.1"'
        in llvm_ir
    )
    assert (
        'getelementptr %"struct.Lockstep_Arena", %"struct.Lockstep_Arena"* %"arena", i32 0, i32 1, i32 %"route_i32_lane0.3"'
        in llvm_ir
    )


def test_emit_llvm_ir_keeps_integer_arithmetic_in_integer_domain():
    llvm_ir = emit_llvm_ir(
        {
            "structs": [],
            "pure_functions": [
                {
                    "name": "sum_ints",
                    "return_type": "int",
                    "params": [{"type": "int", "name": "v"}],
                    "body_ast": [
                        AstReturnStmt(
                            value=AstExprBinary(
                                op="+",
                                left=AstExprVar(path=("v",)),
                                right=AstExprLiteral(kind="int", value="1"),
                            )
                        )
                    ],
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


def test_emit_llvm_ir_promotes_mixed_numeric_binary_operands():
    llvm_ir = emit_llvm_ir(
        {
            "structs": [],
            "pure_functions": [
                {
                    "name": "promote_float_int",
                    "return_type": "double",
                    "params": [
                        {"type": "float", "name": "f"},
                        {"type": "int", "name": "i"},
                        {"type": "double", "name": "d"},
                        {"type": "uint", "name": "u"},
                    ],
                    "body_ast": [
                        AstReturnStmt(
                            value=AstExprBinary(
                                op="+",
                                left=AstExprBinary(
                                    op="*",
                                    left=AstExprVar(path=("f",)),
                                    right=AstExprVar(path=("i",)),
                                ),
                                right=AstExprBinary(
                                    op="/",
                                    left=AstExprVar(path=("d",)),
                                    right=AstExprVar(path=("u",)),
                                ),
                            )
                        )
                    ],
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

    assert "sitofp i32" in llvm_ir
    assert "fmul float" in llvm_ir
    assert "fpext float" in llvm_ir
    assert "uitofp i32" in llvm_ir
    assert "fdiv double" in llvm_ir
    assert "fadd double" in llvm_ir


def test_emit_llvm_ir_promotes_mixed_numeric_operands_in_fused_vectors():
    first_route = "mid = Promote(inp, mid, scale, count);"
    second_route = "out = Copy(mid, out);"
    llvm_ir = emit_llvm_ir(
        {
            "structs": [],
            "shaders": [
                {
                    "name": "Promote",
                    "params": [
                        {"modifier": "in", "type": "float", "name": "inp"},
                        {"modifier": "out", "type": "double", "name": "out"},
                        {"modifier": "uniform", "type": "double", "name": "scale"},
                        {"modifier": "uniform", "type": "uint", "name": "count"},
                    ],
                    "body_ast": [
                        AstAssignStmt(
                            target=("out",),
                            value=AstExprBinary(
                                op="+",
                                left=AstExprBinary(
                                    op="*",
                                    left=AstExprVar(path=("inp",)),
                                    right=AstExprVar(path=("scale",)),
                                ),
                                right=AstExprVar(path=("count",)),
                            ),
                        )
                    ],
                },
                {
                    "name": "Copy",
                    "params": [
                        {"modifier": "in", "type": "double", "name": "inp"},
                        {"modifier": "out", "type": "double", "name": "out"},
                    ],
                    "body_ast": [
                        AstAssignStmt(target=("out",), value=AstExprVar(path=("inp",)))
                    ],
                },
            ],
            "filters": [],
            "pure_functions": [],
            "streams": [
                {"name": "inp", "type": "float", "capacity": 8},
                {"name": "mid", "type": "double", "capacity": 8},
                {"name": "out", "type": "double", "capacity": 8},
            ],
            "accumulators": [],
            "uniforms": [
                {"name": "scale", "type": "double"},
                {"name": "count", "type": "uint"},
            ],
            "bind_routes": [first_route, second_route],
            "bind_routes_ir": [
                {
                    "kind": "kernel",
                    "target": "mid",
                    "kernel": "Promote",
                    "args": ["inp", "mid", "scale", "count"],
                    "route": first_route,
                },
                {
                    "kind": "kernel",
                    "target": "out",
                    "kernel": "Copy",
                    "args": ["mid", "out"],
                    "route": second_route,
                },
            ],
        },
        bind_optimization={
            "optimized_bind_routes": [first_route, second_route],
            "fused_groups": [{"source_routes": [first_route, second_route]}],
        },
    )

    assert "fused_0_body" in llvm_ir
    assert "fpext <8 x float>" in llvm_ir
    assert "uitofp <8 x i32>" in llvm_ir
    assert "fmul <8 x double>" in llvm_ir
    assert "fadd <8 x double>" in llvm_ir


def test_emit_llvm_ir_lowers_select_builtin_for_integers():
    llvm_ir = emit_llvm_ir(
        {
            "structs": [],
            "pure_functions": [
                {
                    "name": "pick",
                    "return_type": "int",
                    "params": [
                        {"type": "bool", "name": "condition"},
                        {"type": "int", "name": "a"},
                        {"type": "int", "name": "b"},
                    ],
                    "body_ast": [
                        AstReturnStmt(
                            value=AstExprCall(
                                name="select",
                                args=(
                                    AstExprVar(path=("condition",)),
                                    AstExprVar(path=("a",)),
                                    AstExprVar(path=("b",)),
                                ),
                            )
                        )
                    ],
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

    assert 'i1 %"condition_val", i32 %"a_val", i32 %"b_val"' in llvm_ir


def test_emit_llvm_ir_lowers_select_builtin_for_structs():
    llvm_ir = emit_llvm_ir(
        {
            "structs": [
                {
                    "name": "Pair",
                    "fields": [
                        {"type": "int", "name": "left"},
                        {"type": "int", "name": "right"},
                    ],
                }
            ],
            "pure_functions": [
                {
                    "name": "pick_pair",
                    "return_type": "Pair",
                    "params": [
                        {"type": "bool", "name": "condition"},
                        {"type": "Pair", "name": "a"},
                        {"type": "Pair", "name": "b"},
                    ],
                    "body_ast": [
                        AstReturnStmt(
                            value=AstExprCall(
                                name="select",
                                args=(
                                    AstExprVar(path=("condition",)),
                                    AstExprVar(path=("a",)),
                                    AstExprVar(path=("b",)),
                                ),
                            )
                        )
                    ],
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

    assert (
        'i1 %"condition_val", %"struct.Pair" %"a_val", %"struct.Pair" %"b_val"'
        in llvm_ir
    )


def test_emit_llvm_ir_defaults_target_triple_and_simd_width_for_fold_reduce():
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

    assert 'target triple = "x86_64-unknown-linux-gnu"' in llvm_ir
    assert 'call fast float @"llvm.vector.reduce.fadd.v8f32"' in llvm_ir


def test_emit_llvm_ir_uses_default_simd_width_when_target_triple_is_unknown():
    llvm_ir = emit_llvm_ir(
        {
            "target_triple": "mystery-unknown-none",
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

    assert 'target triple = "mystery-unknown-none"' in llvm_ir
    assert 'call fast float @"llvm.vector.reduce.fadd.v8f32"' in llvm_ir


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
    assert '%"struct.Lockstep_Arena" = type {float, float}' in llvm_ir
    assert (
        '%"uniform_total_value_byte_ptr" = getelementptr i8, i8* %"uniform_total_arena_bytes", i32 4'
        in llvm_ir
    )
    assert "i32 0, i32 0, i32 4" not in llvm_ir


def test_emit_llvm_ir_strip_mines_fold_larger_than_target_width():
    llvm_ir = emit_llvm_ir(
        {
            "structs": [],
            "pure_functions": [],
            "shaders": [],
            "filters": [],
            "streams": [],
            "accumulators": [{"name": "energy", "type": "float", "size": 17}],
            "uniforms": [{"name": "total", "type": "float"}],
            "bind_routes": ["uniform float total = fold avg(energy);"],
            "bind_routes_ir": [
                {
                    "kind": "fold",
                    "uniform_type": "float",
                    "uniform_name": "total",
                    "operator": "avg",
                    "source": "energy",
                    "route": "uniform float total = fold avg(energy);",
                }
            ],
        },
        target_width=8,
    )

    assert 'br label %"fold_energy_strip_cond"' in llvm_ir
    assert '%"fold_has_full_chunk" = icmp ult i32 %"fold_index", 16' in llvm_ir
    assert '%"fold_index_next" = add i32 %"fold_index", 8' in llvm_ir
    assert "mul i32 16, 4" in llvm_ir
    assert '%"fold_avg" = fdiv float %"fold_reduce", 0x4031000000000000' in llvm_ir


def test_emit_llvm_ir_strip_mines_large_fold_without_truncating_to_target_width():
    llvm_ir = emit_llvm_ir(
        {
            "structs": [],
            "pure_functions": [],
            "shaders": [],
            "filters": [],
            "streams": [],
            "accumulators": [{"name": "energy", "type": "float", "size": 512}],
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
        },
        target_width=8,
    )

    assert '%"fold_has_full_chunk" = icmp ult i32 %"fold_index", 512' in llvm_ir
    assert '%"fold_index_next" = add i32 %"fold_index", 8' in llvm_ir
    assert 'call fast float @"llvm.vector.reduce.fadd.v8f32"' in llvm_ir
    assert '%"struct.Lockstep_Arena" = type {[512 x float], float}' in llvm_ir
    assert '%"fold_energy_chunk_ptr" = bitcast float*' in llvm_ir
    assert (
        '%"fold_chunk_ptr" = phi  <8 x float>* '
        '[%"fold_energy_chunk_ptr", %"entry"], '
        '[%"fold_chunk_ptr_next", %"fold_energy_strip_body"]' in llvm_ir
    )
    assert (
        '%"fold_energy_chunk" = load <8 x float>, <8 x float>* %"fold_chunk_ptr"'
        in llvm_ir
    )
    assert (
        '%"fold_chunk_ptr_next" = getelementptr <8 x float>, '
        '<8 x float>* %"fold_chunk_ptr", i32 1' in llvm_ir
    )
    assert '%"accum_energy_byte_index" = mul i32 %"fold_index", 4' not in llvm_ir
    assert '%"fold_elem_7" = add i32 %"fold_index", 7' not in llvm_ir


def test_compile_lockstep_strip_mines_fold_across_accumulator_route_width():
    source = """
    shader Capture(in float src, accum float energy) { energy = src; }

    pipeline P {
        stream<float, 17> input;
        accumulator<float> energy;
        uniform float total;

        bind {
            energy = Capture(input, energy);
            uniform float total = fold avg(energy);
        }
    }
    """

    result = lockstep_compiler.compile_lockstep(
        source,
        semantic_validator=lambda _tree, **_kwargs: [],
        target_width=8,
    )

    assert (
        '%"struct.Lockstep_Arena" = type {[17 x float], [17 x float], float}'
        in result.llvm_ir
    )
    assert 'br label %"fold_energy_strip_cond"' in result.llvm_ir
    assert '%"fold_has_full_chunk" = icmp ult i32 %"fold_index", 16' in result.llvm_ir
    assert '%"fold_index_next" = add i32 %"fold_index", 8' in result.llvm_ir
    assert 'mul i32 %"idx", 4' in result.llvm_ir
    assert "mul i32 16, 4" in result.llvm_ir
    assert (
        '%"fold_chunk_ptr_next" = getelementptr <8 x float>, '
        '<8 x float>* %"fold_chunk_ptr", i32 1' in result.llvm_ir
    )
    assert '%"accum_energy_byte_index" = mul i32 %"fold_index", 4' not in result.llvm_ir
    assert (
        '%"fold_avg" = fdiv float %"fold_reduce", 0x4031000000000000' in result.llvm_ir
    )


def test_emit_llvm_ir_honors_explicit_target_width_override():
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
        },
        target_width=16,
    )

    assert 'call fast float @"llvm.vector.reduce.fadd.v16f32"' in llvm_ir
    assert 'store float %"fold_reduce"' in llvm_ir


def test_emit_llvm_ir_promotes_mixed_int_float_expression():
    llvm_ir = emit_llvm_ir(
        {
            "structs": [],
            "pure_functions": [
                {
                    "name": "mixed_int_float",
                    "return_type": "float",
                    "params": [],
                    "body_ast": [
                        AstReturnStmt(
                            value=AstExprBinary(
                                op="+",
                                left=AstExprLiteral(kind="int", value="1"),
                                right=AstExprLiteral(kind="float", value="1.0"),
                            )
                        )
                    ],
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

    assert "sitofp i32 1 to float" in llvm_ir
    assert "fadd float" in llvm_ir


def test_compile_lockstep_supports_c_and_function_style_cast_syntax():
    result = compiler_module.compile_lockstep(
        """
        pure float cast_demo(int x) {
            return (float)x + float(x);
        }
        pipeline Main {
            bind { }
        }
        """,
        verbose=False,
    )

    return_expr = result.ast.pure_functions[0].body[0].value
    assert isinstance(return_expr, AstExprBinary)
    assert isinstance(return_expr.left, AstExprCast)
    assert return_expr.left.target_type.name == "float"
    assert isinstance(return_expr.right, AstExprCast)
    assert return_expr.right.target_type.name == "float"


def test_emit_llvm_ir_lowers_explicit_numeric_casts_to_llvm_conversions():
    llvm_ir = emit_llvm_ir(
        {
            "structs": [],
            "pure_functions": [
                {
                    "name": "widen",
                    "return_type": "float",
                    "params": [{"modifier": "in", "type": "int", "name": "x"}],
                    "body_ast": [
                        AstReturnStmt(
                            value=AstExprCast(
                                target_type="float",
                                value=AstExprVar(path=("x",)),
                            )
                        )
                    ],
                },
                {
                    "name": "narrow",
                    "return_type": "int",
                    "params": [{"modifier": "in", "type": "float", "name": "x"}],
                    "body_ast": [
                        AstReturnStmt(
                            value=AstExprCast(
                                target_type="int",
                                value=AstExprVar(path=("x",)),
                            )
                        )
                    ],
                },
                {
                    "name": "narrow_unsigned",
                    "return_type": "uint",
                    "params": [{"modifier": "in", "type": "float", "name": "x"}],
                    "body_ast": [
                        AstReturnStmt(
                            value=AstExprCast(
                                target_type="uint",
                                value=AstExprVar(path=("x",)),
                            )
                        )
                    ],
                },
            ],
            "shaders": [],
            "filters": [],
            "streams": [],
            "accumulators": [],
            "uniforms": [],
            "bind_routes": [],
        }
    )

    assert "sitofp i32" in llvm_ir
    assert "fptosi float" in llvm_ir
    assert "fptoui float" in llvm_ir


def test_emit_llvm_ir_casts_wide_integer_to_bool_with_nonzero_compare():
    llvm_ir = emit_llvm_ir(
        {
            "structs": [],
            "pure_functions": [
                {
                    "name": "int_to_bool",
                    "return_type": "bool",
                    "params": [{"modifier": "in", "type": "int", "name": "x"}],
                    "body_ast": [
                        AstReturnStmt(
                            value=AstExprCast(
                                target_type="bool",
                                value=AstExprVar(path=("x",)),
                            )
                        )
                    ],
                },
                {
                    "name": "uint_to_bool",
                    "return_type": "bool",
                    "params": [{"modifier": "in", "type": "uint", "name": "x"}],
                    "body_ast": [
                        AstReturnStmt(
                            value=AstExprCast(
                                target_type="bool",
                                value=AstExprVar(path=("x",)),
                            )
                        )
                    ],
                },
            ],
            "shaders": [],
            "filters": [],
            "streams": [],
            "accumulators": [],
            "uniforms": [],
            "bind_routes": [],
        }
    )

    assert llvm_ir.count("icmp ne i32") == 2
    assert "trunc i32" not in llvm_ir


def test_emit_llvm_ir_uses_unsigned_float_to_integer_cast_for_uint_targets():
    llvm_ir = emit_llvm_ir(
        {
            "structs": [],
            "pure_functions": [
                {
                    "name": "float_to_uint",
                    "return_type": "uint",
                    "params": [{"modifier": "in", "type": "float", "name": "x"}],
                    "body_ast": [
                        AstVarDeclStmt(
                            declared_type="uint",
                            name="y",
                            initializer=AstExprVar(path=("x",)),
                        ),
                        AstReturnStmt(value=AstExprVar(path=("y",))),
                    ],
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

    assert "fptoui float" in llvm_ir
    assert "fptosi float" not in llvm_ir


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
    assert "#define LOCKSTEP_SIMD_WIDTH 8" in header
    assert "#define LOCKSTEP_OFFSET_STREAM_RAW_POSITIONS 0" in header
    assert "#define LOCKSTEP_OFFSET_ACCUM_ENERGY 12" in header
    assert "#define LOCKSTEP_OFFSET_UNIFORM_DT 16" in header
    assert "void Lockstep_Tick(struct Lockstep_Arena* arena);" in header


def test_emit_c_header_honors_target_width_override_macro():
    header = emit_c_header(
        {"streams": [], "accumulators": [], "uniforms": []}, target_width=16
    )
    assert "#define LOCKSTEP_SIMD_WIDTH 16" in header


def test_emit_c_header_exposes_nested_leaf_offsets_for_soa_layout():
    header = emit_c_header(
        {
            "structs": [
                {
                    "name": "Inner",
                    "fields": [
                        {"type": "float", "name": "x"},
                        {"type": "float", "name": "y"},
                    ],
                },
                {
                    "name": "Outer",
                    "fields": [
                        {"type": "Inner", "name": "pos"},
                        {"type": "float", "name": "mass"},
                    ],
                },
            ],
            "streams": [{"name": "particles", "type": "Outer"}],
            "accumulators": [],
            "uniforms": [],
        }
    )

    assert "float stream_particles_pos_x;" in header
    assert "float stream_particles_pos_y;" in header
    assert "float stream_particles_mass;" in header
    assert "#define LOCKSTEP_OFFSET_STREAM_PARTICLES_POS_X 0" in header
    assert "#define LOCKSTEP_OFFSET_STREAM_PARTICLES_POS_Y 4" in header
    assert "#define LOCKSTEP_OFFSET_STREAM_PARTICLES_MASS 8" in header


def test_emit_c_header_uses_parallel_soa_blocks_for_stream_capacity():
    header = emit_c_header(
        {
            "structs": [
                {
                    "name": "Particle",
                    "fields": [
                        {"type": "float", "name": "x"},
                        {"type": "float", "name": "y"},
                    ],
                }
            ],
            "streams": [{"name": "particles", "type": "Particle", "capacity": "4"}],
            "accumulators": [],
            "uniforms": [],
        }
    )

    assert "float stream_particles_x[4];" in header
    assert "float stream_particles_y[4];" in header
    assert "#define LOCKSTEP_OFFSET_STREAM_PARTICLES_X 0" in header
    assert "#define LOCKSTEP_OFFSET_STREAM_PARTICLES_Y 16" in header
    assert "#define LOCKSTEP_ARENA_BYTES 32" in header


def test_emit_c_header_sizes_folded_array_and_vector_leaves_by_layout_bytes():
    header = emit_c_header(
        {
            "structs": [
                {
                    "name": "Vec",
                    "fields": [
                        {"type": "float[4]", "name": "xs"},
                        {"type": "vector<float,4>", "name": "ys"},
                    ],
                }
            ],
            "streams": [{"name": "values", "type": "Vec", "capacity": "8"}],
            "accumulators": [],
            "uniforms": [],
        }
    )

    assert "float xs[4];" in header
    assert "float ys[4];" in header
    assert "float stream_values_xs[32];" in header
    assert "float stream_values_ys[32];" in header
    assert "#define LOCKSTEP_OFFSET_STREAM_VALUES_XS 0" in header
    assert "#define LOCKSTEP_OFFSET_STREAM_VALUES_YS 128" in header
    assert "#define LOCKSTEP_ARENA_BYTES 256" in header
    assert (
        "_Static_assert(sizeof(struct Lockstep_Arena) == LOCKSTEP_ARENA_BYTES"
        in header
    )


def test_emit_c_header_includes_optional_saturated_write_debug_helpers():
    header = emit_c_header(
        {
            "streams": [{"name": "output_stream", "type": "float", "capacity": "16"}],
            "accumulators": [],
            "uniforms": [],
        }
    )

    assert "#ifdef LOCKSTEP_DEBUG_SATURATED_WRITES" in header
    assert "#include <stdio.h>" in header
    assert "#define LOCKSTEP_CAPACITY_STREAM_OUTPUT_STREAM 16" in header
    assert "#ifndef LOCKSTEP_SATURATED_WRITE_LOG" in header
    assert (
        "static inline size_t Lockstep_SaturatedWriteIndex(size_t index, size_t capacity, const char* stream_name)"
        in header
    )
    assert (
        'LOCKSTEP_SATURATED_WRITE_LOG(stream_name != NULL ? stream_name : "<unnamed>"'
        in header
    )


def test_emit_c_header_raises_lck502_when_cumulative_offsets_overflow(monkeypatch):
    monkeypatch.setattr(c_header_module, "_MAX_U64", 4)

    with pytest.raises(lockstep_compiler.LockstepCompileError) as exc_info:
        emit_c_header(
            {
                "streams": [],
                "accumulators": [
                    {"name": "a", "type": "float"},
                    {"name": "b", "type": "float"},
                ],
                "uniforms": [],
            }
        )

    assert [diag.code for diag in exc_info.value.errors] == ["LCK502"]


def test_emit_c_header_raises_lck502_when_single_leaf_allocation_overflows(monkeypatch):
    monkeypatch.setattr(c_header_module, "_MAX_U64", 4)

    with pytest.raises(lockstep_compiler.LockstepCompileError) as exc_info:
        emit_c_header(
            {
                "streams": [{"name": "samples", "type": "float", "capacity": "2"}],
                "accumulators": [],
                "uniforms": [],
            }
        )

    assert [diag.code for diag in exc_info.value.errors] == ["LCK502"]


def test_build_arena_layout_raises_lck503_for_recursive_struct_layout_cycle():
    with pytest.raises(lockstep_compiler.LockstepCompileError) as exc_info:
        build_arena_layout(
            {
                "structs": [
                    {"name": "A", "fields": [{"name": "b", "type": "B"}]},
                    {"name": "B", "fields": [{"name": "a", "type": "A"}]},
                ]
            }
        )

    assert [diag.code for diag in exc_info.value.errors] == ["LCK503"]


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
        def visit(self, _tree):
            return None

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
        semantic_validator=lambda _tree, **_kwargs: [],
        token_stream_cls=lambda lexer: object(),
        debug_visitor_cls=StubDebugVisitor,
    )

    assert "#define LOCKSTEP_ARENA_BYTES 0" in result.c_header
    assert "void Lockstep_Tick(struct Lockstep_Arena* arena);" in result.c_header


def test_emit_llvm_ir_raises_on_undefined_variable_reference():
    with pytest.raises(CodegenError, match="undefined variable"):
        emit_llvm_ir(
            {
                "pure_functions": [
                    {
                        "name": "demo",
                        "return_type": "float",
                        "params": [],
                        "body_ast": [
                            AstReturnStmt(value=AstExprVar(path=("missing",)))
                        ],
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


def test_emit_llvm_ir_raises_on_intrinsic_type_mismatch():
    with pytest.raises(CodegenError, match="intrinsic 'max' expects float arguments"):
        emit_llvm_ir(
            {
                "pure_functions": [
                    {
                        "name": "max",
                        "return_type": "float",
                        "params": [
                            {"name": "a", "type": "float"},
                            {"name": "b", "type": "float"},
                        ],
                        "intrinsic": True,
                    },
                    {
                        "name": "demo",
                        "return_type": "float",
                        "params": [],
                        "body_ast": [
                            AstReturnStmt(
                                value=AstExprCall(
                                    name="max",
                                    args=(
                                        AstExprLiteral(kind="int", value="1"),
                                        AstExprLiteral(kind="int", value="2"),
                                    ),
                                )
                            )
                        ],
                    },
                ],
                "shaders": [],
                "filters": [],
                "streams": [],
                "accumulators": [],
                "uniforms": [],
                "bind_routes": [],
            }
        )


def test_compile_lockstep_enforces_source_size_limit():
    with pytest.raises(lockstep_compiler.LockstepCompileError) as exc_info:
        lockstep_compiler.compile_lockstep(
            "pipeline P { bind { } }",
            frontend_limits=lockstep_compiler.FrontendLimits(max_source_bytes=8),
        )

    assert exc_info.value.phase == "parse"
    assert exc_info.value.errors[0].code == "LCK003"


def test_compile_lockstep_enforces_expression_nesting_limit():
    source = """
shader S(in float v) {
    float x = (((((((v)))))));
}
pipeline P {
    stream<float,1> input;
    bind {
        input = S(input);
    }
}
"""
    with pytest.raises(lockstep_compiler.LockstepCompileError) as exc_info:
        lockstep_compiler.compile_lockstep(
            source,
            frontend_limits=lockstep_compiler.FrontendLimits(max_expression_nesting=2),
        )

    assert exc_info.value.phase == "parse"
    assert exc_info.value.errors[0].code == "LCK005"


def test_compile_lockstep_enforces_parse_timeout(monkeypatch):
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
    monkeypatch.setattr(
        compiler_module.time, "monotonic", lambda: next(monotonic_values)
    )

    with pytest.raises(lockstep_compiler.LockstepCompileError) as exc_info:
        lockstep_compiler.compile_lockstep(
            "pipeline P { bind { } }",
            frontend_limits=lockstep_compiler.FrontendLimits(parse_timeout_ms=1),
        )

    assert exc_info.value.phase == "parse"
    assert exc_info.value.errors[0].code == "LCK004"


def test_codegen_uses_unsigned_ops_for_uint_math():
    source = """
    pure uint half(uint value) {
        return value / uint(2);
    }

    pure uint remainder(uint value) {
        return value % uint(3);
    }

    pure bool less_than(uint a, uint b) {
        return a < b;
    }

    pure uint shift_right(uint value) {
        return value >> 1;
    }
    """
    result = lockstep_compiler.compile_lockstep(source)
    llvm_ir = result.llvm_ir

    assert "udiv i32" in llvm_ir
    assert "urem i32" in llvm_ir
    assert "icmp ult i32" in llvm_ir
    assert "lshr i32" in llvm_ir


def test_codegen_promotes_asymmetric_uint_binary_operands():
    source = """
    pure uint literal_divided_by_uint(uint value) {
        return 5 / value;
    }

    pure uint literal_remainder_uint(uint value) {
        return 5 % value;
    }

    pure uint literal_divided_by_casted_uint(int value) {
        return 10 / uint(value);
    }

    pure bool literal_less_than_uint(uint value) {
        return 5 < value;
    }

    pure bool literal_less_than_casted_uint(int value) {
        return 5 < uint(value);
    }

    pure uint signed_divided_by_uint(int lhs, uint rhs) {
        return lhs / rhs;
    }

    pure bool signed_less_than_uint(int lhs, uint rhs) {
        return lhs < rhs;
    }

    pure bool nested_uint_expr_compares_unsigned(uint value) {
        return (5 / value) < 10;
    }
    """
    result = lockstep_compiler.compile_lockstep(source)
    llvm_ir = result.llvm_ir

    assert llvm_ir.count("udiv i32") == 4
    assert "urem i32" in llvm_ir
    assert llvm_ir.count("icmp ult i32") == 4
    assert "sdiv i32" not in llvm_ir
    assert "srem i32" not in llvm_ir
    assert "icmp slt i32" not in llvm_ir


def test_codegen_uses_unsigned_ops_for_uint_struct_fields():
    source = """
    struct Pair { uint lhs; uint rhs; };

    pure uint quotient(Pair p) {
        return p.lhs / p.rhs;
    }

    pure uint remainder(Pair p) {
        return p.lhs % p.rhs;
    }

    pure bool less_than(Pair p) {
        return p.lhs < p.rhs;
    }
    """
    result = lockstep_compiler.compile_lockstep(source)
    llvm_ir = result.llvm_ir

    assert "udiv i32" in llvm_ir
    assert "urem i32" in llvm_ir
    assert "icmp ult i32" in llvm_ir


def test_emit_llvm_ir_compacts_filter_outputs_with_return_predicate():
    from lockstep_compiler.ast import (
        AstKernelBindRoute,
        AstKernelDecl,
        AstKernelParam,
        AstPipelineDecl,
        AstStreamDecl,
    )

    program = AstProgram(
        filters=(
            AstKernelDecl(
                name="KeepPositive",
                params=(
                    AstKernelParam(modifier="in", declared_type="float", name="src"),
                    AstKernelParam(modifier="out", declared_type="float", name="dst"),
                ),
                body=(
                    AstAssignStmt(target=("dst",), value=AstExprVar(path=("src",))),
                    AstReturnStmt(
                        value=AstExprBinary(
                            op=">",
                            left=AstExprVar(path=("src",)),
                            right=AstExprLiteral(kind="float", value="0.0"),
                        )
                    ),
                ),
            ),
        ),
        pipelines=(
            AstPipelineDecl(
                name="Main",
                streams=(
                    AstStreamDecl(name="inp", declared_type="float", capacity="4"),
                    AstStreamDecl(name="out", declared_type="float", capacity="4"),
                ),
                bind_routes=(
                    AstKernelBindRoute(
                        target="out",
                        kernel="KeepPositive",
                        args=("inp", "out"),
                        route="out = KeepPositive(inp, out);",
                    ),
                ),
            ),
        ),
    )

    llvm_ir = emit_llvm_ir(program)

    assert 'define i1 @"filter_KeepPositive"' in llvm_ir
    assert '"KeepPositive_write_idx"' in llvm_ir
    assert '"filter_write_select"' in llvm_ir
    assert '"filter_KeepPositive_store"' in llvm_ir
    assert '"filter_KeepPositive_skip"' in llvm_ir
