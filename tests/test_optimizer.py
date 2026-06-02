import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lockstep_compiler.optimizer import optimize_bind_routes


def test_optimize_bind_routes_fuses_linear_shader_filter_chain():
    result = optimize_bind_routes(
        [
            "tmp = Shade(inp, tmp);",
            "out = Blur(tmp, out);",
            "final = Keep(out, final);",
        ],
        shader_names={"Shade", "Keep"},
        filter_names={"Blur"},
    )

    assert result["optimized_bind_routes"] == [
        "final = FUSED[Shade -> Blur -> Keep];",
    ]
    assert result["fused_groups"] == [
        {
            "nodes": ["Shade", "Blur", "Keep"],
            "entry_args": ["inp", "tmp"],
            "output": "final",
            "eliminated_intermediates": ["tmp", "out"],
            "source_routes": [
                "tmp = Shade(inp, tmp);",
                "out = Blur(tmp, out);",
                "final = Keep(out, final);",
            ],
        }
    ]


def test_optimize_bind_routes_keeps_route_when_intermediate_has_multiple_uses():
    result = optimize_bind_routes(
        [
            "tmp = Shade(inp, tmp);",
            "out = Blur(tmp, out);",
            "echo = Keep(tmp, echo);",
        ],
        shader_names={"Shade", "Keep"},
        filter_names={"Blur"},
    )

    assert result["optimized_bind_routes"] == [
        "tmp = Shade(inp, tmp);",
        "out = Blur(tmp, out);",
        "echo = Keep(tmp, echo);",
    ]
    assert result["fused_groups"] == []


def test_optimize_bind_routes_uses_structured_bind_route_ir_when_available():
    result = optimize_bind_routes(
        [
            "tmp = Shade(inp, tmp);",
            "uniform float u0 = fold sum(tmp);",
            "out = Blur(tmp, out);",
        ],
        shader_names={"Shade"},
        filter_names={"Blur"},
        bind_routes_ir=[
            {
                "kind": "kernel",
                "target": "tmp",
                "kernel": "Shade",
                "args": ["inp", "tmp"],
                "route": "tmp = Shade(inp, tmp);",
            },
            {
                "kind": "fold",
                "uniform_type": "float",
                "uniform_name": "u0",
                "operator": "sum",
                "source": "tmp",
                "route": "uniform float u0 = fold sum(tmp);",
            },
            {
                "kind": "kernel",
                "target": "out",
                "kernel": "Blur",
                "args": ["tmp", "out"],
                "route": "out = Blur(tmp, out);",
            },
        ],
    )

    assert result["optimized_bind_routes"] == [
        "tmp = Shade(inp, tmp);",
        "uniform float u0 = fold sum(tmp);",
        "out = Blur(tmp, out);",
    ]
    assert result["fused_groups"] == []


def test_optimize_bind_routes_uses_liveness_not_global_use_count():
    result = optimize_bind_routes(
        [
            "tmp = Shade(inp, tmp);",
            "out = Blur(tmp, out);",
            "tmp = Reset(seed, scratch);",
            "final = Keep(tmp, final);",
        ],
        shader_names={"Shade", "Keep", "Reset"},
        filter_names={"Blur"},
    )

    assert result["optimized_bind_routes"] == [
        "out = FUSED[Shade -> Blur];",
        "final = FUSED[Reset -> Keep];",
    ]
    assert result["fused_groups"] == [
        {
            "nodes": ["Shade", "Blur"],
            "entry_args": ["inp", "tmp"],
            "output": "out",
            "eliminated_intermediates": ["tmp"],
            "source_routes": [
                "tmp = Shade(inp, tmp);",
                "out = Blur(tmp, out);",
            ],
        },
        {
            "nodes": ["Reset", "Keep"],
            "entry_args": ["seed", "scratch"],
            "output": "final",
            "eliminated_intermediates": ["tmp"],
            "source_routes": [
                "tmp = Reset(seed, scratch);",
                "final = Keep(tmp, final);",
            ],
        },
    ]


def test_optimize_bind_routes_fuses_split_join_subgraph_from_dag():
    result = optimize_bind_routes(
        [
            "tmp = Source(inp, tmp);",
            "left = Left(tmp, left);",
            "right = Right(tmp, right);",
            "final = Join(left, right, final);",
        ],
        shader_names={"Source", "Left", "Right", "Join"},
        filter_names=set(),
    )

    assert result["optimized_bind_routes"] == [
        "final = FUSED[Source -> Left -> Right -> Join];",
    ]
    assert result["fused_groups"] == [
        {
            "nodes": ["Source", "Left", "Right", "Join"],
            "entry_args": ["inp", "tmp"],
            "output": "final",
            "eliminated_intermediates": ["tmp", "left", "right"],
            "source_routes": [
                "tmp = Source(inp, tmp);",
                "left = Left(tmp, left);",
                "right = Right(tmp, right);",
                "final = Join(left, right, final);",
            ],
        }
    ]


def test_optimize_bind_routes_eliminates_overwritten_dead_kernel_routes():
    result = optimize_bind_routes(
        [
            "tmp = Shade(inp, tmp);",
            "dead = Dead(tmp, dead);",
            "dead = Reset(seed, scratch);",
            "out = Keep(tmp, out);",
        ],
        shader_names={"Shade", "Dead", "Reset", "Keep"},
        filter_names=set(),
    )

    assert result["optimized_bind_routes"] == [
        "tmp = Shade(inp, tmp);",
        "dead = Reset(seed, scratch);",
        "out = Keep(tmp, out);",
    ]
    assert result["fused_groups"] == []


def test_compile_pipeline_lowers_fused_group_to_single_simd_loop_without_intermediate_stream_writes():
    from llvmlite import binding as llvm

    import lockstep_compiler

    source = """
    shader A(in float src, out float tmp) { tmp = src + 1.0; }
    shader B(in float tmp, out float dst) { dst = tmp * 2.0; }

    pipeline P {
        stream<float, 4> inp;
        stream<float, 4> tmp;
        stream<float, 4> dst;

        bind {
            tmp = A(inp, tmp);
            dst = B(tmp, dst);
        }
    }
    """

    result = lockstep_compiler.compile_lockstep(
        source,
        semantic_validator=lambda _tree, **_kwargs: [],
        target_width=4,
    )

    assert result.entities["optimized_bind_routes"] == ["dst = FUSED[A -> B];"]
    assert result.entities["fused_bind_groups"] == [
        {
            "nodes": ["A", "B"],
            "entry_args": ["inp", "tmp"],
            "output": "dst",
            "eliminated_intermediates": ["tmp"],
            "source_routes": ["tmp=A(inp,tmp);", "dst=B(tmp,dst);"],
        }
    ]
    assert "fused_0_body" in result.llvm_ir
    assert "route_A_body" not in result.llvm_ir
    assert "route_B_body" not in result.llvm_ir
    assert "<4 x float>" in result.llvm_ir
    assert '"fused_inp_vector" = load <4 x float>' in result.llvm_ir
    assert 'store <4 x float> %"fused_math.1"' in result.llvm_ir
    assert '"fused_math" = fadd <4 x float>' in result.llvm_ir
    assert '"fused_math.1" = fmul <4 x float>' in result.llvm_ir
    assert "fused_load_lane_" not in result.llvm_ir
    assert "fused_store_lane_" not in result.llvm_ir
    assert "stream_tmp_value_ptr" not in result.llvm_ir
    assert 'call void @"shader_A"' not in result.llvm_ir
    assert 'call void @"shader_B"' not in result.llvm_ir
    llvm.parse_assembly(result.llvm_ir)


def test_compile_pipeline_vectorizes_fused_struct_field_assignments(monkeypatch):
    from llvmlite import binding as llvm

    import lockstep_compiler
    import lockstep_compiler.compiler as compiler_module

    source = """
    struct Vec2 { float x; float y; };
    struct Particle { Vec2 velocity; float mass; };

    shader Kick(in Particle src, out Particle tmp) {
        tmp.velocity.x = src.velocity.x + 1.0;
        tmp.velocity.y = src.velocity.y + 2.0;
        tmp.mass = src.mass;
    }

    shader Damp(in Particle tmp, out Particle dst) {
        dst.velocity.x = tmp.velocity.x * 0.5;
        dst.velocity.y = tmp.velocity.y * 0.25;
        dst.mass = tmp.mass;
    }

    pipeline P {
        stream<Particle, 4> src;
        stream<Particle, 4> tmp;
        stream<Particle, 4> dst;

        bind {
            tmp = Kick(src, tmp);
            dst = Damp(tmp, dst);
        }
    }
    """

    monkeypatch.setattr(compiler_module, "emit_c_header", lambda *_args, **_kwargs: "")
    result = lockstep_compiler.compile_lockstep(
        source,
        semantic_validator=lambda _tree, **_kwargs: [],
        target_width=4,
    )

    assert result.entities["optimized_bind_routes"] == ["dst = FUSED[Kick -> Damp];"]
    assert "fused_0_body" in result.llvm_ir
    assert "route_Kick_body" not in result.llvm_ir
    assert "route_Damp_body" not in result.llvm_ir
    assert 'call void @"shader_Kick"' not in result.llvm_ir
    assert 'call void @"shader_Damp"' not in result.llvm_ir
    assert "fadd <4 x float>" in result.llvm_ir
    assert "fmul <4 x float>" in result.llvm_ir
    assert "insertelement <4 x float>" in result.llvm_ir
    assert "extractelement <4 x float>" in result.llvm_ir
    llvm.parse_assembly(result.llvm_ir)


def test_compile_passes_per_pipeline_bind_optimization_to_codegen(monkeypatch):
    import lockstep_compiler
    import lockstep_compiler.compiler as compiler_module

    source = """
    shader A(in float src, out float tmp) { tmp = src + 1.0; }
    shader B(in float tmp, out float dst) { dst = tmp * 2.0; }

    pipeline P1 {
        stream<float, 4> inp;
        stream<float, 4> tmp;
        stream<float, 4> dst;

        bind {
            tmp = A(inp, tmp);
            dst = B(tmp, dst);
        }
    }

    pipeline P2 {
        stream<float, 4> inp2;
        stream<float, 4> tmp2;
        stream<float, 4> dst2;

        bind {
            tmp2 = A(inp2, tmp2);
            dst2 = B(tmp2, dst2);
        }
    }
    """

    optimization_calls = []
    captured_bind_optimization = None

    def fake_optimize_bind_routes(bind_routes, **kwargs):
        optimization_calls.append((list(bind_routes), kwargs))
        pipeline_number = len(optimization_calls)
        return {
            "optimized_bind_routes": [f"optimized-pipeline-{pipeline_number}"],
            "fused_groups": [{"pipeline": pipeline_number}],
        }

    def fake_emit_llvm_ir(_program, *, bind_optimization=None, **_kwargs):
        nonlocal captured_bind_optimization
        captured_bind_optimization = bind_optimization
        return '; ModuleID = "lockstep"\n'

    monkeypatch.setattr(
        compiler_module, "optimize_bind_routes", fake_optimize_bind_routes
    )
    monkeypatch.setattr(compiler_module, "emit_llvm_ir", fake_emit_llvm_ir)

    result = lockstep_compiler.compile_lockstep(
        source,
        semantic_validator=lambda _tree, **_kwargs: [],
        target_width=4,
    )

    assert [call[0] for call in optimization_calls] == [
        ["tmp=A(inp,tmp);", "dst=B(tmp,dst);"],
        ["tmp2=A(inp2,tmp2);", "dst2=B(tmp2,dst2);"],
    ]
    assert result.entities["optimized_bind_routes"] == [
        "optimized-pipeline-1",
        "optimized-pipeline-2",
    ]
    assert result.entities["fused_bind_groups"] == [
        {"pipeline": 1},
        {"pipeline": 2},
    ]
    assert captured_bind_optimization == {
        "optimized_bind_routes": ["optimized-pipeline-1", "optimized-pipeline-2"],
        "fused_groups": [{"pipeline": 1}, {"pipeline": 2}],
        "pipeline_optimizations": [
            {
                "optimized_bind_routes": ["optimized-pipeline-1"],
                "fused_groups": [{"pipeline": 1}],
            },
            {
                "optimized_bind_routes": ["optimized-pipeline-2"],
                "fused_groups": [{"pipeline": 2}],
            },
        ],
    }
    assert len(captured_bind_optimization["pipeline_optimizations"]) == len(
        result.ast.pipelines
    )


def test_emit_llvm_ir_honors_per_pipeline_bind_optimization(monkeypatch):
    from llvmlite import binding as llvm

    import lockstep_compiler
    import lockstep_compiler.codegen as codegen_module

    source = """
    shader A(in float src, out float tmp) { tmp = src + 1.0; }
    shader B(in float tmp, out float dst) { dst = tmp * 2.0; }

    pipeline P1 {
        stream<float, 4> inp;
        stream<float, 4> tmp;
        stream<float, 4> dst;

        bind {
            tmp = A(inp, tmp);
            dst = B(tmp, dst);
        }
    }

    pipeline P2 {
        stream<float, 4> inp2;
        stream<float, 4> tmp2;
        stream<float, 4> dst2;

        bind {
            tmp2 = A(inp2, tmp2);
            dst2 = B(tmp2, dst2);
        }
    }
    """
    result = lockstep_compiler.compile_lockstep(
        source,
        semantic_validator=lambda _tree, **_kwargs: [],
        target_width=4,
    )

    def fail_if_codegen_reoptimizes(*_args, **_kwargs):
        raise AssertionError("emit_llvm_ir ignored supplied pipeline optimizations")

    monkeypatch.setattr(
        codegen_module, "optimize_bind_routes", fail_if_codegen_reoptimizes
    )

    llvm_ir = codegen_module.emit_llvm_ir(
        result.ast,
        target_width=4,
        bind_optimization={
            "optimized_bind_routes": [
                "tmp=A(inp,tmp);",
                "dst=B(tmp,dst);",
                "tmp2=A(inp2,tmp2);",
                "dst2=B(tmp2,dst2);",
            ],
            "fused_groups": [],
            "pipeline_optimizations": [
                {
                    "optimized_bind_routes": ["tmp=A(inp,tmp);", "dst=B(tmp,dst);"],
                    "fused_groups": [],
                },
                {
                    "optimized_bind_routes": [
                        "tmp2=A(inp2,tmp2);",
                        "dst2=B(tmp2,dst2);",
                    ],
                    "fused_groups": [],
                },
            ],
        },
    )

    assert "route_A_body" in llvm_ir
    assert "route_B_body" in llvm_ir
    llvm.parse_assembly(llvm_ir)
