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
