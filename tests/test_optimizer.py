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
