import io
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lockstep_compiler.cli import run_cli
from lockstep_compiler.simulator import simulate_pipeline_entities


def test_simulate_pipeline_entities_tracks_route_cardinality_and_fold():
    entities = {
        "streams": [
            {"name": "in_stream", "type": "Vec", "capacity": "8"},
            {"name": "out_stream", "type": "Vec", "capacity": "8"},
        ],
        "accumulators": [{"name": "energy", "type": "float"}],
        "uniforms": [{"name": "dt", "type": "float", "initializer": "0.016"}],
        "shaders": [
            {
                "name": "Apply",
                "params": [
                    {"modifier": "in", "type": "Vec", "name": "src"},
                    {"modifier": "out", "type": "Vec", "name": "dst"},
                    {"modifier": "accum", "type": "float", "name": "energy"},
                ],
            }
        ],
        "filters": [],
        "bind_routes": [
            "out_stream = Apply(in_stream, out_stream, energy);",
            "uniform float total = fold sum(energy);",
        ],
        "bind_routes_ir": [
            {
                "kind": "kernel",
                "target": "out_stream",
                "kernel": "Apply",
                "args": ["in_stream", "out_stream", "energy"],
                "route": "out_stream = Apply(in_stream, out_stream, energy);",
            },
            {
                "kind": "fold",
                "uniform_type": "float",
                "uniform_name": "total",
                "operator": "sum",
                "source": "energy",
                "route": "uniform float total = fold sum(energy);",
            },
        ],
    }

    simulation = simulate_pipeline_entities(
        entities,
        stream_inputs={"in_stream": [{"id": 1}, {"id": 2}]},
    )

    assert simulation["routes"][0]["input_count"] == 2
    assert simulation["routes"][0]["output_count"] == 2
    assert simulation["routes"][1]["kind"] == "fold"
    assert simulation["uniforms"]["total"] == 2


def test_run_cli_simulate_prints_json_for_compiler_result():
    def fake_compiler(_source):
        return {
            "streams": [{"name": "s", "type": "int", "capacity": "4"}],
            "accumulators": [],
            "uniforms": [],
            "shaders": [],
            "filters": [],
            "bind_routes": [],
        }

    stdout = io.StringIO()
    exit_code = run_cli(
        ["--simulate"],
        stdin=io.StringIO("pipeline P { }"),
        stdout=stdout,
        compiler=fake_compiler,
    )

    assert exit_code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["streams"] == {"s": []}


def test_run_cli_simulate_reads_input_file(tmp_path):
    input_file = tmp_path / "sim-input.json"
    input_file.write_text('{"streams": {"s": [1, 2, 3]}}', encoding="utf-8")

    def fake_compiler(_source):
        return {
            "streams": [{"name": "s", "type": "int", "capacity": "4"}],
            "accumulators": [],
            "uniforms": [],
            "shaders": [],
            "filters": [],
            "bind_routes": [],
        }

    stdout = io.StringIO()
    exit_code = run_cli(
        ["--simulate", "--simulate-input", str(input_file)],
        stdin=io.StringIO("pipeline P { }"),
        stdout=stdout,
        compiler=fake_compiler,
    )

    assert exit_code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["streams"]["s"] == [1, 2, 3]


def test_simulate_pipeline_entities_saturates_stream_writes_at_capacity():
    entities = {
        "streams": [
            {"name": "in_stream", "type": "int", "capacity": "5"},
            {"name": "out_stream", "type": "int", "capacity": "2"},
        ],
        "accumulators": [],
        "uniforms": [],
        "shaders": [
            {
                "name": "Project",
                "params": [
                    {"modifier": "in", "type": "int", "name": "src"},
                    {"modifier": "out", "type": "int", "name": "dst"},
                ],
            }
        ],
        "filters": [],
        "bind_routes": ["out_stream = Project(in_stream, out_stream);"],
        "bind_routes_ir": [
            {
                "kind": "kernel",
                "target": "out_stream",
                "kernel": "Project",
                "args": ["in_stream", "out_stream"],
                "route": "out_stream = Project(in_stream, out_stream);",
            }
        ],
    }

    simulation = simulate_pipeline_entities(entities, stream_inputs={"in_stream": [1, 2, 3, 4]})

    assert simulation["streams"]["out_stream"] == [
        {"_source": 1, "_kernel": "Project"},
        {"_source": 4, "_kernel": "Project"},
    ]
