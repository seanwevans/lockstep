import io
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lockstep_compiler.cli import run_cli
from lockstep_compiler.simulator import parse_simulation_inputs, simulate_pipeline_entities


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


def test_run_cli_rejects_simulate_input_without_simulate(tmp_path):
    input_file = tmp_path / "sim-input.json"
    input_file.write_text('{"streams": {"s": [1]}}', encoding="utf-8")

    stderr = io.StringIO()
    exit_code = run_cli(
        ["--simulate-input", str(input_file)],
        stdin=io.StringIO("pipeline P { }"),
        stdout=io.StringIO(),
        stderr=stderr,
        compiler=lambda _source: {},
    )

    assert exit_code == 2
    assert "--simulate-input requires --simulate" in stderr.getvalue()


def test_run_cli_report_prints_single_json_payload_with_entities_and_simulation():
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
        ["--report"],
        stdin=io.StringIO("pipeline P { }"),
        stdout=stdout,
        compiler=fake_compiler,
    )

    assert exit_code == 0
    decoder = json.JSONDecoder()
    payload_text = stdout.getvalue()
    payload, index = decoder.raw_decode(payload_text)
    assert payload["entities"]["streams"][0]["name"] == "s"
    assert payload["simulation"]["streams"] == {"s": []}
    assert payload_text[index:].strip() == ""


def test_run_cli_rejects_combined_dump_and_simulate_modes():
    try:
        run_cli(
            ["--dump", "--simulate"],
            stdin=io.StringIO("pipeline P { }"),
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            compiler=lambda source: source,
        )
    except SystemExit as err:
        assert err.code == 2
    else:
        raise AssertionError("Expected parser to reject combined output-producing flags")

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



def test_parse_simulation_inputs_rejects_non_object_root():
    try:
        parse_simulation_inputs('[1, 2, 3]')
    except ValueError as err:
        assert "<root>" in str(err)
    else:
        raise AssertionError("Expected ValueError for non-object root")


def test_parse_simulation_inputs_rejects_invalid_field_types():
    try:
        parse_simulation_inputs('{"streams": [], "accumulators": {}}')
    except ValueError as err:
        assert "streams" in str(err)
    else:
        raise AssertionError("Expected ValueError for non-object streams field")


def test_parse_simulation_inputs_rejects_non_list_values():
    try:
        parse_simulation_inputs('{"streams": {"s": 42}, "accumulators": {}}')
    except ValueError as err:
        assert "streams.s" in str(err)
    else:
        raise AssertionError("Expected ValueError for non-list stream value")


def test_run_cli_simulate_reports_invalid_simulation_shape(tmp_path):
    input_file = tmp_path / "sim-input.json"
    input_file.write_text('{"streams": {"s": 1}}', encoding="utf-8")

    def fake_compiler(_source):
        return {
            "streams": [{"name": "s", "type": "int", "capacity": "4"}],
            "accumulators": [],
            "uniforms": [],
            "shaders": [],
            "filters": [],
            "bind_routes": [],
        }

    stderr = io.StringIO()
    exit_code = run_cli(
        ["--simulate", "--simulate-input", str(input_file)],
        stdin=io.StringIO("pipeline P { }"),
        stdout=io.StringIO(),
        stderr=stderr,
        compiler=fake_compiler,
    )

    assert exit_code == 1
    assert "Invalid simulation input:" in stderr.getvalue()
    assert "streams.s" in stderr.getvalue()


def test_fold_values_uses_jit_reduce_for_sum_and_avg(monkeypatch):
    calls = []

    def fake_reduce(operator, values):
        calls.append((operator, list(values)))
        return 42.5 if operator == "sum" else 10.625

    monkeypatch.setattr("lockstep_compiler.simulator._jit_numeric_reduce", fake_reduce)

    entities = {
        "streams": [],
        "accumulators": [{"name": "energy", "type": "float"}],
        "uniforms": [
            {"name": "total", "type": "float", "initializer": "0"},
            {"name": "average", "type": "float", "initializer": "0"},
        ],
        "shaders": [],
        "filters": [],
        "bind_routes": [
            "uniform float total = fold sum(energy);",
            "uniform float average = fold avg(energy);",
        ],
        "bind_routes_ir": [
            {
                "kind": "fold",
                "uniform_type": "float",
                "uniform_name": "total",
                "operator": "sum",
                "source": "energy",
                "route": "uniform float total = fold sum(energy);",
            },
            {
                "kind": "fold",
                "uniform_type": "float",
                "uniform_name": "average",
                "operator": "avg",
                "source": "energy",
                "route": "uniform float average = fold avg(energy);",
            },
        ],
    }

    simulation = simulate_pipeline_entities(
        entities,
        accumulator_inputs={"energy": [1.0, 2.0, 3.0, 4.0]},
    )

    assert simulation["uniforms"]["total"] == 42.5
    assert simulation["uniforms"]["average"] == 10.625
    assert calls == [
        ("sum", [1.0, 2.0, 3.0, 4.0]),
        ("avg", [1.0, 2.0, 3.0, 4.0]),
    ]


def test_jit_numeric_reduce_falls_back_to_python_sum_on_error(monkeypatch):
    monkeypatch.setattr("lockstep_compiler.simulator._jit_reduce_callable", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    assert simulate_pipeline_entities(
        {
            "streams": [],
            "accumulators": [{"name": "energy", "type": "float"}],
            "uniforms": [{"name": "total", "type": "float", "initializer": "0"}],
            "shaders": [],
            "filters": [],
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
        accumulator_inputs={"energy": [1.25, 2.75]},
    )["uniforms"]["total"] == 4.0
