import io
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lockstep_compiler.ast import (
    AstAssignStmt,
    AstExprBinary,
    AstExprCall,
    AstExprLiteral,
    AstExprVar,
    AstReturnStmt,
)
from lockstep_compiler.cli import run_cli
from lockstep_compiler.simulator import (
    SimulatorRuntimeError,
    _resolve_sim_path,
    parse_simulation_inputs,
    simulate_pipeline_entities,
    simulate_pipeline_source,
)


def test_resolve_sim_path_treats_zero_as_mapped_value():
    assert _resolve_sim_path({"count": 0, "particle": {"mass": 0}}, ("count",)) == 0
    assert _resolve_sim_path({"particle": {"mass": 0}}, ("particle", "mass")) == 0


def test_resolve_sim_path_rejects_unmapped_direct_lookup():
    try:
        _resolve_sim_path({"count": 0}, ("missing",))
    except SimulatorRuntimeError as err:
        message = str(err)
        assert "Unmapped simulator identifier 'missing'" in message
        assert "available identifiers: count" in message
    else:
        raise AssertionError("Expected SimulatorRuntimeError for unmapped lookup")


def test_resolve_sim_path_rejects_unmapped_nested_lookup():
    try:
        _resolve_sim_path({"particle": {"mass": 1}}, ("particle", "velocity"))
    except SimulatorRuntimeError as err:
        message = str(err)
        assert "Unmapped simulator member 'velocity'" in message
        assert "at 'particle'" in message
        assert "available members: mass" in message
    else:
        raise AssertionError("Expected SimulatorRuntimeError for unmapped member lookup")


def test_simulate_pipeline_source_uses_compiled_runtime_entities():
    source = """
    shader Copy(in float src, out float dst) { dst = src; }

    pipeline P {
        stream<float, 4> inp;
        stream<float, 4> dst;

        bind {
            dst = Copy(inp, dst);
        }
    }
    """

    simulation = simulate_pipeline_source(
        source,
        stream_inputs={"inp": [1.0, 2.0]},
    )

    assert simulation["streams"]["dst"] == [1.0, 2.0]
    assert simulation["routes"] == [
        {
            "route": "dst=Copy(inp,dst);",
            "kind": "shader",
            "input_count": 2,
            "output_count": 2,
            "notes": None,
        }
    ]


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


def test_simulate_pipeline_entities_executes_shader_body_ast_assignments():
    entities = {
        "streams": [
            {"name": "in_stream", "type": "float", "capacity": "8"},
            {"name": "out_stream", "type": "float", "capacity": "8"},
        ],
        "accumulators": [{"name": "energy", "type": "float"}],
        "uniforms": [{"name": "gain", "type": "float", "initializer": "2.5"}],
        "pure_functions": [
            {
                "name": "bias",
                "return_type": "float",
                "params": [{"type": "float", "name": "x"}],
                "body_ast": [
                    AstReturnStmt(
                        value=AstExprBinary(
                            op="+",
                            left=AstExprVar(path=("x",)),
                            right=AstExprLiteral(kind="float", value="1.0"),
                        )
                    )
                ],
            }
        ],
        "shaders": [
            {
                "name": "Scale",
                "params": [
                    {"modifier": "in", "type": "float", "name": "src"},
                    {"modifier": "out", "type": "float", "name": "dst"},
                    {"modifier": "accum", "type": "float", "name": "energy"},
                    {"modifier": "uniform", "type": "float", "name": "gain"},
                ],
                "body_ast": [
                    AstAssignStmt(
                        target=("dst",),
                        value=AstExprBinary(
                            op="*",
                            left=AstExprCall(
                                name="bias", args=(AstExprVar(path=("src",)),)
                            ),
                            right=AstExprVar(path=("gain",)),
                        ),
                    ),
                    AstAssignStmt(
                        target=("energy",),
                        value=AstExprCall(
                            name="abs", args=(AstExprVar(path=("dst",)),)
                        ),
                    ),
                ],
            }
        ],
        "filters": [],
        "bind_routes": ["out_stream = Scale(in_stream, out_stream, energy, gain);"],
        "bind_routes_ir": [
            {
                "kind": "kernel",
                "target": "out_stream",
                "kernel": "Scale",
                "args": ["in_stream", "out_stream", "energy", "gain"],
                "route": "out_stream = Scale(in_stream, out_stream, energy, gain);",
            }
        ],
    }

    simulation = simulate_pipeline_entities(
        entities, stream_inputs={"in_stream": [1.0, -3.0]}
    )

    assert simulation["streams"]["out_stream"] == [5.0, -5.0]
    assert simulation["accumulators"]["energy"] == [5.0, 5.0]


def test_simulate_pipeline_entities_interprets_mapping_body_ast_statements():
    entities = {
        "streams": [
            {"name": "in_stream", "type": "float", "capacity": "8"},
            {"name": "out_stream", "type": "float", "capacity": "8"},
        ],
        "accumulators": [{"name": "energy", "type": "float"}],
        "uniforms": [{"name": "gain", "type": "float", "initializer": "3.0"}],
        "pure_functions": [
            {
                "name": "offset",
                "return_type": "float",
                "params": [{"type": "float", "name": "x"}],
                "body_ast": [
                    {
                        "value": {
                            "op": "+",
                            "left": {"path": ["x"]},
                            "right": {"kind": "float", "value": "2.0"},
                        }
                    }
                ],
            }
        ],
        "shaders": [
            {
                "name": "Scale",
                "params": [
                    {"modifier": "in", "type": "float", "name": "src"},
                    {"modifier": "out", "type": "float", "name": "dst"},
                    {"modifier": "accum", "type": "float", "name": "energy"},
                    {"modifier": "uniform", "type": "float", "name": "gain"},
                ],
                "body_ast": [
                    {
                        "declared_type": "float",
                        "name": "scaled",
                        "initializer": {
                            "op": "*",
                            "left": {"name": "offset", "args": [{"path": ["src"]}]},
                            "right": {"path": ["gain"]},
                        },
                    },
                    {"target": ["dst"], "value": {"path": ["scaled"]}},
                    {
                        "target": ["energy"],
                        "value": {"name": "abs", "args": [{"path": ["dst"]}]},
                    },
                ],
            }
        ],
        "filters": [],
        "bind_routes": ["out_stream = Scale(in_stream, out_stream, energy, gain);"],
        "bind_routes_ir": [
            {
                "kind": "kernel",
                "target": "out_stream",
                "kernel": "Scale",
                "args": ["in_stream", "out_stream", "energy", "gain"],
                "route": "out_stream = Scale(in_stream, out_stream, energy, gain);",
            }
        ],
    }

    simulation = simulate_pipeline_entities(
        entities, stream_inputs={"in_stream": [1.0, -4.0]}
    )

    assert simulation["streams"]["out_stream"] == [9.0, -6.0]
    assert simulation["accumulators"]["energy"] == [9.0, 6.0]


def test_simulate_pipeline_entities_rejects_missing_identifier_reference():
    entities = {
        "streams": [
            {"name": "in_stream", "type": "float", "capacity": "8"},
            {"name": "out_stream", "type": "float", "capacity": "8"},
        ],
        "accumulators": [],
        "uniforms": [],
        "shaders": [
            {
                "name": "BadReference",
                "params": [
                    {"modifier": "in", "type": "float", "name": "src"},
                    {"modifier": "out", "type": "float", "name": "dst"},
                ],
                "body_ast": [
                    AstAssignStmt(target=("dst",), value=AstExprVar(path=("missing",)))
                ],
            }
        ],
        "filters": [],
        "bind_routes": ["out_stream = BadReference(in_stream, out_stream);"],
        "bind_routes_ir": [
            {
                "kind": "kernel",
                "target": "out_stream",
                "kernel": "BadReference",
                "args": ["in_stream", "out_stream"],
                "route": "out_stream = BadReference(in_stream, out_stream);",
            }
        ],
    }

    try:
        simulate_pipeline_entities(entities, stream_inputs={"in_stream": [1.0]})
    except SimulatorRuntimeError as err:
        message = str(err)
        assert "Unmapped simulator identifier 'missing'" in message
        assert "available identifiers: dst, src" in message
    else:
        raise AssertionError("Expected SimulatorRuntimeError for missing identifier")


def test_simulate_pipeline_entities_rejects_missing_struct_member_reference():
    entities = {
        "streams": [
            {"name": "in_stream", "type": "Particle", "capacity": "8"},
            {"name": "out_stream", "type": "float", "capacity": "8"},
        ],
        "accumulators": [],
        "uniforms": [],
        "shaders": [
            {
                "name": "ReadMissingMember",
                "params": [
                    {"modifier": "in", "type": "Particle", "name": "src"},
                    {"modifier": "out", "type": "float", "name": "dst"},
                ],
                "body_ast": [
                    AstAssignStmt(
                        target=("dst",), value=AstExprVar(path=("src", "missing"))
                    )
                ],
            }
        ],
        "filters": [],
        "bind_routes": ["out_stream = ReadMissingMember(in_stream, out_stream);"],
        "bind_routes_ir": [
            {
                "kind": "kernel",
                "target": "out_stream",
                "kernel": "ReadMissingMember",
                "args": ["in_stream", "out_stream"],
                "route": "out_stream = ReadMissingMember(in_stream, out_stream);",
            }
        ],
    }

    try:
        simulate_pipeline_entities(
            entities, stream_inputs={"in_stream": [{"value": 1.0}]}
        )
    except SimulatorRuntimeError as err:
        message = str(err)
        assert "Unmapped simulator member 'missing'" in message
        assert "while resolving 'src.missing'" in message
        assert "available members: value" in message
    else:
        raise AssertionError("Expected SimulatorRuntimeError for missing struct member")


def test_simulate_pipeline_entities_rejects_member_reference_on_scalar():
    entities = {
        "streams": [
            {"name": "in_stream", "type": "float", "capacity": "8"},
            {"name": "out_stream", "type": "float", "capacity": "8"},
        ],
        "accumulators": [],
        "uniforms": [],
        "shaders": [
            {
                "name": "ScalarMember",
                "params": [
                    {"modifier": "in", "type": "float", "name": "src"},
                    {"modifier": "out", "type": "float", "name": "dst"},
                ],
                "body_ast": [
                    AstAssignStmt(
                        target=("dst",), value=AstExprVar(path=("src", "field"))
                    )
                ],
            }
        ],
        "filters": [],
        "bind_routes": ["out_stream = ScalarMember(in_stream, out_stream);"],
        "bind_routes_ir": [
            {
                "kind": "kernel",
                "target": "out_stream",
                "kernel": "ScalarMember",
                "args": ["in_stream", "out_stream"],
                "route": "out_stream = ScalarMember(in_stream, out_stream);",
            }
        ],
    }

    try:
        simulate_pipeline_entities(entities, stream_inputs={"in_stream": [1.0]})
    except SimulatorRuntimeError as err:
        message = str(err)
        assert "Cannot resolve simulator member 'field'" in message
        assert "non-struct value at 'src'" in message
    else:
        raise AssertionError("Expected SimulatorRuntimeError for scalar member reference")


def test_simulate_pipeline_entities_executes_filter_return_predicate_and_struct_fields():
    entities = {
        "streams": [
            {"name": "in_stream", "type": "Particle", "capacity": "8"},
            {"name": "out_stream", "type": "Particle", "capacity": "8"},
        ],
        "accumulators": [],
        "uniforms": [],
        "shaders": [],
        "filters": [
            {
                "name": "KeepLarge",
                "params": [
                    {"modifier": "in", "type": "Particle", "name": "src"},
                    {"modifier": "out", "type": "Particle", "name": "dst"},
                ],
                "body_ast": [
                    AstAssignStmt(
                        target=("dst", "value"),
                        value=AstExprBinary(
                            op="+",
                            left=AstExprVar(path=("src", "value")),
                            right=AstExprLiteral(kind="int", value="10"),
                        ),
                    ),
                    AstReturnStmt(
                        value=AstExprBinary(
                            op=">",
                            left=AstExprVar(path=("src", "value")),
                            right=AstExprLiteral(kind="int", value="1"),
                        )
                    ),
                ],
            }
        ],
        "bind_routes": ["out_stream = KeepLarge(in_stream, out_stream);"],
        "bind_routes_ir": [
            {
                "kind": "kernel",
                "target": "out_stream",
                "kernel": "KeepLarge",
                "args": ["in_stream", "out_stream"],
                "route": "out_stream = KeepLarge(in_stream, out_stream);",
            }
        ],
    }

    simulation = simulate_pipeline_entities(
        entities,
        stream_inputs={"in_stream": [{"value": 1}, {"value": 2}, {"value": 3}]},
    )

    assert simulation["streams"]["out_stream"] == [{"value": 12}, {"value": 13}]


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
    assert "--simulate-input requires --simulate or --report" in stderr.getvalue()


def test_run_cli_report_accepts_simulate_input_file(tmp_path):
    input_file = tmp_path / "sim-input.json"
    input_file.write_text('{"streams": {"s": [10, 20]}}', encoding="utf-8")

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
        ["--report", "--simulate-input", str(input_file)],
        stdin=io.StringIO("pipeline P { }"),
        stdout=stdout,
        compiler=fake_compiler,
    )

    assert exit_code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["simulation"]["streams"]["s"] == [10, 20]


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
        raise AssertionError(
            "Expected parser to reject combined output-producing flags"
        )


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

    simulation = simulate_pipeline_entities(
        entities, stream_inputs={"in_stream": [1, 2, 3, 4]}
    )

    assert simulation["streams"]["out_stream"] == [
        {"_source": 1, "_kernel": "Project"},
        {"_source": 4, "_kernel": "Project"},
    ]


def test_parse_simulation_inputs_rejects_non_object_root():
    try:
        parse_simulation_inputs("[1, 2, 3]")
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

    def fake_reduce(operator, values, **_kwargs):
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


def test_jit_numeric_reduce_has_deterministic_python_behavior_for_mixed_numeric_values():
    from lockstep_compiler.simulator import _jit_numeric_reduce

    values = [1, 2.5, True, False, 4]
    assert _jit_numeric_reduce("sum", values) == 8.5
    assert _jit_numeric_reduce("avg", values) == 1.7


def test_jit_numeric_reduce_preserves_int_result_for_non_bool_int_inputs():
    from lockstep_compiler.simulator import _jit_numeric_reduce

    assert _jit_numeric_reduce("sum", [1, 2, 3]) == 6
    assert isinstance(_jit_numeric_reduce("sum", [1, 2, 3]), int)


def test_jit_numeric_reduce_raises_mode_specific_error_when_llvm_runtime_fails(
    monkeypatch,
):
    from lockstep_compiler.simulator import SimulatorRuntimeError, _jit_numeric_reduce

    monkeypatch.setattr(
        "lockstep_compiler.simulator._run_reduce_subprocess",
        lambda _values: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    try:
        _jit_numeric_reduce("sum", [1.25, 2.75], use_llvm_runtime=True)
    except SimulatorRuntimeError as err:
        assert "LLVM simulation runtime failed" in str(err)
    else:
        raise AssertionError("Expected SimulatorRuntimeError when LLVM runtime fails")


def test_simulate_pipeline_entities_defaults_to_python_mode_without_runtime_binaries(
    monkeypatch,
):
    monkeypatch.setattr(
        "lockstep_compiler.simulator._simulator_runtime_command", lambda: None
    )

    assert (
        simulate_pipeline_entities(
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
            accumulator_inputs={"energy": [1.25, 2.75, True]},
        )["uniforms"]["total"]
        == 5.0
    )
