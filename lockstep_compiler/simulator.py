import json
from dataclasses import dataclass
from functools import lru_cache
import shutil
import subprocess
import tempfile
from typing import Any

from llvmlite import ir

from .compiler import compile_lockstep


@dataclass
class RouteSimulation:
    route: str
    kind: str
    input_count: int
    output_count: int
    notes: str | None = None


def _fold_values(operator: str, values: list[Any]) -> Any:
    numeric = [value for value in values if isinstance(value, (int, float))]
    if not numeric:
        return None
    if operator == "sum":
        return _jit_numeric_reduce("sum", numeric)
    if operator == "avg":
        return _jit_numeric_reduce("avg", numeric)
    if operator == "min":
        return min(numeric)
    if operator == "max":
        return max(numeric)
    return None


def _build_reduce_program_ir(values: list[float]) -> str:
    module = ir.Module(name="lockstep_simulator_fold")
    float_ty = ir.DoubleType()
    int_ty = ir.IntType(32)
    ptr_ty = float_ty.as_pointer()
    fn_ty = ir.FunctionType(float_ty, [ptr_ty, int_ty])
    function = ir.Function(module, fn_ty, name="lockstep_fold_sum")
    data_arg, count_arg = function.args
    data_arg.name = "data"
    count_arg.name = "count"

    entry = function.append_basic_block("entry")
    loop_cond = function.append_basic_block("loop_cond")
    loop_body = function.append_basic_block("loop_body")
    loop_exit = function.append_basic_block("loop_exit")

    builder = ir.IRBuilder(entry)
    builder.branch(loop_cond)

    builder.position_at_end(loop_cond)
    idx_phi = builder.phi(int_ty, name="idx")
    acc_phi = builder.phi(float_ty, name="acc")
    idx_phi.add_incoming(ir.Constant(int_ty, 0), entry)
    acc_phi.add_incoming(ir.Constant(float_ty, 0.0), entry)
    should_continue = builder.icmp_signed("<", idx_phi, count_arg, name="cont")
    builder.cbranch(should_continue, loop_body, loop_exit)

    builder.position_at_end(loop_body)
    row_ptr = builder.gep(data_arg, [idx_phi], inbounds=True, name="row_ptr")
    row_value = builder.load(row_ptr, name="row")
    next_acc = builder.fadd(acc_phi, row_value, name="next_acc")
    next_acc.fastmath.add("fast")
    next_idx = builder.add(idx_phi, ir.Constant(int_ty, 1), name="next_idx")
    builder.branch(loop_cond)

    idx_phi.add_incoming(next_idx, loop_body)
    acc_phi.add_incoming(next_acc, loop_body)

    builder.position_at_end(loop_exit)
    builder.ret(acc_phi)

    array_ty = ir.ArrayType(float_ty, len(values))
    data_const = ir.Constant(array_ty, values)
    data_global = ir.GlobalVariable(module, array_ty, name="fold_data")
    data_global.global_constant = True
    data_global.initializer = data_const

    printf_ty = ir.FunctionType(int_ty, [ir.IntType(8).as_pointer()], var_arg=True)
    printf = ir.Function(module, printf_ty, name="printf")

    fmt_ty = ir.ArrayType(ir.IntType(8), 7)
    fmt_const = ir.Constant(fmt_ty, bytearray(b"%.17g\n\x00"))
    fmt_global = ir.GlobalVariable(module, fmt_ty, name="fmt")
    fmt_global.global_constant = True
    fmt_global.initializer = fmt_const

    main_ty = ir.FunctionType(int_ty, [])
    main_fn = ir.Function(module, main_ty, name="main")
    main_entry = main_fn.append_basic_block("entry")
    main_builder = ir.IRBuilder(main_entry)

    zero = ir.Constant(int_ty, 0)
    data_ptr = main_builder.gep(data_global, [zero, zero], inbounds=True)
    sum_value = main_builder.call(
        function,
        [data_ptr, ir.Constant(int_ty, len(values))],
        name="sum_value",
    )
    fmt_ptr = main_builder.gep(fmt_global, [zero, zero], inbounds=True)
    main_builder.call(printf, [fmt_ptr, sum_value])
    main_builder.ret(zero)

    return str(module)


@lru_cache(maxsize=1)
def _simulator_runtime_command() -> tuple[str, ...] | None:
    clang_path = shutil.which("clang")
    if clang_path:
        return (clang_path,)
    lli_path = shutil.which("lli")
    if lli_path:
        return (lli_path,)
    return None


def _run_reduce_subprocess(values: list[float]) -> float:
    command = _simulator_runtime_command()
    if command is None:
        raise RuntimeError("missing clang/lli runtime toolchain")

    with tempfile.TemporaryDirectory(prefix="lockstep-sim-") as temp_dir:
        ir_path = f"{temp_dir}/fold.ll"
        with open(ir_path, "w", encoding="utf-8") as handle:
            handle.write(_build_reduce_program_ir(values))

        if len(command) == 1 and "clang" in command[0].split("/")[-1]:
            exe_path = f"{temp_dir}/fold_reduce"
            subprocess.run(
                [command[0], "-O2", ir_path, "-o", exe_path],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            run_command = [exe_path]
        else:
            run_command = [command[0], ir_path]

        completed = subprocess.run(
            run_command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )

    return float(completed.stdout.strip())


def _jit_numeric_reduce(operator: str, values: list[Any]) -> Any:
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        return None

    try:
        reduced = _run_reduce_subprocess(numeric_values)
    except Exception:
        reduced = float(sum(numeric_values))

    if operator == "avg":
        reduced /= len(numeric_values)

    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return int(reduced)
    return reduced


def _route_text(route_ir: dict[str, Any]) -> str:
    return str(route_ir.get("route", ""))


def _apply_saturated_writes(rows: list[Any], capacity: int) -> list[Any]:
    if capacity <= 0:
        return []

    saturated_rows: list[Any] = []
    for value in rows:
        if len(saturated_rows) < capacity:
            saturated_rows.append(value)
            continue
        saturated_rows[capacity - 1] = value
    return saturated_rows


def simulate_pipeline_entities(
    entities: dict[str, Any],
    *,
    stream_inputs: dict[str, list[Any]] | None = None,
    accumulator_inputs: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    streams = {
        stream["name"]: {
            "type": stream["type"],
            "capacity": int(stream["capacity"]),
            "rows": list((stream_inputs or {}).get(stream["name"], [])),
        }
        for stream in entities.get("streams", [])
    }
    accumulators = {
        accum["name"]: list((accumulator_inputs or {}).get(accum["name"], []))
        for accum in entities.get("accumulators", [])
    }
    uniforms = {
        uniform["name"]: uniform.get("initializer")
        for uniform in entities.get("uniforms", [])
    }

    kernels = {
        shader["name"]: {"kind": "shader", "params": shader.get("params", [])}
        for shader in entities.get("shaders", [])
    }
    kernels.update(
        {
            flt["name"]: {"kind": "filter", "params": flt.get("params", [])}
            for flt in entities.get("filters", [])
        }
    )

    routes: list[RouteSimulation] = []
    bind_routes_ir = entities.get("bind_routes_ir", [])

    for route_ir in bind_routes_ir:
        route_kind = route_ir.get("kind")
        route_text = _route_text(route_ir)

        if route_kind == "fold":
            source_values = accumulators.get(str(route_ir.get("source", "")), [])
            uniform_name = route_ir.get("uniform_name")
            if isinstance(uniform_name, str) and uniform_name:
                uniforms[uniform_name] = _fold_values(
                    str(route_ir.get("operator", "")), source_values
                )
            routes.append(
                RouteSimulation(
                    route=route_text,
                    kind="fold",
                    input_count=len(source_values),
                    output_count=1,
                )
            )
            continue

        if route_kind == "kernel":
            kernel = kernels.get(str(route_ir.get("kernel", "")))
            if kernel is None:
                routes.append(
                    RouteSimulation(
                        route=route_text,
                        kind="kernel",
                        input_count=0,
                        output_count=0,
                        notes="Unknown kernel",
                    )
                )
                continue

            source_count = 0
            rows: list[Any] = []
            args = (
                route_ir.get("args") if isinstance(route_ir.get("args"), list) else []
            )
            for index, arg_name in enumerate(args):
                params = kernel["params"]
                if index >= len(params):
                    break
                if params[index]["modifier"] == "in" and arg_name in streams:
                    rows = list(streams[arg_name]["rows"])
                    source_count = len(rows)
                    break

            if kernel["kind"] == "filter":
                output_rows = [
                    row
                    for row in rows
                    if not isinstance(row, dict) or row.get("_keep", True)
                ]
            else:
                output_rows = [
                    {"_source": row, "_kernel": route_ir.get("kernel")} for row in rows
                ]

            target = route_ir.get("target")
            if isinstance(target, str) and target in streams:
                cap = streams[target]["capacity"]
                streams[target]["rows"] = _apply_saturated_writes(output_rows, cap)

            for index, arg_name in enumerate(args):
                params = kernel["params"]
                if index >= len(params):
                    break
                if params[index]["modifier"] == "accum" and arg_name in accumulators:
                    accumulators[arg_name].extend([1] * len(output_rows))

            routes.append(
                RouteSimulation(
                    route=route_text,
                    kind=kernel["kind"],
                    input_count=source_count,
                    output_count=len(output_rows),
                )
            )
            continue

        routes.append(
            RouteSimulation(
                route=route_text,
                kind="unknown",
                input_count=0,
                output_count=0,
                notes="Unknown bind route IR kind",
            )
        )

    if entities.get("bind_routes") and not bind_routes_ir:
        routes.append(
            RouteSimulation(
                route="",
                kind="unknown",
                input_count=0,
                output_count=0,
                notes="Missing bind_routes_ir; simulator requires semantic bind route IR",
            )
        )

    return {
        "streams": {name: spec["rows"] for name, spec in streams.items()},
        "accumulators": accumulators,
        "uniforms": uniforms,
        "routes": [
            {
                "route": route.route,
                "kind": route.kind,
                "input_count": route.input_count,
                "output_count": route.output_count,
                "notes": route.notes,
            }
            for route in routes
        ],
    }


def simulate_pipeline_source(
    source_code: str, *, stream_inputs=None, accumulator_inputs=None
) -> dict[str, Any]:
    result = compile_lockstep(source_code, verbose=False)
    return simulate_pipeline_entities(
        result.entities,
        stream_inputs=stream_inputs,
        accumulator_inputs=accumulator_inputs,
    )


def parse_simulation_inputs(
    raw: str,
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    payload = json.loads(raw) if raw.strip() else {}

    if not isinstance(payload, dict):
        raise ValueError("Invalid simulation input at '<root>': expected an object.")

    def _validate_input_map(field: str) -> dict[str, list[Any]]:
        value = payload.get(field, {})
        if not isinstance(value, dict):
            raise ValueError(
                f"Invalid simulation input at '{field}': expected an object map."
            )
        for name, rows in value.items():
            if not isinstance(rows, list):
                raise ValueError(
                    f"Invalid simulation input at '{field}.{name}': expected a list of values."
                )
        return value

    streams = _validate_input_map("streams")
    accumulators = _validate_input_map("accumulators")
    return streams, accumulators
