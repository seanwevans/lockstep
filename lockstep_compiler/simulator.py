import json
import os
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


def _fold_values(
    operator: str, values: list[Any], *, use_llvm_runtime: bool = False
) -> Any:
    numeric = [value for value in values if isinstance(value, (int, float))]
    if not numeric:
        return None
    if operator == "sum":
        return _jit_numeric_reduce("sum", numeric, use_llvm_runtime=use_llvm_runtime)
    if operator == "avg":
        return _jit_numeric_reduce("avg", numeric, use_llvm_runtime=use_llvm_runtime)
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


class SimulatorRuntimeError(RuntimeError):
    """Raised when simulator execution encounters an invalid runtime state."""


def _python_numeric_reduce(operator: str, values: list[Any]) -> Any:
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        return None

    reduced = float(sum(numeric_values))
    if operator == "avg":
        reduced /= len(numeric_values)

    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return int(reduced)
    return reduced


def _jit_numeric_reduce(
    operator: str,
    values: list[Any],
    *,
    use_llvm_runtime: bool = False,
) -> Any:
    reduced_python = _python_numeric_reduce(operator, values)
    if not use_llvm_runtime:
        return reduced_python

    numeric_values = [float(value) for value in values]
    if not numeric_values:
        return None

    try:
        reduced = _run_reduce_subprocess(numeric_values)
    except Exception as exc:
        raise SimulatorRuntimeError(
            "LLVM simulation runtime failed; disable llvm runtime or install clang/lli"
        ) from exc

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


def _coerce_sim_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text == "true":
            return True
        if text == "false":
            return False
        try:
            if any(marker in text for marker in (".", "e", "E")):
                return float(text)
            return int(text)
        except ValueError:
            return value
    return value


def _copy_sim_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_sim_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_sim_value(item) for item in value]
    return value


def _sim_mapping_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _sim_type_text(type_value: Any) -> str | None:
    if type_value is None:
        return None
    if isinstance(type_value, dict):
        name = type_value.get("name")
        return str(name) if name is not None else None
    return str(type_value)


def _sim_path_tuple(path_value: Any) -> tuple[str, ...]:
    if isinstance(path_value, str):
        return tuple(part for part in path_value.split(".") if part)
    if isinstance(path_value, (list, tuple)):
        return tuple(str(part) for part in path_value)
    return ()


def _default_sim_value(type_name: str | None = None) -> Any:
    if type_name in {"float", "double"}:
        return 0.0
    if type_name in {"int", "uint"}:
        return 0
    if type_name == "bool":
        return False
    return {}


def _format_sim_path(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "<empty>"


def _format_available_sim_keys(value: dict[str, Any]) -> str:
    if not value:
        return "none"
    return ", ".join(sorted(str(key) for key in value))


_MISSING_SIM_VALUE = object()


def _raise_unmapped_sim_path(
    *,
    path_kind: str,
    missing_part: str,
    full_path: tuple[str, ...],
    available_scope: dict[str, Any],
    parent_path: tuple[str, ...] | None = None,
) -> None:
    formatted_path = _format_sim_path(full_path)
    available = _format_available_sim_keys(available_scope)
    if parent_path is None:
        raise SimulatorRuntimeError(
            "Unmapped simulator identifier "
            f"'{missing_part}' while resolving '{formatted_path}'; "
            f"available identifiers: {available}."
        )

    raise SimulatorRuntimeError(
        f"Unmapped simulator {path_kind} "
        f"'{missing_part}' at '{_format_sim_path(parent_path)}' while resolving "
        f"'{formatted_path}'; available members: {available}."
    )


def _resolve_sim_path(env: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Resolve a simulator variable/member path without implicit fallbacks."""
    if not path:
        return None

    root = path[0]
    value = env.get(root, _MISSING_SIM_VALUE)
    if value is _MISSING_SIM_VALUE:
        _raise_unmapped_sim_path(
            path_kind="identifier",
            missing_part=root,
            full_path=path,
            available_scope=env,
        )
    resolved_path = (root,)
    for part in path[1:]:
        if not isinstance(value, dict):
            raise SimulatorRuntimeError(
                "Cannot resolve simulator member "
                f"'{part}' on non-struct value at "
                f"'{_format_sim_path(resolved_path)}' while resolving "
                f"'{_format_sim_path(path)}'."
            )
        next_value = value.get(part, _MISSING_SIM_VALUE)
        if next_value is _MISSING_SIM_VALUE:
            _raise_unmapped_sim_path(
                path_kind="member",
                missing_part=part,
                full_path=path,
                available_scope=value,
                parent_path=resolved_path,
            )
        value = next_value
        resolved_path = (*resolved_path, part)
    return _coerce_sim_value(value)


def _assign_sim_path(env: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    if not path:
        return
    if len(path) == 1:
        env[path[0]] = value
        return

    current = env.get(path[0])
    if not isinstance(current, dict):
        current = {}
        env[path[0]] = current
    for part in path[1:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[path[-1]] = value


def _truthy_sim_value(value: Any) -> bool:
    return bool(_coerce_sim_value(value))


def _eval_sim_call(
    name: str,
    args: list[Any],
    pure_functions: dict[str, dict[str, Any]],
) -> Any:
    if name in {"int", "uint"}:
        return int(_coerce_sim_value(args[0])) if args else 0
    if name in {"float", "double"}:
        return float(_coerce_sim_value(args[0])) if args else 0.0
    if name == "bool":
        return _truthy_sim_value(args[0]) if args else False
    if name == "select" and len(args) == 3:
        return args[1] if _truthy_sim_value(args[0]) else args[2]
    if name == "step" and len(args) == 2:
        return 0.0 if float(args[1]) < float(args[0]) else 1.0
    if name == "mix" and len(args) == 3:
        return args[0] * (1.0 - args[2]) + args[1] * args[2]
    if name == "clamp" and len(args) == 3:
        return max(args[1], min(args[0], args[2]))
    if name == "max" and len(args) == 2:
        return max(args[0], args[1])
    if name == "min" and len(args) == 2:
        return min(args[0], args[1])
    if name == "abs" and len(args) == 1:
        return abs(args[0])
    if name == "sign" and len(args) == 1:
        return -1.0 if args[0] < 0 else (1.0 if args[0] > 0 else 0.0)
    if name == "smoothstep" and len(args) == 3:
        edge0, edge1, x = map(float, args)
        if edge0 == edge1:
            t = 0.0
        else:
            t = max(0.0, min((x - edge0) / (edge1 - edge0), 1.0))
        return t * t * (3.0 - 2.0 * t)

    pure = pure_functions.get(name)
    if pure is not None:
        env = {
            param.get("name"): args[index] if index < len(args) else 0
            for index, param in enumerate(pure.get("params", []))
        }
        return _execute_sim_body(pure.get("body_ast", []), env, pure_functions)[0]

    return None


def _eval_sim_expr(
    expr: Any,
    env: dict[str, Any],
    pure_functions: dict[str, dict[str, Any]],
) -> Any:
    from .ast import (
        AstExprBinary,
        AstExprCall,
        AstExprCast,
        AstExprLiteral,
        AstExprUnary,
        AstExprVar,
    )

    is_mapping_expr = isinstance(expr, dict)
    if isinstance(expr, AstExprLiteral) or (
        is_mapping_expr
        and "kind" in expr
        and "value" in expr
        and "op" not in expr
        and "target_type" not in expr
    ):
        kind = _sim_mapping_value(expr, "kind")
        value = _sim_mapping_value(expr, "value")
        if kind == "bool":
            return value == "true" if isinstance(value, str) else bool(value)
        if kind in {"int", "uint"}:
            return int(value)
        if kind in {"float", "double"}:
            return float(value)
        return value
    if isinstance(expr, AstExprVar) or (is_mapping_expr and "path" in expr):
        return _resolve_sim_path(env, _sim_path_tuple(_sim_mapping_value(expr, "path")))
    if isinstance(expr, AstExprUnary) or (is_mapping_expr and "operand" in expr):
        op = _sim_mapping_value(expr, "op")
        operand = _eval_sim_expr(
            _sim_mapping_value(expr, "operand"), env, pure_functions
        )
        if op == "-":
            return -operand
        return not _truthy_sim_value(operand)
    if isinstance(expr, AstExprCast) or (is_mapping_expr and "target_type" in expr):
        return _eval_sim_call(
            str(_sim_type_text(_sim_mapping_value(expr, "target_type"))),
            [_eval_sim_expr(_sim_mapping_value(expr, "value"), env, pure_functions)],
            pure_functions,
        )
    if isinstance(expr, AstExprCall) or (
        is_mapping_expr and "name" in expr and "args" in expr
    ):
        args = _sim_mapping_value(expr, "args", []) or []
        return _eval_sim_call(
            str(_sim_mapping_value(expr, "name")),
            [_eval_sim_expr(arg, env, pure_functions) for arg in args],
            pure_functions,
        )
    if isinstance(expr, AstExprBinary) or (
        is_mapping_expr and "op" in expr and "left" in expr and "right" in expr
    ):
        op = _sim_mapping_value(expr, "op")
        if op == "&&":
            return _truthy_sim_value(
                _eval_sim_expr(_sim_mapping_value(expr, "left"), env, pure_functions)
            ) and _truthy_sim_value(
                _eval_sim_expr(_sim_mapping_value(expr, "right"), env, pure_functions)
            )
        if op == "||":
            return _truthy_sim_value(
                _eval_sim_expr(_sim_mapping_value(expr, "left"), env, pure_functions)
            ) or _truthy_sim_value(
                _eval_sim_expr(_sim_mapping_value(expr, "right"), env, pure_functions)
            )
        left = _eval_sim_expr(_sim_mapping_value(expr, "left"), env, pure_functions)
        right = _eval_sim_expr(_sim_mapping_value(expr, "right"), env, pure_functions)
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op == "%":
            return left % right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "&":
            return int(left) & int(right)
        if op == "|":
            return int(left) | int(right)
        if op == "^":
            return int(left) ^ int(right)
        if op == "<<":
            return int(left) << int(right)
        if op == ">>":
            return int(left) >> int(right)
    return None


def _execute_sim_body(
    body_ast: list[Any],
    env: dict[str, Any],
    pure_functions: dict[str, dict[str, Any]],
) -> tuple[Any, set[str]]:
    from .ast import AstAssignStmt, AstReturnStmt, AstVarDeclStmt

    assigned: set[str] = set()
    return_value = None
    for statement in body_ast:
        is_mapping_statement = isinstance(statement, dict)
        if isinstance(statement, AstVarDeclStmt) or (
            is_mapping_statement
            and "name" in statement
            and ("declared_type" in statement or "initializer" in statement)
        ):
            name = str(_sim_mapping_value(statement, "name"))
            initializer = _sim_mapping_value(statement, "initializer")
            env[name] = (
                _eval_sim_expr(initializer, env, pure_functions)
                if initializer is not None
                else _default_sim_value(
                    _sim_type_text(_sim_mapping_value(statement, "declared_type"))
                )
            )
            assigned.add(name)
            continue
        if isinstance(statement, AstAssignStmt) or (
            is_mapping_statement and "target" in statement and "value" in statement
        ):
            value = _eval_sim_expr(
                _sim_mapping_value(statement, "value"), env, pure_functions
            )
            target = _sim_path_tuple(_sim_mapping_value(statement, "target"))
            _assign_sim_path(env, target, value)
            if target:
                assigned.add(target[0])
            continue
        if isinstance(statement, AstReturnStmt) or (
            is_mapping_statement
            and "value" in statement
            and "target" not in statement
            and "declared_type" not in statement
        ):
            return_value = _eval_sim_expr(
                _sim_mapping_value(statement, "value"), env, pure_functions
            )
            break
    return return_value, assigned


def _simulate_kernel_rows(
    *,
    kernel_name: str,
    kernel: dict[str, Any],
    args: list[Any],
    rows: list[Any],
    streams: dict[str, dict[str, Any]],
    accumulators: dict[str, list[Any]],
    uniforms: dict[str, Any],
    pure_functions: dict[str, dict[str, Any]],
) -> tuple[list[Any], dict[str, list[Any]]]:
    body_ast = kernel.get("body_ast") or []
    params = kernel.get("params", [])
    if not body_ast:
        if kernel["kind"] == "filter":
            return [
                row
                for row in rows
                if not isinstance(row, dict) or row.get("_keep", True)
            ], {}
        return [{"_source": row, "_kernel": kernel_name} for row in rows], {
            arg_name: [1] * len(rows)
            for index, arg_name in enumerate(args)
            if index < len(params)
            and params[index].get("modifier") == "accum"
            and arg_name in accumulators
        }

    output_rows: list[Any] = []
    pending_accum: dict[str, list[Any]] = {
        arg_name: []
        for index, arg_name in enumerate(args)
        if index < len(params)
        and params[index].get("modifier") == "accum"
        and arg_name in accumulators
    }

    for row_index, row in enumerate(rows):
        env: dict[str, Any] = {}
        first_input = _copy_sim_value(row)
        out_param_names: list[str] = []
        accum_param_args: dict[str, str] = {}
        for index, param in enumerate(params):
            param_name = param.get("name")
            modifier = param.get("modifier")
            arg_name = args[index] if index < len(args) else None
            if not isinstance(param_name, str):
                continue
            if modifier == "in" and isinstance(arg_name, str) and arg_name in streams:
                arg_rows = streams[arg_name]["rows"]
                env[param_name] = (
                    _copy_sim_value(arg_rows[row_index])
                    if row_index < len(arg_rows)
                    else _default_sim_value(param.get("type"))
                )
            elif modifier == "out":
                out_param_names.append(param_name)
                env[param_name] = _copy_sim_value(first_input)
            elif modifier == "uniform" and isinstance(arg_name, str):
                env[param_name] = _coerce_sim_value(uniforms.get(arg_name, 0))
            elif modifier == "accum" and isinstance(arg_name, str):
                accum_param_args[param_name] = arg_name
                env[param_name] = _default_sim_value(param.get("type"))

        return_value, assigned = _execute_sim_body(body_ast, env, pure_functions)
        for param_name, arg_name in accum_param_args.items():
            if param_name in assigned:
                pending_accum[arg_name].append(_copy_sim_value(env.get(param_name)))

        keep_row = True
        if kernel["kind"] == "filter" and return_value is not None:
            keep_row = _truthy_sim_value(return_value)
        if not keep_row:
            continue

        if out_param_names:
            output_rows.append(_copy_sim_value(env.get(out_param_names[0])))
        elif return_value is not None and kernel["kind"] != "filter":
            output_rows.append(return_value)
        else:
            output_rows.append(_copy_sim_value(row))

    return output_rows, pending_accum


def simulate_pipeline_entities(
    entities: dict[str, Any],
    *,
    stream_inputs: dict[str, list[Any]] | None = None,
    accumulator_inputs: dict[str, list[Any]] | None = None,
    use_llvm_runtime: bool | None = None,
) -> dict[str, Any]:
    if use_llvm_runtime is None:
        use_llvm_runtime = os.getenv("LOCKSTEP_SIM_USE_LLVM", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

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
    pure_functions = {
        pure["name"]: pure
        for pure in entities.get("pure_functions", [])
        if not pure.get("intrinsic")
    }

    kernels = {
        shader["name"]: {
            "kind": "shader",
            "params": shader.get("params", []),
            "body_ast": shader.get("body_ast", []),
        }
        for shader in entities.get("shaders", [])
    }
    kernels.update(
        {
            flt["name"]: {
                "kind": "filter",
                "params": flt.get("params", []),
                "body_ast": flt.get("body_ast", []),
            }
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
                    str(route_ir.get("operator", "")),
                    source_values,
                    use_llvm_runtime=use_llvm_runtime,
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
            first_input_type: str | None = None
            args = (
                route_ir.get("args") if isinstance(route_ir.get("args"), list) else []
            )
            params = kernel["params"]
            for index, arg_name in enumerate(args):
                if index >= len(params):
                    break
                if params[index]["modifier"] == "in" and arg_name in streams:
                    arg_rows = list(streams[arg_name]["rows"])
                    if first_input_type is None:
                        first_input_type = params[index].get("type")
                        rows = arg_rows
                    source_count = max(source_count, len(arg_rows))

            if len(rows) < source_count:
                rows = rows + [
                    _default_sim_value(first_input_type)
                    for _ in range(source_count - len(rows))
                ]

            output_rows, pending_accum = _simulate_kernel_rows(
                kernel_name=str(route_ir.get("kernel", "")),
                kernel=kernel,
                args=args,
                rows=rows,
                streams=streams,
                accumulators=accumulators,
                uniforms=uniforms,
                pure_functions=pure_functions,
            )

            target = route_ir.get("target")
            if isinstance(target, str) and target in streams:
                cap = streams[target]["capacity"]
                streams[target]["rows"] = _apply_saturated_writes(output_rows, cap)

            for index, arg_name in enumerate(args):
                params = kernel["params"]
                if index >= len(params):
                    break
                if params[index]["modifier"] == "accum" and arg_name in accumulators:
                    accumulators[arg_name].extend(pending_accum.get(arg_name, []))

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
    source_code: str,
    *,
    stream_inputs: dict[str, list[Any]] | None = None,
    accumulator_inputs: dict[str, list[Any]] | None = None,
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
