import json
import re
from dataclasses import dataclass
from typing import Any

from .compiler import compile_lockstep


_BIND_ROUTE_PATTERN = re.compile(
    r"^\s*(?P<target>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"(?P<kernel>[A-Za-z_][A-Za-z0-9_]*)\s*\(\s*(?P<args>[^)]*)\s*\)\s*;\s*$"
)
_FOLD_ROUTE_PATTERN = re.compile(
    r"^\s*uniform\s+(?P<type>[A-Za-z_][A-Za-z0-9_]*)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*fold\s+"
    r"(?P<operator>sum|avg|min|max)\s*\(\s*(?P<source>[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*;\s*$"
)


@dataclass
class RouteSimulation:
    route: str
    kind: str
    input_count: int
    output_count: int
    notes: str | None = None


def _parse_bind_route(route: str) -> dict[str, Any] | None:
    match = _BIND_ROUTE_PATTERN.match(route)
    if match is None:
        return None

    args = [arg.strip() for arg in match.group("args").split(",") if arg.strip()]
    return {
        "kind": "kernel",
        "target": match.group("target"),
        "kernel": match.group("kernel"),
        "args": args,
    }


def _parse_fold_route(route: str) -> dict[str, Any] | None:
    match = _FOLD_ROUTE_PATTERN.match(route)
    if match is None:
        return None
    return {
        "kind": "fold",
        "uniform_type": match.group("type"),
        "uniform_name": match.group("name"),
        "operator": match.group("operator"),
        "source": match.group("source"),
    }


def _fold_values(operator: str, values: list[Any]) -> Any:
    numeric = [value for value in values if isinstance(value, (int, float))]
    if not numeric:
        return None
    if operator == "sum":
        return sum(numeric)
    if operator == "avg":
        return sum(numeric) / len(numeric)
    if operator == "min":
        return min(numeric)
    if operator == "max":
        return max(numeric)
    return None


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
    uniforms = {uniform["name"]: uniform.get("initializer") for uniform in entities.get("uniforms", [])}

    kernels = {shader["name"]: {"kind": "shader", "params": shader.get("params", [])} for shader in entities.get("shaders", [])}
    kernels.update({flt["name"]: {"kind": "filter", "params": flt.get("params", [])} for flt in entities.get("filters", [])})

    routes: list[RouteSimulation] = []
    for route_text in entities.get("bind_routes", []):
        fold_route = _parse_fold_route(route_text)
        if fold_route is not None:
            source_values = accumulators.get(fold_route["source"], [])
            uniforms[fold_route["uniform_name"]] = _fold_values(fold_route["operator"], source_values)
            routes.append(RouteSimulation(route=route_text, kind="fold", input_count=len(source_values), output_count=1))
            continue

        kernel_route = _parse_bind_route(route_text)
        if kernel_route is not None:
            kernel = kernels.get(kernel_route["kernel"])
            if kernel is None:
                routes.append(RouteSimulation(route=route_text, kind="kernel", input_count=0, output_count=0, notes="Unknown kernel"))
                continue

            source_count = 0
            rows: list[Any] = []
            for index, arg_name in enumerate(kernel_route["args"]):
                params = kernel["params"]
                if index >= len(params):
                    break
                if params[index]["modifier"] == "in" and arg_name in streams:
                    rows = list(streams[arg_name]["rows"])
                    source_count = len(rows)
                    break

            if kernel["kind"] == "filter":
                output_rows = [row for row in rows if not isinstance(row, dict) or row.get("_keep", True)]
            else:
                output_rows = [{"_source": row, "_kernel": kernel_route["kernel"]} for row in rows]

            if kernel_route["target"] in streams:
                cap = streams[kernel_route["target"]]["capacity"]
                streams[kernel_route["target"]]["rows"] = output_rows[:cap]

            for index, arg_name in enumerate(kernel_route["args"]):
                params = kernel["params"]
                if index >= len(params):
                    break
                if params[index]["modifier"] == "accum" and arg_name in accumulators:
                    accumulators[arg_name].extend([1] * len(output_rows))

            routes.append(RouteSimulation(route=route_text, kind=kernel["kind"], input_count=source_count, output_count=len(output_rows)))
            continue

        routes.append(RouteSimulation(route=route_text, kind="unknown", input_count=0, output_count=0, notes="Unable to parse bind route"))

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


def simulate_pipeline_source(source_code: str, *, stream_inputs=None, accumulator_inputs=None) -> dict[str, Any]:
    result = compile_lockstep(source_code, verbose=False)
    return simulate_pipeline_entities(result.entities, stream_inputs=stream_inputs, accumulator_inputs=accumulator_inputs)


def parse_simulation_inputs(raw: str) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    payload = json.loads(raw) if raw.strip() else {}
    return payload.get("streams", {}), payload.get("accumulators", {})
