import json
from dataclasses import dataclass
from typing import Any

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
        return sum(numeric)
    if operator == "avg":
        return sum(numeric) / len(numeric)
    if operator == "min":
        return min(numeric)
    if operator == "max":
        return max(numeric)
    return None


def _route_text(route_ir: dict[str, Any]) -> str:
    return str(route_ir.get("route", ""))


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
    bind_routes_ir = entities.get("bind_routes_ir", [])

    for route_ir in bind_routes_ir:
        route_kind = route_ir.get("kind")
        route_text = _route_text(route_ir)

        if route_kind == "fold":
            source_values = accumulators.get(str(route_ir.get("source", "")), [])
            uniform_name = route_ir.get("uniform_name")
            if isinstance(uniform_name, str) and uniform_name:
                uniforms[uniform_name] = _fold_values(str(route_ir.get("operator", "")), source_values)
            routes.append(RouteSimulation(route=route_text, kind="fold", input_count=len(source_values), output_count=1))
            continue

        if route_kind == "kernel":
            kernel = kernels.get(str(route_ir.get("kernel", "")))
            if kernel is None:
                routes.append(RouteSimulation(route=route_text, kind="kernel", input_count=0, output_count=0, notes="Unknown kernel"))
                continue

            source_count = 0
            rows: list[Any] = []
            args = route_ir.get("args") if isinstance(route_ir.get("args"), list) else []
            for index, arg_name in enumerate(args):
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
                output_rows = [{"_source": row, "_kernel": route_ir.get("kernel")} for row in rows]

            target = route_ir.get("target")
            if isinstance(target, str) and target in streams:
                cap = streams[target]["capacity"]
                streams[target]["rows"] = output_rows[:cap]

            for index, arg_name in enumerate(args):
                params = kernel["params"]
                if index >= len(params):
                    break
                if params[index]["modifier"] == "accum" and arg_name in accumulators:
                    accumulators[arg_name].extend([1] * len(output_rows))

            routes.append(RouteSimulation(route=route_text, kind=kernel["kind"], input_count=source_count, output_count=len(output_rows)))
            continue

        routes.append(RouteSimulation(route=route_text, kind="unknown", input_count=0, output_count=0, notes="Unknown bind route IR kind"))

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


def simulate_pipeline_source(source_code: str, *, stream_inputs=None, accumulator_inputs=None) -> dict[str, Any]:
    result = compile_lockstep(source_code, verbose=False)
    return simulate_pipeline_entities(result.entities, stream_inputs=stream_inputs, accumulator_inputs=accumulator_inputs)


def parse_simulation_inputs(raw: str) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    payload = json.loads(raw) if raw.strip() else {}
    return payload.get("streams", {}), payload.get("accumulators", {})
