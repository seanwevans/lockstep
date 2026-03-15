from __future__ import annotations

from dataclasses import dataclass
import re


_BIND_ROUTE_RE = re.compile(
    r"^\s*(?P<target>[A-Za-z_]\w*)\s*=\s*(?P<callee>[A-Za-z_]\w*)\s*\((?P<args>[^)]*)\)\s*;\s*$"
)


@dataclass(frozen=True)
class ParsedBindRoute:
    target: str
    callee: str
    args: tuple[str, ...]
    raw: str


def _parse_bind_route(route: str) -> ParsedBindRoute | None:
    match = _BIND_ROUTE_RE.match(route)
    if not match:
        return None

    args_text = match.group("args").strip()
    args = (
        tuple(arg.strip() for arg in args_text.split(",") if arg.strip())
        if args_text
        else ()
    )
    return ParsedBindRoute(
        target=match.group("target"),
        callee=match.group("callee"),
        args=args,
        raw=route,
    )


def optimize_bind_routes(
    bind_routes: list[str],
    *,
    shader_names: set[str],
    filter_names: set[str],
    bind_routes_ir: list[dict] | None = None,
) -> dict[str, list]:
    """Build a bind DAG, eliminate dead kernels, and fuse fusible subgraphs."""

    if bind_routes_ir:
        parsed_routes = []
        for index, route in enumerate(bind_routes_ir):
            if route.get("kind") != "kernel":
                parsed_routes.append(None)
                continue
            parsed_routes.append(
                ParsedBindRoute(
                    target=route.get("target", ""),
                    callee=route.get("kernel", ""),
                    args=tuple(route.get("args", [])),
                    raw=route.get(
                        "route", bind_routes[index] if index < len(bind_routes) else ""
                    ),
                )
            )
        if len(parsed_routes) < len(bind_routes):
            parsed_routes.extend(
                _parse_bind_route(route) for route in bind_routes[len(parsed_routes) :]
            )
    else:
        parsed_routes = [_parse_bind_route(route) for route in bind_routes]
    valid_kernel_names = shader_names | filter_names

    live_after_route: list[set[str]] = [set() for _ in bind_routes]
    live: set[str] = set()
    for idx in range(len(parsed_routes) - 1, -1, -1):
        parsed = parsed_routes[idx]
        live_after_route[idx] = set(live)
        if parsed is None:
            continue
        # target is defined by the route, so the previous value is dead unless
        # the route reads it as an argument.
        live.discard(parsed.target)
        live.update(parsed.args)

    last_definition_index: dict[str, int] = {}
    for idx, parsed in enumerate(parsed_routes):
        if parsed is None:
            continue
        last_definition_index[parsed.target] = idx

    # Dead-code elimination over kernel routes: drop kernels whose target value
    # is never consumed, unless they explicitly read/accumulate their target.
    active_kernel_indices: set[int] = set()
    for idx, parsed in enumerate(parsed_routes):
        if parsed is None or parsed.callee not in valid_kernel_names:
            continue
        if (
            parsed.target in live_after_route[idx]
            or last_definition_index.get(parsed.target) == idx
        ):
            active_kernel_indices.add(idx)

    # Build producer/consumer links between active kernels.
    producer_by_symbol: dict[str, int] = {}
    deps: dict[int, set[int]] = {idx: set() for idx in active_kernel_indices}
    consumers: dict[int, set[int]] = {idx: set() for idx in active_kernel_indices}
    for idx, parsed in enumerate(parsed_routes):
        if idx not in active_kernel_indices or parsed is None:
            continue
        for arg in parsed.args:
            producer = producer_by_symbol.get(arg)
            if producer is not None:
                deps[idx].add(producer)
                consumers[producer].add(idx)
        producer_by_symbol[parsed.target] = idx

    fused_groups: list[dict[str, object]] = []
    fused_replacement_at: dict[int, str] = {}
    removed_indices: set[int] = set()

    # Segment kernel DAGs by non-kernel barriers so serialization order is stable.
    segment: list[int] = []

    def flush_segment() -> None:
        if not segment:
            return
        segment_set = set(segment)
        sinks = [idx for idx in segment if not (consumers[idx] & segment_set)]
        taken: set[int] = set()
        for sink_idx in sinks:
            if sink_idx in taken:
                continue

            group = {sink_idx}
            worklist = [sink_idx]
            while worklist:
                current = worklist.pop()
                for dep in deps[current] & segment_set:
                    if dep in group:
                        continue
                    group.add(dep)
                    worklist.append(dep)

            # Fusion requires closed internal dataflow for intermediates.
            closure_ok = True
            for idx in group:
                if idx == sink_idx:
                    continue
                if consumers[idx] - group:
                    closure_ok = False
                    break
                if parsed_routes[idx].target in live_after_route[sink_idx]:
                    closure_ok = False
                    break
            if not closure_ok or len(group) <= 1:
                continue

            ordered_nodes = sorted(group)
            entry_args: list[str] = []
            produced_so_far: set[str] = set()
            first_idx = ordered_nodes[0]
            for idx in ordered_nodes:
                route = parsed_routes[idx]
                for arg in route.args:
                    if arg in produced_so_far:
                        continue
                    if arg == route.target and idx != first_idx:
                        continue
                    if arg not in entry_args:
                        entry_args.append(arg)
                produced_so_far.add(route.target)

            fused_groups.append(
                {
                    "nodes": [parsed_routes[idx].callee for idx in ordered_nodes],
                    "entry_args": entry_args,
                    "output": parsed_routes[sink_idx].target,
                    "eliminated_intermediates": [
                        parsed_routes[idx].target
                        for idx in ordered_nodes
                        if idx != sink_idx
                    ],
                    "source_routes": [parsed_routes[idx].raw for idx in ordered_nodes],
                }
            )
            fused_replacement_at[sink_idx] = (
                f"{parsed_routes[sink_idx].target} = FUSED["
                f"{' -> '.join(parsed_routes[idx].callee for idx in ordered_nodes)}];"
            )
            removed_indices.update(set(ordered_nodes) - {sink_idx})
            taken.update(group)

        segment.clear()

    for idx, parsed in enumerate(parsed_routes):
        is_active_kernel = idx in active_kernel_indices and parsed is not None
        if is_active_kernel:
            segment.append(idx)
            continue
        flush_segment()
    flush_segment()

    optimized_bind_routes: list[str] = []
    for idx, route in enumerate(bind_routes):
        parsed = parsed_routes[idx]
        if (
            parsed is not None
            and parsed.callee in valid_kernel_names
            and idx not in active_kernel_indices
        ):
            continue
        if idx in removed_indices:
            continue
        replacement = fused_replacement_at.get(idx)
        optimized_bind_routes.append(replacement if replacement is not None else route)

    return {
        "optimized_bind_routes": optimized_bind_routes,
        "fused_groups": fused_groups,
    }
