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
    args = tuple(arg.strip() for arg in args_text.split(",") if arg.strip()) if args_text else ()
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
    """Fuse adjacent shader/filter bind routes that shuttle temporary SoA streams."""

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
                    raw=route.get("route", bind_routes[index] if index < len(bind_routes) else ""),
                )
            )
        if len(parsed_routes) < len(bind_routes):
            parsed_routes.extend(_parse_bind_route(route) for route in bind_routes[len(parsed_routes) :])
    else:
        parsed_routes = [_parse_bind_route(route) for route in bind_routes]
    valid_kernel_names = shader_names | filter_names

    use_count: dict[str, int] = {}
    for parsed in parsed_routes:
        if parsed is None:
            continue
        for arg in parsed.args:
            if arg == parsed.target:
                continue
            use_count[arg] = use_count.get(arg, 0) + 1

    fused_groups: list[dict[str, object]] = []
    optimized_bind_routes: list[str] = []
    idx = 0

    while idx < len(bind_routes):
        current = parsed_routes[idx]
        if current is None or current.callee not in valid_kernel_names:
            optimized_bind_routes.append(bind_routes[idx])
            idx += 1
            continue

        chain = [current]
        chain_end = idx

        while chain_end + 1 < len(bind_routes):
            nxt = parsed_routes[chain_end + 1]
            prev = chain[-1]
            if nxt is None or nxt.callee not in valid_kernel_names:
                break
            if use_count.get(prev.target, 0) != 1:
                break
            if prev.target not in nxt.args:
                break
            chain.append(nxt)
            chain_end += 1

        if len(chain) == 1:
            optimized_bind_routes.append(bind_routes[idx])
            idx += 1
            continue

        fused_groups.append(
            {
                "nodes": [route.callee for route in chain],
                "entry_args": list(chain[0].args),
                "output": chain[-1].target,
                "eliminated_intermediates": [route.target for route in chain[:-1]],
                "source_routes": [route.raw for route in chain],
            }
        )

        optimized_bind_routes.append(
            f"{chain[-1].target} = FUSED[{ ' -> '.join(route.callee for route in chain) }];"
        )
        idx = chain_end + 1

    return {
        "optimized_bind_routes": optimized_bind_routes,
        "fused_groups": fused_groups,
    }
