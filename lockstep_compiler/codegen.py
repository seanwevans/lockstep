from __future__ import annotations

from typing import Any


_PRIMITIVE_TYPE_MAP = {
    "bool": "i1",
    "int": "i32",
    "uint": "i32",
    "float": "float",
    "double": "double",
}


def _sanitize_symbol(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)


def _llvm_type(type_name: str, known_structs: set[str]) -> str:
    if type_name in _PRIMITIVE_TYPE_MAP:
        return _PRIMITIVE_TYPE_MAP[type_name]
    if type_name in known_structs:
        return f"%struct.{_sanitize_symbol(type_name)}"
    return "i8*"


def emit_llvm_ir(entities: dict[str, Any]) -> str:
    """Generate a baseline LLVM IR module for frontend-discovered entities."""

    structs = entities.get("structs", [])
    shaders = entities.get("shaders", [])
    filters = entities.get("filters", [])
    pure_functions = entities.get("pure_functions", [])
    streams = entities.get("streams", [])
    accumulators = entities.get("accumulators", [])
    uniforms = entities.get("uniforms", [])
    bind_routes = entities.get("bind_routes", [])

    known_structs = set(structs)
    lines: list[str] = [
        '; ModuleID = "lockstep"',
        'source_filename = "lockstep"',
        "",
    ]

    for struct_name in structs:
        safe_name = _sanitize_symbol(struct_name)
        lines.append(f"%struct.{safe_name} = type {{ i8 }}")

    if structs:
        lines.append("")

    for pure in pure_functions:
        name = _sanitize_symbol(pure["name"])
        return_type = _llvm_type(pure["return_type"], known_structs)
        lines.append(f"declare {return_type} @pure_{name}()")

    for shader in shaders:
        lines.append(f"declare void @shader_{_sanitize_symbol(shader['name'])}()")

    for flt in filters:
        lines.append(f"declare void @filter_{_sanitize_symbol(flt['name'])}()")

    if pure_functions or shaders or filters:
        lines.append("")

    for stream in streams:
        stream_name = _sanitize_symbol(stream["name"])
        stream_type = _llvm_type(stream["type"], known_structs)
        lines.append(f"@stream_{stream_name} = external global {stream_type}")

    for accum in accumulators:
        accum_name = _sanitize_symbol(accum["name"])
        accum_type = _llvm_type(accum["type"], known_structs)
        lines.append(f"@accum_{accum_name} = external global {accum_type}")

    for uniform in uniforms:
        uniform_name = _sanitize_symbol(uniform["name"])
        uniform_type = _llvm_type(uniform["type"], known_structs)
        lines.append(f"@uniform_{uniform_name} = external global {uniform_type}")

    if streams or accumulators or uniforms:
        lines.append("")

    lines.extend(
        [
            "define void @Lockstep_Tick() {",
            "entry:",
        ]
    )

    for route in bind_routes:
        route_text = str(route).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'  call void asm sideeffect "; bind: {route_text}", ""()')

    lines.extend(["  ret void", "}"])

    return "\n".join(lines) + "\n"
