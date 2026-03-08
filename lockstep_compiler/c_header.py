from __future__ import annotations

from typing import Any

from .ast import AstProgram, ast_to_entities
from .utils import sanitize_symbol as _sanitize_symbol

_PRIMITIVE_C_TYPE = {
    "bool": "uint8_t",
    "int": "int32_t",
    "uint": "uint32_t",
    "float": "float",
    "double": "double",
}

_PRIMITIVE_SIZE = {
    "bool": 1,
    "int": 4,
    "uint": 4,
    "float": 4,
    "double": 8,
}


def _normalize_structs(structs: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for struct_decl in structs:
        if isinstance(struct_decl, str):
            normalized.append({"name": struct_decl, "fields": []})
        elif isinstance(struct_decl, dict) and struct_decl.get("name"):
            fields = struct_decl.get("fields") if isinstance(struct_decl.get("fields"), list) else []
            normalized.append({"name": struct_decl["name"], "fields": fields})
    return normalized


def _field_size(type_name: str, struct_sizes: dict[str, int]) -> int:
    if type_name in _PRIMITIVE_SIZE:
        return _PRIMITIVE_SIZE[type_name]
    if type_name in struct_sizes:
        return struct_sizes[type_name]
    return 8


def _resolve_struct_layouts(normalized_structs: list[dict[str, Any]]) -> tuple[dict[str, int], set[str]]:
    struct_sizes: dict[str, int] = {}
    unresolved = {struct["name"] for struct in normalized_structs}
    struct_map = {struct["name"]: struct for struct in normalized_structs}
    opaque_structs: set[str] = set()

    while unresolved:
        progress = False
        for struct_name in list(unresolved):
            fields = struct_map[struct_name].get("fields", [])
            can_resolve = True
            size = 0
            for field in fields:
                field_type_name = field.get("type", "float")
                if field_type_name in unresolved:
                    can_resolve = False
                    break
                size += _field_size(field_type_name, struct_sizes)
            if not can_resolve:
                continue
            struct_sizes[struct_name] = size
            unresolved.remove(struct_name)
            progress = True

        if not progress:
            for struct_name in unresolved:
                struct_sizes[struct_name] = 1
                opaque_structs.add(struct_name)
            break

    return struct_sizes, opaque_structs


def _c_type(type_name: str, known_structs: set[str]) -> str:
    if type_name in _PRIMITIVE_C_TYPE:
        return _PRIMITIVE_C_TYPE[type_name]
    if type_name in known_structs:
        return f"struct Lockstep_{_sanitize_symbol(type_name)}"
    return "void*"


def emit_c_header(program_or_entities: AstProgram | dict[str, Any], guard: str = "LOCKSTEP_GENERATED_H") -> str:
    entities = ast_to_entities(program_or_entities) if isinstance(program_or_entities, AstProgram) else program_or_entities

    normalized_structs = _normalize_structs(entities.get("structs", []))
    known_structs = {struct["name"] for struct in normalized_structs}
    struct_sizes, opaque_structs = _resolve_struct_layouts(normalized_structs)

    arena_fields: list[tuple[str, str, str]] = []
    for stream in entities.get("streams", []):
        arena_fields.append(("stream", stream["name"], stream["type"]))
    for accumulator in entities.get("accumulators", []):
        arena_fields.append(("accum", accumulator["name"], accumulator["type"]))
    for uniform in entities.get("uniforms", []):
        arena_fields.append(("uniform", uniform["name"], uniform["type"]))

    offsets: list[tuple[str, str, int]] = []
    cursor = 0
    for kind, name, type_name in arena_fields:
        offsets.append((kind, name, cursor))
        cursor += _field_size(type_name, struct_sizes)

    lines = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stdint.h>",
        "#include <stddef.h>",
        "#ifdef LOCKSTEP_DEBUG_SATURATED_WRITES",
        "#include <stdio.h>",
        "#endif",
        "",
        "#if defined(_MSC_VER)",
        "#define LOCKSTEP_PACKED_STRUCT(definition) __pragma(pack(push, 1)) definition __pragma(pack(pop))",
        "#else",
        "#define LOCKSTEP_PACKED_STRUCT(definition) definition __attribute__((packed))",
        "#endif",
        "",
    ]

    for struct_decl in normalized_structs:
        struct_name = struct_decl["name"]
        c_struct_name = f"Lockstep_{_sanitize_symbol(struct_name)}"
        if struct_name in opaque_structs:
            lines.append(
                f"LOCKSTEP_PACKED_STRUCT(struct {c_struct_name} {{ uint8_t _opaque; }});"
            )
            lines.append("")
            continue

        lines.append(f"LOCKSTEP_PACKED_STRUCT(struct {c_struct_name} {{")
        for field in struct_decl.get("fields", []):
            field_type = _c_type(field.get("type", "float"), known_structs)
            field_name = _sanitize_symbol(field.get("name", "field"))
            lines.append(f"    {field_type} {field_name};")
        lines.append("});")
        lines.append("")

    lines.append("LOCKSTEP_PACKED_STRUCT(struct Lockstep_Arena {")
    for kind, name, type_name in arena_fields:
        c_type_name = _c_type(type_name, known_structs)
        lines.append(f"    {c_type_name} {kind}_{_sanitize_symbol(name)};")
    lines.append("});")
    lines.append("")

    lines.append(f"#define LOCKSTEP_ARENA_BYTES {cursor}")
    for kind, name, offset in offsets:
        macro_suffix = f"{kind}_{_sanitize_symbol(name)}".upper()
        lines.append(f"#define LOCKSTEP_OFFSET_{macro_suffix} {offset}")
    for stream in entities.get("streams", []):
        stream_name = _sanitize_symbol(stream["name"]).upper()
        stream_capacity = int(stream.get("capacity", 0)) if stream.get("capacity") is not None else 0
        lines.append(f"#define LOCKSTEP_CAPACITY_STREAM_{stream_name} {stream_capacity}")
    lines.append("")

    lines.extend(
        [
            "#ifndef LOCKSTEP_SATURATED_WRITE_LOG",
            "#define LOCKSTEP_SATURATED_WRITE_LOG(stream_name, index, capacity, saturated_index) \\",
            "    fprintf(stderr, \"[lockstep] saturated write stream=%s index=%zu capacity=%zu -> %zu\\n\", \\",
            "            (stream_name), (size_t)(index), (size_t)(capacity), (size_t)(saturated_index))",
            "#endif",
            "",
            "static inline size_t Lockstep_SaturatedWriteIndex(size_t index, size_t capacity, const char* stream_name) {",
            "    if (capacity == 0) {",
            "        return 0;",
            "    }",
            "    if (index < capacity) {",
            "        return index;",
            "    }",
            "    const size_t saturated_index = capacity - 1;",
            "#ifdef LOCKSTEP_DEBUG_SATURATED_WRITES",
            "    LOCKSTEP_SATURATED_WRITE_LOG(stream_name != NULL ? stream_name : \"<unnamed>\", index, capacity, saturated_index);",
            "#endif",
            "    return saturated_index;",
            "}",
            "",
        ]
    )

    lines.extend(
        [
            "#ifdef __cplusplus",
            'extern "C" {',
            "#endif",
            "",
            "void Lockstep_Tick(struct Lockstep_Arena* arena);",
            "",
            "#ifdef __cplusplus",
            "}",
            "#endif",
            "",
            "#endif",
            "",
        ]
    )

    return "\n".join(lines)
