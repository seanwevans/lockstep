from __future__ import annotations

from .ast import AstProgram, ast_to_entities
from .arena_layout import build_arena_layout
from .errors import LockstepCompileError
from .models import LockstepDiagnostic
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

_MAX_U64 = (1 << 64) - 1


def _c_type(type_name: str, known_structs: set[str]) -> str:
    if type_name in _PRIMITIVE_C_TYPE:
        return _PRIMITIVE_C_TYPE[type_name]
    if type_name in known_structs:
        return f"struct Lockstep_{_sanitize_symbol(type_name)}"
    return "void*"


def emit_c_header(
    program: AstProgram,
    guard: str = "LOCKSTEP_GENERATED_H",
    *,
    target_width: int = 8,
) -> str:
    entities = ast_to_entities(program)

    layout = build_arena_layout(entities)
    normalized_structs = layout.normalized_structs
    known_structs = layout.known_structs
    opaque_structs = layout.opaque_structs

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
        struct_name = struct_decl.name
        c_struct_name = f"Lockstep_{_sanitize_symbol(struct_name)}"
        if struct_name in opaque_structs:
            lines.append(
                f"LOCKSTEP_PACKED_STRUCT(struct {c_struct_name} {{ uint8_t _opaque; }});"
            )
            lines.append("")
            continue

        lines.append(f"LOCKSTEP_PACKED_STRUCT(struct {c_struct_name} {{")
        for field in struct_decl.fields:
            field_type = _c_type(field.type_name, known_structs)
            field_name = _sanitize_symbol(field.name)
            lines.append(f"    {field_type} {field_name};")
        lines.append("});")
        lines.append("")

    lines.append("LOCKSTEP_PACKED_STRUCT(struct Lockstep_Arena {")
    cumulative_layout_total = 0
    for leaf in layout.leaves:
        leaf_size = _PRIMITIVE_SIZE.get(leaf.type_name, 1)
        leaf_allocation_size = leaf_size * leaf.element_count
        if cumulative_layout_total > _MAX_U64 - leaf_allocation_size:
            diagnostic = LockstepDiagnostic(
                severity="error",
                code="LCK502",
                message=(
                    "Arena structural offset generation overflowed the platform "
                    "allocation threshold (2^64-1 bytes)."
                ),
                line=1,
                column=0,
                source_file="<c_header>",
                hint=(
                    "Reduce aggregate stream or uniform layout size so cumulative "
                    "arena bytes fit within unsigned 64-bit limits."
                ),
            )
            raise LockstepCompileError(
                [diagnostic],
                diagnostics=[diagnostic],
                phase="codegen",
                source_file=diagnostic.source_file,
            )
        cumulative_layout_total += leaf_allocation_size
        c_type_name = _c_type(leaf.type_name, known_structs)
        path_suffix = "_".join(_sanitize_symbol(part) for part in leaf.path) if leaf.path else "value"
        field_name = f"{leaf.kind}_{_sanitize_symbol(leaf.binding_name)}_{path_suffix}"
        if leaf.element_count > 1:
            lines.append(f"    {c_type_name} {field_name}[{leaf.element_count}];")
        else:
            lines.append(f"    {c_type_name} {field_name};")
    lines.append("});")
    lines.append("")

    lines.append(f"#define LOCKSTEP_ARENA_BYTES {layout.total_size}")
    lines.append(f"#define LOCKSTEP_SIMD_WIDTH {int(target_width)}")
    lines.extend(
        [
            "#if defined(__cplusplus) && (__cplusplus >= 201103L)",
            'static_assert(LOCKSTEP_ARENA_BYTES <= SIZE_MAX, "LOCKSTEP_ARENA_BYTES must fit in size_t on the target architecture");',
            "#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)",
            '_Static_assert(LOCKSTEP_ARENA_BYTES <= SIZE_MAX, "LOCKSTEP_ARENA_BYTES must fit in size_t on the target architecture");',
            "#endif",
        ]
    )
    for kind, name, offset in layout.top_level_offsets:
        macro_suffix = f"{kind}_{_sanitize_symbol(name)}".upper()
        lines.append(f"#define LOCKSTEP_OFFSET_{macro_suffix} {offset}")
    for leaf in layout.leaves:
        if not leaf.path:
            continue
        leaf_suffix = "_".join(_sanitize_symbol(part) for part in leaf.path).upper()
        macro_suffix = f"{leaf.kind}_{_sanitize_symbol(leaf.binding_name)}_{leaf_suffix}".upper()
        lines.append(f"#define LOCKSTEP_OFFSET_{macro_suffix} {leaf.offset}")
    for stream in entities.get("streams", []):
        stream_name = _sanitize_symbol(stream["name"]).upper()
        stream_capacity = (
            int(stream.get("capacity", 0)) if stream.get("capacity") is not None else 0
        )
        lines.append(
            f"#define LOCKSTEP_CAPACITY_STREAM_{stream_name} {stream_capacity}"
        )
    lines.append("")

    lines.extend(
        [
            "#ifndef LOCKSTEP_SATURATED_WRITE_LOG",
            "#define LOCKSTEP_SATURATED_WRITE_LOG(stream_name, index, capacity, saturated_index) \\",
            '    fprintf(stderr, "[lockstep] saturated write stream=%s index=%zu capacity=%zu -> %zu\\n", \\',
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
            '    LOCKSTEP_SATURATED_WRITE_LOG(stream_name != NULL ? stream_name : "<unnamed>", index, capacity, saturated_index);',
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
