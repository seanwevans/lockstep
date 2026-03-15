from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_PRIMITIVE_SIZE = {
    "bool": 1,
    "int": 4,
    "uint": 4,
    "float": 4,
    "double": 8,
}


@dataclass(frozen=True)
class ArenaLeaf:
    kind: str
    binding_name: str
    path: tuple[str, ...]
    type_name: str
    offset: int
    size: int


@dataclass(frozen=True)
class ArenaLayout:
    normalized_structs: list[dict[str, Any]]
    known_structs: set[str]
    struct_sizes: dict[str, int]
    opaque_structs: set[str]
    leaves: list[ArenaLeaf]
    top_level_offsets: list[tuple[str, str, int]]
    total_size: int


def normalize_structs(structs: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for struct_decl in structs:
        if isinstance(struct_decl, str):
            normalized.append({"name": struct_decl, "fields": []})
        elif isinstance(struct_decl, dict) and struct_decl.get("name"):
            fields = (
                struct_decl.get("fields")
                if isinstance(struct_decl.get("fields"), list)
                else []
            )
            normalized.append({"name": struct_decl["name"], "fields": fields})
    return normalized


def field_size(type_name: str, struct_sizes: dict[str, int]) -> int:
    if type_name in _PRIMITIVE_SIZE:
        return _PRIMITIVE_SIZE[type_name]
    if type_name in struct_sizes:
        return struct_sizes[type_name]
    return 8


def resolve_struct_layouts(
    normalized_structs: list[dict[str, Any]],
) -> tuple[dict[str, int], set[str]]:
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
                size += field_size(field_type_name, struct_sizes)
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


def _flatten_type_leaves(
    type_name: str,
    struct_map: dict[str, dict[str, Any]],
    opaque_structs: set[str],
    *,
    path: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], str]]:
    if type_name in _PRIMITIVE_SIZE:
        return [(path, type_name)]
    struct_decl = struct_map.get(type_name)
    if struct_decl is None or type_name in opaque_structs:
        return [(path, type_name)]
    leaves: list[tuple[tuple[str, ...], str]] = []
    for field in struct_decl.get("fields", []):
        child_name = str(field.get("name", "field"))
        child_type = str(field.get("type", "float"))
        leaves.extend(
            _flatten_type_leaves(
                child_type,
                struct_map,
                opaque_structs,
                path=path + (child_name,),
            )
        )
    if not leaves:
        return [(path, type_name)]
    return leaves


def build_arena_layout(entities: dict[str, Any]) -> ArenaLayout:
    normalized_structs = normalize_structs(entities.get("structs", []))
    known_structs = {struct["name"] for struct in normalized_structs}
    struct_sizes, opaque_structs = resolve_struct_layouts(normalized_structs)
    struct_map = {struct["name"]: struct for struct in normalized_structs}

    bindings: list[tuple[str, str, str]] = []
    for stream in entities.get("streams", []):
        bindings.append(("stream", stream["name"], stream["type"]))
    for accumulator in entities.get("accumulators", []):
        bindings.append(("accum", accumulator["name"], accumulator["type"]))
    for uniform in entities.get("uniforms", []):
        bindings.append(("uniform", uniform["name"], uniform["type"]))

    leaves: list[ArenaLeaf] = []
    top_level_offsets: list[tuple[str, str, int]] = []
    cursor = 0
    for kind, binding_name, type_name in bindings:
        top_level_offsets.append((kind, binding_name, cursor))
        type_leaves = _flatten_type_leaves(type_name, struct_map, opaque_structs)
        for path, leaf_type in type_leaves:
            size = field_size(leaf_type, struct_sizes)
            leaves.append(
                ArenaLeaf(
                    kind=kind,
                    binding_name=binding_name,
                    path=path,
                    type_name=leaf_type,
                    offset=cursor,
                    size=size,
                )
            )
            cursor += size

    return ArenaLayout(
        normalized_structs=normalized_structs,
        known_structs=known_structs,
        struct_sizes=struct_sizes,
        opaque_structs=opaque_structs,
        leaves=leaves,
        top_level_offsets=top_level_offsets,
        total_size=cursor,
    )
