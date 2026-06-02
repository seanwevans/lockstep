from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .ast import AstProgram, AstStructDecl, AstType

from .errors import LockstepCompileError
from .models import LockstepDiagnostic

_TYPE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[][<>,]")


@dataclass(frozen=True)
class _ArraySuffix:
    size: int


@dataclass(frozen=True)
class _GenericSuffix:
    type_name: "_ParsedTypeName"
    width: int | None


@dataclass(frozen=True)
class _ParsedTypeName:
    base: str
    suffixes: tuple[_ArraySuffix | _GenericSuffix, ...]


def _tokenize_type_name(type_name: str) -> list[str] | None:
    tokens = _TYPE_TOKEN_RE.findall(type_name)
    if not tokens or "".join(tokens) != type_name:
        return None
    return tokens


def _parse_type_name_tokens(
    tokens: list[str], index: int = 0
) -> tuple[_ParsedTypeName | None, int]:
    if (
        index >= len(tokens)
        or not tokens[index][0].isalpha()
        and tokens[index][0] != "_"
    ):
        return None, index

    base = tokens[index]
    index += 1
    suffixes: list[_ArraySuffix | _GenericSuffix] = []

    while index < len(tokens):
        token = tokens[index]
        if token == "[":
            if (
                index + 2 >= len(tokens)
                or not tokens[index + 1].isdigit()
                or tokens[index + 2] != "]"
            ):
                return None, index
            suffixes.append(_ArraySuffix(size=int(tokens[index + 1])))
            index += 3
            continue

        if token == "<":
            inner, index = _parse_type_name_tokens(tokens, index + 1)
            if inner is None:
                return None, index
            width: int | None = None
            if index < len(tokens) and tokens[index] == ",":
                if index + 1 >= len(tokens) or not tokens[index + 1].isdigit():
                    return None, index
                width = int(tokens[index + 1])
                index += 2
            if index >= len(tokens) or tokens[index] != ">":
                return None, index
            suffixes.append(_GenericSuffix(type_name=inner, width=width))
            index += 1
            continue

        break

    return _ParsedTypeName(base=base, suffixes=tuple(suffixes)), index


def _parse_type_name(type_name: str) -> _ParsedTypeName | None:
    tokens = _tokenize_type_name(type_name)
    if tokens is None:
        return None
    parsed, index = _parse_type_name_tokens(tokens)
    if parsed is None or index != len(tokens):
        return None
    return parsed


def _type_multiplier(parsed_type: _ParsedTypeName) -> int:
    multiplier = 1
    for suffix in parsed_type.suffixes:
        if isinstance(suffix, _ArraySuffix):
            multiplier *= max(suffix.size, 1)
        else:
            multiplier *= max(suffix.width or 1, 1)
    return multiplier


def _structural_type_name(parsed_type: _ParsedTypeName) -> str:
    has_generic_suffix = any(
        isinstance(suffix, _GenericSuffix) for suffix in parsed_type.suffixes
    )
    if not has_generic_suffix:
        return parsed_type.base
    for suffix in parsed_type.suffixes:
        if isinstance(suffix, _GenericSuffix):
            return _structural_type_name(suffix.type_name)
    return parsed_type.base


_PRIMITIVE_SIZE = {
    "bool": 1,
    "int": 4,
    "uint": 4,
    "float": 4,
    "double": 8,
}


@dataclass(frozen=True)
class LayoutStructField:
    name: str
    type_name: str


@dataclass(frozen=True)
class LayoutStructDecl:
    name: str
    fields: tuple[LayoutStructField, ...] = ()


@dataclass(frozen=True)
class ArenaLeaf:
    kind: str
    binding_name: str
    path: tuple[str, ...]
    type_name: str
    offset: int
    size: int
    element_count: int


@dataclass(frozen=True)
class ArenaLayout:
    normalized_structs: tuple[LayoutStructDecl, ...]
    known_structs: set[str]
    struct_sizes: dict[str, int]
    opaque_structs: set[str]
    leaves: list[ArenaLeaf]
    top_level_offsets: list[tuple[str, str, int]]
    total_size: int


def normalize_structs(structs: list[Any]) -> tuple[LayoutStructDecl, ...]:
    normalized: list[LayoutStructDecl] = []
    for struct_decl in structs:
        if isinstance(struct_decl, str):
            normalized.append(LayoutStructDecl(name=struct_decl))
        elif isinstance(struct_decl, dict) and struct_decl.get("name"):
            raw_fields = (
                struct_decl.get("fields")
                if isinstance(struct_decl.get("fields"), list)
                else []
            )
            fields = tuple(
                LayoutStructField(
                    name=str(field.get("name", "field")),
                    type_name=str(field.get("type", "float")),
                )
                for field in raw_fields
            )
            normalized.append(
                LayoutStructDecl(name=str(struct_decl["name"]), fields=fields)
            )
    return tuple(normalized)


def normalize_ast_structs(
    structs: tuple[AstStructDecl, ...],
) -> tuple[LayoutStructDecl, ...]:
    return tuple(
        LayoutStructDecl(
            name=struct.name,
            fields=tuple(
                LayoutStructField(
                    name=field.name, type_name=_ast_type_name(field.declared_type)
                )
                for field in struct.fields
            ),
        )
        for struct in structs
    )


def _ast_type_name(type_ref: AstType | str) -> str:
    return type_ref.name if isinstance(type_ref, AstType) else type_ref


def field_size(type_name: str, struct_sizes: dict[str, int]) -> int:
    parsed = _parse_type_name(type_name)
    if parsed is None:
        return 8
    base_type = _structural_type_name(parsed)
    if base_type in _PRIMITIVE_SIZE:
        base_size = _PRIMITIVE_SIZE[base_type]
    elif base_type in struct_sizes:
        base_size = struct_sizes[base_type]
    else:
        base_size = 8
    return base_size * _type_multiplier(parsed)


def resolve_struct_layouts(
    normalized_structs: tuple[LayoutStructDecl, ...],
) -> tuple[dict[str, int], set[str]]:
    struct_sizes: dict[str, int] = {}
    unresolved = {struct.name for struct in normalized_structs}
    struct_map = {struct.name: struct for struct in normalized_structs}
    opaque_structs: set[str] = set()

    while unresolved:
        progress = False
        for struct_name in list(unresolved):
            fields = struct_map[struct_name].fields
            can_resolve = True
            size = 0
            for field in fields:
                field_type_name = field.type_name
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
            cycle_structs = sorted(unresolved)
            diagnostic = LockstepDiagnostic(
                severity="error",
                code="LCK503",
                message=(
                    "Arena layout contains a recursive struct dependency cycle: "
                    + " -> ".join(cycle_structs)
                ),
                line=1,
                column=0,
                source_file="<arena_layout>",
                hint=(
                    "Break recursive struct references in pipeline resource layouts "
                    "so each struct can be resolved to a finite byte size."
                ),
            )
            raise LockstepCompileError(
                [diagnostic],
                diagnostics=[diagnostic],
                phase="codegen",
                source_file=diagnostic.source_file,
            )

    return struct_sizes, opaque_structs


def _flatten_type_leaves(
    type_name: str,
    struct_map: dict[str, LayoutStructDecl],
    opaque_structs: set[str],
    *,
    path: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], str, int]]:
    parsed = _parse_type_name(type_name)
    if parsed is None:
        return [(path, type_name, 1)]

    multiplier = _type_multiplier(parsed)
    structural_type = _structural_type_name(parsed)
    if structural_type in _PRIMITIVE_SIZE:
        return [(path, structural_type, multiplier)]
    struct_decl = struct_map.get(structural_type)
    if struct_decl is None or structural_type in opaque_structs:
        return [(path, structural_type, multiplier)]
    leaves: list[tuple[tuple[str, ...], str, int]] = []
    for field in struct_decl.fields:
        child_name = field.name
        child_type = field.type_name
        for child_path, child_leaf_type, child_multiplier in _flatten_type_leaves(
            child_type,
            struct_map,
            opaque_structs,
            path=path + (child_name,),
        ):
            leaves.append((child_path, child_leaf_type, child_multiplier * multiplier))
    if not leaves:
        return [(path, structural_type, multiplier)]
    return leaves


def _build_layout_from_bindings(
    normalized_structs: tuple[LayoutStructDecl, ...],
    bindings: list[tuple[str, str, str, int]],
) -> ArenaLayout:
    known_structs = {struct.name for struct in normalized_structs}
    struct_sizes, opaque_structs = resolve_struct_layouts(normalized_structs)
    struct_map = {struct.name: struct for struct in normalized_structs}

    leaves: list[ArenaLeaf] = []
    top_level_offsets: list[tuple[str, str, int]] = []
    cursor = 0
    for kind, binding_name, type_name, element_count in bindings:
        top_level_offsets.append((kind, binding_name, cursor))
        type_leaves = _flatten_type_leaves(type_name, struct_map, opaque_structs)
        for path, leaf_type, leaf_multiplier in type_leaves:
            size = field_size(leaf_type, struct_sizes) * leaf_multiplier
            leaves.append(
                ArenaLeaf(
                    kind=kind,
                    binding_name=binding_name,
                    path=path,
                    type_name=leaf_type,
                    offset=cursor,
                    size=size,
                    element_count=element_count,
                )
            )
            cursor += size * element_count

    return ArenaLayout(
        normalized_structs=normalized_structs,
        known_structs=known_structs,
        struct_sizes=struct_sizes,
        opaque_structs=opaque_structs,
        leaves=leaves,
        top_level_offsets=top_level_offsets,
        total_size=cursor,
    )


def build_arena_layout(entities: dict[str, Any]) -> ArenaLayout:
    normalized_structs = normalize_structs(entities.get("structs", []))
    bindings: list[tuple[str, str, str, int]] = []
    for stream in entities.get("streams", []):
        capacity = int(stream.get("capacity", 1))
        bindings.append(("stream", stream["name"], stream["type"], max(capacity, 1)))
    for accumulator in entities.get("accumulators", []):
        element_count = (
            int(accumulator.get("size", 1))
            if accumulator.get("size") is not None
            else 1
        )
        bindings.append(
            ("accum", accumulator["name"], accumulator["type"], max(element_count, 1))
        )
    for uniform in entities.get("uniforms", []):
        bindings.append(("uniform", uniform["name"], uniform["type"], 1))
    return _build_layout_from_bindings(normalized_structs, bindings)


def build_ast_arena_layout(
    program: AstProgram,
    *,
    accumulator_sizes: dict[str, int] | None = None,
) -> ArenaLayout:
    normalized_structs = normalize_ast_structs(program.structs)
    accumulator_sizes = accumulator_sizes or {}
    bindings: list[tuple[str, str, str, int]] = []
    for pipeline in program.pipelines:
        for stream in pipeline.streams:
            bindings.append(
                (
                    "stream",
                    stream.name,
                    _ast_type_name(stream.declared_type),
                    max(int(stream.capacity), 1),
                )
            )
        for accumulator in pipeline.accumulators:
            bindings.append(
                (
                    "accum",
                    accumulator.name,
                    _ast_type_name(accumulator.declared_type),
                    max(int(accumulator_sizes.get(accumulator.name, 1)), 1),
                )
            )
        for uniform in pipeline.uniforms:
            bindings.append(
                ("uniform", uniform.name, _ast_type_name(uniform.declared_type), 1)
            )
    return _build_layout_from_bindings(normalized_structs, bindings)
