from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches

from .models import (
    ParsedTypeArraySuffix,
    ParsedTypeGenericSuffix,
    ParsedTypeName,
    SemanticStructField,
)


@dataclass(frozen=True)
class TypeValidationIssue:
    referenced_type: str
    suggestions: tuple[str, ...]


def consume_identifier(type_name: str, index: int) -> tuple[str | None, int]:
    if index >= len(type_name) or not (
        type_name[index].isalpha() or type_name[index] == "_"
    ):
        return None, index
    start = index
    index += 1
    while index < len(type_name) and (
        type_name[index].isalnum() or type_name[index] == "_"
    ):
        index += 1
    return type_name[start:index], index


def consume_digits(type_name: str, index: int) -> tuple[str | None, int]:
    if index >= len(type_name) or not type_name[index].isdigit():
        return None, index
    start = index
    index += 1
    while index < len(type_name) and type_name[index].isdigit():
        index += 1
    return type_name[start:index], index


def parse_type_name(
    type_name: str, index: int = 0
) -> tuple[ParsedTypeName | None, int]:
    base_name, index = consume_identifier(type_name, index)
    if base_name is None:
        return None, index

    suffixes: list[ParsedTypeArraySuffix | ParsedTypeGenericSuffix] = []
    while index < len(type_name):
        if type_name[index] == "[":
            index += 1
            length, index = consume_digits(type_name, index)
            if length is None or index >= len(type_name) or type_name[index] != "]":
                return None, index
            suffixes.append(ParsedTypeArraySuffix(size=int(length)))
            index += 1
            continue

        if type_name[index] == "<":
            index += 1
            inner_type, index = parse_type_name(type_name, index)
            if inner_type is None:
                return None, index
            arity = None
            if index < len(type_name) and type_name[index] == ",":
                index += 1
                digits, index = consume_digits(type_name, index)
                if digits is None:
                    return None, index
                arity = int(digits)
            if index >= len(type_name) or type_name[index] != ">":
                return None, index
            suffixes.append(ParsedTypeGenericSuffix(type_name=inner_type, arity=arity))
            index += 1
            continue
        break

    return ParsedTypeName(base=base_name, inner=tuple(suffixes)), index


def collect_referenced_type_names(parsed_type: ParsedTypeName) -> list[str]:
    has_generic_suffix = any(
        isinstance(s, ParsedTypeGenericSuffix) for s in parsed_type.inner
    )
    referenced_names = [] if has_generic_suffix else [parsed_type.base]
    for suffix in parsed_type.inner:
        if isinstance(suffix, ParsedTypeGenericSuffix):
            referenced_names.extend(collect_referenced_type_names(suffix.type_name))
    return referenced_names


def validate_type_name(
    type_name: str, known_types: set[str]
) -> tuple[bool, list[TypeValidationIssue]]:
    parsed_type, index = parse_type_name(type_name)
    if parsed_type is None or index != len(type_name):
        return False, []

    issues: list[TypeValidationIssue] = []
    for referenced_type in collect_referenced_type_names(parsed_type):
        if referenced_type in known_types:
            continue
        suggestions = tuple(
            get_close_matches(referenced_type, sorted(known_types), n=2, cutoff=0.6)
        )
        issues.append(
            TypeValidationIssue(
                referenced_type=referenced_type, suggestions=suggestions
            )
        )
    return len(issues) == 0, issues


def build_struct_field_type_index(
    structs: dict[str, dict[str, SemanticStructField | str]],
) -> dict[str, dict[str, str]]:
    """Return a plain ``struct -> field -> type`` index.

    Compiler semantic indexes store ``SemanticStructField`` objects, while LSP
    entity and regex-recovery paths already carry serialized type strings.  Keep
    this helper shared by accepting either representation so cached LSP analysis
    and compiler semantic validation resolve fields identically.
    """

    return {
        struct_name: {
            field_name: (field if isinstance(field, str) else field.declared_type)
            for field_name, field in fields.items()
        }
        for struct_name, fields in structs.items()
    }
