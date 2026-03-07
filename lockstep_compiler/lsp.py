from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .compiler import compile_lockstep
from .errors import LockstepCompileError


@dataclass(frozen=True)
class MemberDefinition:
    struct_name: str
    field_name: str
    line: int
    column: int


_STRUCT_RE = re.compile(r"\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", re.MULTILINE)
_FIELD_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*;")
_TYPED_NAME_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\b")
_PARAM_RE = re.compile(
    r"\b(?:in|out|accum|uniform)\s+([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
_SHADER_DEF_RE = re.compile(r"\bshader\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_PURE_DEF_RE = re.compile(
    r"\bpure\s+[A-Za-z_][A-Za-z0-9_]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_BIND_BLOCK_RE = re.compile(r"\bbind\s*\{(?P<body>[\s\S]*?)\}", re.MULTILINE)
_MEMBER_ACCESS_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)")


def _is_inside_bind_block(source: str, line: int, column: int) -> bool:
    """Return whether the given cursor position is inside any `bind { ... }` body."""

    if line < 0 or column < 0:
        return False

    lines = source.splitlines(keepends=True)
    if line >= len(lines):
        return False

    offset = sum(len(existing) for existing in lines[:line]) + min(column, len(lines[line]))

    for match in re.finditer(r"\bbind\s*\{", source):
        block_start = match.end()
        depth = 1
        index = block_start
        while index < len(source) and depth > 0:
            token = source[index]
            if token == "{":
                depth += 1
            elif token == "}":
                depth -= 1
            index += 1

        if depth != 0:
            continue

        block_end = index - 1
        if block_start <= offset <= block_end:
            return True

    return False


def _normalize_completion_name(entry: Any) -> str | None:
    if isinstance(entry, dict):
        return entry.get("name")
    if isinstance(entry, str):
        return entry
    return str(entry) if entry else None


def compile_for_lsp(source: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compile source and return entities + diagnostics in dictionary form."""

    try:
        result = compile_lockstep(source, verbose=False)
        diagnostics = [
            {
                "severity": diagnostic.severity,
                "code": diagnostic.code,
                "message": diagnostic.message,
                "line": diagnostic.line,
                "column": diagnostic.column,
                "hint": diagnostic.hint,
            }
            for diagnostic in result.diagnostics
        ]
        return result.entities, diagnostics
    except LockstepCompileError as error:
        diagnostics = [
            {
                "severity": diagnostic.severity,
                "code": diagnostic.code,
                "message": diagnostic.message,
                "line": diagnostic.line,
                "column": diagnostic.column,
                "hint": diagnostic.hint,
            }
            for diagnostic in error.diagnostics
        ]
        return {}, diagnostics


def build_struct_member_index(source: str) -> dict[str, dict[str, MemberDefinition]]:
    """Index struct field declarations with source locations (0-based)."""

    index: dict[str, dict[str, MemberDefinition]] = {}

    for struct_match in _STRUCT_RE.finditer(source):
        struct_name = struct_match.group(1)
        body_start = struct_match.end()
        body_end = source.find("}", body_start)
        if body_end == -1:
            continue
        body = source[body_start:body_end]
        index[struct_name] = {}

        for field_match in _FIELD_RE.finditer(body):
            field_name = field_match.group(2)
            absolute_offset = body_start + field_match.start(2)
            line = source.count("\n", 0, absolute_offset)
            line_start = source.rfind("\n", 0, absolute_offset)
            if line_start == -1:
                column = absolute_offset
            else:
                column = absolute_offset - line_start - 1
            index[struct_name][field_name] = MemberDefinition(
                struct_name=struct_name,
                field_name=field_name,
                line=line,
                column=column,
            )

    return index


def infer_variable_types(source: str) -> dict[str, str]:
    """Best-effort type inference for names declared as `<Type> <name>`."""

    inferred: dict[str, str] = {}
    for declared_type, name in _PARAM_RE.findall(source):
        inferred.setdefault(name, declared_type)

    for declared_type, name in _TYPED_NAME_RE.findall(source):
        if declared_type in {"return", "if", "for", "while", "bind", "stream", "uniform"}:
            continue
        inferred.setdefault(name, declared_type)
    return inferred


def find_member_definition(
    source: str,
    line: int,
    column: int,
) -> MemberDefinition | None:
    """Resolve `foo.bar` usages to `struct` field declaration locations."""

    lines = source.splitlines()
    if line < 0 or line >= len(lines):
        return None

    line_text = lines[line]
    for match in _MEMBER_ACCESS_RE.finditer(line_text):
        start, end = match.span(0)
        if not (start <= column <= end):
            continue
        variable_name, field_name = match.groups()
        variable_types = infer_variable_types(source)
        struct_name = variable_types.get(variable_name)
        if not struct_name:
            return None
        struct_index = build_struct_member_index(source)
        return struct_index.get(struct_name, {}).get(field_name)
    return None


def provide_bind_completion_items(
    source: str,
    *,
    line: int | None = None,
    column: int | None = None,
) -> list[dict[str, Any]]:
    """Return ranked completion entries for bind routes and callable symbols."""

    entities, _ = compile_for_lsp(source)
    completion_entries: list[dict[str, Any]] = []
    seen_labels: set[str] = set()

    inside_bind = (
        line is not None
        and column is not None
        and _is_inside_bind_block(source, line=line, column=column)
    )

    bind_routes = entities.get("bind_routes", [])
    if not bind_routes:
        for bind_match in _BIND_BLOCK_RE.finditer(source):
            bind_body = bind_match.group("body")
            for statement in bind_body.split(";"):
                normalized = re.sub(r"\s+", "", statement)
                if normalized:
                    bind_routes.append(f"{normalized};")

    if inside_bind:
        for route in bind_routes:
            if route in seen_labels:
                continue
            completion_entries.append(
                {
                    "label": route,
                    "detail": "Bind route template",
                    "sort_text": f"1-{route}",
                    "kind": "snippet",
                }
            )
            seen_labels.add(route)

    shaders = entities.get("shaders", [])
    filters = entities.get("filters", [])
    if not shaders:
        shaders = [{"name": name} for name in _SHADER_DEF_RE.findall(source)]
    if not filters:
        filters = []

    shader_filter_names = sorted(
        {
            f"{name}(...)"
            for name in [
                _normalize_completion_name(entry)
                for entry in [*shaders, *filters]
            ]
            if name
        }
    )
    for callable_name in shader_filter_names:
        if callable_name in seen_labels:
            continue
        completion_entries.append(
            {
                "label": callable_name,
                "detail": "Shader/filter callable",
                "sort_text": f"2-{callable_name}",
                "kind": "function",
            }
        )
        seen_labels.add(callable_name)

    pure_functions = entities.get("pure_functions", [])
    if not pure_functions:
        pure_functions = [{"name": name} for name in _PURE_DEF_RE.findall(source)]

    pure_function_names = sorted(
        {
            f"{name}(...)"
            for name in [_normalize_completion_name(entry) for entry in pure_functions]
            if name
        }
    )
    for function_name in pure_function_names:
        if function_name in seen_labels:
            continue
        completion_entries.append(
            {
                "label": function_name,
                "detail": "Pure function callable",
                "sort_text": f"3-{function_name}",
                "kind": "function",
            }
        )
        seen_labels.add(function_name)

    return completion_entries


def provide_hover_info(
    source: str,
    line: int,
    column: int,
) -> str | None:
    """Return hover text for the identifier at the given position."""

    lines = source.splitlines()
    if line < 0 or line >= len(lines):
        return None

    line_text = lines[line]

    # Try member access first (foo.bar)
    for match in _MEMBER_ACCESS_RE.finditer(line_text):
        start, end = match.span(0)
        if not (start <= column <= end):
            continue
        variable_name, field_name = match.groups()
        variable_types = infer_variable_types(source)
        struct_name = variable_types.get(variable_name)
        if struct_name:
            struct_index = build_struct_member_index(source)
            field_def = struct_index.get(struct_name, {}).get(field_name)
            if field_def:
                # Find the field type from struct declarations
                for struct_match in _STRUCT_RE.finditer(source):
                    if struct_match.group(1) != struct_name:
                        continue
                    body_start = struct_match.end()
                    body_end = source.find("}", body_start)
                    if body_end == -1:
                        continue
                    body = source[body_start:body_end]
                    for fm in _FIELD_RE.finditer(body):
                        if fm.group(2) == field_name:
                            return f"(field) `{struct_name}.{field_name}: {fm.group(1)}`"
            return f"(field) `{struct_name}.{field_name}`"
        return None

    # Try plain identifier
    id_pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
    for match in id_pattern.finditer(line_text):
        start, end = match.span(1)
        if not (start <= column < end):
            continue
        name = match.group(1)
        variable_types = infer_variable_types(source)
        if name in variable_types:
            return f"(variable) `{name}: {variable_types[name]}`"

        # Check if it's a struct name
        for struct_match in _STRUCT_RE.finditer(source):
            if struct_match.group(1) == name:
                return f"(struct) `struct {name}`"

        # Check shaders/filters/pure functions
        for shader_match in _SHADER_DEF_RE.finditer(source):
            if shader_match.group(1) == name:
                return f"(shader) `shader {name}(...)`"
        for pure_match in _PURE_DEF_RE.finditer(source):
            if pure_match.group(1) == name:
                return f"(pure) `pure {name}(...)`"
        return None

    return None


def run_lsp_server() -> int:
    """Run the Lockstep LSP server via pygls if installed."""

    try:
        from lsprotocol import types
        from pygls.server import LanguageServer
    except Exception as exc:  # pragma: no cover - exercised only with optional deps
        raise RuntimeError(
            "LSP dependencies are missing. Install with: pip install pygls lsprotocol"
        ) from exc

    server = LanguageServer("lockstep-lsp", "0.1.0")

    def _to_lsp_diagnostic(diag: dict[str, Any]) -> types.Diagnostic:
        severity_map = {
            "error": types.DiagnosticSeverity.Error,
            "warning": types.DiagnosticSeverity.Warning,
            "info": types.DiagnosticSeverity.Information,
        }
        line = max((diag.get("line") or 1) - 1, 0)
        column = max((diag.get("column") or 1) - 1, 0)
        return types.Diagnostic(
            range=types.Range(
                start=types.Position(line=line, character=column),
                end=types.Position(line=line, character=column + 1),
            ),
            severity=severity_map.get(diag.get("severity", "info"), types.DiagnosticSeverity.Information),
            code=diag.get("code"),
            message=diag.get("message", ""),
        )

    @server.feature(types.TEXT_DOCUMENT_DID_OPEN)
    @server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
    def validate(params):
        document = server.workspace.get_text_document(params.text_document.uri)
        _, diagnostics = compile_for_lsp(document.source)
        server.publish_diagnostics(document.uri, [_to_lsp_diagnostic(d) for d in diagnostics])

    @server.feature(types.TEXT_DOCUMENT_COMPLETION)
    def completion(params: types.CompletionParams):
        document = server.workspace.get_text_document(params.text_document.uri)
        entries = provide_bind_completion_items(
            document.source,
            line=params.position.line,
            column=params.position.character,
        )
        return types.CompletionList(
            is_incomplete=False,
            items=[
                types.CompletionItem(
                    label=entry["label"],
                    detail=entry["detail"],
                    sort_text=entry["sort_text"],
                    kind=(
                        types.CompletionItemKind.Snippet
                        if entry["kind"] == "snippet"
                        else types.CompletionItemKind.Function
                    ),
                )
                for entry in entries
            ],
        )

    @server.feature(types.TEXT_DOCUMENT_HOVER)
    def hover(params: types.HoverParams):
        document = server.workspace.get_text_document(params.text_document.uri)
        info = provide_hover_info(
            document.source,
            params.position.line,
            params.position.character,
        )
        if info is None:
            return None
        return types.Hover(
            contents=types.MarkupContent(
                kind=types.MarkupKind.Markdown,
                value=info,
            ),
        )

    @server.feature(types.TEXT_DOCUMENT_DEFINITION)
    def definition(params: types.DefinitionParams):
        document = server.workspace.get_text_document(params.text_document.uri)
        member = find_member_definition(
            document.source,
            params.position.line,
            params.position.character,
        )
        if member is None:
            return None
        target = types.Range(
            start=types.Position(line=member.line, character=member.column),
            end=types.Position(line=member.line, character=member.column + len(member.field_name)),
        )
        return types.Location(uri=document.uri, range=target)

    server.start_io()
    return 0
