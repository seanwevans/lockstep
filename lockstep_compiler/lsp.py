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


@dataclass(frozen=True)
class AnalysisContext:
    variable_types: dict[str, str]
    struct_member_index: dict[str, dict[str, MemberDefinition]]
    struct_field_types: dict[str, dict[str, str]]


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
_FILTER_DEF_RE = re.compile(r"\bfilter\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
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
                "source_file": diagnostic.source_file,
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
                "source_file": diagnostic.source_file,
            }
            for diagnostic in error.diagnostics
        ]
        return {}, diagnostics


def build_struct_member_index(source: str) -> dict[str, dict[str, MemberDefinition]]:
    """Index struct field declarations with source locations (0-based)."""

    entities, _ = compile_for_lsp(source)
    if entities.get("structs"):
        return _build_struct_member_index_from_entities(source, entities)

    return _build_struct_member_index_from_regex(source)


def _mask_comments(source: str) -> str:
    """Replace comment contents with spaces while preserving offsets."""

    chars = list(source)
    index = 0
    while index < len(chars):
        if chars[index] == "/" and index + 1 < len(chars):
            if chars[index + 1] == "/":
                chars[index] = " "
                chars[index + 1] = " "
                index += 2
                while index < len(chars) and chars[index] != "\n":
                    chars[index] = " "
                    index += 1
                continue
            if chars[index + 1] == "*":
                chars[index] = " "
                chars[index + 1] = " "
                index += 2
                while index + 1 < len(chars):
                    if chars[index] == "*" and chars[index + 1] == "/":
                        chars[index] = " "
                        chars[index + 1] = " "
                        index += 2
                        break
                    if chars[index] != "\n":
                        chars[index] = " "
                    index += 1
                continue
        index += 1
    return "".join(chars)


def _offset_to_line_column(source: str, offset: int) -> tuple[int, int]:
    line = source.count("\n", 0, offset)
    line_start = source.rfind("\n", 0, offset)
    if line_start == -1:
        return line, offset
    return line, offset - line_start - 1


def _find_matching_brace(source: str, brace_index: int) -> int:
    depth = 0
    for idx in range(brace_index, len(source)):
        token = source[idx]
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _build_struct_member_index_from_entities(
    source: str,
    entities: dict[str, Any],
) -> dict[str, dict[str, MemberDefinition]]:
    masked = _mask_comments(source)
    index: dict[str, dict[str, MemberDefinition]] = {}

    for struct in entities.get("structs", []):
        struct_name = struct.get("name")
        if not struct_name:
            continue
        field_names = [field.get("name") for field in struct.get("fields", []) if field.get("name")]
        struct_match = re.search(rf"\bstruct\s+{re.escape(struct_name)}\b", masked)
        if not struct_match:
            continue

        open_brace = masked.find("{", struct_match.end())
        if open_brace == -1:
            continue
        close_brace = _find_matching_brace(masked, open_brace)
        if close_brace == -1:
            continue

        struct_fields: dict[str, MemberDefinition] = {}
        body = masked[open_brace + 1 : close_brace]
        body_start = open_brace + 1
        for field_name in field_names:
            field_match = re.search(rf"\b{re.escape(field_name)}\b\s*;", body)
            if not field_match:
                continue
            field_offset = body_start + field_match.start()
            line, column = _offset_to_line_column(source, field_offset)
            struct_fields[field_name] = MemberDefinition(
                struct_name=struct_name,
                field_name=field_name,
                line=line,
                column=column,
            )
        index[struct_name] = struct_fields

    return index


def _build_struct_member_index_from_regex(source: str) -> dict[str, dict[str, MemberDefinition]]:
    """Fallback indexer used when compilation cannot provide entities."""

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
            line, column = _offset_to_line_column(source, absolute_offset)
            index[struct_name][field_name] = MemberDefinition(
                struct_name=struct_name,
                field_name=field_name,
                line=line,
                column=column,
            )

    return index


def infer_variable_types(source: str) -> dict[str, str]:
    """Best-effort type inference for names declared as `<Type> <name>`."""

    entities, _ = compile_for_lsp(source)
    if entities:
        return _infer_variable_types_from_entities(entities)

    return _infer_variable_types_from_regex(source)


def _expr_inferred_type(expr: Any, known_types: dict[str, str]) -> str | None:
    path = getattr(expr, "path", None)
    if path:
        return known_types.get(path[0])

    literal_kind = getattr(expr, "kind", None)
    if literal_kind == "number":
        value = getattr(expr, "value", "")
        return "float" if "." in value else "int"
    if literal_kind == "boolean":
        return "bool"

    return None


def _infer_variable_types_from_entities(entities: dict[str, Any]) -> dict[str, str]:
    inferred: dict[str, str] = {}

    for kernel in [*entities.get("shaders", []), *entities.get("filters", [])]:
        for param in kernel.get("params", []):
            name = param.get("name")
            declared_type = param.get("type")
            if name and declared_type:
                inferred.setdefault(name, declared_type)

        for stmt in kernel.get("body_ast", []):
            name = getattr(stmt, "name", None)
            if not name:
                continue
            declared_type = getattr(stmt, "declared_type", None)
            if declared_type is not None:
                inferred.setdefault(name, str(declared_type))
                continue
            initializer = getattr(stmt, "initializer", None)
            initializer_type = _expr_inferred_type(initializer, inferred) if initializer else None
            if initializer_type:
                inferred.setdefault(name, initializer_type)

    for pure in entities.get("pure_functions", []):
        for param in pure.get("params", []):
            name = param.get("name")
            declared_type = param.get("type")
            if name and declared_type:
                inferred.setdefault(name, declared_type)

    return inferred


def _infer_variable_types_from_regex(source: str) -> dict[str, str]:
    """Fallback inference used when compilation cannot provide entities."""

    inferred: dict[str, str] = {}
    for declared_type, name in _PARAM_RE.findall(source):
        inferred.setdefault(name, declared_type)

    for declared_type, name in _TYPED_NAME_RE.findall(source):
        if declared_type in {"return", "if", "for", "while", "bind", "stream", "uniform", "shader", "filter", "pure", "struct", "pipeline"}:
            continue
        inferred.setdefault(name, declared_type)
    return inferred


def _build_struct_field_type_index(source: str) -> dict[str, dict[str, str]]:
    """Index struct field type declarations for hover text generation."""

    index: dict[str, dict[str, str]] = {}
    for struct_match in _STRUCT_RE.finditer(source):
        struct_name = struct_match.group(1)
        body_start = struct_match.end()
        body_end = source.find("}", body_start)
        if body_end == -1:
            continue

        body = source[body_start:body_end]
        index[struct_name] = {}
        for field_match in _FIELD_RE.finditer(body):
            index[struct_name][field_match.group(2)] = field_match.group(1)

    return index


def build_analysis_context(source: str) -> AnalysisContext:
    """Build reusable per-request analysis context for member-based LSP features."""

    return AnalysisContext(
        variable_types=infer_variable_types(source),
        struct_member_index=build_struct_member_index(source),
        struct_field_types=_build_struct_field_type_index(source),
    )


def find_member_definition(
    source: str,
    line: int,
    column: int,
    analysis_context: AnalysisContext | None = None,
) -> MemberDefinition | None:
    """Resolve `foo.bar` usages to `struct` field declaration locations."""

    lines = source.splitlines()
    if line < 0 or line >= len(lines):
        return None

    line_text = lines[line]
    context = analysis_context or build_analysis_context(source)
    variable_types = context.variable_types
    struct_index = context.struct_member_index

    for match in _MEMBER_ACCESS_RE.finditer(line_text):
        start, end = match.span(0)
        if not (start <= column <= end):
            continue
        variable_name, field_name = match.groups()
        struct_name = variable_types.get(variable_name)
        if not struct_name:
            return None
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
    analysis_context: AnalysisContext | None = None,
) -> str | None:
    """Return hover text for the identifier at the given position."""

    lines = source.splitlines()
    if line < 0 or line >= len(lines):
        return None

    line_text = lines[line]
    context = analysis_context or build_analysis_context(source)
    variable_types = context.variable_types
    struct_index = context.struct_member_index
    field_types = context.struct_field_types

    # Try member access first (foo.bar)
    for match in _MEMBER_ACCESS_RE.finditer(line_text):
        start, end = match.span(0)
        if not (start <= column <= end):
            continue
        variable_name, field_name = match.groups()
        struct_name = variable_types.get(variable_name)
        if struct_name:
            field_def = struct_index.get(struct_name, {}).get(field_name)
            if field_def:
                field_type = field_types.get(struct_name, {}).get(field_name)
                if field_type:
                    return f"(field) `{struct_name}.{field_name}: {field_type}`"
            return f"(field) `{struct_name}.{field_name}`"
        return None

    # Try plain identifier
    id_pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
    for match in id_pattern.finditer(line_text):
        start, end = match.span(1)
        if not (start <= column < end):
            continue
        name = match.group(1)
        if name in variable_types:
            return f"(variable) `{name}: {variable_types[name]}`"

        entities, _ = compile_for_lsp(source)
        if entities:
            if any(struct.get("name") == name for struct in entities.get("structs", [])):
                return f"(struct) `struct {name}`"
            if any(shader.get("name") == name for shader in entities.get("shaders", [])):
                return f"(shader) `shader {name}(...)`"
            if any(filter_decl.get("name") == name for filter_decl in entities.get("filters", [])):
                return f"(filter) `filter {name}(...)`"
            if any(
                pure.get("name") == name and not pure.get("intrinsic")
                for pure in entities.get("pure_functions", [])
            ):
                return f"(pure) `pure {name}(...)`"
        else:
            # Regex fallback only when compile failed hard.
            for struct_match in _STRUCT_RE.finditer(source):
                if struct_match.group(1) == name:
                    return f"(struct) `struct {name}`"

            for shader_match in _SHADER_DEF_RE.finditer(source):
                if shader_match.group(1) == name:
                    return f"(shader) `shader {name}(...)`"
            for filter_match in _FILTER_DEF_RE.finditer(source):
                if filter_match.group(1) == name:
                    return f"(filter) `filter {name}(...)`"
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
        analysis_context = build_analysis_context(document.source)
        info = provide_hover_info(
            document.source,
            params.position.line,
            params.position.character,
            analysis_context=analysis_context,
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
        analysis_context = build_analysis_context(document.source)
        member = find_member_definition(
            document.source,
            params.position.line,
            params.position.character,
            analysis_context=analysis_context,
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
