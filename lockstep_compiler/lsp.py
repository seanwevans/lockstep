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
        entities = dict(result.entities)
        entities["__ast__"] = result.ast
        return entities, diagnostics
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


def _offset_to_line_column(source: str, offset: int) -> tuple[int, int]:
    line = source.count("\n", 0, offset)
    line_start = source.rfind("\n", 0, offset)
    column = offset if line_start == -1 else offset - line_start - 1
    return line, column


def _iter_code_tokens(source: str, start: int = 0, end: int | None = None):
    limit = len(source) if end is None else min(end, len(source))
    index = max(start, 0)
    while index < limit:
        char = source[index]
        if char.isspace():
            index += 1
            continue
        if char == "/" and index + 1 < limit:
            nxt = source[index + 1]
            if nxt == "/":
                index += 2
                while index < limit and source[index] != "\n":
                    index += 1
                continue
            if nxt == "*":
                index += 2
                while index + 1 < limit and not (
                    source[index] == "*" and source[index + 1] == "/"
                ):
                    index += 1
                index = min(index + 2, limit)
                continue
        if char in {'"', "'"}:
            quote = char
            index += 1
            while index < limit:
                if source[index] == "\\":
                    index += 2
                    continue
                if source[index] == quote:
                    index += 1
                    break
                index += 1
            continue
        if char.isalpha() or char == "_":
            token_start = index
            index += 1
            while index < limit and (source[index].isalnum() or source[index] == "_"):
                index += 1
            yield "ident", source[token_start:index], token_start
            continue
        yield "symbol", char, index
        index += 1


def _find_matching_brace(source: str, opening_brace_offset: int) -> int | None:
    depth = 1
    for token_kind, token_value, token_offset in _iter_code_tokens(
        source, start=opening_brace_offset + 1
    ):
        if token_kind != "symbol":
            continue
        if token_value == "{":
            depth += 1
        elif token_value == "}":
            depth -= 1
            if depth == 0:
                return token_offset
    return None


def _scan_struct_blocks(source: str) -> list[tuple[str, int, int]]:
    blocks: list[tuple[str, int, int]] = []
    tokens = list(_iter_code_tokens(source))
    index = 0
    while index + 2 < len(tokens):
        kind, value, _ = tokens[index]
        if kind == "ident" and value == "struct":
            next_kind, struct_name, _ = tokens[index + 1]
            brace_kind, brace_value, brace_offset = tokens[index + 2]
            if next_kind == "ident" and brace_kind == "symbol" and brace_value == "{":
                body_end = _find_matching_brace(source, brace_offset)
                if body_end is not None:
                    blocks.append((struct_name, brace_offset + 1, body_end))
                index += 3
                continue
        index += 1
    return blocks


def _scan_struct_fields(
    source: str,
    body_start: int,
    body_end: int,
    allowed_fields: set[str] | None,
) -> list[tuple[str, int]]:
    fields: list[tuple[str, int]] = []
    body_tokens = list(_iter_code_tokens(source, start=body_start, end=body_end))
    index = 0
    while index + 2 < len(body_tokens):
        type_kind, _, _ = body_tokens[index]
        field_kind, field_name, field_offset = body_tokens[index + 1]
        semi_kind, semi_value, _ = body_tokens[index + 2]
        if (
            type_kind == "ident"
            and field_kind == "ident"
            and semi_kind == "symbol"
            and semi_value == ";"
        ):
            if allowed_fields is None or field_name in allowed_fields:
                fields.append((field_name, field_offset))
            index += 3
            continue
        index += 1
    return fields


def build_struct_member_index(source: str) -> dict[str, dict[str, MemberDefinition]]:
    """Index struct field declarations with source locations (0-based)."""

    entities, _ = compile_for_lsp(source)
    ast_program = entities.get("__ast__")

    expected_fields: dict[str, set[str]] = {}
    if ast_program is not None:
        for struct_decl in ast_program.structs:
            expected_fields[struct_decl.name] = {field.name for field in struct_decl.fields}
    else:
        for struct_decl in entities.get("structs", []):
            if isinstance(struct_decl, dict) and struct_decl.get("name"):
                expected_fields[struct_decl["name"]] = {
                    field.get("name")
                    for field in struct_decl.get("fields", [])
                    if isinstance(field, dict) and field.get("name")
                }

    index: dict[str, dict[str, MemberDefinition]] = {}
    for struct_name, body_start, body_end in _scan_struct_blocks(source):
        if expected_fields and struct_name not in expected_fields:
            continue
        allowed_fields = expected_fields.get(struct_name) if expected_fields else None
        if allowed_fields == set():
            allowed_fields = None
        struct_fields = _scan_struct_fields(source, body_start, body_end, allowed_fields)
        if not struct_fields:
            continue
        index.setdefault(struct_name, {})
        for field_name, absolute_offset in struct_fields:
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


def provide_bind_completion_items(source: str) -> list[str]:
    """Return completion labels for bind routes and callable kernels."""

    entities, _ = compile_for_lsp(source)
    items: list[str] = []

    bind_routes = entities.get("bind_routes", [])
    if not bind_routes:
        for bind_match in _BIND_BLOCK_RE.finditer(source):
            bind_body = bind_match.group("body")
            for statement in bind_body.split(";"):
                normalized = re.sub(r"\s+", "", statement)
                if normalized:
                    bind_routes.append(f"{normalized};")

    for route in bind_routes:
        if route not in items:
            items.append(route)

    shaders = entities.get("shaders", [])
    if not shaders:
        shaders = [{"name": name} for name in _SHADER_DEF_RE.findall(source)]

    for shader in shaders:
        shader_name = shader.get("name") if isinstance(shader, dict) else str(shader)
        if shader_name and shader_name not in items:
            items.append(f"{shader_name}(...)")

    pure_functions = entities.get("pure_functions", [])
    if not pure_functions:
        pure_functions = [{"name": name} for name in _PURE_DEF_RE.findall(source)]

    for pure_function in pure_functions:
        function_name = (
            pure_function.get("name")
            if isinstance(pure_function, dict)
            else str(pure_function)
        )
        if function_name and function_name not in items:
            items.append(f"{function_name}(...)")

    return items


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
        labels = provide_bind_completion_items(document.source)
        return types.CompletionList(
            is_incomplete=False,
            items=[types.CompletionItem(label=label, kind=types.CompletionItemKind.Function) for label in labels],
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
