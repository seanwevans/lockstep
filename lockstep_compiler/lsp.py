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
