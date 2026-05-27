from __future__ import annotations

import asyncio
import hashlib
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
    callable_index: dict[str, "DefinitionTarget"]
    entities: dict[str, Any]


@dataclass(frozen=True)
class CompiledLspContext:
    entities: dict[str, Any]
    diagnostics: list[dict[str, Any]]


@dataclass(frozen=True)
class DefinitionTarget:
    line: int
    column: int
    symbol: str


@dataclass(frozen=True)
class DocumentSnapshotKey:
    uri: str
    version: int | None
    source_hash: str


@dataclass
class CachedLspSnapshot:
    compiled_context: CompiledLspContext
    analysis_context: AnalysisContext | None = None


_MEMBER_ACCESS_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)")
_BIND_BLOCK_RE = re.compile(r"\bbind\s*\{(?P<body>[\s\S]*?)\}", re.MULTILINE)


def _diagnostics_to_dicts(diagnostics: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "severity": diagnostic.severity,
            "code": diagnostic.code,
            "message": diagnostic.message,
            "line": diagnostic.line,
            "column": diagnostic.column,
            "hint": diagnostic.hint,
            "source_file": diagnostic.source_file,
        }
        for diagnostic in diagnostics
    ]


def _is_inside_bind_block(source: str, line: int, column: int) -> bool:
    if line < 0 or column < 0:
        return False

    lines = source.splitlines(keepends=True)
    if line >= len(lines):
        return False

    offset = sum(len(existing) for existing in lines[:line]) + min(
        column, len(lines[line])
    )

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


def compile_for_lsp(source: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    context = compile_context_for_lsp(source)
    return context.entities, context.diagnostics


def compile_context_for_lsp(source: str) -> CompiledLspContext:
    compile_diagnostics: list[dict[str, Any]]
    entities: dict[str, Any]
    try:
        result = compile_lockstep(source, verbose=False)
        entities = result.entities
        compile_diagnostics = _diagnostics_to_dicts(result.diagnostics)
    except LockstepCompileError as error:
        compile_diagnostics = _diagnostics_to_dicts(error.diagnostics)
        entities = {}
        try:
            recovery = compile_lockstep(
                source,
                verbose=False,
                semantic_validator=lambda *_args, **_kwargs: [],
            )
            # Keep semantic-index entities from recovery, but preserve diagnostics
            # from the original compile failure so editors can surface true errors.
            entities = recovery.entities
        except Exception:
            entities = {}
    return CompiledLspContext(
        entities=entities,
        diagnostics=compile_diagnostics,
    )


def _infer_variable_types_from_entities(entities: dict[str, Any]) -> dict[str, str]:
    inferred: dict[str, str] = {}
    callable_returns: dict[str, str] = {
        pure.get("name"): pure.get("return_type")
        for pure in entities.get("pure_functions", [])
        if pure.get("name") and pure.get("return_type")
    }

    def _infer_expr_type(expr: Any) -> str | None:
        if expr is None:
            return None
        expr_type_name = type(expr).__name__
        if expr_type_name == "AstExprInt":
            return "int"
        if expr_type_name == "AstExprFloat":
            return "float"
        if expr_type_name == "AstExprBool":
            return "bool"
        if expr_type_name == "AstExprString":
            return "string"
        if expr_type_name == "AstExprVar":
            return inferred.get(getattr(expr, "name", ""))
        if expr_type_name == "AstExprCall":
            return callable_returns.get(getattr(expr, "name", ""))
        if expr_type_name == "AstExprUnary":
            return _infer_expr_type(getattr(expr, "operand", None))
        if expr_type_name == "AstExprBinary":
            left_type = _infer_expr_type(getattr(expr, "left", None))
            right_type = _infer_expr_type(getattr(expr, "right", None))
            if left_type is not None and left_type == right_type:
                return left_type
        return None

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
            initializer_type = _infer_expr_type(initializer)
            if initializer_type is not None:
                inferred.setdefault(name, initializer_type)

    for pure in entities.get("pure_functions", []):
        for param in pure.get("params", []):
            name = param.get("name")
            declared_type = param.get("type")
            if name and declared_type:
                inferred.setdefault(name, declared_type)

    return inferred


def _build_struct_member_index_from_entities(
    entities: dict[str, Any],
) -> dict[str, dict[str, MemberDefinition]]:
    index: dict[str, dict[str, MemberDefinition]] = {}

    for struct in entities.get("structs", []):
        struct_name = struct.get("name")
        if not struct_name:
            continue
        index[struct_name] = {}
        for field in struct.get("fields", []):
            field_name = field.get("name")
            if not field_name:
                continue
            line = max((field.get("line") or 1) - 1, 0)
            column = max(field.get("column") or 0, 0)
            index[struct_name][field_name] = MemberDefinition(
                struct_name=struct_name,
                field_name=field_name,
                line=line,
                column=column,
            )

    return index


def _build_struct_field_type_index_from_entities(
    entities: dict[str, Any],
) -> dict[str, dict[str, str]]:
    return {
        struct.get("name"): {
            field.get("name"): field.get("type")
            for field in struct.get("fields", [])
            if field.get("name") and field.get("type")
        }
        for struct in entities.get("structs", [])
        if struct.get("name")
    }


def build_struct_member_index(source: str) -> dict[str, dict[str, MemberDefinition]]:
    entities, _ = compile_for_lsp(source)
    return _build_struct_member_index_from_entities(entities)


def infer_variable_types(source: str) -> dict[str, str]:
    entities, _ = compile_for_lsp(source)
    return _infer_variable_types_from_entities(entities)


def build_analysis_context(
    source: str,
    compiled_context: CompiledLspContext | None = None,
) -> AnalysisContext:
    context = compiled_context or compile_context_for_lsp(source)
    entities = context.entities
    return AnalysisContext(
        variable_types=_infer_variable_types_from_entities(entities),
        struct_member_index=_build_struct_member_index_from_entities(entities),
        struct_field_types=_build_struct_field_type_index_from_entities(entities),
        callable_index=_build_callable_index(source, entities),
        entities=entities,
    )


def _offset_to_line_column(source: str, offset: int) -> tuple[int, int]:
    clamped = min(max(offset, 0), len(source))
    line = source.count("\n", 0, clamped)
    line_start = source.rfind("\n", 0, clamped)
    if line_start == -1:
        return line, clamped
    return line, clamped - line_start - 1


def _line_column_to_offset(source: str, line: int, column: int) -> int | None:
    if line < 0 or column < 0:
        return None
    lines = source.splitlines(keepends=True)
    if line >= len(lines):
        return None
    return sum(len(existing) for existing in lines[:line]) + min(column, len(lines[line]))


def _code_mask(source: str) -> list[bool]:
    mask = [True] * len(source)
    in_string = False
    escaped = False
    in_comment = False
    for index, char in enumerate(source):
        if in_comment:
            mask[index] = False
            if char == "\n":
                in_comment = False
            continue

        if in_string:
            mask[index] = False
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == "/" and index + 1 < len(source) and source[index + 1] == "/":
            mask[index] = False
            mask[index + 1] = False
            in_comment = True
            continue

        if char == '"':
            mask[index] = False
            in_string = True
            escaped = False
    return mask


def _identifier_at_position(source: str, line: int, column: int) -> str | None:
    offset = _line_column_to_offset(source, line, column)
    if offset is None or offset >= len(source):
        return None

    code_mask = _code_mask(source)
    if not code_mask[offset]:
        return None

    index = offset
    if not (source[index].isalnum() or source[index] == "_"):
        if index > 0 and (source[index - 1].isalnum() or source[index - 1] == "_"):
            index -= 1
        else:
            return None

    start = index
    while start > 0 and (source[start - 1].isalnum() or source[start - 1] == "_"):
        start -= 1
    end = index + 1
    while end < len(source) and (source[end].isalnum() or source[end] == "_"):
        end += 1

    if any(not code_mask[pos] for pos in range(start, end)):
        return None
    return source[start:end]


def _member_access_at_position(source: str, line: int, column: int) -> tuple[str, str] | None:
    lines = source.splitlines()
    if line < 0 or line >= len(lines):
        return None
    line_text = lines[line]
    code_mask = _code_mask(source)
    line_start = _line_column_to_offset(source, line, 0)
    if line_start is None:
        return None
    for match in _MEMBER_ACCESS_RE.finditer(line_text):
        start, end = match.span(0)
        if not (start <= column < end):
            continue
        if any(not code_mask[line_start + pos] for pos in range(start, end)):
            continue
        return match.group(1), match.group(2)
    return None


def _find_callable_definition(
    source: str,
    pattern: re.Pattern[str],
    symbol: str,
) -> DefinitionTarget | None:
    match = pattern.search(source)
    if match is None:
        return None

    line, column = _offset_to_line_column(source, match.start("name"))
    return DefinitionTarget(line=line, column=column, symbol=symbol)


def _location_target_from_entity(entity: dict[str, Any]) -> DefinitionTarget | None:
    name = entity.get("name")
    line = entity.get("line")
    column = entity.get("column")
    if not name or not isinstance(line, int) or not isinstance(column, int):
        return None
    if line <= 0 or column < 0:
        return None
    return DefinitionTarget(line=line - 1, column=column, symbol=name)


def _build_callable_index(
    source: str,
    entities: dict[str, Any],
) -> dict[str, DefinitionTarget]:
    callable_index: dict[str, DefinitionTarget] = {}

    for shader in entities.get("shaders", []):
        name = shader.get("name")
        if not name or name in callable_index:
            continue
        target = _location_target_from_entity(shader)
        if target is None:
            target = _find_callable_definition(
                source,
                re.compile(rf"\bshader\s+(?P<name>{re.escape(name)})\s*\("),
                name,
            )
        if target is not None:
            callable_index[name] = target

    for filter_decl in entities.get("filters", []):
        name = filter_decl.get("name")
        if not name or name in callable_index:
            continue
        target = _location_target_from_entity(filter_decl)
        if target is None:
            target = _find_callable_definition(
                source,
                re.compile(rf"\bfilter\s+(?P<name>{re.escape(name)})\s*\("),
                name,
            )
        if target is not None:
            callable_index[name] = target

    for pure in entities.get("pure_functions", []):
        name = pure.get("name")
        if not name or pure.get("intrinsic") or name in callable_index:
            continue
        target = _location_target_from_entity(pure)
        if target is None:
            target = _find_callable_definition(
                source,
                re.compile(
                    rf"\bpure\b[\s\S]*?\b(?P<name>{re.escape(name)})\s*\("
                ),
                name,
            )
        if target is not None:
            callable_index[name] = target

    return callable_index


def find_member_definition(
    source: str,
    line: int,
    column: int,
    analysis_context: AnalysisContext | None = None,
) -> MemberDefinition | None:
    context = analysis_context or build_analysis_context(source)
    member_access = _member_access_at_position(source, line, column)
    if member_access is None:
        return None
    variable_name, field_name = member_access
    struct_name = context.variable_types.get(variable_name)
    if not struct_name:
        return None
    return context.struct_member_index.get(struct_name, {}).get(field_name)


def find_definition_target(
    source: str,
    line: int,
    column: int,
    analysis_context: AnalysisContext | None = None,
) -> DefinitionTarget | None:
    member = find_member_definition(
        source,
        line,
        column,
        analysis_context=analysis_context,
    )
    if member is not None:
        return DefinitionTarget(
            line=member.line,
            column=member.column,
            symbol=member.field_name,
        )

    lines = source.splitlines()
    if line < 0 or line >= len(lines):
        return None

    context = analysis_context or build_analysis_context(source)
    name = _identifier_at_position(source, line, column)
    if name is None:
        return None
    return context.callable_index.get(name)


def provide_bind_completion_items(
    source: str,
    *,
    line: int | None = None,
    column: int | None = None,
    analysis_context: AnalysisContext | None = None,
) -> list[dict[str, Any]]:
    context = analysis_context or build_analysis_context(source)
    entities = context.entities
    completion_entries: list[dict[str, Any]] = []
    seen_labels: set[str] = set()

    inside_bind = (
        line is not None
        and column is not None
        and _is_inside_bind_block(source, line=line, column=column)
    )

    if inside_bind:
        bind_route_labels = [
            entry.get("route")
            for entry in entities.get("bind_routes_meta", [])
            if isinstance(entry, dict) and entry.get("route")
        ]
        if not bind_route_labels:
            bind_route_labels = [route for route in entities.get("bind_routes", []) if route]

        for route in bind_route_labels:
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

    shader_filter_names = sorted(
        {
            f"{entry.get('name')}(...)"
            for entry in [*entities.get("shaders", []), *entities.get("filters", [])]
            if entry.get("name")
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

    pure_function_names = sorted(
        {
            f"{entry.get('name')}(...)"
            for entry in entities.get("pure_functions", [])
            if entry.get("name") and not entry.get("intrinsic")
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
    context = analysis_context or build_analysis_context(source)
    member_access = _member_access_at_position(source, line, column)
    if member_access is not None:
        variable_name, field_name = member_access
        struct_name = context.variable_types.get(variable_name)
        if struct_name:
            field_type = context.struct_field_types.get(struct_name, {}).get(field_name)
            if field_type:
                return f"(field) `{struct_name}.{field_name}: {field_type}`"
            return f"(field) `{struct_name}.{field_name}`"
        return None

    name = _identifier_at_position(source, line, column)
    if name is None:
        return None
    if name in context.variable_types:
        return f"(variable) `{name}: {context.variable_types[name]}`"

    entities = context.entities
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
    return None


def run_lsp_server() -> int:
    try:
        from lsprotocol import types
        from pygls.server import LanguageServer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "LSP dependencies are missing. Install with: pip install pygls lsprotocol"
        ) from exc

    server = LanguageServer("lockstep-lsp", "0.1.0")
    debounce_seconds = 0.15
    # Cache strategy:
    # - Keys are (uri, version, source_hash) snapshots.
    # - Version is preferred when provided by client; hash guards against stale
    #   state for clients that omit/skip versions.
    # - Close events clear all snapshots for that URI.
    #
    # Concurrency behavior:
    # - Handlers execute on the event loop. We avoid duplicate compile work by
    #   keeping a per-snapshot in-flight task map and awaiting shared tasks.
    snapshot_cache: dict[DocumentSnapshotKey, CachedLspSnapshot] = {}
    pending_compiles: dict[DocumentSnapshotKey, asyncio.Task[CompiledLspContext]] = {}
    latest_snapshot_key_by_uri: dict[str, DocumentSnapshotKey] = {}
    pending_tasks: dict[str, asyncio.Task[Any]] = {}

    def _snapshot_key(uri: str, source: str, version: int | None) -> DocumentSnapshotKey:
        return DocumentSnapshotKey(
            uri=uri,
            version=version,
            source_hash=hashlib.sha1(source.encode("utf-8")).hexdigest(),
        )

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
            severity=severity_map.get(
                diag.get("severity", "info"), types.DiagnosticSeverity.Information
            ),
            code=diag.get("code"),
            message=diag.get("message", ""),
        )

    async def _compiled_context_for_document(
        *,
        uri: str,
        source: str,
        version: int | None,
    ) -> tuple[DocumentSnapshotKey, CompiledLspContext]:
        key = _snapshot_key(uri, source, version)
        cached = snapshot_cache.get(key)
        if cached is not None:
            latest_snapshot_key_by_uri[uri] = key
            return key, cached.compiled_context

        task = pending_compiles.get(key)
        if task is None:
            task = asyncio.create_task(asyncio.to_thread(compile_context_for_lsp, source))
            pending_compiles[key] = task
        compiled = await task
        pending_compiles.pop(key, None)
        snapshot_cache[key] = CachedLspSnapshot(compiled_context=compiled)
        latest_snapshot_key_by_uri[uri] = key
        return key, compiled

    def _analysis_context_for_snapshot(
        key: DocumentSnapshotKey,
        source: str,
    ) -> AnalysisContext:
        cached = snapshot_cache[key]
        if cached.analysis_context is None:
            cached.analysis_context = build_analysis_context(
                source,
                compiled_context=cached.compiled_context,
            )
        return cached.analysis_context

    async def _validate(uri: str, *, debounced: bool) -> None:
        if debounced:
            await asyncio.sleep(debounce_seconds)
        document = server.workspace.get_text_document(uri)
        _, compiled = await _compiled_context_for_document(
            uri=uri,
            source=document.source,
            version=getattr(document, "version", None),
        )
        server.publish_diagnostics(
            uri, [_to_lsp_diagnostic(d) for d in compiled.diagnostics]
        )
        pending_tasks.pop(uri, None)

    def _schedule_validate(uri: str, *, debounced: bool = True) -> None:
        task = pending_tasks.get(uri)
        if task is not None and not task.done():
            task.cancel()
        pending_tasks[uri] = asyncio.create_task(_validate(uri, debounced=debounced))

    def _clear_document_context(uri: str) -> None:
        task = pending_tasks.pop(uri, None)
        if task is not None and not task.done():
            task.cancel()
        latest_snapshot_key_by_uri.pop(uri, None)
        for key in [existing for existing in snapshot_cache if existing.uri == uri]:
            snapshot_cache.pop(key, None)
            pending = pending_compiles.pop(key, None)
            if pending is not None and not pending.done():
                pending.cancel()

    @server.feature(types.TEXT_DOCUMENT_DID_OPEN)  # type: ignore[untyped-decorator]
    @server.feature(types.TEXT_DOCUMENT_DID_CHANGE)  # type: ignore[untyped-decorator]
    def validate(params: Any) -> None:
        _schedule_validate(params.text_document.uri)

    @server.feature(types.TEXT_DOCUMENT_DID_SAVE)  # type: ignore[untyped-decorator]
    def validate_on_save(params: Any) -> None:
        _schedule_validate(params.text_document.uri, debounced=False)

    @server.feature(types.TEXT_DOCUMENT_DID_CLOSE)  # type: ignore[untyped-decorator]
    def close_document(params: Any) -> None:
        _clear_document_context(params.text_document.uri)
        server.publish_diagnostics(params.text_document.uri, [])

    @server.feature(types.TEXT_DOCUMENT_COMPLETION)  # type: ignore[untyped-decorator]
    async def completion(
        params: types.CompletionParams,
    ) -> types.CompletionList:
        document = server.workspace.get_text_document(params.text_document.uri)
        key, _compiled = await _compiled_context_for_document(
            uri=document.uri,
            source=document.source,
            version=getattr(document, "version", None),
        )
        analysis_context = _analysis_context_for_snapshot(key, document.source)
        entries = provide_bind_completion_items(
            document.source,
            line=params.position.line,
            column=params.position.character,
            analysis_context=analysis_context,
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

    @server.feature(types.TEXT_DOCUMENT_HOVER)  # type: ignore[untyped-decorator]
    async def hover(params: types.HoverParams) -> types.Hover | None:
        document = server.workspace.get_text_document(params.text_document.uri)
        key, _compiled = await _compiled_context_for_document(
            uri=document.uri,
            source=document.source,
            version=getattr(document, "version", None),
        )
        analysis_context = _analysis_context_for_snapshot(
            key,
            document.source,
        )
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

    @server.feature(types.TEXT_DOCUMENT_DEFINITION)  # type: ignore[untyped-decorator]
    async def definition(
        params: types.DefinitionParams,
    ) -> types.Location | None:
        document = server.workspace.get_text_document(params.text_document.uri)
        key, _compiled = await _compiled_context_for_document(
            uri=document.uri,
            source=document.source,
            version=getattr(document, "version", None),
        )
        analysis_context = _analysis_context_for_snapshot(
            key,
            document.source,
        )
        target = find_definition_target(
            document.source,
            params.position.line,
            params.position.character,
            analysis_context=analysis_context,
        )
        if target is None:
            return None
        target_range = types.Range(
            start=types.Position(line=target.line, character=target.column),
            end=types.Position(
                line=target.line, character=target.column + len(target.symbol)
            ),
        )
        return types.Location(uri=document.uri, range=target_range)

    server.start_io()
    return 0
