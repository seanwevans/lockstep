import functools
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from antlr4 import CommonTokenStream, InputStream

from .ast import AstKernelParam, AstProgram, AstPureDecl, AstType, ast_to_entities, build_program_ast
from .c_header import emit_c_header
from .codegen import CodegenError, emit_llvm_ir
from .errors import LockstepCompileError, ParseErrorCollector
from .models import (
    LockstepCompileResult,
    LockstepDiagnostic,
    normalize_diagnostics,
)
from .optimizer import optimize_bind_routes
from .prelude import load_intrinsics
from .visitors import validate_semantics as _validate_semantics


DEFAULT_SOURCE_FILE = "<stdin>"
DEFAULT_MAX_SOURCE_BYTES = 1_048_576
DEFAULT_PARSE_TIMEOUT_MS = 2_000
DEFAULT_MAX_EXPRESSION_NESTING = 128
DEFAULT_INVALID_FRONTEND_LIMIT_CODE = "LCK006"
_DEPENDENCY_DECL_PATTERN = re.compile(
    r'^\s*(?:import|#include)\s+"((?:[^"\\]|\\.)+)"\s*;\s*$',
    re.MULTILINE,
)


@dataclass(frozen=True)
class FrontendLimits:
    max_source_bytes: int | None = DEFAULT_MAX_SOURCE_BYTES
    parse_timeout_ms: int | None = DEFAULT_PARSE_TIMEOUT_MS
    max_expression_nesting: int | None = DEFAULT_MAX_EXPRESSION_NESTING


class FrontendLimitExceeded(RuntimeError):
    def __init__(self, diagnostic: LockstepDiagnostic):
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


class _DeadlineTokenStream(CommonTokenStream):
    def __init__(self, lexer, *, deadline: float | None = None, source_file: str):
        super().__init__(lexer)
        self._deadline = deadline
        self._source_file = source_file

    def _enforce_deadline(self) -> None:
        if self._deadline is None:
            return
        if time.monotonic() > self._deadline:
            raise FrontendLimitExceeded(
                LockstepDiagnostic(
                    severity="error",
                    code="LCK004",
                    message=(
                        "Parsing exceeded the configured runtime limit "
                        "(parse_timeout_ms)."
                    ),
                    line=1,
                    column=0,
                    source_file=self._source_file,
                    hint=(
                        "Increase --parse-timeout-ms (or parse_timeout_ms via API) "
                        "for trusted large inputs, or set it to 0 to disable."
                    ),
                )
            )

    def LA(self, i: int):
        self._enforce_deadline()
        return super().LA(i)

    def LT(self, i: int):
        self._enforce_deadline()
        return super().LT(i)

    def consume(self):
        self._enforce_deadline()
        return super().consume()


def _normalize_frontend_limit_value(
    value: int | None,
    *,
    name: str,
    source_file: str,
) -> int | None:
    if value is None:
        return None
    if value < 0:
        diagnostic = LockstepDiagnostic(
            severity="error",
            code=DEFAULT_INVALID_FRONTEND_LIMIT_CODE,
            message=(
                f"Invalid frontend limit '{name}': {value}. "
                "Expected a non-negative integer."
            ),
            line=1,
            column=0,
            source_file=source_file,
            hint=(
                f"Set {name} to 0 to disable, or provide a positive integer "
                "to enforce the limit."
            ),
        )
        raise LockstepCompileError(
            [diagnostic],
            diagnostics=[diagnostic],
            phase="parse",
            source_file=source_file,
        )
    if value == 0:
        return None
    return value


def _normalize_frontend_limits(
    limits: FrontendLimits | None,
    *,
    source_file: str,
) -> FrontendLimits:
    resolved = limits or FrontendLimits()
    return FrontendLimits(
        max_source_bytes=_normalize_frontend_limit_value(
            resolved.max_source_bytes,
            name="max_source_bytes",
            source_file=source_file,
        ),
        parse_timeout_ms=_normalize_frontend_limit_value(
            resolved.parse_timeout_ms,
            name="parse_timeout_ms",
            source_file=source_file,
        ),
        max_expression_nesting=_normalize_frontend_limit_value(
            resolved.max_expression_nesting,
            name="max_expression_nesting",
            source_file=source_file,
        ),
    )


def _enforce_source_size_limit(
    source_code: str,
    *,
    source_file: str,
    limits: FrontendLimits,
) -> None:
    if limits.max_source_bytes is None:
        return
    source_size = len(source_code.encode("utf-8"))
    if source_size <= limits.max_source_bytes:
        return
    diagnostic = LockstepDiagnostic(
        severity="error",
        code="LCK003",
        message=(
            "Source size exceeds the configured limit "
            f"({source_size} bytes > {limits.max_source_bytes} bytes)."
        ),
        line=1,
        column=0,
        source_file=source_file,
        hint=(
            "Increase --max-source-bytes (or max_source_bytes via API) for trusted "
            "inputs, or set it to 0 to disable."
        ),
    )
    raise LockstepCompileError(
        [diagnostic],
        diagnostics=[diagnostic],
        phase="parse",
        source_file=source_file,
    )


def _max_expression_nesting(parse_tree: Any) -> int:
    if not hasattr(parse_tree, "getChildCount"):
        return 0

    max_depth = 0

    def _walk(node: Any, depth: int) -> None:
        nonlocal max_depth
        class_name = node.__class__.__name__
        depth_increase = 0
        if class_name == "PrimaryExprContext" and node.getChildCount() >= 3:
            if node.getChild(0).getText() == "(" and node.getChild(2).getText() == ")":
                depth_increase = 1
        elif class_name == "UnaryExprContext" and node.getChildCount() >= 2:
            if node.getChild(1).__class__.__name__ == "UnaryExprContext":
                depth_increase = 1
        current_depth = depth + depth_increase
        if current_depth > max_depth:
            max_depth = current_depth
        for index in range(node.getChildCount()):
            child = node.getChild(index)
            if hasattr(child, "getChildCount"):
                _walk(child, current_depth)

    _walk(parse_tree, 0)
    return max_depth


def _enforce_expression_nesting_limit(
    parse_tree: Any,
    *,
    source_file: str,
    limits: FrontendLimits,
) -> None:
    if limits.max_expression_nesting is None:
        return
    depth = _max_expression_nesting(parse_tree)
    if depth <= limits.max_expression_nesting:
        return
    diagnostic = LockstepDiagnostic(
        severity="error",
        code="LCK005",
        message=(
            "Expression nesting exceeds the configured limit "
            f"({depth} > {limits.max_expression_nesting})."
        ),
        line=1,
        column=0,
        source_file=source_file,
        hint=(
            "Reduce nested expressions, raise --max-expr-nesting "
            "(or max_expression_nesting via API), or set it to 0 to disable."
        ),
    )
    raise LockstepCompileError(
        [diagnostic],
        diagnostics=[diagnostic],
        phase="parse",
        source_file=source_file,
    )


def _merge_intrinsic_pure_functions_into_ast(program: AstProgram) -> AstProgram:
    existing_names = {pure.name for pure in program.pure_functions}
    intrinsic_decls = tuple(
        AstPureDecl(
            name=intrinsic.name,
            return_type=AstType(name=intrinsic.return_type, kind="primitive"),
            params=tuple(
                AstKernelParam(
                    modifier="value",
                    declared_type=AstType(name=param.type_name, kind="primitive"),
                    name=param.name,
                )
                for param in intrinsic.params
            ),
            intrinsic=True,
        )
        for intrinsic in load_intrinsics().values()
        if intrinsic.name not in existing_names
    )
    if not intrinsic_decls:
        return program
    return AstProgram(
        structs=program.structs,
        shaders=program.shaders,
        filters=program.filters,
        pure_functions=(*program.pure_functions, *intrinsic_decls),
        pipelines=program.pipelines,
    )


def _line_count(source: str) -> int:
    return source.count("\n") + 1


def _build_combined_source(
    source_code: str,
    *,
    source_file: str,
    library_sources: list[str] | None,
    library_source_files: list[str] | None,
) -> tuple[str, list[tuple[int, int, str]]]:
    libraries = library_sources or []
    all_sources = [*libraries, source_code]
    resolved_library_files = library_source_files or [
        f"<library:{idx + 1}>" for idx in range(len(libraries))
    ]
    if len(resolved_library_files) < len(libraries):
        resolved_library_files = [
            *resolved_library_files,
            *[
                f"<library:{idx + 1}>"
                for idx in range(len(resolved_library_files), len(libraries))
            ],
        ]
    all_source_files = [*resolved_library_files[: len(libraries)], source_file]
    mapping: list[tuple[int, int, str]] = []
    current_line = 1
    for index, part in enumerate(all_sources):
        part_line_count = _line_count(part)
        start_line = current_line
        end_line = start_line + part_line_count - 1
        mapping.append((start_line, end_line, all_source_files[index]))
        current_line = end_line + 1
        if index < len(all_sources) - 1:
            current_line += 1
    return "\n\n".join(all_sources), mapping


def _decode_dependency_path(path_literal: str) -> str:
    return bytes(path_literal, "utf-8").decode("unicode_escape")


def _dependency_parse_error(
    *,
    message: str,
    source_file: str,
    line: int = 1,
) -> LockstepCompileError:
    diagnostic = LockstepDiagnostic(
        severity="error",
        code="LCK002",
        message=message,
        line=line,
        column=0,
        source_file=source_file,
        hint="Fix dependency declarations before compilation can continue.",
    )
    raise LockstepCompileError(
        [diagnostic],
        diagnostics=[diagnostic],
        phase="parse",
        source_file=source_file,
    )


def _extract_dependency_references(source_code: str) -> list[tuple[str, int]]:
    dependencies: list[tuple[str, int]] = []
    for match in _DEPENDENCY_DECL_PATTERN.finditer(source_code):
        path_literal = match.group(1)
        line = source_code.count("\n", 0, match.start()) + 1
        dependencies.append((_decode_dependency_path(path_literal), line))
    return dependencies


def _resolve_dependency_sources(
    source_code: str,
    *,
    source_file: str,
) -> tuple[list[str], list[str]]:
    if source_file.startswith("<") and source_file.endswith(">"):
        base_directory = Path.cwd()
    else:
        base_directory = Path(source_file).resolve().parent

    visited: set[Path] = set()
    in_stack: list[Path] = []
    ordered_sources: list[str] = []
    ordered_source_files: list[str] = []

    def _resolve_reference(reference: str, parent_file: str) -> Path:
        candidate = Path(reference)
        if candidate.is_absolute():
            return candidate.resolve()
        if parent_file.startswith("<") and parent_file.endswith(">"):
            return (base_directory / candidate).resolve()
        return (Path(parent_file).resolve().parent / candidate).resolve()

    def _walk(current_source: str, current_file: str) -> None:
        for reference, line in _extract_dependency_references(current_source):
            dependency_path = _resolve_reference(reference, current_file)
            if dependency_path in in_stack:
                cycle_chain = [*in_stack, dependency_path]
                cycle_text = " -> ".join(str(path) for path in cycle_chain)
                _dependency_parse_error(
                    message=(
                        f"Circular dependency detected while resolving '{reference}': "
                        f"{cycle_text}"
                    ),
                    source_file=current_file,
                    line=line,
                )

            if dependency_path in visited:
                continue

            try:
                dependency_source = dependency_path.read_text(encoding="utf-8")
            except OSError as exc:
                _dependency_parse_error(
                    message=(
                        f"Unable to resolve dependency '{reference}' from "
                        f"'{current_file}': {exc}"
                    ),
                    source_file=current_file,
                    line=line,
                )

            in_stack.append(dependency_path)
            _walk(dependency_source, str(dependency_path))
            in_stack.pop()

            visited.add(dependency_path)
            ordered_sources.append(dependency_source)
            ordered_source_files.append(str(dependency_path))

    _walk(source_code, source_file)
    return ordered_sources, ordered_source_files


def _remap_diagnostic(
    diagnostic: LockstepDiagnostic,
    source_map: list[tuple[int, int, str]],
    default_source_file: str,
) -> LockstepDiagnostic:
    if diagnostic.source_file:
        return diagnostic
    for start_line, end_line, mapped_source in source_map:
        if start_line <= diagnostic.line <= end_line:
            return replace(
                diagnostic,
                source_file=mapped_source,
                line=(diagnostic.line - start_line) + 1,
            )
    return replace(diagnostic, source_file=default_source_file)


def _remap_diagnostics(
    diagnostics: list[LockstepDiagnostic],
    *,
    source_map: list[tuple[int, int, str]],
    default_source_file: str,
) -> list[LockstepDiagnostic]:
    return [
        _remap_diagnostic(diagnostic, source_map, default_source_file)
        for diagnostic in diagnostics
    ]


def _compile_lockstep_with_dependencies(
    source_code: str,
    *,
    verbose: bool = True,
    source_file: str = DEFAULT_SOURCE_FILE,
    source_map: list[tuple[int, int, str]] | None = None,
    lexer_cls,
    parser_cls,
    visitor_cls,
    semantic_validator=None,
    token_stream_cls=CommonTokenStream,
    debug_visitor_cls=None,
    frontend_limits: FrontendLimits | None = None,
) -> LockstepCompileResult:
    source_map = source_map or [(1, _line_count(source_code), source_file)]
    resolved_limits = _normalize_frontend_limits(frontend_limits, source_file=source_file)
    _enforce_source_size_limit(
        source_code,
        source_file=source_file,
        limits=resolved_limits,
    )

    input_stream = InputStream(source_code)
    lexer = lexer_cls(input_stream)
    error_listener = ParseErrorCollector(source_file=None)
    lexer.removeErrorListeners()
    lexer.addErrorListener(error_listener)
    if resolved_limits.parse_timeout_ms is not None:
        deadline = time.monotonic() + (resolved_limits.parse_timeout_ms / 1_000)

        def _build_token_stream(inner_lexer):
            return _DeadlineTokenStream(
                inner_lexer,
                deadline=deadline,
                source_file=source_file,
            )

        stream_builder = _build_token_stream
    else:
        stream_builder = token_stream_cls

    stream = stream_builder(lexer)

    parser = parser_cls(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(error_listener)
    try:
        tree = parser.program()
    except FrontendLimitExceeded as exc:
        diagnostic = _remap_diagnostic(
            exc.diagnostic,
            source_map,
            source_file,
        )
        raise LockstepCompileError(
            [diagnostic],
            diagnostics=[diagnostic],
            phase="parse",
            source_file=diagnostic.source_file,
        ) from exc

    _enforce_expression_nesting_limit(
        tree,
        source_file=source_file,
        limits=resolved_limits,
    )

    if error_listener.errors:
        parse_errors = _remap_diagnostics(
            error_listener.errors,
            source_map=source_map,
            default_source_file=source_file,
        )
        raise LockstepCompileError(
            parse_errors,
            diagnostics=parse_errors,
            source_file=parse_errors[0].source_file if parse_errors else source_file,
        )

    typed_ast = _merge_intrinsic_pure_functions_into_ast(
        build_program_ast(tree, visitor_cls)
    )

    semantic_validator = semantic_validator or (
        lambda parse_tree, *, typed_ast: validate_semantics(
            parse_tree, visitor_cls, typed_ast=typed_ast
        )
    )
    try:
        semantic_diagnostics = normalize_diagnostics(
            semantic_validator(tree, typed_ast=typed_ast)
        )
    except TypeError:
        semantic_diagnostics = normalize_diagnostics(semantic_validator(tree))
    semantic_diagnostics = _remap_diagnostics(
        semantic_diagnostics,
        source_map=source_map,
        default_source_file=source_file,
    )
    semantic_errors = [d for d in semantic_diagnostics if d.severity == "error"]
    if semantic_errors:
        raise LockstepCompileError(
            semantic_errors,
            diagnostics=semantic_diagnostics,
            phase="semantic",
            source_file=semantic_errors[0].source_file,
        )

    entities = ast_to_entities(typed_ast)

    all_diagnostics = normalize_diagnostics(semantic_diagnostics)
    bind_optimization = optimize_bind_routes(
        entities["bind_routes"],
        shader_names={shader["name"] for shader in entities["shaders"]},
        filter_names={flt["name"] for flt in entities["filters"]},
        bind_routes_ir=entities.get("bind_routes_ir"),
    )
    entities = {
        **entities,
        "optimized_bind_routes": bind_optimization["optimized_bind_routes"],
        "fused_bind_groups": bind_optimization["fused_groups"],
    }

    try:
        llvm_ir = emit_llvm_ir(typed_ast)
    except CodegenError as error:
        codegen_diagnostic = LockstepDiagnostic(
            severity="error",
            code="LCK501",
            message=str(error),
            line=1,
            column=0,
            source_file=source_file,
            hint="Fix semantic inconsistencies that prevent LLVM IR emission.",
        )
        raise LockstepCompileError(
            [codegen_diagnostic],
            diagnostics=normalize_diagnostics([*all_diagnostics, codegen_diagnostic]),
            phase="codegen",
            source_file=source_file,
        ) from error

    return LockstepCompileResult(
        parse_tree=tree,
        entities=entities,
        ast=typed_ast,
        llvm_ir=llvm_ir,
        c_header=emit_c_header(typed_ast),
        diagnostics=all_diagnostics,
    )


@functools.lru_cache(maxsize=1)
def load_default_parser_classes() -> tuple[Any, Any, Any]:
    """Load and cache the default generated parser classes.

    The first call imports generated parser modules; subsequent calls reuse the
    cached class tuple to avoid repeated import work.
    """
    from generated.parser.LockstepLexer import LockstepLexer
    from generated.parser.LockstepParser import LockstepParser
    from generated.parser.LockstepVisitor import LockstepVisitor

    return LockstepLexer, LockstepParser, LockstepVisitor


def validate_semantics(parse_tree: Any, visitor_cls=None, *, typed_ast=None):
    if visitor_cls is None:
        _, _, visitor_cls = load_default_parser_classes()
    return _validate_semantics(parse_tree, visitor_cls, typed_ast=typed_ast)


def compile_lockstep(
    source_code: str,
    *,
    verbose: bool = True,
    source_file: str = DEFAULT_SOURCE_FILE,
    library_sources: list[str] | None = None,
    library_source_files: list[str] | None = None,
    lexer_cls=None,
    parser_cls=None,
    visitor_cls=None,
    semantic_validator=None,
    token_stream_cls=CommonTokenStream,
    debug_visitor_cls=None,
    frontend_limits: FrontendLimits | None = None,
) -> LockstepCompileResult:
    resolved_library_sources: list[str] = list(library_sources or [])
    resolved_library_source_files: list[str] = list(library_source_files or [])

    dependency_sources, dependency_source_files = _resolve_dependency_sources(
        source_code,
        source_file=source_file,
    )
    resolved_library_sources.extend(dependency_sources)
    resolved_library_source_files.extend(dependency_source_files)

    source_code, source_map = _build_combined_source(
        source_code,
        source_file=source_file,
        library_sources=resolved_library_sources,
        library_source_files=resolved_library_source_files,
    )

    if lexer_cls is None or parser_cls is None or visitor_cls is None:
        default_lexer_cls, default_parser_cls, default_visitor_cls = (
            _load_default_parser_classes()
        )
        lexer_cls = lexer_cls or default_lexer_cls
        parser_cls = parser_cls or default_parser_cls
        visitor_cls = visitor_cls or default_visitor_cls

    return _compile_lockstep_with_dependencies(
        source_code,
        verbose=verbose,
        source_file=source_file,
        source_map=source_map,
        lexer_cls=lexer_cls,
        parser_cls=parser_cls,
        visitor_cls=visitor_cls,
        semantic_validator=semantic_validator,
        token_stream_cls=token_stream_cls,
        debug_visitor_cls=debug_visitor_cls,
        frontend_limits=frontend_limits,
    )


# Backward-compatible alias for existing internal imports.
_load_default_parser_classes = load_default_parser_classes
