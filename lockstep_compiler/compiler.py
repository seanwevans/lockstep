import functools
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import replace
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
_DEPENDENCY_DECL_PATTERN = re.compile(
    r'^\s*(?:import|#include)\s+"((?:[^"\\]|\\.)+)"\s*;\s*$',
    re.MULTILINE,
)


@dataclass(frozen=True)
class ParserResourceLimits:
    """Configurable limits for parser resource consumption."""

    max_file_size_bytes: int | None = 1024 * 1024
    max_expression_nesting_depth: int | None = 256
    parse_timeout_seconds: float | None = 2.0


class ParserLimitError(RuntimeError):
    pass


def _parser_limit_diagnostic(
    *,
    code: str,
    message: str,
    source_file: str,
    hint: str,
) -> LockstepDiagnostic:
    return LockstepDiagnostic(
        severity="error",
        code=code,
        message=message,
        line=1,
        column=0,
        source_file=source_file,
        hint=hint,
    )


def _validate_parser_limits(limits: ParserResourceLimits):
    if (
        limits.max_file_size_bytes is not None
        and limits.max_file_size_bytes <= 0
    ):
        raise ValueError("max_file_size_bytes must be positive when provided.")
    if (
        limits.max_expression_nesting_depth is not None
        and limits.max_expression_nesting_depth <= 0
    ):
        raise ValueError(
            "max_expression_nesting_depth must be positive when provided."
        )
    if limits.parse_timeout_seconds is not None and limits.parse_timeout_seconds <= 0:
        raise ValueError("parse_timeout_seconds must be positive when provided.")


@contextmanager
def _parse_timeout(timeout_seconds: float | None):
    if timeout_seconds is None:
        yield
        return
    if not hasattr(signal, "setitimer"):
        yield
        return

    def _handle_timeout(_signum, _frame):
        raise ParserLimitError(f"Parse timeout exceeded ({timeout_seconds:.3f}s).")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _expr_nesting_depth(expr: Any) -> int:
    from .ast import AstExprBinary, AstExprCall, AstExprCast, AstExprUnary

    if isinstance(expr, AstExprBinary):
        return 1 + max(_expr_nesting_depth(expr.left), _expr_nesting_depth(expr.right))
    if isinstance(expr, AstExprUnary):
        return 1 + _expr_nesting_depth(expr.operand)
    if isinstance(expr, AstExprCall):
        if not expr.args:
            return 1
        return 1 + max(_expr_nesting_depth(arg) for arg in expr.args)
    if isinstance(expr, AstExprCast):
        return 1 + _expr_nesting_depth(expr.value)
    return 1


def _statement_expr_depth(stmt: Any) -> int:
    from .ast import AstAssignStmt, AstReturnStmt, AstVarDeclStmt

    if isinstance(stmt, AstAssignStmt):
        return _expr_nesting_depth(stmt.value)
    if isinstance(stmt, AstReturnStmt):
        return _expr_nesting_depth(stmt.value)
    if isinstance(stmt, AstVarDeclStmt) and stmt.initializer is not None:
        return _expr_nesting_depth(stmt.initializer)
    return 0


def _typed_ast_max_expr_nesting(typed_ast: Any) -> int:
    if typed_ast is None:
        return 0
    max_depth = 0
    for pure_decl in getattr(typed_ast, "pure_functions", ()):
        for stmt in pure_decl.body:
            max_depth = max(max_depth, _statement_expr_depth(stmt))
    for kernel_decl in (*getattr(typed_ast, "shaders", ()), *getattr(typed_ast, "filters", ())):
        for stmt in kernel_decl.body:
            max_depth = max(max_depth, _statement_expr_depth(stmt))
    return max_depth


def _intrinsic_to_entity(intrinsic: IntrinsicSignature) -> PureFunctionEntity:
    return PureFunctionEntity(
        name=intrinsic.name,
        return_type=intrinsic.return_type,
        params=tuple(
            PureFunctionParamEntity(type=param.type_name, name=param.name)
            for param in intrinsic.params
        ),
        body=(),
        intrinsic=True,
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
    parser_limits: ParserResourceLimits | None = None,
) -> LockstepCompileResult:
    source_map = source_map or [(1, _line_count(source_code), source_file)]
    parser_limits = parser_limits or ParserResourceLimits()
    _validate_parser_limits(parser_limits)

    max_file_size_bytes = parser_limits.max_file_size_bytes
    if max_file_size_bytes is not None and len(source_code.encode("utf-8")) > max_file_size_bytes:
        file_size_diagnostic = _parser_limit_diagnostic(
            code="LCK002",
            message=(
                "Source file exceeds parser file size limit "
                f"({max_file_size_bytes} bytes)."
            ),
            source_file=source_file,
            hint="Increase --max-file-size-bytes or reduce input size.",
        )
        raise LockstepCompileError(
            [file_size_diagnostic],
            diagnostics=[file_size_diagnostic],
            source_file=source_file,
        )

    input_stream = InputStream(source_code)
    lexer = lexer_cls(input_stream)
    error_listener = ParseErrorCollector(source_file=None)
    lexer.removeErrorListeners()
    lexer.addErrorListener(error_listener)
    stream = token_stream_cls(lexer)

    parser = parser_cls(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(error_listener)
    try:
        with _parse_timeout(parser_limits.parse_timeout_seconds):
            tree = parser.program()
    except ParserLimitError as error:
        timeout_diagnostic = _parser_limit_diagnostic(
            code="LCK004",
            message=str(error),
            source_file=source_file,
            hint="Increase --parse-timeout-seconds or simplify pathological input.",
        )
        raise LockstepCompileError(
            [timeout_diagnostic],
            diagnostics=[timeout_diagnostic],
            source_file=source_file,
        ) from error

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

    if (
        parser_limits.max_expression_nesting_depth is not None
        and typed_ast is not None
    ):
        max_depth = _typed_ast_max_expr_nesting(typed_ast)
        if max_depth > parser_limits.max_expression_nesting_depth:
            depth_diagnostic = _parser_limit_diagnostic(
                code="LCK003",
                message=(
                    "Expression nesting depth exceeds parser limit "
                    f"({max_depth} > {parser_limits.max_expression_nesting_depth})."
                ),
                source_file=source_file,
                hint="Increase --max-expression-nesting-depth or simplify nested expressions.",
            )
            raise LockstepCompileError(
                [depth_diagnostic],
                diagnostics=[depth_diagnostic],
                source_file=source_file,
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
    parser_limits: ParserResourceLimits | None = None,
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
        parser_limits=parser_limits,
    )


# Backward-compatible alias for existing internal imports.
_load_default_parser_classes = load_default_parser_classes
