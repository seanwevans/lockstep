import functools
from typing import Any

from antlr4 import CommonTokenStream, InputStream

from .errors import LockstepCompileError, ParseErrorCollector
from .models import LockstepCompileResult, normalize_diagnostics
from .optimizer import optimize_bind_routes
from .visitors import build_debug_visitor, validate_semantics as _validate_semantics


def _compile_lockstep_with_dependencies(
    source_code: str,
    *,
    verbose: bool = True,
    lexer_cls,
    parser_cls,
    visitor_cls,
    semantic_validator=None,
    token_stream_cls=CommonTokenStream,
    debug_visitor_cls=None,
) -> LockstepCompileResult:
    input_stream = InputStream(source_code)
    lexer = lexer_cls(input_stream)
    error_listener = ParseErrorCollector()
    lexer.removeErrorListeners()
    lexer.addErrorListener(error_listener)
    stream = token_stream_cls(lexer)

    parser = parser_cls(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(error_listener)
    tree = parser.program()

    if error_listener.errors:
        raise LockstepCompileError(error_listener.errors, diagnostics=error_listener.errors)

    semantic_validator = semantic_validator or (lambda parse_tree: validate_semantics(parse_tree, visitor_cls))
    semantic_diagnostics = normalize_diagnostics(semantic_validator(tree))
    semantic_errors = [d for d in semantic_diagnostics if d.severity == "error"]
    if semantic_errors:
        raise LockstepCompileError(
            semantic_errors,
            diagnostics=semantic_diagnostics,
            phase="semantic",
        )

    debug_visitor_cls = debug_visitor_cls or build_debug_visitor(visitor_cls)
    visitor = debug_visitor_cls(verbose=verbose)
    visitor.visit(tree)
    all_diagnostics = normalize_diagnostics([*semantic_diagnostics, *visitor.diagnostics])
    bind_optimization = optimize_bind_routes(
        visitor.bind_routes,
        shader_names={shader["name"] for shader in visitor.shaders},
        filter_names={flt["name"] for flt in visitor.filters},
    )

    return LockstepCompileResult(
        parse_tree=tree,
        entities={
            "structs": visitor.structs,
            "shaders": visitor.shaders,
            "filters": visitor.filters,
            "pure_functions": visitor.pure_functions,
            "streams": visitor.streams,
            "accumulators": visitor.accumulators,
            "uniforms": visitor.uniforms,
            "bind_routes": visitor.bind_routes,
            "optimized_bind_routes": bind_optimization["optimized_bind_routes"],
            "fused_bind_groups": bind_optimization["fused_groups"],
        },
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


def validate_semantics(parse_tree: Any, visitor_cls=None):
    if visitor_cls is None:
        _, _, visitor_cls = load_default_parser_classes()
    return _validate_semantics(parse_tree, visitor_cls)


def compile_lockstep(
    source_code: str,
    *,
    verbose: bool = True,
    lexer_cls=None,
    parser_cls=None,
    visitor_cls=None,
    semantic_validator=None,
    token_stream_cls=CommonTokenStream,
    debug_visitor_cls=None,
) -> LockstepCompileResult:
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
        lexer_cls=lexer_cls,
        parser_cls=parser_cls,
        visitor_cls=visitor_cls,
        semantic_validator=semantic_validator,
        token_stream_cls=token_stream_cls,
        debug_visitor_cls=debug_visitor_cls,
    )


# Backward-compatible alias for existing internal imports.
_load_default_parser_classes = load_default_parser_classes
