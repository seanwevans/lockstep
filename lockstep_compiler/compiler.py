from typing import Any

from antlr4 import CommonTokenStream, InputStream

from .errors import LockstepCompileError, ParseErrorCollector
from .models import LockstepCompileResult, normalize_diagnostics
from .visitors import build_debug_visitor, validate_semantics


def compile_lockstep(
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
        },
        diagnostics=all_diagnostics,
    )
