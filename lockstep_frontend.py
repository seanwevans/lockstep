import argparse
import sys
from pathlib import Path
from typing import Any

from antlr4 import CommonTokenStream, InputStream

PARSER_DIR = Path(__file__).parent / "generated" / "parser"
if str(PARSER_DIR) not in sys.path:
    sys.path.insert(0, str(PARSER_DIR))

from LockstepLexer import LockstepLexer
from LockstepParser import LockstepParser

from lockstep_compiler import (
    LockstepCompileError,
    LockstepCompileResult,
    LockstepDebugVisitor,
    LockstepDiagnostic,
    LockstepSemanticValidator,
    ParseErrorCollector,
    normalize_diagnostics,
)


def validate_semantics(parse_tree: Any) -> list[LockstepDiagnostic]:
    """Validate semantic constraints after syntactic parsing succeeds."""

    validator = LockstepSemanticValidator()
    return validator.validate(parse_tree)


def compile_lockstep(source_code: str, verbose: bool = True) -> LockstepCompileResult:
    input_stream = InputStream(source_code)
    lexer = LockstepLexer(input_stream)
    error_listener = ParseErrorCollector()
    lexer.removeErrorListeners()
    lexer.addErrorListener(error_listener)
    stream = CommonTokenStream(lexer)

    parser = LockstepParser(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(error_listener)
    tree = parser.program()

    if error_listener.errors:
        raise LockstepCompileError(error_listener.errors, diagnostics=error_listener.errors)

    semantic_diagnostics = normalize_diagnostics(validate_semantics(tree))
    semantic_errors = [
        diagnostic for diagnostic in semantic_diagnostics if diagnostic.severity == "error"
    ]
    if semantic_errors:
        raise LockstepCompileError(
            semantic_errors,
            diagnostics=semantic_diagnostics,
            phase="semantic",
        )

    visitor = LockstepDebugVisitor(verbose=verbose)
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


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Lockstep compiler frontend for parsing and semantic validation."
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Optional path to a Lockstep source file. Reads from stdin when omitted.",
    )
    return parser


def run_cli(argv=None, *, stdin=None, stderr=None, compiler=compile_lockstep):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    stdin = sys.stdin if stdin is None else stdin
    stderr = sys.stderr if stderr is None else stderr

    if args.path:
        source_path = Path(args.path)
        try:
            source = source_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"Unable to read '{source_path}': file not found.", file=stderr)
            return 1
        except PermissionError as err:
            reason = err.strerror or "permission denied"
            print(f"Unable to read '{source_path}': {reason}.", file=stderr)
            return 1
        except UnicodeDecodeError as err:
            print(
                f"Unable to read '{source_path}': invalid UTF-8 ({err.reason}).",
                file=stderr,
            )
            return 1
    else:
        source = stdin.read()

    try:
        compiler(source)
    except LockstepCompileError as err:
        count = len(err.errors)
        suffix = "" if count == 1 else "s"
        print(
            f"Compilation failed with {count} {err.phase} error{suffix}.",
            file=stderr,
        )
        for error in err.errors:
            print(f"line {error.line}:{error.column} {error.message}", file=stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
