import argparse
import inspect
import json
import sys
import traceback
from pathlib import Path

from .errors import LockstepCompileError
from .simulator import parse_simulation_inputs, simulate_pipeline_entities
from .formatter import format_lockstep_source


def _read_source_file(path: Path, *, stderr) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Unable to read '{path}': file not found.", file=stderr)
    except PermissionError as err:
        reason = err.strerror or "permission denied"
        print(f"Unable to read '{path}': {reason}.", file=stderr)
    except UnicodeDecodeError as err:
        print(
            f"Unable to read '{path}': invalid UTF-8 ({err.reason}).",
            file=stderr,
        )
    return None


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Debug parser for Lockstep source files."
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Optional path to a Lockstep source file. Reads from stdin when omitted.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the Lockstep compiler version and exit.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dump",
        action="store_true",
        help="Print compiled entities (pipeline topology and bounds) as JSON.",
    )
    mode_group.add_argument(
        "--format",
        action="store_true",
        help="Format Lockstep source code into canonical straight-line style.",
    )
    mode_group.add_argument(
        "--emit-ir",
        action="store_true",
        help="Emit generated LLVM IR to stdout.",
    )
    mode_group.add_argument(
        "--emit-header",
        action="store_true",
        help="Emit generated C host header to stdout.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed exception diagnostics and traceback on failures.",
    )
    parser.add_argument(
        "--lib",
        action="append",
        default=[],
        metavar="PATH",
        help="Path to a Lockstep source library file containing shared structs/pure functions.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate bind-route execution on small in-memory datasets.",
    )
    parser.add_argument(
        "--simulate-input",
        help="Path to a JSON file containing simulation inputs with `streams` and optional `accumulators` maps.",
    )
    return parser


def run_cli(argv=None, *, stdin=None, stdout=None, stderr=None, compiler=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr

    if args.simulate_input and not args.simulate:
        print(
            "usage error: --simulate-input requires --simulate.",
            file=stderr,
        )
        return 2

    if args.version:
        from importlib.metadata import version as _pkg_version, PackageNotFoundError
        try:
            ver = _pkg_version("Lockstep")
        except PackageNotFoundError:
            ver = "0.1.0"
        print(f"lockstepc {ver}", file=stdout)
        return 0

    if args.path:
        source_path = Path(args.path)
        source = _read_source_file(source_path, stderr=stderr)
        if source is None:
            return 1
    else:
        source = stdin.read()

    library_sources = []
    for library_path in args.lib:
        library_source = _read_source_file(Path(library_path), stderr=stderr)
        if library_source is None:
            return 1
        library_sources.append(library_source)

    if args.format:
        print(format_lockstep_source(source), end="", file=stdout)
        return 0

    if compiler is None:
        from .compiler import compile_lockstep

        compiler = compile_lockstep

    if not callable(compiler):
        print(
            "Compiler configuration error: compiler must be callable.",
            file=stderr,
        )
        return 1

    supports_verbose = False
    supports_library_sources = False
    try:
        signature = inspect.signature(compiler)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        supports_verbose = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            or parameter.name == "verbose"
            for parameter in signature.parameters.values()
        )
        supports_library_sources = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            or parameter.name == "library_sources"
            for parameter in signature.parameters.values()
        )

    compile_kwargs = {}
    if supports_verbose:
        compile_kwargs["verbose"] = False
    if supports_library_sources:
        compile_kwargs["library_sources"] = library_sources

    try:
        if compile_kwargs:
            result = compiler(source, **compile_kwargs)
        else:
            result = compiler(source)
    except LockstepCompileError as err:
        count = len(err.errors)
        suffix = "" if count == 1 else "s"
        print(
            f"Compilation failed with {count} {err.phase} error{suffix}.",
            file=stderr,
        )
        for error in err.errors:
            print(f"line {error.line}:{error.column} {error.message}", file=stderr)
        if args.debug:
            if err.diagnostics:
                print("diagnostics:", file=stderr)
                print(
                    json.dumps(
                        [
                            {
                                "severity": diagnostic.severity,
                                "code": diagnostic.code,
                                "message": diagnostic.message,
                                "line": diagnostic.line,
                                "column": diagnostic.column,
                                "hint": diagnostic.hint,
                            }
                            for diagnostic in err.diagnostics
                        ],
                        indent=2,
                        sort_keys=True,
                    ),
                    file=stderr,
                )
            print(f"{type(err).__name__}: {err}", file=stderr)
            traceback.print_exc(file=stderr)
        return 1
    except Exception as err:
        print("Compilation failed due to an internal error.", file=stderr)
        if args.debug:
            print(f"{type(err).__name__}: {err}", file=stderr)
            traceback.print_exc(file=stderr)
        return 1

    if args.emit_ir:
        llvm_ir = getattr(result, "llvm_ir", "")
        print(llvm_ir, file=stdout)
        return 0

    if args.emit_header:
        c_header = getattr(result, "c_header", "")
        print(c_header, file=stdout)
        return 0

    if args.dump:
        entities = getattr(result, "entities", result)
        print(json.dumps(entities, indent=2, sort_keys=True, default=str), file=stdout)

    if args.simulate:
        input_payload = ""
        if args.simulate_input:
            input_path = Path(args.simulate_input)
            try:
                input_payload = input_path.read_text(encoding="utf-8")
            except OSError as err:
                print(f"Unable to read simulation input '{input_path}': {err}", file=stderr)
                return 1
        try:
            stream_inputs, accumulator_inputs = parse_simulation_inputs(input_payload)
        except json.JSONDecodeError as err:
            print(f"Unable to parse simulation input JSON: {err}", file=stderr)
            return 1
        except ValueError as err:
            print(f"Invalid simulation input: {err}", file=stderr)
            return 1

        entities = getattr(result, "entities", result)
        simulation = simulate_pipeline_entities(
            entities,
            stream_inputs=stream_inputs,
            accumulator_inputs=accumulator_inputs,
        )
        print(json.dumps(simulation, indent=2, sort_keys=True, default=str), file=stdout)

    return 0


def main(argv=None):
    from .compiler import compile_lockstep

    sys.exit(run_cli(argv, compiler=compile_lockstep))
