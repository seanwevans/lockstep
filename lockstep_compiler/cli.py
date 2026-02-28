import argparse
import json
import sys
from pathlib import Path

from .errors import LockstepCompileError


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
        "--dump",
        action="store_true",
        help="Print compiled entities (pipeline topology and bounds) as JSON.",
    )
    return parser


def run_cli(argv=None, *, stdin=None, stdout=None, stderr=None, compiler=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
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

    if compiler is None:
        print(
            "Compiler configuration error: no compiler callable was provided.",
            file=stderr,
        )
        return 1

    if not callable(compiler):
        print(
            "Compiler configuration error: compiler must be callable.",
            file=stderr,
        )
        return 1

    try:
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
        return 1
    except Exception:
        print("Compilation failed due to an internal error.", file=stderr)
        return 1

    if args.dump:
        entities = getattr(result, "entities", result)
        print(json.dumps(entities, indent=2, sort_keys=True, default=str), file=stdout)

    return 0
