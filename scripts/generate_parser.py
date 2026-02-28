#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
import urllib.request
from pathlib import Path

ANTLR_VERSION = "4.13.2"
ANTLR_SHA256 = "eae2dfa119a64327444672aff63e9ec35a20180dc5b8090b7a6ab85125df4d76"
DEFAULT_ANTLR_JAR = Path("tools") / f"antlr-{ANTLR_VERSION}-complete.jar"
GRAMMAR_FILE = Path("Lockstep.g4")
OUTPUT_DIR = Path("generated") / "parser"


class AntlrJarError(RuntimeError):
    """Raised when the ANTLR JAR is missing or fails integrity checks."""


def verify_antlr_jar(jar_path: Path) -> None:
    sha256 = hashlib.sha256()
    with jar_path.open("rb") as jar_file:
        while chunk := jar_file.read(1024 * 1024):
            sha256.update(chunk)

    actual = sha256.hexdigest()
    if actual != ANTLR_SHA256:
        raise AntlrJarError(
            "ANTLR JAR checksum mismatch for "
            f"{jar_path}: expected {ANTLR_SHA256}, got {actual}. "
            "Re-download the jar from https://www.antlr.org/download/ "
            "or provide a trusted jar via --jar-path."
        )


def ensure_antlr_jar(jar_path: Path, offline: bool) -> None:
    if jar_path.exists():
        verify_antlr_jar(jar_path)
        return

    if offline:
        raise AntlrJarError(
            f"ANTLR JAR not found at {jar_path} and --offline was set. "
            "Provide the jar with --jar-path or disable --offline to allow download."
        )

    jar_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.antlr.org/download/antlr-{ANTLR_VERSION}-complete.jar"
    print(f"Downloading ANTLR {ANTLR_VERSION} from {url} to {jar_path}...")
    urllib.request.urlretrieve(url, jar_path)
    verify_antlr_jar(jar_path)


def generate_parser(jar_path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "java",
        "-jar",
        str(jar_path),
        "-Dlanguage=Python3",
        "-visitor",
        "-o",
        str(OUTPUT_DIR),
        str(GRAMMAR_FILE),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate ANTLR parser files.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not download ANTLR jar; fail if it is missing.",
    )
    parser.add_argument(
        "--jar-path",
        type=Path,
        default=DEFAULT_ANTLR_JAR,
        help=f"Path to ANTLR {ANTLR_VERSION} complete jar (default: %(default)s).",
    )
    args = parser.parse_args()

    try:
        ensure_antlr_jar(args.jar_path, args.offline)
        generate_parser(args.jar_path)
    except AntlrJarError as exc:
        parser.error(str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
