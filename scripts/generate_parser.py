#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
import urllib.request
from pathlib import Path

ANTLR_VERSION = "4.13.2"
ANTLR_JAR = Path("tools") / f"antlr-{ANTLR_VERSION}-complete.jar"
ANTLR_JAR_SHA256 = "eae2dfa119a64327444672aff63e9ec35a20180dc5b8090b7a6ab85125df4d76"
GRAMMAR_FILE = Path("Lockstep.g4")
OUTPUT_DIR = Path("generated") / "parser"


def antlr_mismatch_message(actual_sha256: str) -> str:
    return (
        f"Integrity check failed for {ANTLR_JAR}.\n"
        f"Expected SHA-256: {ANTLR_JAR_SHA256}\n"
        f"Actual SHA-256:   {actual_sha256}\n"
        "Delete the file and rerun this script to re-download the ANTLR jar.\n"
        f"  rm -f {ANTLR_JAR}"
    )


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_antlr_jar(path: Path) -> None:
    actual_sha256 = compute_sha256(path)
    if actual_sha256 != ANTLR_JAR_SHA256:
        raise RuntimeError(antlr_mismatch_message(actual_sha256))


def ensure_antlr_jar() -> None:
    if ANTLR_JAR.exists():
        verify_antlr_jar(ANTLR_JAR)
        return

    ANTLR_JAR.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.antlr.org/download/antlr-{ANTLR_VERSION}-complete.jar"
    print(f"Downloading ANTLR {ANTLR_VERSION} from {url}...")
    urllib.request.urlretrieve(url, ANTLR_JAR)
    verify_antlr_jar(ANTLR_JAR)


def generate_parser() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "java",
        "-jar",
        str(ANTLR_JAR),
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
        "--verify-toolchain",
        action="store_true",
        help="Only verify ANTLR jar integrity and exit.",
    )
    args = parser.parse_args()

    ensure_antlr_jar()
    if args.verify_toolchain:
        print(f"ANTLR toolchain verified: {ANTLR_JAR}")
        return 0

    generate_parser()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
