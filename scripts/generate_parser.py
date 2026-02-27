#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import urllib.request
from pathlib import Path

ANTLR_VERSION = "4.13.2"
ANTLR_JAR = Path("tools") / f"antlr-{ANTLR_VERSION}-complete.jar"
GRAMMAR_FILE = Path("Lockstep.g4")
OUTPUT_DIR = Path("generated") / "parser"


def ensure_antlr_jar() -> None:
    if ANTLR_JAR.exists():
        return
    ANTLR_JAR.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.antlr.org/download/antlr-{ANTLR_VERSION}-complete.jar"
    print(f"Downloading ANTLR {ANTLR_VERSION} from {url}...")
    urllib.request.urlretrieve(url, ANTLR_JAR)


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
    parser.parse_args()
    ensure_antlr_jar()
    generate_parser()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
