#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-help}"

generate() {
  antlr4 -Dlanguage=Python3 -visitor Lockstep.g4
}

run_sample() {
  python3 debug_compiler.py
}

clean() {
  rm -f \
    LockstepLexer.py LockstepParser.py LockstepVisitor.py LockstepListener.py \
    LockstepLexer.tokens Lockstep.tokens \
    LockstepLexer.interp Lockstep.interp
}

help() {
  cat <<'USAGE'
Usage: ./make.sh <target>

Targets:
  generate    Regenerate parser artifacts from Lockstep.g4
  run-sample  Run debug_compiler.py with the built-in sample source
  clean       Remove generated ANTLR artifacts
USAGE
}

case "$TARGET" in
  generate)
    generate
    ;;
  run-sample)
    run_sample
    ;;
  clean)
    clean
    ;;
  help|-h|--help)
    help
    ;;
  *)
    echo "Unknown target: $TARGET" >&2
    help
    exit 1
    ;;
esac
