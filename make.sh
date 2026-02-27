#!/usr/bin/env bash
set -euo pipefail

echo "make.sh is a thin wrapper. Prefer: make generate-parser"
make generate-parser
