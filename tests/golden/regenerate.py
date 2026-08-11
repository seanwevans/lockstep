#!/usr/bin/env python3
"""Regenerate every golden IR/header snapshot under tests/golden/.

Run this after an intentional codegen change and review the diff before
committing:

    python tests/golden/regenerate.py

This is equivalent to ``LOCKSTEP_UPDATE_GOLDEN=1 pytest tests/test_golden_ir.py``
but does not require pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the exact snapshotting logic the tests use so the two never diverge.
sys.path.insert(0, str(REPO_ROOT / "tests"))
from test_golden_ir import regenerate  # noqa: E402


def main() -> int:
    written = regenerate()
    for path in written:
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    print(f"\n{len(written)} golden artifact(s) regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
