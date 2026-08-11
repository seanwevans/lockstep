"""Golden-IR snapshot tests for the code generator.

``codegen.py`` is the largest module in the compiler and, before these tests,
its output was only checked by a handful of ``grep``-style assertions on IR
substrings. That leaves most of the emitted LLVM IR and the generated C header
unguarded: a refactor can silently change the ABI (arena layout, offset macros)
or the lowering of a whole class of programs and no test would notice.

These tests pin the *exact* emitted IR and header for a small, curated corpus
that exercises the main lowering paths:

* ``minimal``              — a plain shader over a struct stream with a uniform;
* ``physics_accumulator``  — a single accumulating shader plus a ``fold`` reduce;
* ``filter_aggregation``   — a filter (compacting store) feeding an accumulator;
* ``fused_accumulator``    — a pure multi-shader accumulator pipeline that the
                             backend fuses into one vector loop.

Each program's IR and header are snapshotted under ``tests/golden/``. Any codegen
change that alters the output makes the corresponding test fail with a unified
diff, so the change is reviewed deliberately rather than slipping through.

The generated IR is deterministic and target-pinned (the module triple defaults
to ``x86_64-unknown-linux-gnu`` with an empty data layout), so the snapshots are
portable across the CI matrix.

## Regenerating the snapshots

When a codegen change is intentional, regenerate the golden files and review the
diff before committing:

    LOCKSTEP_UPDATE_GOLDEN=1 pytest tests/test_golden_ir.py

or run the helper directly:

    python tests/golden/regenerate.py
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path

import pytest

from lockstep_compiler.compiler import compile_lockstep

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
PROGRAM_DIR = GOLDEN_DIR / "programs"

PROGRAMS = sorted(p.stem for p in PROGRAM_DIR.glob("*.lock"))

# A single knob regenerates every snapshot; the helper script sets it too.
_UPDATE = os.environ.get("LOCKSTEP_UPDATE_GOLDEN") == "1"


def _artifacts(name: str) -> dict[str, str]:
    source = (PROGRAM_DIR / f"{name}.lock").read_text(encoding="utf-8")
    result = compile_lockstep(source, verbose=False)
    return {
        "ll": result.llvm_ir or "",
        "h": result.c_header or "",
    }


def _golden_path(name: str, ext: str) -> Path:
    return GOLDEN_DIR / f"{name}.{ext}"


def regenerate() -> list[Path]:
    """(Re)write every golden artifact. Returns the paths written."""
    written: list[Path] = []
    for name in PROGRAMS:
        for ext, text in _artifacts(name).items():
            path = _golden_path(name, ext)
            path.write_text(text, encoding="utf-8")
            written.append(path)
    return written


def _check(name: str, ext: str, label: str) -> None:
    actual = _artifacts(name)[ext]
    path = _golden_path(name, ext)

    if _UPDATE:
        path.write_text(actual, encoding="utf-8")
        pytest.skip(f"regenerated {path.name}")

    assert path.exists(), (
        f"missing golden {path.name}; regenerate with "
        f"LOCKSTEP_UPDATE_GOLDEN=1 pytest {Path(__file__).name}"
    )
    expected = path.read_text(encoding="utf-8")
    if actual != expected:
        diff = "\n".join(
            difflib.unified_diff(
                expected.splitlines(),
                actual.splitlines(),
                fromfile=f"golden/{path.name}",
                tofile=f"emitted/{path.name}",
                lineterm="",
            )
        )
        pytest.fail(
            f"{label} for '{name}' drifted from its golden snapshot.\n"
            f"If this change is intended, regenerate with "
            f"LOCKSTEP_UPDATE_GOLDEN=1 pytest {Path(__file__).name}\n\n{diff}"
        )


@pytest.mark.parametrize("name", PROGRAMS)
def test_golden_llvm_ir(name: str) -> None:
    _check(name, "ll", "LLVM IR")


@pytest.mark.parametrize("name", PROGRAMS)
def test_golden_c_header(name: str) -> None:
    _check(name, "h", "C header")


def test_corpus_is_present() -> None:
    """Guard against an empty corpus silently passing every parametrized test."""
    assert PROGRAMS, "no golden programs found under tests/golden/programs/"
    assert "fused_accumulator" in PROGRAMS
