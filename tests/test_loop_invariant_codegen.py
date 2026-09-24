"""Per-stage row loops stay analyzable by LLVM (see the alias-analysis probe in
``benchmarks/RESULTS.md``).

Two codegen details decide whether clang can vectorize a per-stage route:

* a ``uniform`` argument is loaded once before the row loop, not on every row
  -- LLVM can't prove the row stores miss the uniform's arena slot, so a load
  left in the loop blocks LICM and then the vectorizer;
* row indices are clamped with a scalar ``smax``/``smin`` that scalar evolution
  understands, not a ``<4 x i32>`` splat/select/extract idiom it can't see
  through ("cannot identify array bounds").
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from lockstep_compiler.compiler import compile_lockstep

REPO_ROOT = Path(__file__).resolve().parents[1]

BRIGHTEN = """
struct Pixel { float r; float g; float b; };
shader Brighten(in Pixel src, out Pixel dst, uniform float gain) {
    dst.r = src.r * gain;
    dst.g = src.g * gain;
    dst.b = src.b * gain;
}
pipeline Image {
    stream<Pixel, 4096> pixelsIn;
    stream<Pixel, 4096> pixelsOut;
    uniform float gain = 1.5;
    bind { pixelsOut = Brighten(pixelsIn, pixelsOut, gain); }
}
"""


def _tick(ir: str) -> str:
    start = ir.index('define void @"Lockstep_Tick"')
    return ir[start : ir.index("\n}\n", start)]


def test_uniform_is_loaded_before_the_row_loop() -> None:
    lines = _tick(compile_lockstep(BRIGHTEN, verbose=False).llvm_ir).splitlines()
    loop_start = lines.index("route_Brighten_cond:")
    uniform_loads = [
        index
        for index, line in enumerate(lines)
        if '%"uniform_gain_val' in line and "= load" in line
    ]
    assert len(uniform_loads) == 1
    assert uniform_loads[0] < loop_start


def test_row_index_clamp_is_scalar() -> None:
    tick = _tick(compile_lockstep(BRIGHTEN, verbose=False).llvm_ir)
    assert "route_clamp" in tick
    assert "<4 x i32>" not in tick


@pytest.mark.skipif(shutil.which("clang") is None, reason="clang not on PATH")
def test_clang_vectorizes_the_per_stage_loop() -> None:
    ir = compile_lockstep(BRIGHTEN, verbose=False).llvm_ir
    with tempfile.TemporaryDirectory(prefix="lsloop_") as tmp:
        ir_path = Path(tmp) / "brighten.ll"
        ir_path.write_text(ir, encoding="utf-8")
        proc = subprocess.run(
            [
                shutil.which("clang") or "clang",
                "-O3",
                "-Wno-override-module",
                "-c",
                str(ir_path),
                "-o",
                str(Path(tmp) / "brighten.o"),
                "-Rpass=loop-vectorize",
                "-Rpass-analysis=loop-vectorize",
            ],
            capture_output=True,
            text=True,
        )
    assert proc.returncode == 0, proc.stderr
    assert "vectorized loop" in proc.stderr, proc.stderr
    assert "unsafe dependent memory operations" not in proc.stderr
