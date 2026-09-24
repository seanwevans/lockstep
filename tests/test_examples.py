"""End-to-end smoke tests for the programs in ``examples/``.

Each example is built exactly as ``examples/README.md`` describes: ``lockstepc
--emit-ir`` / ``--emit-header``, clang for the IR, and the example's C host
compiled against the generated header. The particle simulation's output is then
checked frame by frame against the simulator, which exercises the whole host
contract -- priming streams and uniforms, reading folded uniforms back, and
reading a filtered stream's live row count before its rows.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from differential.native import toolchain_problem
from lockstep_compiler.cli import run_cli
from lockstep_compiler.compiler import compile_lockstep
from lockstep_compiler.simulator import simulate_pipeline_entities

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"

pytestmark = pytest.mark.skipif(
    toolchain_problem() is not None,
    reason=f"native execution unavailable: {toolchain_problem()}",
)


def _lockstepc(*argv: str) -> str:
    stdout, stderr = io.StringIO(), io.StringIO()
    code = run_cli(list(argv), stdout=stdout, stderr=stderr, compiler=compile_lockstep)
    assert code == 0, stderr.getvalue()
    return stdout.getvalue()


def _build_and_run(program: str, host: str, work: Path) -> str:
    source = EXAMPLES / program
    (work / "program.ll").write_text(_lockstepc(str(source), "--emit-ir"), encoding="utf-8")
    (work / "lockstep_generated.h").write_text(
        _lockstepc(str(source), "--emit-header"), encoding="utf-8"
    )
    clang = shutil.which("clang")
    assert clang is not None
    subprocess.run(
        [clang, "-O2", "-Wno-override-module", "-c", str(work / "program.ll"),
         "-o", str(work / "program.o")],
        check=True, capture_output=True,
    )
    subprocess.run(
        [clang, "-std=c11", "-O2", str(EXAMPLES / host), str(work / "program.o"),
         f"-I{work}", "-o", str(work / "host")],
        check=True, capture_output=True,
    )
    return subprocess.run(
        [str(work / "host")], check=True, capture_output=True, text=True, timeout=60
    ).stdout


def test_minimal_example_matches_readme(tmp_path: Path) -> None:
    output = _build_and_run("minimal.lock", "minimal_host.c", tmp_path)
    assert output.strip() == "After tick: pos=10.00 vel=2.00 dt=0.50"


def _f32(value: float) -> float:
    import struct

    return struct.unpack("<f", struct.pack("<f", value))[0]


def _initial_particles(count: int) -> list[dict[str, float]]:
    # Mirrors the priming loop in examples/particles_host.c.
    return [
        {
            "x": _f32((i % 125) * _f32(0.9)),
            "y": _f32((i % 17) * 0.5),
            "vx": float((i % 11) - 5),
            "vy": 0.0,
            "mass": 1.0 + float(i % 3),
        }
        for i in range(count)
    ]


def test_particle_example_matches_simulator(tmp_path: Path) -> None:
    output = _build_and_run("particles.lock", "particles_host.c", tmp_path)
    frames = re.findall(
        r"frame (\d+): kinetic energy ([-\d.]+), inside (\d+)/(\d+) "
        r"\(kept (\d+)\), mean x ([-\d.]+)",
        output,
    )
    assert len(frames) == 3, output

    result = compile_lockstep((EXAMPLES / "particles.lock").read_text(), verbose=False)
    particles = _initial_particles(1000)
    for _frame, energy, inside, total, kept, mean_x in frames:
        simulated = simulate_pipeline_entities(result, stream_inputs={"particles": particles})
        particles = simulated["streams"]["particles"]
        tallied = simulated["streams"]["tallied"]
        assert int(total) == 1000
        assert int(inside) == len(tallied)
        assert int(kept) == simulated["uniforms"]["keptCount"]
        assert float(energy) == pytest.approx(
            simulated["uniforms"]["kineticEnergy"], rel=1e-4
        )
        expected_mean = sum(row["x"] for row in tallied) / len(tallied)
        assert float(mean_x) == pytest.approx(expected_mean, abs=2e-3)
