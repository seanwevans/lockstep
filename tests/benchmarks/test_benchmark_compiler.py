import random
import pathlib
import sys
from typing import Any

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lockstep_compiler.compiler import compile_lockstep
from lockstep_compiler.simulator import simulate_pipeline_entities


MEDIUM_LOCKSTEP_PROGRAM = """
struct Scalar {
    float v;
};

pure float p0(float x) {
    return x;
}

pure float p1(float x) {
    return x;
}

pure float p2(float x) {
    return x;
}

pure float p3(float x) {
    return x;
}

pure float p4(float x) {
    return x;
}

shader Stage0() {
}

shader Stage1() {
}

shader Stage2() {
}

shader Stage3() {
}

shader Stage4() {
}

pipeline MediumBenchmark {
    stream<Scalar, 8192> source;
    stream<Scalar, 8192> stage_a;
    stream<Scalar, 8192> stage_b;
    stream<Scalar, 8192> stage_c;
    stream<Scalar, 8192> sink;
    accumulator<float> energy;
    uniform float dt = 0.016;

    bind {
        uniform float total_energy = fold sum(energy);
    }
}
"""


def _stream_heavy_entities(capacity: int) -> dict[str, Any]:
    return {
        "streams": [
            {"name": "source", "type": "Particle", "capacity": str(capacity)},
            {"name": "stage1", "type": "Particle", "capacity": str(capacity)},
            {"name": "stage2", "type": "Particle", "capacity": str(capacity)},
            {"name": "stage3", "type": "Particle", "capacity": str(capacity)},
            {"name": "stage4", "type": "Particle", "capacity": str(capacity)},
            {"name": "sink", "type": "Particle", "capacity": str(capacity)},
        ],
        "accumulators": [],
        "uniforms": [],
        "shaders": [
            {
                "name": "StageA",
                "params": [
                    {"modifier": "in", "type": "Particle", "name": "src"},
                    {"modifier": "out", "type": "Particle", "name": "dst"},
                ],
            },
            {
                "name": "StageB",
                "params": [
                    {"modifier": "in", "type": "Particle", "name": "src"},
                    {"modifier": "out", "type": "Particle", "name": "dst"},
                ],
            },
            {
                "name": "StageC",
                "params": [
                    {"modifier": "in", "type": "Particle", "name": "src"},
                    {"modifier": "out", "type": "Particle", "name": "dst"},
                ],
            },
            {
                "name": "StageD",
                "params": [
                    {"modifier": "in", "type": "Particle", "name": "src"},
                    {"modifier": "out", "type": "Particle", "name": "dst"},
                ],
            },
        ],
        "filters": [
            {
                "name": "KeepAlive",
                "params": [
                    {"modifier": "in", "type": "Particle", "name": "src"},
                    {"modifier": "out", "type": "Particle", "name": "dst"},
                ],
            }
        ],
        "bind_routes": [
            "stage1 = StageA(source, stage1);",
            "stage2 = KeepAlive(stage1, stage2);",
            "stage3 = StageB(stage2, stage3);",
            "stage4 = StageC(stage3, stage4);",
            "sink = StageD(stage4, sink);",
        ],
        "bind_routes_ir": [
            {
                "kind": "kernel",
                "target": "stage1",
                "kernel": "StageA",
                "args": ["source", "stage1"],
                "route": "stage1 = StageA(source, stage1);",
            },
            {
                "kind": "kernel",
                "target": "stage2",
                "kernel": "KeepAlive",
                "args": ["stage1", "stage2"],
                "route": "stage2 = KeepAlive(stage1, stage2);",
            },
            {
                "kind": "kernel",
                "target": "stage3",
                "kernel": "StageB",
                "args": ["stage2", "stage3"],
                "route": "stage3 = StageB(stage2, stage3);",
            },
            {
                "kind": "kernel",
                "target": "stage4",
                "kernel": "StageC",
                "args": ["stage3", "stage4"],
                "route": "stage4 = StageC(stage3, stage4);",
            },
            {
                "kind": "kernel",
                "target": "sink",
                "kernel": "StageD",
                "args": ["stage4", "sink"],
                "route": "sink = StageD(stage4, sink);",
            },
        ],
    }


def _stream_rows(size: int, *, seed: int = 101) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for idx in range(size):
        rows.append(
            {
                "id": idx,
                "x": rng.random(),
                "y": rng.random(),
                "vx": rng.random(),
                "vy": rng.random(),
                "_keep": rng.random() > 0.15,
            }
        )
    return rows


def test_benchmark_compile_medium_program(benchmark: Any):
    benchmark.pedantic(
        lambda: compile_lockstep(MEDIUM_LOCKSTEP_PROGRAM, verbose=False),
        rounds=3,
        iterations=1,
        warmup_rounds=0,
    )


@pytest.mark.parametrize("row_count", [1_000, 10_000, 100_000])
def test_benchmark_simulate_stream_heavy_bind_graph(benchmark: Any, row_count: int):
    entities = _stream_heavy_entities(row_count)
    source_rows = _stream_rows(row_count, seed=2024 + row_count)

    benchmark.pedantic(
        lambda: simulate_pipeline_entities(entities, stream_inputs={"source": source_rows}),
        rounds=3,
        iterations=1,
        warmup_rounds=0,
    )
