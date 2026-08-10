"""ABI agreement between the generated C header and the LLVM IR arena.

A fold accumulator is lowered by the backend as one slot per stream row (the
per-row partials the reduction pass folds), so the arena reserves ``capacity``
elements for it. The C header must reserve the same footprint: a host that
follows the documented flow -- allocate ``LOCKSTEP_ARENA_BYTES``, call
``Lockstep_Tick`` -- otherwise under-allocates and the kernel writes past the
end of the arena.

These tests pin that the header and IR agree on the arena size and that the
accumulator leaf is sized to the stream capacity rather than a lone scalar.
"""

from __future__ import annotations

import re

from lockstep_compiler.arena_layout import infer_accumulator_sizes
from lockstep_compiler.compiler import compile_lockstep

_LLVM_ELEMENT_BYTES = {
    "i1": 1,
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "i64": 8,
    "float": 4,
    "double": 8,
}

FOLD_PROGRAM = """
struct Particle {
    int id;
    float vx;
    float vy;
    float mass;
};

shader IntegrateAndAccumulate(
    in Particle src,
    out Particle dst,
    accum float energy
) {
    float speed2 = (src.vx * src.vx) + (src.vy * src.vy);
    energy = energy + (0.5 * src.mass * speed2);
    dst.id = src.id;
    dst.vx = src.vx;
    dst.vy = src.vy;
    dst.mass = src.mass;
}

pipeline Sim {
    stream<Particle, 4096> particlesIn;
    stream<Particle, 4096> particlesOut;
    accumulator<float> energy;
    bind {
        particlesOut = IntegrateAndAccumulate(particlesIn, particlesOut, energy);
        uniform float total = fold sum(energy);
    }
}
"""


def _ir_arena_bytes(ir_text: str) -> int:
    match = re.search(
        r'%"struct\.Lockstep_Arena"\s*=\s*type\s*\{(.+?)\}', ir_text, re.S
    )
    assert match is not None, "arena struct type missing from IR"
    total = 0
    for array in re.finditer(r"\[(\d+)\s*x\s*(\w+)\]", match.group(1)):
        total += int(array.group(1)) * _LLVM_ELEMENT_BYTES[array.group(2)]
    return total


def _header_arena_bytes(header_text: str) -> int:
    match = re.search(r"#define LOCKSTEP_ARENA_BYTES (\d+)", header_text)
    assert match is not None, "LOCKSTEP_ARENA_BYTES missing from header"
    return int(match.group(1))


def test_infer_accumulator_sizes_uses_route_trip_count():
    entities = {
        "accumulators": [{"name": "energy", "type": "float"}],
        "streams": [
            {"name": "pin", "type": "P", "capacity": "4096"},
            {"name": "pout", "type": "P", "capacity": "4096"},
        ],
        "shaders": [
            {
                "name": "K",
                "params": [
                    {"modifier": "in", "type": "P", "name": "src"},
                    {"modifier": "out", "type": "P", "name": "dst"},
                    {"modifier": "accum", "type": "float", "name": "energy"},
                ],
            }
        ],
        "filters": [],
        "bind_routes_ir": [
            {
                "kind": "kernel",
                "kernel": "K",
                "target": "pout",
                "args": ["pin", "pout", "energy"],
            }
        ],
    }
    assert infer_accumulator_sizes(entities) == {"energy": 4096}


def test_infer_accumulator_sizes_defaults_to_scalar_without_routes():
    entities = {
        "accumulators": [{"name": "loose", "type": "float"}],
        "streams": [],
        "shaders": [],
        "filters": [],
        "bind_routes_ir": [],
    }
    assert infer_accumulator_sizes(entities) == {"loose": 1}


def test_header_and_ir_agree_on_fold_arena_size():
    result = compile_lockstep(FOLD_PROGRAM, verbose=False)
    header_bytes = _header_arena_bytes(result.c_header)
    ir_bytes = _ir_arena_bytes(result.llvm_ir)
    assert header_bytes == ir_bytes


def test_header_sizes_accumulator_leaf_to_stream_capacity():
    result = compile_lockstep(FOLD_PROGRAM, verbose=False)
    # The accumulator leaf must be a capacity-sized array, not a lone scalar,
    # or a documented allocate-LOCKSTEP_ARENA_BYTES host would corrupt memory.
    assert "float accum_energy_value[4096];" in result.c_header
    assert "float accum_energy_value;" not in result.c_header
