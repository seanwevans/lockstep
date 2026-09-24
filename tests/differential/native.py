"""Compile a Lockstep program with clang and run one ``Lockstep_Tick`` on it.

The runner is deliberately program-agnostic: a single fixed C driver reads a raw
arena image from a file, calls ``Lockstep_Tick`` once, and writes the arena back
out.  All packing and unpacking of stream rows, accumulators, and uniforms
happens in Python against the arena layout the C header is generated from, so
the oracle exercises exactly the ABI a host sees.
"""

from __future__ import annotations

import functools
import platform
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lockstep_compiler.arena_layout import ArenaLayout, build_arena_layout

REPO_ROOT = Path(__file__).resolve().parents[2]
INTRINSICS_C = REPO_ROOT / "benchmarks" / "native" / "lockstep_intrinsics.c"

_DRIVER_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void Lockstep_Tick(void* arena);

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    size_t bytes = (size_t)strtoull(argv[1], NULL, 10);
    size_t padded = ((bytes + 63) / 64) * 64 + 64;
    unsigned char* arena = (unsigned char*)aligned_alloc(64, padded);
    if (!arena) return 3;
    memset(arena, 0, padded);
    FILE* in = fopen(argv[2], "rb");
    if (!in) return 4;
    if (bytes && fread(arena, 1, bytes, in) != bytes) return 5;
    fclose(in);
    Lockstep_Tick(arena);
    FILE* out = fopen(argv[3], "wb");
    if (!out) return 6;
    if (bytes && fwrite(arena, 1, bytes, out) != bytes) return 7;
    fclose(out);
    free(arena);
    return 0;
}
"""

_SCALAR_FORMATS = {"float": "<f", "double": "<d", "int": "<i", "uint": "<I", "bool": "<?"}


class NativeToolchainUnavailable(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def toolchain_problem() -> str | None:
    """Return why native execution is unavailable here, or ``None`` if it works.

    The generated IR hard-codes an ``x86_64-unknown-linux-gnu`` triple, so only a
    Linux x86-64 host can link and run it.
    """
    if shutil.which("clang") is None:
        return "clang not on PATH"
    if platform.system() != "Linux" or platform.machine().lower() not in {
        "x86_64",
        "amd64",
    }:
        return "generated IR targets x86_64-unknown-linux-gnu"
    return None


@functools.lru_cache(maxsize=1)
def _support_objects() -> tuple[str, str]:
    """Compile the driver and intrinsics once per process."""
    clang = shutil.which("clang")
    assert clang is not None
    work = Path(tempfile.mkdtemp(prefix="lsoracle_support_"))
    driver_c = work / "driver.c"
    driver_c.write_text(_DRIVER_C, encoding="utf-8")
    objects = []
    for src in (driver_c, INTRINSICS_C):
        obj = work / (src.stem + ".o")
        proc = subprocess.run(
            [clang, "-O2", "-c", str(src), "-o", str(obj)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise NativeToolchainUnavailable(proc.stderr)
        objects.append(str(obj))
    return objects[0], objects[1]


@dataclass
class ArenaCodec:
    """Pack/unpack host-visible values against the header's arena layout."""

    layout: ArenaLayout

    @classmethod
    def for_entities(cls, entities: dict[str, Any]) -> "ArenaCodec":
        return cls(build_arena_layout(entities))

    def _leaves(self, kind: str, name: str):
        return [
            leaf
            for leaf in self.layout.leaves
            if leaf.kind == kind and leaf.binding_name == name
        ]

    def new_image(self) -> bytearray:
        return bytearray(self.layout.total_size)

    def write_rows(self, image: bytearray, stream: str, rows: list[Any]) -> None:
        for leaf in self._leaves("stream", stream):
            fmt = _SCALAR_FORMATS[leaf.type_name]
            for index, row in enumerate(rows[: leaf.element_count]):
                value = _path_get(row, leaf.path)
                struct.pack_into(fmt, image, leaf.offset + index * leaf.size, value)

    def leaf_types(self, stream: str) -> dict[tuple[str, ...], str]:
        return {leaf.path: leaf.type_name for leaf in self._leaves("stream", stream)}

    def has_leaf(self, kind: str, name: str) -> bool:
        return bool(self._leaves(kind, name))

    def scalar_type(self, kind: str, name: str) -> str:
        (leaf,) = self._leaves(kind, name)
        return leaf.type_name

    def read_rows(self, image: bytes, stream: str, count: int) -> list[Any]:
        leaves = self._leaves("stream", stream)
        rows: list[Any] = []
        for index in range(count):
            row: Any = {}
            for leaf in leaves:
                fmt = _SCALAR_FORMATS[leaf.type_name]
                (value,) = struct.unpack_from(fmt, image, leaf.offset + index * leaf.size)
                if not leaf.path:
                    row = value
                else:
                    _path_set(row, leaf.path, value)
            rows.append(row)
        return rows

    def write_scalar(self, image: bytearray, kind: str, name: str, value: Any) -> None:
        (leaf,) = self._leaves(kind, name)
        struct.pack_into(_SCALAR_FORMATS[leaf.type_name], image, leaf.offset, value)

    def read_scalar(self, image: bytes, kind: str, name: str) -> Any:
        (leaf,) = self._leaves(kind, name)
        return struct.unpack_from(_SCALAR_FORMATS[leaf.type_name], image, leaf.offset)[0]


def _path_get(row: Any, path: tuple[str, ...]) -> Any:
    for part in path:
        row = row[part]
    return row


def _path_set(row: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    for part in path[:-1]:
        row = row.setdefault(part, {})
    row[path[-1]] = value


def run_tick(llvm_ir: str, image: bytes, *, keep_dir: Path | None = None) -> bytes:
    """Link ``llvm_ir`` against the fixed driver, tick once over ``image``."""
    problem = toolchain_problem()
    if problem is not None:
        raise NativeToolchainUnavailable(problem)
    clang = shutil.which("clang")
    assert clang is not None
    driver_o, intrinsics_o = _support_objects()
    with tempfile.TemporaryDirectory(prefix="lsoracle_") as tmp:
        work = Path(tmp)
        ir_path = work / "mod.ll"
        exe = work / "tick"
        in_path = work / "arena.in"
        out_path = work / "arena.out"
        ir_path.write_text(llvm_ir, encoding="utf-8")
        in_path.write_bytes(bytes(image))
        proc = subprocess.run(
            [
                clang,
                "-O2",
                "-Wno-override-module",
                str(ir_path),
                driver_o,
                intrinsics_o,
                "-o",
                str(exe),
                "-lm",
            ],
            capture_output=True,
            text=True,
        )
        if keep_dir is not None:
            keep_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(ir_path, keep_dir / "mod.ll")
        if proc.returncode != 0:
            raise RuntimeError(f"clang failed:\n{proc.stderr}")
        run = subprocess.run(
            [str(exe), str(len(image)), str(in_path), str(out_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if run.returncode != 0:
            raise RuntimeError(
                f"tick driver exited {run.returncode}: {run.stderr.strip()}"
            )
        return out_path.read_bytes()
