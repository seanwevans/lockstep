from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any

_INTRINSIC_DECL_RE = re.compile(
    r"^\s*intrinsic\s+(?P<return_type>[A-Za-z_][A-Za-z0-9_]*)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<params>[^)]*)\)\s*;\s*$"
)
_PARAM_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*$")


@functools.lru_cache(maxsize=1)
def load_intrinsics() -> dict[str, dict[str, Any]]:
    prelude_path = Path(__file__).with_name("prelude.lock")
    intrinsics: dict[str, dict[str, Any]] = {}
    for line in prelude_path.read_text(encoding="utf-8").splitlines():
        match = _INTRINSIC_DECL_RE.match(line)
        if match is None:
            continue
        params_raw = match.group("params").strip()
        params: list[dict[str, str]] = []
        if params_raw:
            for piece in params_raw.split(","):
                param_match = _PARAM_RE.match(piece)
                if param_match is None:
                    continue
                params.append(
                    {"type": param_match.group(1), "name": param_match.group(2)}
                )

        intrinsics[match.group("name")] = {
            "name": match.group("name"),
            "return_type": match.group("return_type"),
            "params": params,
            "body": [],
            "intrinsic": True,
        }
    return intrinsics
