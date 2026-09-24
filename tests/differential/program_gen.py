"""Seeded generator of random, *valid* Lockstep programs plus their inputs.

Each generated case is a pipeline the semantic validator accepts, together with
full-capacity input rows for its source streams and values for its uniforms.
The differential oracle (``tests/test_differential_oracle.py``) runs every case
through both the Python simulator and the compiled ``Lockstep_Tick`` and checks
that their observable outputs agree.

The generator is plain ``random.Random(seed)`` rather than Hypothesis: every
case costs a clang invocation, so shrinking is too expensive to be useful, and a
seed is a complete, stable reproducer
(``PYTHONPATH=.:tests python -m differential.oracle SEED``).

Shapes covered:

* 1-4 stage kernel chains over 1-3 struct types with ``float``/``int``/
  ``bool``/``double`` fields and nested struct fields (so stage fusion, SoA
  leaf vectors, and bool byte columns are all hit);
* shaders, keep-all filters (no ``return``, or ``return true``) anywhere in a
  chain, and data-dependent filters as the last stage of a chain;
* in-place routes (``s = K(s, s)``), a second ``in`` stream read alongside the
  chain input (fan-in), and a side route reading an intermediate stream
  (fan-out -- which blocks fusion);
* ``uniform`` parameters, locals, user ``pure`` functions (which keep a group
  off the vector path), every branchless intrinsic, casts, integer
  division/remainder/shift/bitwise ops, and ``&&``/``||``/``!``;
* ``accum`` parameters (``float``/``int``/``double``) consumed by one or two
  ``fold sum/avg/min/max`` routes, including accumulators written by two
  kernels, and fold results read back by a later stage as a ``uniform``;
* capacities below, at, and above the SIMD width, with and without a scalar
  tail.

Shapes deliberately *not* generated are the semantic gaps the oracle found that
need a language-design decision rather than a bug fix; each one is pinned by a
strict ``xfail`` in ``tests/test_differential_oracle.py`` instead:

* streams of different capacities in one route (saturation semantics differ);
* a data-dependent filter whose output feeds a later stage (compiled code has no
  live row count);
* an accumulator written by two kernels and folded with ``avg``/``min``/``max``
  (per-row vs. per-contribution semantics).

Two restrictions are about the comparison, not about semantics: float
``sum``/``avg`` folds reassociate, so only exact folds (integer ones, and float
``min``/``max``) feed later stages; and values are kept in ranges where 32-bit
``int`` arithmetic cannot hit undefined behavior (division by zero, oversized
shifts, out-of-range float-to-int conversion).
"""

from __future__ import annotations

import random
import struct as _struct
from dataclasses import dataclass, field
from typing import Any

SCALAR_TYPES = ("float", "int", "bool", "double")
CAPACITIES = (1, 3, 4, 7, 8, 9, 16, 21, 64)
FOLD_OPERATORS = ("sum", "avg", "min", "max")


def f32(value: float) -> float:
    """Round a Python float to the nearest IEEE single."""
    return _struct.unpack("<f", _struct.pack("<f", value))[0]


@dataclass
class StructSpec:
    name: str
    fields: list[tuple[str, str]]  # (field name, scalar or struct type)


@dataclass
class GeneratedCase:
    seed: int
    source: str
    capacity: int
    # stream name -> rows (dict per struct row, scalar per primitive row)
    stream_inputs: dict[str, list[Any]]
    # uniform name -> value the host primes into the arena
    uniform_values: dict[str, Any]
    # streams whose final contents are observable (never read by a later route)
    sink_streams: list[str]
    # uniforms written by a ``fold`` route
    fold_uniforms: list[str]
    stream_types: dict[str, str] = field(default_factory=dict)


class _Scope:
    """Readable expressions, by scalar type."""

    def __init__(self) -> None:
        self.by_type: dict[str, list[str]] = {name: [] for name in SCALAR_TYPES}

    def add(self, type_name: str, text: str) -> None:
        self.by_type[type_name].append(text)


@dataclass
class _Pure:
    name: str
    params: list[tuple[str, str]]  # (type, name)
    return_type: str
    body: list[str]


class _ExprGen:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.pures: list[_Pure] = []

    # -- literals ------------------------------------------------------------
    def float_lit(self, nonzero: bool = False) -> str:
        choices = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 0.1, 0.3, 7.5, 10.0, 100.0]
        if nonzero:
            choices = [c for c in choices if c != 0.0]
        return f"{self.rng.choice(choices)!r}"

    def int_lit(self, nonzero: bool = False) -> str:
        return str(self.rng.randint(1 if nonzero else 0, 9))

    # -- typed expressions -----------------------------------------------------
    def expr(self, type_name: str, scope: _Scope, depth: int) -> str:
        return {
            "float": self.float_expr,
            "int": self.int_expr,
            "bool": self.bool_expr,
            "double": self.double_expr,
        }[type_name](scope, depth)

    def _leaf(self, type_name: str, scope: _Scope) -> str:
        names = scope.by_type[type_name]
        if names and self.rng.random() < 0.75:
            return self.rng.choice(names)
        if type_name == "float":
            return self.float_lit()
        if type_name == "int":
            return self.int_lit()
        if type_name == "double":
            return f"((double) {self.float_lit()})"
        return self.rng.choice(["true", "false"])

    def _pure_call(self, type_name: str, scope: _Scope, depth: int) -> str | None:
        candidates = [pure for pure in self.pures if pure.return_type == type_name]
        if not candidates:
            return None
        pure = self.rng.choice(candidates)
        args = ", ".join(self.expr(ptype, scope, depth) for ptype, _ in pure.params)
        return f"{pure.name}({args})"

    def float_expr(self, scope: _Scope, depth: int) -> str:
        if depth <= 0:
            return self._leaf("float", scope)
        d = depth - 1
        rng = self.rng
        kind = rng.randrange(19)
        if kind <= 2:
            return self._leaf("float", scope)
        if kind == 3:
            op = rng.choice(["+", "-", "*"])
            return f"({self.float_expr(scope, d)} {op} {self.float_expr(scope, d)})"
        if kind == 4:
            return f"({self.float_expr(scope, d)} / {self.float_lit(nonzero=True)})"
        if kind == 5:
            return f"(-{self.float_expr(scope, d)})"
        if kind == 6:
            return (
                f"select({self.bool_expr(scope, d)}, {self.float_expr(scope, d)}, "
                f"{self.float_expr(scope, d)})"
            )
        if kind == 7:
            return (
                f"mix({self.float_expr(scope, d)}, {self.float_expr(scope, d)}, "
                f"{self.float_expr(scope, d)})"
            )
        if kind == 8:
            return f"clamp({self.float_expr(scope, d)}, -5.0, 5.0)"
        if kind == 9:
            fn = rng.choice(["min", "max"])
            return f"{fn}({self.float_expr(scope, d)}, {self.float_expr(scope, d)})"
        if kind == 10:
            return f"abs({self.float_expr(scope, d)})"
        if kind == 11:
            return f"sign({self.float_expr(scope, d)})"
        if kind == 12:
            return f"step({self.float_expr(scope, d)}, {self.float_expr(scope, d)})"
        if kind == 13:
            return (
                f"smoothstep({self.float_expr(scope, d)}, {self.float_expr(scope, d)}, "
                f"{self.float_expr(scope, d)})"
            )
        if kind == 14:
            return f"((float) {self.int_expr(scope, d)})"
        if kind == 15:
            return f"((float) {self.bool_expr(scope, d)})"
        if kind == 16:
            return f"((float) {self.double_expr(scope, d)})"
        if kind == 17:
            call = self._pure_call("float", scope, d)
            if call is not None:
                return call
        return f"({self.float_expr(scope, d)} * {self.float_lit()})"

    def double_expr(self, scope: _Scope, depth: int) -> str:
        if depth <= 0:
            return self._leaf("double", scope)
        d = depth - 1
        rng = self.rng
        kind = rng.randrange(9)
        if kind <= 2:
            return self._leaf("double", scope)
        if kind == 3:
            op = rng.choice(["+", "-", "*"])
            return f"({self.double_expr(scope, d)} {op} {self.double_expr(scope, d)})"
        if kind == 4:
            return (
                f"({self.double_expr(scope, d)} / "
                f"((double) {self.float_lit(nonzero=True)}))"
            )
        if kind == 5:
            return f"(-{self.double_expr(scope, d)})"
        if kind == 6:
            return (
                f"select({self.bool_expr(scope, d)}, {self.double_expr(scope, d)}, "
                f"{self.double_expr(scope, d)})"
            )
        if kind == 7:
            return f"((double) {self.float_expr(scope, d)})"
        return f"((double) {self.int_expr(scope, d)})"

    def int_expr(self, scope: _Scope, depth: int) -> str:
        if depth <= 0:
            return self._leaf("int", scope)
        d = depth - 1
        rng = self.rng
        kind = rng.randrange(13)
        if kind <= 2:
            return self._leaf("int", scope)
        if kind == 3:
            op = rng.choice(["+", "-", "*"])
            return f"({self.int_expr(scope, d)} {op} {self.int_expr(scope, d)})"
        if kind == 4:
            op = rng.choice(["/", "%"])
            return f"({self.int_expr(scope, d)} {op} {self.int_lit(nonzero=True)})"
        if kind == 5:
            return f"(-{self.int_expr(scope, d)})"
        if kind == 6:
            return (
                f"select({self.bool_expr(scope, d)}, {self.int_expr(scope, d)}, "
                f"{self.int_expr(scope, d)})"
            )
        if kind == 7:
            # Keep float->int conversions in range: out-of-range fptosi is poison.
            return f"((int) clamp({self.float_expr(scope, d)}, -1000.0, 1000.0))"
        if kind == 8:
            op = rng.choice(["&", "|", "^"])
            return f"({self.int_expr(scope, d)} {op} {self.int_expr(scope, d)})"
        if kind == 9:
            op = rng.choice(["<<", ">>"])
            return f"({self.int_expr(scope, d)} {op} {rng.randint(0, 4)})"
        if kind == 10:
            return f"((int) {self.bool_expr(scope, d)})"
        if kind == 11:
            call = self._pure_call("int", scope, d)
            if call is not None:
                return call
        return f"({self.int_expr(scope, d)} * {self.int_lit()})"

    def bool_expr(self, scope: _Scope, depth: int) -> str:
        if depth <= 0:
            return self._leaf("bool", scope)
        d = depth - 1
        rng = self.rng
        kind = rng.randrange(10)
        if kind <= 1:
            return self._leaf("bool", scope)
        cmp_op = rng.choice(["<", "<=", ">", ">=", "==", "!="])
        if kind == 2:
            return f"({self.float_expr(scope, d)} {cmp_op} {self.float_expr(scope, d)})"
        if kind == 3:
            return f"({self.int_expr(scope, d)} {cmp_op} {self.int_expr(scope, d)})"
        if kind == 4:
            return f"({self.double_expr(scope, d)} {cmp_op} {self.double_expr(scope, d)})"
        if kind == 5:
            op = rng.choice(["&&", "||"])
            return f"({self.bool_expr(scope, d)} {op} {self.bool_expr(scope, d)})"
        if kind == 6:
            return f"(!{self.bool_expr(scope, d)})"
        if kind == 7:
            return f"((bool) {self.int_expr(scope, d)})"
        if kind == 8:
            call = self._pure_call("bool", scope, d)
            if call is not None:
                return call
        return f"({self.float_expr(scope, d)} > {self.float_lit()})"


def _input_value(rng: random.Random, type_name: str) -> Any:
    if type_name == "float":
        # Mostly exactly-representable values, some arbitrary ones.
        if rng.random() < 0.7:
            return rng.randint(-40, 40) * 0.25
        return f32(rng.uniform(-20.0, 20.0))
    if type_name == "double":
        if rng.random() < 0.5:
            return rng.randint(-40, 40) * 0.25
        return rng.uniform(-20.0, 20.0)
    if type_name == "int":
        return rng.randint(-50, 50)
    return rng.random() < 0.5


@dataclass
class _Kernel:
    name: str
    kind: str  # "shader" | "filter"
    params: list[tuple[str, str, str]]  # (modifier, type, name)
    body: list[str]


@dataclass
class _Route:
    target: str
    kernel: _Kernel
    args: list[str]

    def render(self) -> str:
        return f"{self.target} = {self.kernel.name}({', '.join(self.args)});"


class _ProgramBuilder:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.gen = _ExprGen(self.rng)
        self.depth = self.rng.randint(1, 3)
        self.capacity = self.rng.choice(CAPACITIES)
        self.structs: list[StructSpec] = []
        self.kernels: list[_Kernel] = []
        self.streams: list[tuple[str, str]] = []
        self.accumulators: list[tuple[str, str]] = []
        self.uniform_decls: list[tuple[str, str, Any]] = []
        # Rendered bind statements, in order (kernel routes and folds).
        self.bind: list[str] = []
        self.fold_uniforms: list[str] = []
        # Fold results a later stage may read: (type, uniform name).
        self.readable_folds: list[tuple[str, str]] = []
        self.shareable_accums: set[str] = set()
        self._pending_uniform_args: list[str] = []

    # -- types ---------------------------------------------------------------
    def build_structs(self) -> None:
        rng = self.rng
        for struct_index in range(rng.randint(1, 3)):
            fields: list[tuple[str, str]] = []
            for field_index in range(rng.randint(1, 3)):
                if self.structs and rng.random() < 0.15:
                    ftype = rng.choice(self.structs).name
                else:
                    ftype = rng.choice(["float", "float", "int", "bool", "double"])
                fields.append((f"f{field_index}", ftype))
            if all(ftype == "bool" for _, ftype in fields):
                fields[0] = (fields[0][0], rng.choice(["float", "int"]))
            self.structs.append(StructSpec(f"S{struct_index}", fields))

    def leaves(self, type_name: str, prefix: str = "") -> list[tuple[str, str]]:
        """Flatten a stream element type to ``(access path, scalar type)``."""
        if type_name in SCALAR_TYPES:
            return [(prefix, type_name)]
        spec = next(s for s in self.structs if s.name == type_name)
        out: list[tuple[str, str]] = []
        for fname, ftype in spec.fields:
            out.extend(self.leaves(ftype, f"{prefix}.{fname}" if prefix else fname))
        return out

    def element_type(self) -> str:
        if self.rng.random() < 0.1:
            return self.rng.choice(["float", "int"])
        return self.rng.choice(self.structs).name

    def input_row(self, type_name: str) -> Any:
        if type_name in SCALAR_TYPES:
            return _input_value(self.rng, type_name)
        spec = next(s for s in self.structs if s.name == type_name)
        return {fname: self.input_row(ftype) for fname, ftype in spec.fields}

    # -- pure functions --------------------------------------------------------
    def build_pures(self) -> list[_Pure]:
        rng = self.rng
        pures: list[_Pure] = []
        for pure_index in range(rng.randint(0, 2)):
            params = [
                (rng.choice(["float", "int", "bool"]), f"a{i}")
                for i in range(rng.randint(1, 2))
            ]
            return_type = rng.choice(["float", "int", "bool"])
            scope = _Scope()
            for ptype, pname in params:
                scope.add(ptype, pname)
            body: list[str] = []
            if rng.random() < 0.5:
                ltype = rng.choice(["float", "int", "bool"])
                body.append(f"{ltype} l0 = {self.gen.expr(ltype, scope, 1)};")
                scope.add(ltype, "l0")
            body.append(f"return {self.gen.expr(return_type, scope, 2)};")
            pure = _Pure(f"fn{pure_index}", params, return_type, body)
            pures.append(pure)
            # Later pure functions may call earlier ones.
            self.gen.pures.append(pure)
        return pures

    # -- kernels ---------------------------------------------------------------
    def make_kernel(
        self,
        in_type: str,
        out_type: str,
        *,
        kind: str,
        keep_all: bool,
        extra_in: tuple[str, str] | None,
        accum_ok: bool,
    ) -> tuple[_Kernel, list[str]]:
        rng = self.rng
        scope = _Scope()
        params: list[tuple[str, str, str]] = [("in", in_type, "src")]
        for path, ltype in self.leaves(in_type):
            scope.add(ltype, f"src.{path}" if path else "src")
        if extra_in is not None:
            params.append(("in", extra_in[1], "side"))
            for path, ltype in self.leaves(extra_in[1]):
                scope.add(ltype, f"side.{path}" if path else "side")
        params.append(("out", out_type, "dst"))
        uniform_args: list[str] = []
        for utype, uname, _ in self.uniform_decls:
            if rng.random() < 0.6:
                params.append(("uniform", utype, f"p_{uname}"))
                scope.add(utype, f"p_{uname}")
                uniform_args.append(uname)
        for utype, uname in self.readable_folds:
            if rng.random() < 0.7:
                params.append(("uniform", utype, f"p_{uname}"))
                scope.add(utype, f"p_{uname}")
                uniform_args.append(uname)
        body: list[str] = []
        for local_index in range(rng.randint(0, 2)):
            ltype = rng.choice(SCALAR_TYPES)
            lname = f"t{local_index}"
            body.append(f"{ltype} {lname} = {self.gen.expr(ltype, scope, self.depth)};")
            scope.add(ltype, lname)
        for path, ltype in self.leaves(out_type):
            target = f"dst.{path}" if path else "dst"
            body.append(f"{target} = {self.gen.expr(ltype, scope, self.depth)};")
        accum_args: list[str] = []
        if kind == "shader" and accum_ok:
            for _ in range(rng.choice([0, 0, 1, 1, 2])):
                shared = [
                    (aname, atype)
                    for aname, atype in self.accumulators
                    if aname in self.shareable_accums and aname not in accum_args
                ]
                if shared and rng.random() < 0.3:
                    aname, atype = rng.choice(shared)
                else:
                    atype = rng.choice(["float", "float", "int", "double"])
                    aname = f"acc{len(self.accumulators)}"
                    self.accumulators.append((aname, atype))
                pname = f"acc_{len(accum_args)}"
                params.append(("accum", atype, pname))
                body.append(f"{pname} = {pname} + {self.gen.expr(atype, scope, self.depth)};")
                accum_args.append(aname)
        if kind == "filter":
            if not keep_all:
                body.append(f"return {self.gen.bool_expr(scope, self.depth)};")
            elif rng.random() < 0.3:
                body.append("return true;")
        kernel = _Kernel(f"K{len(self.kernels)}", kind, params, body)
        self.kernels.append(kernel)
        # Order the uniform args to match the params.
        self._pending_uniform_args = uniform_args
        return kernel, accum_args

    def route(
        self, kernel: _Kernel, target: str, src: str, extra: str | None, accs: list[str]
    ) -> _Route:
        uniform_args = list(self._pending_uniform_args)
        args: list[str] = []
        accs = list(accs)
        for modifier, _ptype, pname in kernel.params:
            if modifier == "in":
                args.append(src if pname == "src" else str(extra))
            elif modifier == "out":
                args.append(target)
            elif modifier == "uniform":
                args.append(uniform_args.pop(0))
            else:
                args.append(accs.pop(0))
        return _Route(target, kernel, args)

    def build(self, seed: int) -> GeneratedCase:
        rng = self.rng
        self.build_structs()
        pures = self.build_pures()
        # Accumulators left open for a second writer are folded (``sum`` only,
        # see the module docstring) at the end of the bind block.
        accum_fold_plan: dict[str, list[str]] = {}

        uniform_values: dict[str, Any] = {}
        for uniform_index in range(rng.randint(0, 2)):
            utype = rng.choice(["float", "int", "double"])
            name = f"u{uniform_index}"
            value = _input_value(rng, utype)
            self.uniform_decls.append((utype, name, value))
            uniform_values[name] = value

        input_type = self.element_type()
        self.streams.append(("s0", input_type))
        source_streams = ["s0"]
        if rng.random() < 0.3:
            self.streams.append(("aux", self.element_type()))
            source_streams.append("aux")
        stream_types = dict(self.streams)

        def plan_folds(new_accums: list[tuple[str, str]], writer_is_last: bool) -> None:
            """Emit fold routes for freshly created accumulators, right away."""
            for aname, atype in new_accums:
                ops = [rng.choice(FOLD_OPERATORS)]
                if rng.random() < 0.2:
                    ops.append(rng.choice(FOLD_OPERATORS))
                # Keep some accumulators open for a second writer (sum only),
                # folding them at the end of the bind block instead.
                if not writer_is_last and rng.random() < 0.25:
                    accum_fold_plan[aname] = ["sum"] * len(ops)
                    self.shareable_accums.add(aname)
                    continue
                for fold_index, op in enumerate(ops):
                    uname = f"r_{aname}_{fold_index}"
                    self.bind.append(f"uniform {atype} {uname} = fold {op}({aname});")
                    self.fold_uniforms.append(uname)
                    exact = atype == "int" or op in {"min", "max"}
                    if exact and atype != "double" and rng.random() < 0.6:
                        self.readable_folds.append((atype, uname))

        n_stages = rng.randint(1, 4)
        current_stream, current_type = "s0", input_type
        chain_streams: list[tuple[str, str]] = []
        for stage in range(n_stages):
            is_last = stage == n_stages - 1
            roll = rng.random()
            if roll < 0.2:
                kind, keep_all = "filter", True
            elif roll < 0.35 and is_last:
                kind, keep_all = "filter", False
            else:
                kind, keep_all = "shader", True
            in_place = (
                kind == "shader"
                and current_stream not in source_streams
                and rng.random() < 0.15
            )
            # Filters pass rows through unchanged in type.
            out_type = (
                current_type if kind == "filter" or in_place else self.element_type()
            )
            extra_in = None
            if "aux" in source_streams and rng.random() < 0.5:
                extra_in = ("aux", stream_types["aux"])
            accums_before = len(self.accumulators)
            kernel, accs = self.make_kernel(
                current_type,
                out_type,
                kind=kind,
                keep_all=keep_all,
                extra_in=extra_in,
                accum_ok=True,
            )
            target = current_stream if in_place else f"s{stage + 1}"
            if not in_place:
                self.streams.append((target, out_type))
                stream_types[target] = out_type
            self.bind.append(
                self.route(
                    kernel, target, current_stream, extra_in[0] if extra_in else None, accs
                ).render()
            )
            chain_streams.append((target, out_type))
            plan_folds(self.accumulators[accums_before:], is_last)
            current_stream, current_type = target, out_type

        # Fan-out: a side route reading an intermediate (or the source) stream.
        if rng.random() < 0.3:
            candidates = [("s0", input_type)] + chain_streams[:-1]
            src_name, src_type = rng.choice(candidates)
            side_kind = "filter" if rng.random() < 0.3 else "shader"
            out_type = src_type if side_kind == "filter" else self.element_type()
            accums_before = len(self.accumulators)
            kernel, accs = self.make_kernel(
                src_type,
                out_type,
                kind=side_kind,
                keep_all=rng.random() < 0.5,
                extra_in=None,
                accum_ok=True,
            )
            self.streams.append(("side_out", out_type))
            stream_types["side_out"] = out_type
            self.bind.append(self.route(kernel, "side_out", src_name, None, accs).render())
            plan_folds(self.accumulators[accums_before:], True)

        for aname, ops in accum_fold_plan.items():
            atype = dict(self.accumulators)[aname]
            for fold_index, op in enumerate(ops):
                uname = f"r_{aname}_{fold_index}"
                self.bind.append(f"uniform {atype} {uname} = fold {op}({aname});")
                self.fold_uniforms.append(uname)

        source = self.render(pures)
        stream_inputs = {
            name: [self.input_row(stream_types[name]) for _ in range(self.capacity)]
            for name in source_streams
        }
        return GeneratedCase(
            seed=seed,
            source=source,
            capacity=self.capacity,
            stream_inputs=stream_inputs,
            uniform_values=uniform_values,
            sink_streams=[],  # derived from the compiled program by the oracle
            fold_uniforms=list(self.fold_uniforms),
            stream_types=stream_types,
        )

    def render(self, pures: list[_Pure]) -> str:
        lines: list[str] = []
        for spec in self.structs:
            members = " ".join(f"{ftype} {fname};" for fname, ftype in spec.fields)
            lines.append(f"struct {spec.name} {{ {members} }};")
        lines.append("")
        for pure in pures:
            params = ", ".join(f"{ptype} {pname}" for ptype, pname in pure.params)
            lines.append(f"pure {pure.return_type} {pure.name}({params}) {{")
            lines.extend(f"    {stmt}" for stmt in pure.body)
            lines.append("}")
            lines.append("")
        for kernel in self.kernels:
            params = ", ".join(f"{m} {t} {n}" for m, t, n in kernel.params)
            lines.append(f"{kernel.kind} {kernel.name}({params}) {{")
            lines.extend(f"    {stmt}" for stmt in kernel.body)
            lines.append("}")
            lines.append("")
        lines.append("pipeline P {")
        for sname, stype in self.streams:
            lines.append(f"    stream<{stype}, {self.capacity}> {sname};")
        for aname, atype in self.accumulators:
            lines.append(f"    accumulator<{atype}> {aname};")
        for utype, uname, value in self.uniform_decls:
            lines.append(f"    uniform {utype} {uname} = {_literal(utype, value)};")
        lines.append("    bind {")
        lines.extend(f"        {stmt}" for stmt in self.bind)
        lines.append("    }")
        lines.append("}")
        return "\n".join(lines) + "\n"


def _literal(type_name: str, value: Any) -> str:
    if type_name == "int":
        return f"-{abs(int(value))}" if int(value) < 0 else str(int(value))
    text = repr(abs(float(value)))
    if "." not in text and "e" not in text:
        text += ".0"
    return f"-{text}" if float(value) < 0 else text


def generate_case(seed: int) -> GeneratedCase:
    return _ProgramBuilder(seed).build(seed)
