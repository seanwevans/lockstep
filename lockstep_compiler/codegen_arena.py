"""Arena addressing, live row counts, and hoisted uniform loads for ``Lockstep_Tick``.

Part of the ``Lockstep_Tick`` code generator: ``codegen._TickCodegen`` combines
this mixin with the others and holds the per-tick state they share through
``self`` (the arena pointer, IR builder, arena layout, and lowering caches).
"""

from __future__ import annotations

from collections.abc import Sequence
from llvmlite import ir

from .ast import AstKernelBindRoute
from .codegen_lowerer import _type_name
from .utils import sanitize_symbol as _sanitize_symbol


class ArenaAccessMixin:
    """Arena addressing, live row counts, and hoisted uniform loads for ``Lockstep_Tick``."""

    def _infer_accumulator_sizes(self) -> dict[str, int]:
        inferred_sizes = {accum.name: 1 for accum in self.accumulators}
        for name, size in self.explicit_accumulator_sizes.items():
            if name in inferred_sizes:
                inferred_sizes[name] = max(inferred_sizes[name], size)
        for pipeline in self.program.pipelines:
            pipeline_stream_capacities = {
                stream.name: int(stream.capacity) for stream in pipeline.streams
            }
            pipeline_accumulators = {accum.name for accum in pipeline.accumulators}
            for route in pipeline.bind_routes:
                if not isinstance(route, AstKernelBindRoute):
                    continue
                signature = self.kernel_signatures.get(route.kernel)
                if signature is None:
                    continue
                _, params = signature
                trip_count = 0
                for index, arg_name in enumerate(route.args):
                    if index >= len(params):
                        break
                    if (
                        params[index].modifier == "in"
                        and arg_name in pipeline_stream_capacities
                    ):
                        trip_count = max(
                            trip_count, pipeline_stream_capacities[arg_name]
                        )
                if route.target in pipeline_stream_capacities:
                    trip_count = max(
                        trip_count, pipeline_stream_capacities[route.target]
                    )
                trip_count = max(trip_count, 1)
                for index, arg_name in enumerate(route.args):
                    if index >= len(params):
                        break
                    if (
                        params[index].modifier == "accum"
                        and arg_name in pipeline_accumulators
                    ):
                        inferred_sizes[arg_name] = max(
                            inferred_sizes[arg_name], trip_count
                        )
        return inferred_sizes

    def _hoist_uniform_loads(
        self, routes: Sequence[AstKernelBindRoute], *, vector_leaves: bool = False
    ) -> None:
        # A uniform is constant for the whole route, but loading it inside the
        # row loop leaves LLVM to prove the row stores never overwrite it.  The
        # arena is one pointer and BasicAA can't bound the row index, so it
        # can't: the load stays in the loop and blocks vectorization.  Loading
        # it once before the loop sidesteps the question entirely.
        self._hoisted_uniforms.clear()
        self._hoisted_uniform_leaves.clear()
        for route in routes:
            callee, params = self._kernel_function_and_params(route.kernel)
            if callee is None:
                continue
            for param, arg_name in zip(params, route.args):
                if not vector_leaves:
                    break
                if param.modifier != "uniform" or arg_name not in self.uniform_slots:
                    continue
                leaves = self._vectorizable_leaf_fields(_type_name(param.declared_type))
                for rel_path, (leaf_ty, leaf_type_name) in (leaves or {}).items():
                    leaf_key = (arg_name, rel_path, str(leaf_ty))
                    if leaf_key not in self._hoisted_uniform_leaves:
                        self._hoisted_uniform_leaves[leaf_key] = self._load_value(
                            "uniform", arg_name, leaf_ty, leaf_type_name, rel_path
                        )
            for index, param in enumerate(params):
                if index >= len(route.args) or index >= len(callee.args):
                    continue
                arg_name = route.args[index]
                if param.modifier != "uniform" or arg_name not in self.uniform_slots:
                    continue
                llvm_type = callee.args[index].type
                key = (arg_name, str(llvm_type))
                if key not in self._hoisted_uniforms:
                    self._hoisted_uniforms[key] = self._load_tick_param(
                        "uniform", arg_name, llvm_type
                    )

    def _i32(self, value: int) -> ir.Constant:
        return ir.Constant(ir.IntType(32), int(value))

    def _count_bound(self, value: ir.Value, bound: int) -> int:
        return int(value.constant) if isinstance(value, ir.Constant) else bound

    def _stream_live_count(self, name: str) -> ir.Value:
        return self._live_counts.get(name) or self._i32(
            max(self.stream_capacities.get(name, 1), 1)
        )

    def _is_dynamic(self, value: ir.Value) -> bool:
        return not isinstance(value, ir.Constant)

    def _same_count(self, lhs: ir.Value, rhs: ir.Value) -> bool:
        if lhs is rhs:
            return True
        return (
            isinstance(lhs, ir.Constant)
            and isinstance(rhs, ir.Constant)
            and int(lhs.constant) == int(rhs.constant)
        )

    def _max_count(self, values: list[tuple[ir.Value, int]]) -> ir.Value:
        # ``values`` pairs each count with its static upper bound (capacity).
        constants = [int(v.constant) for v, _ in values if isinstance(v, ir.Constant)]
        dynamic = [(v, bound) for v, bound in values if self._is_dynamic(v)]
        best_const = max(constants, default=0)
        dynamic = [(v, bound) for v, bound in dynamic if bound > best_const]
        if not dynamic:
            return self._i32(best_const)
        result: ir.Value = dynamic[0][0]
        for value, _bound in dynamic[1:]:
            if value is result:
                continue
            bigger = self.tick_builder.icmp_unsigned(
                ">", value, result, name="live_max_cmp"
            )
            result = self.tick_builder.select(bigger, value, result, name="live_max")
        if constants:
            floor = self._i32(best_const)
            bigger = self.tick_builder.icmp_unsigned(
                ">", floor, result, name="live_max_cmp"
            )
            result = self.tick_builder.select(bigger, floor, result, name="live_max")
        return result

    def _route_in_streams(self, route: AstKernelBindRoute) -> list[str]:
        _, params = self.kernel_signatures.get(route.kernel, (None, ()))
        return [
            arg
            for param, arg in zip(params, route.args)
            if param.modifier == "in" and arg in self.stream_capacities
        ]

    def _dynamic_route_trip(self, route: AstKernelBindRoute) -> ir.Value | None:
        # ``None`` when every input stream is at a compile-time row count, so
        # the route keeps its static trip count (and codegen is unchanged).
        inputs = self._route_in_streams(route)
        if not any(self._is_dynamic(self._stream_live_count(name)) for name in inputs):
            return None
        return self._max_count(
            [
                (self._stream_live_count(name), self.stream_capacities[name])
                for name in inputs
            ]
        )

    def _pad_counts_for(
        self, route: AstKernelBindRoute, trip: ir.Value
    ) -> dict[str, ir.Value]:
        # Inputs that may hold fewer live rows than the route's trip count read
        # as zero past their end (the simulator pads short inputs the same way).
        return {
            name: self._stream_live_count(name)
            for name in self._route_in_streams(route)
            if self._is_dynamic(self._stream_live_count(name))
            and not self._same_count(self._stream_live_count(name), trip)
        }

    def _publish_count(self, stream: str, value: ir.Value) -> None:
        self._live_counts[stream] = value
        if stream not in self.counted_stream_names:
            return
        ptr = self._leaf_ptr("count", stream, (), ir.IntType(32))
        if ptr is not None:
            self.tick_builder.store(value, ptr)

    def _zero_value(self, llvm_type: ir.Type) -> ir.Value:
        if isinstance(llvm_type, ir.VoidType):
            return ir.Constant(ir.IntType(32), 0)
        return ir.Constant(llvm_type, None)

    def _leaf_ptr(
        self,
        kind: str,
        name: str,
        path: tuple[str, ...],
        leaf_type: ir.Type,
        loop_index_reg: ir.Value | None = None,
    ) -> ir.Value | None:
        leaf_key = (kind, name, path)
        if leaf_key not in self.leaf_specs:
            return None
        leaf_offset, leaf_size = self.leaf_specs[leaf_key]
        byte_offset: ir.Value = ir.Constant(ir.IntType(32), leaf_offset)
        if kind in {"stream", "accum"} and loop_index_reg is not None:
            index_type = loop_index_reg.type
            if isinstance(index_type, ir.IntType) and index_type != byte_offset.type:
                byte_offset = ir.Constant(index_type, leaf_offset)
            stride = ir.Constant(byte_offset.type, leaf_size)
            scaled_index = self.tick_builder.mul(
                loop_index_reg,
                stride,
                name=f"{kind}_{_sanitize_symbol(name)}_byte_index",
            )
            byte_offset = self.tick_builder.add(
                byte_offset,
                scaled_index,
                name=f"{kind}_{_sanitize_symbol(name)}_byte_offset",
            )

        # Address leaves with raw byte arithmetic.  The arena struct layout is
        # still useful for ABI/type descriptions, but using it as the GEP base
        # would make LLVM scale offsets by the selected field type instead of
        # treating ``leaf_offset`` as an exact byte position.
        arena_byte_ptr_type = ir.IntType(8).as_pointer()
        bytes_ptr = self.tick_builder.bitcast(
            self.arena_ptr,
            arena_byte_ptr_type,
            name=f"{kind}_{_sanitize_symbol(name)}_arena_bytes",
        )
        leaf_byte_addr = self.tick_builder.gep(
            bytes_ptr,
            [byte_offset],
            name=f"{kind}_{_sanitize_symbol(name)}_{'_'.join(path) if path else 'value'}_byte_ptr",
        )
        typed_ptr = self.tick_builder.bitcast(leaf_byte_addr, leaf_type.as_pointer())
        if typed_ptr.type.pointee != leaf_type:
            return None
        return typed_ptr

    def _load_value(
        self,
        kind: str,
        name: str,
        value_type: ir.Type,
        declared_type: str,
        path: tuple[str, ...] = (),
        element_index: ir.Value | None = None,
    ) -> ir.Value:
        if declared_type in self.struct_fields and isinstance(
            value_type, ir.IdentifiedStructType
        ):
            aggregate = ir.Constant(value_type, ir.Undefined)
            fields = self.struct_fields.get(declared_type, ())
            for index, element_type in enumerate(value_type.elements):
                field = fields[index] if index < len(fields) else None
                child_name = field.name if field is not None else f"field{index}"
                child_type = (
                    _type_name(field.declared_type) if field is not None else "float"
                )
                field_value = self._load_value(
                    kind,
                    name,
                    element_type,
                    child_type,
                    path + (child_name,),
                    element_index,
                )
                aggregate = self.tick_builder.insert_value(
                    aggregate, field_value, index
                )
            return aggregate

        ptr = self._leaf_ptr(kind, name, path, value_type, element_index)
        if ptr is None:
            return self._zero_value(value_type)
        return self.tick_builder.load(ptr, name=f"{kind}_{_sanitize_symbol(name)}_val")

    def _store_value(
        self,
        kind: str,
        name: str,
        value: ir.Value,
        declared_type: str,
        path: tuple[str, ...] = (),
        element_index: ir.Value | None = None,
    ):
        value_type = value.type
        if declared_type in self.struct_fields and isinstance(
            value_type, ir.IdentifiedStructType
        ):
            fields = self.struct_fields.get(declared_type, ())
            for index, element_type in enumerate(value_type.elements):
                part = self.tick_builder.extract_value(value, index)
                field = fields[index] if index < len(fields) else None
                child_name = field.name if field is not None else f"field{index}"
                child_type = (
                    _type_name(field.declared_type) if field is not None else "float"
                )
                self._store_value(
                    kind, name, part, child_type, path + (child_name,), element_index
                )
            return
        ptr = self._leaf_ptr(kind, name, path, value_type, element_index)
        if ptr is None:
            return
        self.tick_builder.store(value, ptr)

    def _load_tick_param(
        self,
        kind: str,
        name: str,
        field_type: ir.Type,
        element_index: ir.Value | None = None,
    ) -> ir.Value:
        declared_type = self.binding_declared_types.get((kind, name), "float")
        return self._load_value(
            kind, name, field_type, declared_type, element_index=element_index
        )

    def _load_tick_param_ptr(
        self,
        kind: str,
        name: str,
        field_type: ir.Type,
        element_index: ir.Value | None = None,
    ) -> ir.Value:
        return self._leaf_ptr(kind, name, (), field_type, element_index)

    def _clamp_i32(self, value: ir.Value, lo: ir.Value, hi: ir.Value) -> ir.Value:
        # A plain scalar smax/smin: LLVM's scalar evolution understands it, so
        # the vectorizer can bound every clamped row index, and instcombine
        # drops the clamp entirely when the trip count never exceeds the
        # capacity.  (A <4 x i32> splat/select/extract formulation used to be
        # emitted here; SCEV cannot see through it, which left per-stage loops
        # scalar with "cannot identify array bounds".)
        above = self.tick_builder.icmp_signed(">", value, lo, name="route_clamp_lo_cmp")
        raised = self.tick_builder.select(above, value, lo, name="route_clamp_lo")
        below = self.tick_builder.icmp_signed(
            "<", raised, hi, name="route_clamp_hi_cmp"
        )
        return self.tick_builder.select(below, raised, hi, name="route_clamp")

    def _dynamic_trip_bound(self, route: AstKernelBindRoute) -> int:
        return max(
            (self.stream_capacities[name] for name in self._route_in_streams(route)),
            default=1,
        )

    def _clamped_stream_index(
        self, name: str, current: ir.Value, kind: str = "stream"
    ) -> ir.Value:
        raw_capacity = (
            self.accum_sizes.get(name, 0)
            if kind == "accum"
            else self.stream_capacities.get(name, 0)
        )
        safe_capacity = max(int(raw_capacity), 1)
        if (
            kind == "stream"
            and self._row_bound
            and self._row_bound[-1] <= safe_capacity
        ):
            # The loop visits fewer rows than this stream holds, so the index
            # is already in range.  With a run-time row count LLVM can't prove
            # that itself, and the clamp would stop it from bounding the loop's
            # addresses (the vectorizer's "cannot identify array bounds").
            return current
        max_index = ir.Constant(ir.IntType(32), safe_capacity - 1)
        return self._clamp_i32(current, ir.Constant(ir.IntType(32), 0), max_index)
