"""``fold`` lowering: buffer strip-mining, register-carried reductions, and fold-into-kernel fusion.

Part of the ``Lockstep_Tick`` code generator: ``codegen._TickCodegen`` combines
this mixin with the others and holds the per-tick state they share through
``self`` (the arena pointer, IR builder, arena layout, and lowering caches).
"""

from __future__ import annotations

from llvmlite import ir

from .ast import AstFoldBindRoute, AstKernelBindRoute
from .codegen_lowerer import _type_name
from .utils import sanitize_symbol as _sanitize_symbol


class ReductionMixin:
    """``fold`` lowering: buffer strip-mining, register-carried reductions, and fold-into-kernel fusion."""

    def _get_vector_reduce_intrinsic(
        self, name: str, ret_ty: ir.Type, arg_tys: list[ir.Type]
    ) -> ir.Function:
        fn_ty = ir.FunctionType(ret_ty, arg_tys)
        intrinsic = self.module.globals.get(name)
        if intrinsic is None:
            intrinsic = ir.Function(self.module, fn_ty, name=name)
        return intrinsic

    def _average_over_count(
        self, total: ir.Value, count: ir.Value, uniform_type: ir.Type, name: str
    ) -> ir.Value:
        # ``avg`` over a run-time row count; zero rows averages to 0 (the
        # simulator's empty fold) instead of 0/0.
        is_empty = self.tick_builder.icmp_unsigned(
            "==", count, self._i32(0), name=f"{name}_empty"
        )
        safe_count = self.tick_builder.select(
            is_empty, self._i32(1), count, name=f"{name}_n"
        )
        if isinstance(uniform_type, (ir.FloatType, ir.DoubleType)):
            divisor = self.tick_builder.uitofp(
                safe_count, uniform_type, name=f"{name}_nf"
            )
            quotient = self.tick_builder.fdiv(total, divisor, name=name)
        else:
            divisor = self.lowerer._coerce_value_to_type(
                safe_count, uniform_type, "int"
            )
            quotient = self.tick_builder.sdiv(total, divisor, name=name)
        return self.tick_builder.select(
            is_empty, ir.Constant(uniform_type, 0), quotient, name=f"{name}_safe"
        )

    def _reduce_fold_dynamic(
        self, operator: str, source_name: str, uniform_type: ir.Type, count: ir.Value
    ) -> ir.Value:
        # Fold the first ``count`` slots of an accumulator buffer, where
        # ``count`` is only known at run time (the writer ran after a filter).
        # A scalar loop with a ``fast`` combine: LLVM vectorizes it.
        symbol = _sanitize_symbol(source_name)
        acc_slot = self.tick_builder.alloca(uniform_type, name=f"fold_{symbol}_dyn_acc")
        self.tick_builder.store(
            self._reduction_identity(uniform_type, operator), acc_slot
        )
        index_ptr = self.tick_builder.alloca(
            ir.IntType(32), name=f"fold_{symbol}_dyn_idx"
        )
        self.tick_builder.store(self._i32(0), index_ptr)
        loop_cond = self.tick.append_basic_block(f"fold_{symbol}_dyn_cond")
        loop_body = self.tick.append_basic_block(f"fold_{symbol}_dyn_body")
        loop_exit = self.tick.append_basic_block(f"fold_{symbol}_dyn_exit")
        self.tick_builder.branch(loop_cond)
        self.tick_builder.position_at_end(loop_cond)
        current = self.tick_builder.load(index_ptr, name="fold_dyn_idx")
        active = self.tick_builder.icmp_unsigned(
            "<", current, count, name="fold_dyn_active"
        )
        self.tick_builder.cbranch(active, loop_body, loop_exit)
        self.tick_builder.position_at_end(loop_body)
        value = self._load_tick_param("accum", source_name, uniform_type, current)
        running = self.tick_builder.load(acc_slot, name="fold_dyn_cur")
        self.tick_builder.store(
            self._reduction_combine(
                operator, running, value, uniform_type, "fold_dyn_next"
            ),
            acc_slot,
        )
        self.tick_builder.store(
            self.tick_builder.add(current, self._i32(1), name="fold_dyn_idx_next"),
            index_ptr,
        )
        self.tick_builder.branch(loop_cond)
        self.tick_builder.position_at_end(loop_exit)
        reduced: ir.Value = self.tick_builder.load(acc_slot, name="fold_dyn_result")
        if operator == "avg":
            reduced = self._average_over_count(
                reduced, count, uniform_type, "fold_dyn_avg"
            )
        return reduced

    def _reduce_fold(
        self, operator: str, source_name: str, uniform_type: ir.Type
    ) -> ir.Value:
        live_count = self._accum_live_counts.get(source_name)
        if live_count is not None and self._is_dynamic(live_count):
            return self._reduce_fold_dynamic(
                operator, source_name, uniform_type, live_count
            )
        lane_count = max(int(self.accum_sizes.get(source_name, 1)), 1)
        vector_ty = ir.VectorType(uniform_type, self.simd_width)

        # Map (is_float, operator) -> intrinsic suffix for reduction ops.
        is_float = isinstance(uniform_type, (ir.FloatType, ir.DoubleType))
        is_int = isinstance(uniform_type, ir.IntType)
        _REDUCE_INTRINSIC = {
            (True, "sum"): "fadd",
            (True, "avg"): "fadd",
            (True, "min"): "fmin",
            (True, "max"): "fmax",
            (False, "sum"): "add",
            (False, "avg"): "add",
            (False, "min"): "smin",
            (False, "max"): "smax",
        }
        intrinsic_suffix = (
            _REDUCE_INTRINSIC.get((is_float, operator))
            if (is_float or is_int)
            else None
        )
        if intrinsic_suffix is None:
            return self._zero_value(uniform_type)

        def _identity_value() -> ir.Constant:
            if is_float:
                if operator == "min":
                    return ir.Constant(uniform_type, float("inf"))
                if operator == "max":
                    return ir.Constant(uniform_type, float("-inf"))
                return ir.Constant(uniform_type, 0.0)
            if isinstance(uniform_type, ir.IntType):
                if operator == "min":
                    max_signed = (1 << (uniform_type.width - 1)) - 1
                    return ir.Constant(uniform_type, max_signed)
                if operator == "max":
                    min_signed = -(1 << (uniform_type.width - 1))
                    return ir.Constant(uniform_type, min_signed)
                return ir.Constant(uniform_type, 0)
            return ir.Constant(uniform_type, None)

        identity_value = _identity_value()
        vector_accumulator = ir.Constant(vector_ty, [identity_value] * self.simd_width)

        def _load_accum_chunk_vector(chunk_vector_ptr: ir.Value) -> ir.Value:
            # The strip-mined fold consumes a contiguous SIMD-width chunk of the
            # accumulator.  The strip loop carries a vector pointer induction
            # variable, so the loop body performs one uniform vector load and
            # one vector-strided pointer increment instead of rebuilding scalar
            # byte offsets independently for every SIMD lane.
            # The arena is packed, so an accumulator column starts at an
            # arbitrary byte offset: the load must not assume the vector type's
            # natural alignment (which lowers to a faulting ``movaps``).
            chunk = self.tick_builder.load(
                chunk_vector_ptr, name=f"fold_{_sanitize_symbol(source_name)}_chunk"
            )
            chunk.align = 1
            return chunk

        def _insert_accum_chunk_lane(
            vector_value: ir.Value, lane: int, element_index: ir.Value
        ) -> ir.Value:
            lane_value = self._load_tick_param(
                "accum",
                source_name,
                uniform_type,
                element_index,
            )
            return self.tick_builder.insert_element(
                vector_value,
                lane_value,
                ir.Constant(ir.IntType(32), lane),
                name=f"fold_lane_{lane}",
            )

        def _combine_vectors(lhs: ir.Value, rhs: ir.Value, name: str) -> ir.Value:
            if operator in {"sum", "avg"}:
                if is_float:
                    return self.tick_builder.fadd(lhs, rhs, name=name)
                return self.tick_builder.add(lhs, rhs, name=name)
            if operator == "min":
                if is_float:
                    predicate = self.tick_builder.fcmp_ordered(
                        "<", rhs, lhs, name=f"{name}_cmp"
                    )
                else:
                    predicate = self.tick_builder.icmp_signed(
                        "<", rhs, lhs, name=f"{name}_cmp"
                    )
                return self.tick_builder.select(predicate, rhs, lhs, name=name)
            if operator == "max":
                if is_float:
                    predicate = self.tick_builder.fcmp_ordered(
                        ">", rhs, lhs, name=f"{name}_cmp"
                    )
                else:
                    predicate = self.tick_builder.icmp_signed(
                        ">", rhs, lhs, name=f"{name}_cmp"
                    )
                return self.tick_builder.select(predicate, rhs, lhs, name=name)
            return lhs

        # Strip-mine accumulator buffers that are wider than the active hardware
        # vector width.  Each loop iteration folds one vector-sized block into a
        # lane-wise partial accumulator; the horizontal reduction runs only once
        # after the strip loop and any scalar tail have been merged.
        full_chunk_limit = (lane_count // self.simd_width) * self.simd_width
        if full_chunk_limit:
            first_chunk_ptr = self._leaf_ptr("accum", source_name, (), uniform_type)
            if first_chunk_ptr is not None:
                first_chunk_vector_ptr = self.tick_builder.bitcast(
                    first_chunk_ptr,
                    vector_ty.as_pointer(),
                    name=f"fold_{_sanitize_symbol(source_name)}_chunk_ptr",
                )
                preheader_block = self.tick_builder.block
                loop_cond = self.tick.append_basic_block(
                    f"fold_{_sanitize_symbol(source_name)}_strip_cond"
                )
                loop_body = self.tick.append_basic_block(
                    f"fold_{_sanitize_symbol(source_name)}_strip_body"
                )
                loop_exit = self.tick.append_basic_block(
                    f"fold_{_sanitize_symbol(source_name)}_strip_exit"
                )

                self.tick_builder.branch(loop_cond)
                self.tick_builder.position_at_end(loop_cond)
                loop_index = self.tick_builder.phi(ir.IntType(32), name="fold_index")
                loop_chunk_ptr = self.tick_builder.phi(
                    first_chunk_vector_ptr.type, name="fold_chunk_ptr"
                )
                loop_accumulator = self.tick_builder.phi(
                    vector_ty, name="fold_vector_acc"
                )
                loop_index.add_incoming(ir.Constant(ir.IntType(32), 0), preheader_block)
                loop_chunk_ptr.add_incoming(first_chunk_vector_ptr, preheader_block)
                loop_accumulator.add_incoming(vector_accumulator, preheader_block)
                in_full_chunks = self.tick_builder.icmp_unsigned(
                    "<",
                    loop_index,
                    ir.Constant(ir.IntType(32), full_chunk_limit),
                    name="fold_has_full_chunk",
                )
                self.tick_builder.cbranch(in_full_chunks, loop_body, loop_exit)

                self.tick_builder.position_at_end(loop_body)
                chunk_vector = _load_accum_chunk_vector(loop_chunk_ptr)
                next_accumulator = _combine_vectors(
                    loop_accumulator, chunk_vector, "fold_vector_next"
                )
                next_index = self.tick_builder.add(
                    loop_index,
                    ir.Constant(ir.IntType(32), self.simd_width),
                    name="fold_index_next",
                )
                next_chunk_ptr = self.tick_builder.gep(
                    loop_chunk_ptr,
                    [ir.Constant(ir.IntType(32), 1)],
                    name="fold_chunk_ptr_next",
                )
                self.tick_builder.branch(loop_cond)
                loop_index.add_incoming(next_index, self.tick_builder.block)
                loop_chunk_ptr.add_incoming(next_chunk_ptr, self.tick_builder.block)
                loop_accumulator.add_incoming(next_accumulator, self.tick_builder.block)

                self.tick_builder.position_at_end(loop_exit)
                vector_accumulator = loop_accumulator

        tail_count = lane_count - full_chunk_limit
        if tail_count:
            tail_vector = ir.Constant(vector_ty, [identity_value] * self.simd_width)
            for lane in range(tail_count):
                tail_vector = _insert_accum_chunk_lane(
                    tail_vector,
                    lane,
                    ir.Constant(ir.IntType(32), full_chunk_limit + lane),
                )
            vector_accumulator = _combine_vectors(
                vector_accumulator, tail_vector, "fold_tail_acc"
            )

        intrinsic_name = f"llvm.vector.reduce.{intrinsic_suffix}.v{self.simd_width}{uniform_type.intrinsic_name}"
        # fadd requires a starting accumulator argument
        needs_start_value = is_float and operator in {"sum", "avg"}
        if needs_start_value:
            intrinsic = self._get_vector_reduce_intrinsic(
                intrinsic_name, uniform_type, [uniform_type, vector_ty]
            )
            reduced = self.tick_builder.call(
                intrinsic,
                [ir.Constant(uniform_type, 0.0), vector_accumulator],
                name="fold_reduce",
            )
        else:
            intrinsic = self._get_vector_reduce_intrinsic(
                intrinsic_name, uniform_type, [vector_ty]
            )
            reduced = self.tick_builder.call(
                intrinsic, [vector_accumulator], name="fold_reduce"
            )

        if is_float:
            reduced.fastmath.add("fast")

        if operator == "avg":
            if is_float:
                reduced = self.tick_builder.fdiv(
                    reduced,
                    ir.Constant(uniform_type, float(lane_count)),
                    name="fold_avg",
                )
            else:
                reduced = self.tick_builder.sdiv(
                    reduced, ir.Constant(uniform_type, lane_count), name="fold_avg"
                )

        return reduced

    def _horizontal_reduce_vector(
        self, operator: str, vector_value: ir.Value, uniform_type: ir.Type, name: str
    ) -> ir.Value:
        # Collapse a ``<simd_width x T>`` lane-wise partial accumulator to a
        # scalar with the matching ``llvm.vector.reduce.*`` intrinsic -- the same
        # horizontal step ``_reduce_fold`` performs, factored out so the fused
        # group's register-carried accumulator can reuse it without a buffer.
        is_float = isinstance(uniform_type, (ir.FloatType, ir.DoubleType))
        suffix = {
            (True, "sum"): "fadd",
            (True, "avg"): "fadd",
            (True, "min"): "fmin",
            (True, "max"): "fmax",
            (False, "sum"): "add",
            (False, "avg"): "add",
            (False, "min"): "smin",
            (False, "max"): "smax",
        }.get((is_float, operator))
        if suffix is None:
            return self._zero_value(uniform_type)
        vector_ty = ir.VectorType(uniform_type, self.simd_width)
        intrinsic_name = f"llvm.vector.reduce.{suffix}.v{self.simd_width}{uniform_type.intrinsic_name}"
        if is_float and operator in {"sum", "avg"}:
            intrinsic = self._get_vector_reduce_intrinsic(
                intrinsic_name, uniform_type, [uniform_type, vector_ty]
            )
            reduced = self.tick_builder.call(
                intrinsic, [ir.Constant(uniform_type, 0.0), vector_value], name=name
            )
        else:
            intrinsic = self._get_vector_reduce_intrinsic(
                intrinsic_name, uniform_type, [vector_ty]
            )
            reduced = self.tick_builder.call(intrinsic, [vector_value], name=name)
        if is_float:
            reduced.fastmath.add("fast")
        return reduced

    def _lower_fold_route(self, route: AstFoldBindRoute) -> None:
        source_name = route.source
        uniform_name = route.uniform_name
        if (
            source_name not in self.accum_slots
            or uniform_name not in self.uniform_slots
        ):
            return
        # If a fused filter-group already reduced this fold's accumulator in a
        # loop-carried register (keyed per fold uniform, so one accumulator can
        # feed several folds), consume that scalar directly.
        group_reduced = self._fused_group_reductions.get(uniform_name)
        if group_reduced is not None:
            reduced_value, fused_type_name = group_reduced
            self._store_value("uniform", uniform_name, reduced_value, fused_type_name)
            return
        # If the writing kernel route already reduced this accumulator in-register
        # (fold-into-kernel fusion), consume that value instead of re-reducing the
        # per-row buffer -- which the fused route never materialized.
        fused = self._fused_reductions.get(source_name)
        if fused is not None:
            reduced_value, fused_type_name = fused
            self._store_value("uniform", uniform_name, reduced_value, fused_type_name)
            return
        uniform_type_name = _type_name(route.uniform_type)
        uniform_type = self.lowerer._llvm_type(uniform_type_name, self.known_structs)
        reduced = self._reduce_fold(route.operator, source_name, uniform_type)
        self._store_value("uniform", uniform_name, reduced, uniform_type_name)

    # --- Fold-into-kernel fusion -------------------------------------------
    #
    # A per-row ``accum`` buffer that is consumed by exactly one ``fold`` is a
    # pure reduction intermediate: the kernel writes each row's partial and the
    # fold strip-mines the buffer back down to a scalar.  Materializing the whole
    # buffer every tick just to re-read it is the throughput gap against
    # hand-written C (which keeps the reduction in a register).  When an
    # accumulator qualifies, the writing route keeps the same scalar per-row loop
    # (which clang auto-vectorizes into contiguous vector loads/stores) but folds
    # each row's partial into a loop-carried *local* register accumulator --
    # ``fadd fast`` for sum/avg so LLVM reassociates it into a vector reduction,
    # ``fcmp``+``select`` for min/max -- and never touches the O(rows) buffer.  The
    # per-row partial comes from a scratch slot seeded to zero (the arena's
    # zero-init value), so the fused scalar equals a reduction over the same
    # per-tick contributions the buffer path folds -- the linear "consumed by a
    # fold" contract; ``tests/test_fold_reduction_fusion.py`` checks it against
    # both the strip-mine path and an independent host sum.  The arena still
    # reserves the buffer (ABI unchanged); it is simply left untouched.  Only
    # standalone kernel routes take this path; fused multi-stage groups still
    # materialize the buffer.
    def _reduction_identity(self, uniform_type: ir.Type, operator: str) -> ir.Constant:
        is_float = isinstance(uniform_type, (ir.FloatType, ir.DoubleType))
        if is_float:
            if operator == "min":
                return ir.Constant(uniform_type, float("inf"))
            if operator == "max":
                return ir.Constant(uniform_type, float("-inf"))
            return ir.Constant(uniform_type, 0.0)
        if isinstance(uniform_type, ir.IntType):
            if operator == "min":
                return ir.Constant(uniform_type, (1 << (uniform_type.width - 1)) - 1)
            if operator == "max":
                return ir.Constant(uniform_type, -(1 << (uniform_type.width - 1)))
            return ir.Constant(uniform_type, 0)
        return ir.Constant(uniform_type, None)

    def _reduction_combine(
        self,
        operator: str,
        lhs: ir.Value,
        rhs: ir.Value,
        uniform_type: ir.Type,
        name: str,
    ) -> ir.Value:
        # Scalar fold step run once per row into the loop-carried accumulator.
        # For float sum/avg the add carries ``fast`` so LLVM's loop vectorizer is
        # free to reassociate it into a vector reduction -- exactly what lets the
        # whole loop stay vectorized without the per-row arena buffer.
        is_float = isinstance(uniform_type, (ir.FloatType, ir.DoubleType))
        if operator in {"sum", "avg"}:
            if is_float:
                return self.tick_builder.fadd(lhs, rhs, name=name, flags=["fast"])
            return self.tick_builder.add(lhs, rhs, name=name)
        if operator in {"min", "max"}:
            cmp_op = "<" if operator == "min" else ">"
            if is_float:
                predicate = self.tick_builder.fcmp_ordered(
                    cmp_op, rhs, lhs, name=f"{name}_cmp"
                )
            else:
                predicate = self.tick_builder.icmp_signed(
                    cmp_op, rhs, lhs, name=f"{name}_cmp"
                )
            return self.tick_builder.select(predicate, rhs, lhs, name=name)
        return lhs

    def _route_reduction_fusible(self, route: AstKernelBindRoute) -> bool:
        if not isinstance(route, AstKernelBindRoute):
            return False
        if route.kernel in self.filter_names:
            return False
        signature = self.kernel_signatures.get(route.kernel)
        if signature is None:
            return False
        _, params = signature
        accum_args = [
            (index, param)
            for index, param in enumerate(params)
            if param.modifier == "accum"
        ]
        if not accum_args:
            return False
        for index, _param in accum_args:
            arg_name = route.args[index] if index < len(route.args) else ""
            if arg_name not in self._reducible_accums:
                return False
            if self._reducible_writer.get(arg_name) != id(route):
                return False
        return True

    def _lower_reduction_route(self, route: AstKernelBindRoute) -> None:
        # Same scalar per-row loop as ``_lower_kernel_route`` (which clang
        # auto-vectorizes into contiguous vector loads/stores), but the
        # accumulator is redirected from the O(rows) arena buffer to a per-row
        # scratch slot that is folded into a loop-carried register accumulator.
        # LLVM promotes both allocas out of memory, so the buffer traffic
        # disappears and the fold reduces in-register -- matching hand-written C.
        _, params = self.kernel_signatures[route.kernel]
        dynamic_trip = self._dynamic_route_trip(route)
        trip_value: ir.Value = (
            dynamic_trip
            if dynamic_trip is not None
            else self._i32(self._kernel_route_trip_count(route))
        )
        pad_counts = self._pad_counts_for(route, trip_value) if dynamic_trip else None
        kernel_symbol = _sanitize_symbol(route.kernel)
        if dynamic_trip is not None:
            self._row_bound.append(self._dynamic_trip_bound(route))

        # accum_name -> (acc_slot, operator, uniform_type, uniform_type_name)
        reductions: dict[str, tuple[ir.AllocaInstr, str, ir.Type, str]] = {}
        for index, param in enumerate(params):
            if param.modifier != "accum":
                continue
            accum_name = route.args[index] if index < len(route.args) else ""
            info = self._reducible_accums.get(accum_name)
            if info is None:
                continue
            operator, uniform_type_name = info
            uniform_type = self.lowerer._llvm_type(
                uniform_type_name, self.known_structs
            )
            acc_slot = self.tick_builder.alloca(
                uniform_type, name=f"reduce_{_sanitize_symbol(accum_name)}_acc"
            )
            self.tick_builder.store(
                self._reduction_identity(uniform_type, operator), acc_slot
            )
            reductions[accum_name] = (
                acc_slot,
                operator,
                uniform_type,
                uniform_type_name,
            )

        self._hoist_uniform_loads([route])
        index_ptr = self.tick_builder.alloca(
            ir.IntType(32), name=f"reduce_{kernel_symbol}_idx"
        )
        self.tick_builder.store(ir.Constant(ir.IntType(32), 0), index_ptr)
        loop_cond = self.tick.append_basic_block(f"reduce_{kernel_symbol}_cond")
        loop_body = self.tick.append_basic_block(f"reduce_{kernel_symbol}_body")
        loop_exit = self.tick.append_basic_block(f"reduce_{kernel_symbol}_exit")
        self.tick_builder.branch(loop_cond)

        self.tick_builder.position_at_end(loop_cond)
        current = self.tick_builder.load(index_ptr, name="reduce_idx")
        cond = self.tick_builder.icmp_signed(
            "<", current, trip_value, name="reduce_active"
        )
        self.tick_builder.cbranch(cond, loop_body, loop_exit)

        self.tick_builder.position_at_end(loop_body)
        # A fresh per-row scratch slot per accumulator, seeded to zero (the
        # arena's zero-init value) so the kernel body yields this row's partial.
        scratch: dict[str, ir.Value] = {}
        for accum_name, (_slot, _op, uniform_type, _utn) in reductions.items():
            row_slot = self.tick_builder.alloca(
                uniform_type, name=f"reduce_{_sanitize_symbol(accum_name)}_row"
            )
            self.tick_builder.store(ir.Constant(uniform_type, None), row_slot)
            scratch[accum_name] = row_slot
        self._emit_kernel_call(
            route, current, accum_overrides=scratch, pad_counts=pad_counts
        )
        for accum_name, (acc_slot, operator, uniform_type, _utn) in reductions.items():
            delta = self.tick_builder.load(
                scratch[accum_name],
                name=f"reduce_{_sanitize_symbol(accum_name)}_row_val",
            )
            current_acc = self.tick_builder.load(
                acc_slot, name=f"reduce_{_sanitize_symbol(accum_name)}_cur"
            )
            combined = self._reduction_combine(
                operator,
                current_acc,
                delta,
                uniform_type,
                f"reduce_{_sanitize_symbol(accum_name)}_next",
            )
            self.tick_builder.store(combined, acc_slot)
        next_index = self.tick_builder.add(
            current, ir.Constant(ir.IntType(32), 1), name="reduce_idx_next"
        )
        self.tick_builder.store(next_index, index_ptr)
        self.tick_builder.branch(loop_cond)

        self.tick_builder.position_at_end(loop_exit)
        for accum_name, (
            acc_slot,
            operator,
            uniform_type,
            uniform_type_name,
        ) in reductions.items():
            reduced: ir.Value = self.tick_builder.load(
                acc_slot, name=f"reduce_{_sanitize_symbol(accum_name)}_final"
            )
            if operator == "avg":
                if dynamic_trip is not None:
                    reduced = self._average_over_count(
                        reduced,
                        trip_value,
                        uniform_type,
                        f"reduce_{_sanitize_symbol(accum_name)}_avg",
                    )
                else:
                    lane_count = max(int(self.accum_sizes.get(accum_name, 1)), 1)
                    if isinstance(uniform_type, (ir.FloatType, ir.DoubleType)):
                        reduced = self.tick_builder.fdiv(
                            reduced,
                            ir.Constant(uniform_type, float(lane_count)),
                            name=f"reduce_{_sanitize_symbol(accum_name)}_avg",
                        )
                    else:
                        reduced = self.tick_builder.sdiv(
                            reduced,
                            ir.Constant(uniform_type, lane_count),
                            name=f"reduce_{_sanitize_symbol(accum_name)}_avg",
                        )
            self._fused_reductions[accum_name] = (reduced, uniform_type_name)
        self._hoist_uniform_loads(())
        if dynamic_trip is not None:
            self._row_bound.pop()
        self._finish_route_counts(route, trip_value, trip_value)
