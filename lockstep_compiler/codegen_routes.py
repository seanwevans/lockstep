"""Per-stage lowering of one kernel bind route: a scalar row loop that calls the kernel.

Part of the ``Lockstep_Tick`` code generator: ``codegen._TickCodegen`` combines
this mixin with the others and holds the per-tick state they share through
``self`` (the arena pointer, IR builder, arena layout, and lowering caches).
"""

from __future__ import annotations

from llvmlite import ir

from .ast import AstKernelBindRoute, AstKernelParam
from .codegen_lowerer import CodegenError
from .utils import sanitize_symbol as _sanitize_symbol


class RouteLoweringMixin:
    """Per-stage lowering of one kernel bind route: a scalar row loop that calls the kernel."""

    def _kernel_function_and_params(
        self,
        kernel_name: str,
    ) -> tuple[ir.Function | None, tuple[AstKernelParam, ...]]:
        callee = self.function_map.get(
            f"shader_{_sanitize_symbol(kernel_name)}"
        ) or self.function_map.get(f"filter_{_sanitize_symbol(kernel_name)}")
        signature = self.kernel_signatures.get(kernel_name)
        return callee, signature[1] if signature is not None else ()

    def _kernel_route_trip_count(self, route: AstKernelBindRoute) -> int:
        callee, params = self._kernel_function_and_params(route.kernel)
        if callee is None:
            raise CodegenError(
                f"undefined shader/filter '{route.kernel}' in bind route: {route.route}"
            )
        trip_count = 0
        for index, arg_name in enumerate(route.args):
            if index >= len(params):
                break
            if params[index].modifier == "in" and arg_name in self.stream_capacities:
                trip_count = max(trip_count, self.stream_capacities[arg_name])
        if route.target in self.stream_capacities:
            trip_count = max(trip_count, self.stream_capacities[route.target])
        return max(trip_count, 1)

    def _route_arg_value(
        self,
        *,
        arg_name: str,
        param: ir.Argument,
        modifier: str | None,
        current: ir.Value,
        local_slots: dict[str, ir.AllocaInstr] | None = None,
        local_out_slots: dict[str, ir.AllocaInstr] | None = None,
        accum_overrides: dict[str, ir.Value] | None = None,
        out_index: ir.Value | None = None,
        pad_counts: dict[str, ir.Value] | None = None,
    ) -> ir.Value:
        # ``local_slots`` holds values forwarded *into* this route by earlier
        # stages of a fused group; ``local_out_slots`` receives this route's own
        # eliminated outputs.  They are separate because an in-place stage
        # (``s = K(s, s)``) reads the previous ``s`` while producing a new one.
        local_slots = local_slots or {}
        if local_out_slots is None:
            local_out_slots = local_slots
        if modifier == "in" and arg_name in local_slots:
            return self.tick_builder.load(
                local_slots[arg_name], name=f"fused_{_sanitize_symbol(arg_name)}"
            )
        if modifier == "out" and arg_name in local_out_slots:
            return local_out_slots[arg_name]
        if modifier in {"in", "out"} and arg_name in self.stream_slots:
            if modifier == "out":
                # ``out_index`` is the compacted write position when an earlier
                # filter in a fused group dropped rows.
                clamped_index = self._clamped_stream_index(
                    arg_name, current if out_index is None else out_index
                )
                return self._load_tick_param_ptr(
                    "stream", arg_name, param.type.pointee, clamped_index
                )
            clamped_index = self._clamped_stream_index(arg_name, current)
            value = self._load_tick_param("stream", arg_name, param.type, clamped_index)
            if pad_counts and arg_name in pad_counts:
                live = self.tick_builder.icmp_unsigned(
                    "<", current, pad_counts[arg_name], name="in_row_live"
                )
                value = self.tick_builder.select(
                    live, value, self._zero_value(param.type), name="in_row_padded"
                )
            return value
        if modifier == "accum" and accum_overrides and arg_name in accum_overrides:
            # Fold-into-kernel fusion: point the accumulator at a per-row scratch
            # slot instead of the O(rows) arena buffer.  The caller reads the
            # scratch back and folds it into a register accumulator.
            return accum_overrides[arg_name]
        if modifier == "accum" and arg_name in self.accum_slots:
            return self._load_tick_param_ptr(
                "accum", arg_name, param.type.pointee, current
            )
        if modifier == "uniform" and arg_name in self.uniform_slots:
            hoisted = self._hoisted_uniforms.get((arg_name, str(param.type)))
            if hoisted is not None:
                return hoisted
            return self._load_tick_param("uniform", arg_name, param.type)
        return self._zero_value(param.type)

    def _emit_kernel_call(
        self,
        route: AstKernelBindRoute,
        current: ir.Value,
        *,
        local_slots: dict[str, ir.AllocaInstr] | None = None,
        local_out_slots: dict[str, ir.AllocaInstr] | None = None,
        output_index: ir.Value | None = None,
        accum_overrides: dict[str, ir.Value] | None = None,
        pad_counts: dict[str, ir.Value] | None = None,
    ) -> ir.Value | None:
        callee, params = self._kernel_function_and_params(route.kernel)
        out_slots = local_slots if local_out_slots is None else local_out_slots
        if callee is None:
            return None
        store_index = output_index if output_index is not None else current
        call_args = []
        copy_out_slots: list[tuple[str, ir.AllocaInstr, str]] = []
        for index, param in enumerate(callee.args):
            arg_name = route.args[index] if index < len(route.args) else ""
            modifier = params[index].modifier if index < len(params) else None
            # A filter's out stream is normally written through its compacting
            # store (below).  But when that out is an eliminated intermediate
            # forwarded through a fused group's scalar tail (``local_slots``),
            # the value stays in a register for the next stage -- there is no
            # stream to compact into -- so route it like an ordinary out param.
            is_filter_output = (
                route.kernel in self.filter_names
                and modifier == "out"
                and arg_name in self.stream_slots
                and not (out_slots and arg_name in out_slots)
            )
            call_arg = (
                None
                if is_filter_output
                else self._route_arg_value(
                    arg_name=arg_name,
                    param=param,
                    modifier=modifier,
                    current=current,
                    local_slots=local_slots,
                    local_out_slots=local_out_slots,
                    accum_overrides=accum_overrides,
                    out_index=output_index,
                    pad_counts=pad_counts,
                )
            )
            if call_arg is None and modifier == "out" and arg_name in self.stream_slots:
                if not hasattr(param.type, "pointee"):
                    raise CodegenError(
                        f"output parameter '{param.name}' for route '{route.route}' is not a pointer"
                    )
                declared_type = self.binding_declared_types.get(
                    ("stream", arg_name), "float"
                )
                slot = self.tick_builder.alloca(
                    param.type.pointee,
                    name=f"route_{_sanitize_symbol(arg_name)}_out_slot",
                )
                initial_value = self._load_tick_param(
                    "stream", arg_name, param.type.pointee, store_index
                )
                self.tick_builder.store(initial_value, slot)
                call_arg = slot
                copy_out_slots.append((arg_name, slot, declared_type))
            if call_arg is None:
                raise CodegenError(
                    f"could not lower argument '{arg_name}' for route '{route.route}'"
                )
            call_args.append(call_arg)
        result = self.tick_builder.call(callee, call_args)
        if route.kernel in self.filter_names and copy_out_slots:
            keep_bool = result
            if not (
                isinstance(keep_bool.type, ir.IntType) and keep_bool.type.width == 1
            ):
                keep_bool = self.lowerer._coerce_value_to_type(
                    keep_bool, ir.IntType(1), "bool"
                )
            store_block = self.tick.append_basic_block(
                f"filter_{_sanitize_symbol(route.kernel)}_store"
            )
            skip_block = self.tick.append_basic_block(
                f"filter_{_sanitize_symbol(route.kernel)}_skip"
            )
            after_block = self.tick.append_basic_block(
                f"filter_{_sanitize_symbol(route.kernel)}_after"
            )
            self.tick_builder.cbranch(keep_bool, store_block, skip_block)
            self.tick_builder.position_at_end(store_block)
            for arg_name, slot, declared_type in copy_out_slots:
                updated_value = self.tick_builder.load(
                    slot, name=f"route_{_sanitize_symbol(arg_name)}_out_value"
                )
                self._store_value(
                    "stream",
                    arg_name,
                    updated_value,
                    declared_type,
                    element_index=store_index,
                )
            self.tick_builder.branch(after_block)
            self.tick_builder.position_at_end(skip_block)
            self.tick_builder.branch(after_block)
            self.tick_builder.position_at_end(after_block)
            return result
        for arg_name, slot, declared_type in copy_out_slots:
            updated_value = self.tick_builder.load(
                slot, name=f"route_{_sanitize_symbol(arg_name)}_out_value"
            )
            self._store_value(
                "stream",
                arg_name,
                updated_value,
                declared_type,
                element_index=store_index,
            )
        return result

    def _lower_kernel_route(
        self, route: AstKernelBindRoute, *, allow_reduction_fusion: bool = False
    ):
        # Reduction fusion only applies to a standalone kernel route (dispatched
        # on its own), never to a route lowered as part of a fused group's
        # per-stage fallback -- there the per-row accumulator buffer is still the
        # interface between stages and the differential fusion tests rely on it.
        if allow_reduction_fusion and self._route_reduction_fusible(route):
            self._lower_reduction_route(route)
            return
        dynamic_trip = self._dynamic_route_trip(route)
        trip_value: ir.Value = (
            dynamic_trip
            if dynamic_trip is not None
            else self._i32(self._kernel_route_trip_count(route))
        )
        pad_counts = self._pad_counts_for(route, trip_value) if dynamic_trip else None
        kernel_name = route.kernel
        if dynamic_trip is not None:
            self._row_bound.append(self._dynamic_trip_bound(route))
        self._hoist_uniform_loads([route])

        index_ptr = self.tick_builder.alloca(
            ir.IntType(32), name=f"{_sanitize_symbol(kernel_name)}_idx"
        )
        self.tick_builder.store(ir.Constant(ir.IntType(32), 0), index_ptr)
        write_index_ptr = None
        if kernel_name in self.filter_names:
            write_index_ptr = self.tick_builder.alloca(
                ir.IntType(32), name=f"{_sanitize_symbol(kernel_name)}_write_idx"
            )
            self.tick_builder.store(ir.Constant(ir.IntType(32), 0), write_index_ptr)

        loop_cond = self.tick.append_basic_block(
            f"route_{_sanitize_symbol(kernel_name)}_cond"
        )
        loop_body = self.tick.append_basic_block(
            f"route_{_sanitize_symbol(kernel_name)}_body"
        )
        loop_exit = self.tick.append_basic_block(
            f"route_{_sanitize_symbol(kernel_name)}_exit"
        )
        self.tick_builder.branch(loop_cond)

        self.tick_builder.position_at_end(loop_cond)
        current = self.tick_builder.load(index_ptr, name="idx")
        cond = self.tick_builder.icmp_signed(
            "<", current, trip_value, name="route_active"
        )
        self.tick_builder.cbranch(cond, loop_body, loop_exit)

        self.tick_builder.position_at_end(loop_body)
        output_index = current
        if write_index_ptr is not None:
            output_index = self.tick_builder.load(
                write_index_ptr, name="filter_write_idx"
            )
        keep_value = self._emit_kernel_call(
            route, current, output_index=output_index, pad_counts=pad_counts
        )
        if write_index_ptr is not None:
            keep_bool = keep_value
            if keep_bool is None:
                keep_bool = ir.Constant(ir.IntType(1), 1)
            if not (
                isinstance(keep_bool.type, ir.IntType) and keep_bool.type.width == 1
            ):
                keep_bool = self.lowerer._coerce_value_to_type(
                    keep_bool, ir.IntType(1), "bool"
                )
            write_next = self.tick_builder.add(
                output_index, ir.Constant(ir.IntType(32), 1), name="filter_write_next"
            )
            selected_write = self.tick_builder.select(
                keep_bool, write_next, output_index, name="filter_write_select"
            )
            self.tick_builder.store(selected_write, write_index_ptr)
        next_index = self.tick_builder.add(
            current, ir.Constant(ir.IntType(32), 1), name="idx_next"
        )
        self.tick_builder.store(next_index, index_ptr)
        self.tick_builder.branch(loop_cond)

        self.tick_builder.position_at_end(loop_exit)
        self._hoist_uniform_loads(())
        if dynamic_trip is not None:
            self._row_bound.pop()
        self._finish_route_counts(
            route,
            trip_value,
            (
                self.tick_builder.load(write_index_ptr, name="filter_kept")
                if write_index_ptr is not None
                and not self._filter_always_keeps(kernel_name)
                else trip_value
            ),
        )

    def _finish_route_counts(
        self, route: AstKernelBindRoute, trip_value: ir.Value, output_count: ir.Value
    ) -> None:
        # Record how many rows the route produced and how many accumulator
        # slots it filled (every processed row writes its accumulators, even a
        # row a filter then drops).
        self._publish_count(route.target, output_count)
        self._note_accum_rows(route, trip_value)

    def _note_accum_rows(self, route: AstKernelBindRoute, trip_value: ir.Value) -> None:
        # With several writers, the buffer holds the longest writer's rows.
        _, params = self.kernel_signatures.get(route.kernel, (None, ()))
        for param, arg in zip(params, route.args):
            if param.modifier != "accum" or arg not in self.accum_slots:
                continue
            bound = max(int(self.accum_sizes.get(arg, 1)), 1)
            previous = self._accum_live_counts.get(arg)
            self._accum_live_counts[arg] = (
                trip_value
                if previous is None
                else self._max_count([(previous, bound), (trip_value, bound)])
            )
