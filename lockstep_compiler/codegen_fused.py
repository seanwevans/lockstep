"""Multi-stage fusion: one vector loop per fused group, including through filters.

Part of the ``Lockstep_Tick`` code generator: ``codegen._TickCodegen`` combines
this mixin with the others and holds the per-tick state they share through
``self`` (the arena pointer, IR builder, arena layout, and lowering caches).
"""

from __future__ import annotations

from llvmlite import ir

from .ast import (
    AstAssignStmt,
    AstExprBinary,
    AstExprCall,
    AstExprCast,
    AstExprLiteral,
    AstExprUnary,
    AstExprVar,
    AstKernelBindRoute,
    AstReturnStmt,
    AstType,
    AstVarDeclStmt,
)
from .codegen_lowerer import CodegenError, _type_name
from .utils import sanitize_symbol as _sanitize_symbol
from .codegen_vector import _FusedVectorLowerer


class FusedGroupMixin:
    """Multi-stage fusion: one vector loop per fused group, including through filters."""

    def _can_vectorize_fused_group(
        self, routes: tuple[AstKernelBindRoute, ...]
    ) -> bool:
        supported_statements = (AstAssignStmt, AstVarDeclStmt)
        supported_exprs = (
            AstExprLiteral,
            AstExprVar,
            AstExprUnary,
            AstExprBinary,
            AstExprCast,
            AstExprCall,
        )

        type_env: dict[str, str] = {}

        def path_type(path: tuple[str, ...]) -> str | None:
            if not path:
                return None
            current_type_name = type_env.get(_sanitize_symbol(path[0]))
            if current_type_name is None:
                return None
            for field_name in path[1:]:
                current_ty = self.lowerer._llvm_type(
                    current_type_name, self.known_structs
                )
                field_info = self._field_index_and_type(current_ty, field_name)
                if field_info is None:
                    return None
                _, _, current_type_name = field_info
            return current_type_name

        def type_vectorizable(type_name: str | AstType | None) -> bool:
            return (
                type_name is not None
                and self._vectorizable_leaf_fields(type_name) is not None
            )

        def scalar_type_vectorizable(type_name: str | AstType | None) -> bool:
            if type_name is None:
                return False
            scalar_ty = self.lowerer._llvm_type(type_name, self.known_structs)
            return self._vector_type_for_scalar(scalar_ty) is not None

        def expr_supported(expr) -> bool:
            if not isinstance(expr, supported_exprs):
                return False
            if isinstance(expr, AstExprLiteral):
                return True
            if isinstance(expr, AstExprVar):
                return scalar_type_vectorizable(path_type(expr.path))
            if isinstance(expr, AstExprUnary):
                return expr_supported(expr.operand)
            if isinstance(expr, AstExprBinary):
                return expr_supported(expr.left) and expr_supported(expr.right)
            if isinstance(expr, AstExprCast):
                return scalar_type_vectorizable(expr.target_type) and expr_supported(
                    expr.value
                )
            if isinstance(expr, AstExprCall):
                return expr.name in {
                    "select",
                    "step",
                    "mix",
                    "min",
                    "max",
                    "clamp",
                    "abs",
                    "sign",
                    "smoothstep",
                    "int",
                    "uint",
                    "float",
                    "double",
                    "bool",
                } and all(expr_supported(arg) for arg in expr.args)
            return False

        for route in routes:
            signature = self.kernel_signatures.get(route.kernel)
            if signature is None:
                return False
            kernel_decl, params = signature
            type_env = {}
            for param in params:
                param_type_name = _type_name(param.declared_type)
                if self._vectorizable_leaf_fields(param_type_name) is None:
                    return False
                # ``accum`` parameters are per-row read-modify-write leaves: the
                # fused loop loads the current accumulator slot vector, runs the
                # kernel body, and stores the result back (see
                # _emit_vector_fused_chunk).  The horizontal ``fold`` reduction
                # over the buffer runs later, unchanged, so fusing an
                # accumulating stage is value-identical to the per-stage loops.
                type_env[_sanitize_symbol(param.name)] = param_type_name
            for statement in kernel_decl.body:
                if (
                    isinstance(statement, AstReturnStmt)
                    and route.kernel in self.filter_names
                ):
                    # A filter's keep flag becomes the fused loop's lane mask.
                    if not expr_supported(statement.value):
                        return False
                    continue
                if not isinstance(statement, supported_statements):
                    return False
                if isinstance(statement, AstAssignStmt):
                    if not scalar_type_vectorizable(
                        path_type(statement.target)
                    ) or not expr_supported(statement.value):
                        return False
                elif isinstance(statement, AstVarDeclStmt):
                    declared = statement.declared_type or AstType("float")
                    declared_name = _type_name(declared)
                    if self._vectorizable_leaf_fields(declared_name) is None:
                        return False
                    if statement.initializer is not None and not expr_supported(
                        statement.initializer
                    ):
                        return False
                    type_env[_sanitize_symbol(statement.name)] = declared_name
        return True

    def _emit_vector_fused_chunk(
        self,
        routes: tuple[AstKernelBindRoute, ...],
        current: ir.Value,
        eliminated_targets: set[str],
        chunk_trip_count: int,
        carried_accums: set[str] | None = None,
        first_drop: int | None = None,
        write_index: ir.Value | None = None,
    ) -> tuple[dict[str, tuple[ir.Value, ir.Value | None]], ir.Value | None]:
        # ``carried_accums`` names accumulators the caller reduces in a
        # loop-carried register instead of the arena buffer: their slot is seeded
        # to the zero identity (never loaded from the buffer), the buffer is never
        # written, and each row's partial vector is returned for the caller to
        # combine, with the lane mask in force when it was written (``None``: all
        # lanes live).
        #
        # From ``first_drop`` (the position of the first filter that can drop
        # rows) on, a filter's ``return`` narrows the chunk's lane mask and the
        # group's sink is compress-stored at ``write_index``.  Returns
        # ``(partials, final_mask)``.
        carried_accums = carried_accums or set()
        carried_partials: dict[str, tuple[ir.Value, ir.Value | None]] = {}
        route_values: dict[str, dict[tuple[str, ...], ir.Value]] = {}
        lane_mask: ir.Value | None = None
        for position, route in enumerate(routes):
            compacting = first_drop is not None and position >= first_drop
            drops_rows = (
                route.kernel in self.filter_names
                and not self._filter_always_keeps(route.kernel)
            )
            mask_before_route = lane_mask
            signature = self.kernel_signatures[route.kernel]
            kernel_decl, params = signature
            vector_lowerer = _FusedVectorLowerer(self)
            out_params: list[tuple[str, str, str]] = []
            accum_params: list[tuple[str, str, str]] = []
            carried_params: list[tuple[str, str, str]] = []
            for index, param in enumerate(params):
                arg_name = route.args[index] if index < len(route.args) else ""
                param_type_name = _type_name(param.declared_type)
                vector_lowerer.define_slot(param.name, param_type_name)
                if param.modifier == "in" and arg_name in route_values:
                    for rel_path, value in route_values[arg_name].items():
                        vector_lowerer.set_slot_leaf(param.name, rel_path, value)
                elif param.modifier == "in" and arg_name in self.stream_slots:
                    for rel_path, value in self._load_stream_binding_vectors(
                        arg_name,
                        param_type_name,
                        current,
                        chunk_trip_count,
                    ).items():
                        vector_lowerer.set_slot_leaf(param.name, rel_path, value)
                elif param.modifier == "uniform" and arg_name in self.uniform_slots:
                    uniform_values = self._vectorizable_leaf_fields(param_type_name)
                    if uniform_values is None:
                        raise CodegenError(
                            f"type '{param_type_name}' cannot be SIMD-vectorized"
                        )
                    for rel_path, (leaf_ty, leaf_type_name) in uniform_values.items():
                        scalar = self._hoisted_uniform_leaves.get(
                            (arg_name, rel_path, str(leaf_ty))
                        ) or self._load_value(
                            "uniform", arg_name, leaf_ty, leaf_type_name, rel_path
                        )
                        vector_lowerer.set_slot_leaf(
                            param.name,
                            rel_path,
                            self._splat_to_vector(
                                scalar,
                                ir.VectorType(leaf_ty, self.simd_width),
                                leaf_type_name,
                            ),
                        )
                elif param.modifier == "out":
                    if arg_name in self.stream_slots:
                        # A compacted sink row starts from the row it will
                        # overwrite, like the per-stage filter path.
                        out_row = (
                            write_index
                            if compacting
                            and write_index is not None
                            and arg_name not in eliminated_targets
                            else current
                        )
                        for rel_path, value in self._load_stream_binding_vectors(
                            arg_name,
                            param_type_name,
                            out_row,
                            chunk_trip_count,
                        ).items():
                            vector_lowerer.set_slot_leaf(param.name, rel_path, value)
                    out_params.append((arg_name, param.name, param_type_name))
                elif param.modifier == "accum" and arg_name in carried_accums:
                    # Register-carried accumulator: leave the slot at its zero
                    # identity (``define_slot`` already zeroed it) so ``acc = acc +
                    # delta`` yields this row's partial, exactly as the standalone
                    # fold-into-kernel path seeds its scratch.  The buffer is
                    # neither read nor written; the partial is returned below.
                    carried_params.append((arg_name, param.name, param_type_name))
                elif param.modifier == "accum" and arg_name in self.accum_slots:
                    # An accumulator slot is a per-row read-modify-write buffer:
                    # seed the vector lowerer with the current window so kernel
                    # bodies that read the accumulator (``acc = acc + delta``)
                    # observe the same value the scalar path would, then store
                    # the updated window back below.
                    for rel_path, value in self._load_stream_binding_vectors(
                        arg_name,
                        param_type_name,
                        current,
                        chunk_trip_count,
                        "accum",
                    ).items():
                        vector_lowerer.set_slot_leaf(param.name, rel_path, value)
                    accum_params.append((arg_name, param.name, param_type_name))
            for statement in kernel_decl.body:
                if isinstance(statement, AstReturnStmt):
                    # Validated as the last statement; a keep-all filter's
                    # ``return true`` needs no mask.
                    if drops_rows:
                        keep = self._coerce_vector_value(
                            vector_lowerer.lower_expr(statement.value),
                            ir.VectorType(ir.IntType(1), self.simd_width),
                            "bool",
                        )
                        lane_mask = (
                            keep
                            if lane_mask is None
                            else self.tick_builder.and_(
                                lane_mask, keep, name="fused_keep"
                            )
                        )
                    break
                vector_lowerer.lower_statement(statement)
            for arg_name, param_name, param_type_name in out_params:
                values = vector_lowerer.slot_leaf_values(param_name)
                if (
                    not values
                    and vector_lowerer._key((param_name,)) in vector_lowerer.values
                ):
                    values = {
                        (): vector_lowerer.values[vector_lowerer._key((param_name,))]
                    }
                if arg_name in eliminated_targets:
                    route_values[arg_name] = values
                elif compacting and lane_mask is not None and write_index is not None:
                    self._compress_store_binding_vectors(
                        arg_name, param_type_name, values, write_index, lane_mask
                    )
                elif arg_name in self.stream_slots:
                    self._store_stream_binding_vectors(
                        arg_name, param_type_name, values, current, chunk_trip_count
                    )
            for arg_name, param_name, param_type_name in accum_params:
                values = vector_lowerer.slot_leaf_values(param_name)
                if (
                    not values
                    and vector_lowerer._key((param_name,)) in vector_lowerer.values
                ):
                    values = {
                        (): vector_lowerer.values[vector_lowerer._key((param_name,))]
                    }
                self._store_stream_binding_vectors(
                    arg_name,
                    param_type_name,
                    values,
                    current,
                    chunk_trip_count,
                    "accum",
                )
            for arg_name, param_name, _param_type_name in carried_params:
                values = vector_lowerer.slot_leaf_values(param_name)
                partial = values.get(())
                if partial is None:
                    root_key = vector_lowerer._key((param_name,))
                    partial = vector_lowerer.values.get(root_key)
                if partial is not None:
                    # Every row a stage processes contributes, including one its
                    # own filter then drops -- so the mask *before* this route.
                    carried_partials[arg_name] = (partial, mask_before_route)
        return carried_partials, lane_mask

    def _filter_always_keeps(self, kernel_name: str) -> bool:
        # A filter keeps every row unconditionally when its body never returns a
        # keep flag: the generated function falls through to ``ret i1 1`` (see
        # ``lower_function``), matching the simulator's ``keep_row`` default. Such
        # a filter is semantically an identity copy -- its compacting store
        # degenerates to a straight store at the read index -- so it can fuse like
        # a shader. A ``return true;`` is treated the same. Any other return makes
        # the keep flag data-dependent, so the filter is *not* an unconditional
        # pass and must keep its scalar compacting loop.
        signature = self.kernel_signatures.get(kernel_name)
        if signature is None:
            return False
        kernel_decl, _ = signature
        for statement in kernel_decl.body:
            if isinstance(statement, AstReturnStmt):
                value = statement.value
                if not (
                    isinstance(value, AstExprLiteral) and str(value.value) == "true"
                ):
                    return False
        return True

    def _group_carry_reductions(
        self,
        routes: tuple[AstKernelBindRoute, ...],
    ) -> tuple[set[str], list[dict]] | None:
        # Decide whether the group's accumulators can be carried in loop-carried
        # registers (no arena buffer).  Returns ``(carried_accum_names,
        # reductions)`` when every accumulator written in the group qualifies, or
        # ``None`` when any does not -- in which case the caller keeps the per-row
        # buffer for the whole group (unchanged behavior).  An accumulator
        # qualifies when it has a single writer, that writer is in this group, its
        # scalar type is reducible, and each fold over it uses sum/min/max on the
        # matching type (``avg`` keeps the buffer -- its divisor is the buffer
        # width).  An accumulator with no fold at all still qualifies: its buffer
        # is write-only dead state, so it is simply dropped.
        group_ids = {id(route) for route in routes}
        group_accums: list[str] = []
        seen: set[str] = set()
        for route in routes:
            _, params = self.kernel_signatures.get(route.kernel, (None, ()))
            for index, param in enumerate(params):
                if param.modifier != "accum":
                    continue
                accum = route.args[index] if index < len(route.args) else ""
                if accum in self.accum_slots and accum not in seen:
                    seen.add(accum)
                    group_accums.append(accum)
        if not group_accums:
            return set(), []
        reductions: list[dict] = []
        for accum in group_accums:
            writers = self._accum_writer_ids.get(accum, [])
            if len(writers) != 1 or writers[0] not in group_ids:
                return None
            accum_type_name = self.binding_declared_types.get(("accum", accum), "float")
            accum_ty = self.lowerer._llvm_type(accum_type_name, self.known_structs)
            if not isinstance(accum_ty, (ir.FloatType, ir.DoubleType, ir.IntType)):
                return None
            for fold_route in self._accum_fold_routes.get(accum, []):
                if fold_route.operator not in {"sum", "min", "max"}:
                    return None
                uniform_type_name = _type_name(fold_route.uniform_type)
                uniform_type = self.lowerer._llvm_type(
                    uniform_type_name, self.known_structs
                )
                if uniform_type != accum_ty:
                    return None
                reductions.append(
                    {
                        "uniform": fold_route.uniform_name,
                        "accum": accum,
                        "op": fold_route.operator,
                        "uty": uniform_type,
                        "utn": uniform_type_name,
                    }
                )
        return set(group_accums), reductions

    def _lower_fused_kernel_group(
        self, routes: tuple[AstKernelBindRoute, ...], group_index: int
    ):
        if not routes:
            return
        trip_count = max(self._kernel_route_trip_count(route) for route in routes)
        # An intermediate that the sink stage overwrites in place (``s = A(..);
        # s = B(s, s)``) is the group's output, not an eliminated temporary: it
        # must still be stored.
        eliminated_targets = {route.target for route in routes[:-1]} - {
            routes[-1].target
        }
        if not eliminated_targets:
            for route in routes:
                self._lower_kernel_route(route)
            return
        # The group's rows: its entry streams (read from the arena rather than
        # forwarded between stages) must all hold the same live row count,
        # since one fused loop walks them in lockstep.  Short inputs that would
        # need zero padding take the per-stage path instead.
        produced_in_group: set[str] = set()
        entry_streams: list[str] = []
        for route in routes:
            for name in self._route_in_streams(route):
                if name not in produced_in_group and name not in entry_streams:
                    entry_streams.append(name)
            produced_in_group.add(route.target)
        entry_counts = [self._stream_live_count(name) for name in entry_streams]
        dynamic_trip = any(self._is_dynamic(count) for count in entry_counts)
        if dynamic_trip and not all(
            self._same_count(count, entry_counts[0]) for count in entry_counts
        ):
            for route in routes:
                self._lower_kernel_route(route)
            return
        trip_value: ir.Value = (
            entry_counts[0] if dynamic_trip else self._i32(trip_count)
        )

        # Filters.  A keep-all filter (no data-dependent ``return``) is an
        # identity copy and fuses like a shader.  A filter that drops rows
        # fuses too: its ``return`` becomes a lane mask, every later stage
        # computes on all lanes but only kept lanes reach the sink (a
        # ``llvm.masked.compressstore`` at a running write index) or a fold (a
        # masked lane contributes the reduction identity).  That needs every
        # accumulator carried in registers, and every stage after the first
        # dropping filter to read only values forwarded from earlier stages:
        # an arena stream there would pair compacted rows with raw row indices.
        drop_positions = [
            position
            for position, route in enumerate(routes)
            if route.kernel in self.filter_names
            and not self._filter_always_keeps(route.kernel)
        ]
        first_drop = drop_positions[0] if drop_positions else None
        if not self._can_vectorize_fused_group(routes):
            for route in routes:
                self._lower_kernel_route(route)
            return

        carry = self._group_carry_reductions(routes)
        carried_accums: set[str] = set() if carry is None else carry[0]
        reductions: list[dict] = [] if carry is None else carry[1]

        if first_drop is not None:
            # Rows stay aligned lane for lane only if, after a dropping filter,
            # a stage reads values produced at or after the most recent dropping
            # filter (which compacted them identically).
            misaligned = False
            for position, route in enumerate(routes):
                last_drop = max(
                    (d for d in drop_positions if d < position), default=None
                )
                if last_drop is None:
                    continue
                for name in self._route_in_streams(route):
                    producers = [p for p in range(position) if routes[p].target == name]
                    if (
                        name not in eliminated_targets
                        or not producers
                        or producers[-1] < last_drop
                    ):
                        misaligned = True
            if carry is None or misaligned:
                for route in routes:
                    self._lower_kernel_route(route)
                return

        self._hoist_uniform_loads(routes, vector_leaves=True)
        if dynamic_trip:
            self._row_bound.append(
                max((self.stream_capacities[name] for name in entry_streams), default=1)
            )
        index_ptr = self.tick_builder.alloca(
            ir.IntType(32), name=f"fused_{group_index}_idx"
        )
        self.tick_builder.store(ir.Constant(ir.IntType(32), 0), index_ptr)
        write_index_ptr = None
        if first_drop is not None:
            write_index_ptr = self.tick_builder.alloca(
                ir.IntType(32), name=f"fused_{group_index}_write_idx"
            )
            self.tick_builder.store(self._i32(0), write_index_ptr)
        if dynamic_trip:
            full_trip_value: ir.Value = self.tick_builder.mul(
                self.tick_builder.udiv(
                    trip_value, self._i32(self.simd_width), name="fused_chunks"
                ),
                self._i32(self.simd_width),
                name="fused_full_trip",
            )
        else:
            full_trip_value = self._i32(
                (trip_count // self.simd_width) * self.simd_width
            )

        # Loop-carried register accumulators for the group's folds: one vector
        # partial per fold (carried across the vector loop) plus a scalar partial
        # (folded across the scalar tail), each seeded to the operator identity.
        for reduction in reductions:
            uniform_type = reduction["uty"]
            vector_ty = ir.VectorType(uniform_type, self.simd_width)
            identity = self._reduction_identity(uniform_type, reduction["op"])
            vec_slot = self.tick_builder.alloca(
                vector_ty, name=f"fused_{group_index}_{reduction['uniform']}_vec"
            )
            self.tick_builder.store(
                ir.Constant(vector_ty, [identity] * self.simd_width), vec_slot
            )
            tail_slot = self.tick_builder.alloca(
                uniform_type, name=f"fused_{group_index}_{reduction['uniform']}_tail"
            )
            self.tick_builder.store(identity, tail_slot)
            reduction["vec"] = vec_slot
            reduction["tail"] = tail_slot

        loop_cond = self.tick.append_basic_block(f"fused_{group_index}_cond")
        loop_body = self.tick.append_basic_block(f"fused_{group_index}_body")
        loop_exit = self.tick.append_basic_block(f"fused_{group_index}_exit")
        self.tick_builder.branch(loop_cond)

        self.tick_builder.position_at_end(loop_cond)
        current = self.tick_builder.load(index_ptr, name="fused_idx")
        cond = self.tick_builder.icmp_signed(
            "<",
            current,
            full_trip_value,
            name="fused_vector_active",
        )
        self.tick_builder.cbranch(cond, loop_body, loop_exit)

        self.tick_builder.position_at_end(loop_body)
        chunk_write_index = (
            self.tick_builder.load(write_index_ptr, name="fused_write_idx")
            if write_index_ptr is not None
            else None
        )
        partials, kept_mask = self._emit_vector_fused_chunk(
            routes,
            current,
            eliminated_targets,
            trip_count,
            carried_accums=carried_accums,
            first_drop=first_drop,
            write_index=chunk_write_index,
        )
        if write_index_ptr is not None and kept_mask is not None:
            kept = self.tick_builder.call(
                self._get_vector_reduce_intrinsic(
                    f"llvm.vector.reduce.add.v{self.simd_width}i32",
                    ir.IntType(32),
                    [ir.VectorType(ir.IntType(32), self.simd_width)],
                ),
                [
                    self.tick_builder.zext(
                        kept_mask,
                        ir.VectorType(ir.IntType(32), self.simd_width),
                        name="fused_kept_lanes",
                    )
                ],
                name="fused_kept",
            )
            self.tick_builder.store(
                self.tick_builder.add(chunk_write_index, kept, name="fused_write_next"),
                write_index_ptr,
            )
        for reduction in reductions:
            partial_and_mask = partials.get(reduction["accum"])
            if partial_and_mask is None:
                continue
            partial, lane_mask = partial_and_mask
            if lane_mask is not None:
                # A dropped lane contributes the operator identity.
                identity = self._reduction_identity(reduction["uty"], reduction["op"])
                partial = self.tick_builder.select(
                    lane_mask,
                    partial,
                    ir.Constant(partial.type, [identity] * self.simd_width),
                    name="fused_carry_masked",
                )
            running = self.tick_builder.load(reduction["vec"], name="fused_carry_cur")
            combined = self._reduction_combine(
                reduction["op"],
                running,
                partial,
                reduction["uty"],
                "fused_carry_next",
            )
            self.tick_builder.store(combined, reduction["vec"])
        next_index = self.tick_builder.add(
            current, ir.Constant(ir.IntType(32), self.simd_width), name="fused_idx_next"
        )
        self.tick_builder.store(next_index, index_ptr)
        self.tick_builder.branch(loop_cond)

        self.tick_builder.position_at_end(loop_exit)

        if (
            dynamic_trip
            or (trip_count // self.simd_width) * self.simd_width < trip_count
        ):
            tail_index_ptr = self.tick_builder.alloca(
                ir.IntType(32), name=f"fused_{group_index}_tail_idx"
            )
            self.tick_builder.store(full_trip_value, tail_index_ptr)
            tail_cond = self.tick.append_basic_block(f"fused_{group_index}_tail_cond")
            tail_body = self.tick.append_basic_block(f"fused_{group_index}_tail_body")
            tail_exit = self.tick.append_basic_block(f"fused_{group_index}_tail_exit")
            self.tick_builder.branch(tail_cond)
            self.tick_builder.position_at_end(tail_cond)
            tail_current = self.tick_builder.load(tail_index_ptr, name="fused_tail_idx")
            tail_active = self.tick_builder.icmp_signed(
                "<",
                tail_current,
                trip_value,
                name="fused_tail_active",
            )
            self.tick_builder.cbranch(tail_active, tail_body, tail_exit)
            self.tick_builder.position_at_end(tail_body)
            # After a dropping filter rejects the row, skip the rest of it.
            tail_row_done = (
                self.tick.append_basic_block(f"fused_{group_index}_tail_row_done")
                if first_drop is not None
                else None
            )
            # Eliminated intermediates produced so far in this row, forwarded to
            # later stages; each producing stage gets a fresh slot (see
            # ``_route_arg_value``).
            local_slots: dict[str, ir.AllocaInstr] = {}
            for position, route in enumerate(routes):
                callee, params = self._kernel_function_and_params(route.kernel)
                if callee is None:
                    continue
                route_out_slots: dict[str, ir.AllocaInstr] = {}
                for index, param in enumerate(callee.args):
                    if index >= len(route.args) or index >= len(params):
                        continue
                    if (
                        params[index].modifier == "out"
                        and route.args[index] in eliminated_targets
                        and route.args[index] not in route_out_slots
                        and hasattr(param.type, "pointee")
                    ):
                        slot_name = _sanitize_symbol(route.args[index])
                        route_out_slots[route.args[index]] = self.tick_builder.alloca(
                            param.type.pointee, name=f"fused_{slot_name}_tail_slot"
                        )
                # Carried accumulators the scalar tail row writes: redirect them
                # to per-row scratch (never the buffer) and fold each partial into
                # the reduction's scalar tail accumulator.
                tail_scratch: dict[str, tuple[ir.AllocaInstr, ir.Type]] = {}
                for index, param in enumerate(params):
                    if param.modifier != "accum":
                        continue
                    accum = route.args[index] if index < len(route.args) else ""
                    if accum not in carried_accums or accum in tail_scratch:
                        continue
                    accum_ty = self.lowerer._llvm_type(
                        self.binding_declared_types.get(("accum", accum), "float"),
                        self.known_structs,
                    )
                    scratch_slot = self.tick_builder.alloca(
                        accum_ty, name=f"fused_{_sanitize_symbol(accum)}_tail_acc"
                    )
                    self.tick_builder.store(ir.Constant(accum_ty, None), scratch_slot)
                    tail_scratch[accum] = (scratch_slot, accum_ty)
                is_sink = position == len(routes) - 1
                row_write_index = None
                if write_index_ptr is not None and is_sink:
                    row_write_index = self.tick_builder.load(
                        write_index_ptr, name="fused_tail_write_idx"
                    )
                keep_value = self._emit_kernel_call(
                    route,
                    tail_current,
                    local_slots=local_slots,
                    local_out_slots=route_out_slots,
                    output_index=row_write_index,
                    accum_overrides={
                        accum: slot for accum, (slot, _ty) in tail_scratch.items()
                    }
                    or None,
                )
                local_slots = {**local_slots, **route_out_slots}
                row_kept: ir.Value | None = None
                if position in drop_positions and keep_value is not None:
                    row_kept = keep_value
                    if not (
                        isinstance(row_kept.type, ir.IntType)
                        and row_kept.type.width == 1
                    ):
                        row_kept = self.lowerer._coerce_value_to_type(
                            row_kept, ir.IntType(1), "bool"
                        )
                for accum, (scratch_slot, _ty) in tail_scratch.items():
                    delta = self.tick_builder.load(
                        scratch_slot, name="fused_tail_acc_val"
                    )
                    for reduction in reductions:
                        if reduction["accum"] != accum:
                            continue
                        current_tail = self.tick_builder.load(
                            reduction["tail"], name="fused_tail_cur"
                        )
                        self.tick_builder.store(
                            self._reduction_combine(
                                reduction["op"],
                                current_tail,
                                delta,
                                reduction["uty"],
                                "fused_tail_next_acc",
                            ),
                            reduction["tail"],
                        )
                if is_sink and row_write_index is not None:
                    # The sink stored this row (a sink filter compacts on its
                    # own keep flag): advance the shared write index.
                    advance = (
                        self.tick_builder.zext(row_kept, ir.IntType(32))
                        if row_kept is not None
                        else self._i32(1)
                    )
                    self.tick_builder.store(
                        self.tick_builder.add(
                            row_write_index, advance, name="fused_tail_write_next"
                        ),
                        write_index_ptr,
                    )
                elif row_kept is not None and tail_row_done is not None:
                    continue_block = self.tick.append_basic_block(
                        f"fused_{group_index}_tail_kept_{position}"
                    )
                    self.tick_builder.cbranch(row_kept, continue_block, tail_row_done)
                    self.tick_builder.position_at_end(continue_block)
            if tail_row_done is not None:
                self.tick_builder.branch(tail_row_done)
                self.tick_builder.position_at_end(tail_row_done)
            tail_next = self.tick_builder.add(
                tail_current, ir.Constant(ir.IntType(32), 1), name="fused_tail_next"
            )
            self.tick_builder.store(tail_next, tail_index_ptr)
            self.tick_builder.branch(tail_cond)
            self.tick_builder.position_at_end(tail_exit)

        # Horizontally reduce each carried vector partial and fold in the scalar
        # tail partial, then publish the scalar for the fold route to consume.
        for reduction in reductions:
            vector_partial = self.tick_builder.load(
                reduction["vec"], name="fused_carry_final_vec"
            )
            reduced = self._horizontal_reduce_vector(
                reduction["op"],
                vector_partial,
                reduction["uty"],
                "fused_carry_reduce",
            )
            tail_partial = self.tick_builder.load(
                reduction["tail"], name="fused_carry_final_tail"
            )
            reduced = self._reduction_combine(
                reduction["op"],
                reduced,
                tail_partial,
                reduction["uty"],
                "fused_carry_final",
            )
            self._fused_group_reductions[reduction["uniform"]] = (
                reduced,
                reduction["utn"],
            )
        self._hoist_uniform_loads(())
        if dynamic_trip:
            self._row_bound.pop()

        sink_count: ir.Value = (
            self.tick_builder.load(write_index_ptr, name=f"fused_{group_index}_kept")
            if write_index_ptr is not None
            else trip_value
        )
        self._publish_count(routes[-1].target, sink_count)
        for route in routes:
            self._note_accum_rows(route, trip_value)
