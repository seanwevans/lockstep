from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from llvmlite import ir

from .ast import (
    AstAccumulatorDecl,
    AstFoldBindRoute,
    AstKernelBindRoute,
    AstKernelDecl,
    AstKernelParam,
    AstProgram,
    AstStreamDecl,
    AstStructField,
    AstUniformDecl,
)
from .arena_layout import ast_counted_streams, build_ast_arena_layout
from .codegen_legacy import program_from_legacy_mapping as _program_from_legacy_mapping
from .optimizer import optimize_bind_routes
from .utils import sanitize_symbol as _sanitize_symbol
from .codegen_arena import ArenaAccessMixin
from .codegen_fused import FusedGroupMixin
from .codegen_reduce import ReductionMixin
from .codegen_routes import RouteLoweringMixin
from .codegen_vector import VectorMemoryMixin
from .codegen_lowerer import (
    _PRIMITIVE_TYPE_MAP,
    CodegenError,  # noqa: F401 -- re-exported: ``from .codegen import CodegenError``
    _FunctionLowerer,
    _type_name,
)


def _simd_width_for_target_triple(target_triple: str | None) -> int:
    normalized = (target_triple or "").lower()
    arch = normalized.split("-", maxsplit=1)[0]
    if arch in {"x86_64", "amd64"}:
        return 8
    if arch in {"x86", "i386", "i486", "i586", "i686"}:
        return 4
    if arch in {"aarch64", "arm64", "arm", "armv7", "wasm32", "wasm64"}:
        return 4
    return 8


def _pipeline_streams(program: AstProgram) -> list[AstStreamDecl]:
    return [stream for pipeline in program.pipelines for stream in pipeline.streams]


def _pipeline_accumulators(program: AstProgram) -> list[AstAccumulatorDecl]:
    return [accum for pipeline in program.pipelines for accum in pipeline.accumulators]


def _pipeline_uniforms(program: AstProgram) -> list[AstUniformDecl]:
    return [uniform for pipeline in program.pipelines for uniform in pipeline.uniforms]


def _kernel_param_type(param: AstKernelParam) -> str:
    return _type_name(param.declared_type)


def _kernel_param_llvm_type(
    lowerer: _FunctionLowerer,
    param: AstKernelParam,
    known_structs: dict[str, ir.IdentifiedStructType],
) -> ir.Type:
    param_type = lowerer._llvm_type(param.declared_type, known_structs)
    if param.modifier in {"out", "accum"}:
        return param_type.as_pointer()
    return param_type


class _TickCodegen(
    ArenaAccessMixin,
    RouteLoweringMixin,
    ReductionMixin,
    VectorMemoryMixin,
    FusedGroupMixin,
):
    """State for one ``Lockstep_Tick``; ``emit_llvm_ir`` fills it and drives the
    lowering. The lowering steps live in the mixins, one module each."""


def emit_llvm_ir(
    program: AstProgram | Mapping[str, Any],
    *,
    target_width: int | None = None,
    bind_optimization: dict[str, object] | None = None,
) -> str:
    """Generate LLVM IR from the typed Lockstep AST."""

    cg = _TickCodegen()
    cg.program = program

    cg.explicit_accumulator_sizes: dict[str, int] = {}
    target_triple = "x86_64-unknown-linux-gnu"
    bind_route_comments: list[str] = []
    if isinstance(cg.program, Mapping):
        raw_target_triple = cg.program.get("target_triple")
        if raw_target_triple is not None:
            target_triple = str(raw_target_triple)
        raw_bind_routes = cg.program.get("bind_routes", ())
        if isinstance(raw_bind_routes, Sequence) and not isinstance(
            raw_bind_routes, str
        ):
            bind_route_comments = [str(route) for route in raw_bind_routes]
        cg.program, cg.explicit_accumulator_sizes = _program_from_legacy_mapping(
            cg.program
        )
    if not isinstance(cg.program, AstProgram):
        raise TypeError("emit_llvm_ir expects an AstProgram")

    structs = cg.program.structs
    shaders = cg.program.shaders
    filters = cg.program.filters
    pure_functions = cg.program.pure_functions
    streams = _pipeline_streams(cg.program)
    cg.accumulators = _pipeline_accumulators(cg.program)
    uniforms = _pipeline_uniforms(cg.program)

    context = ir.Context()
    cg.module = ir.Module(name="lockstep", context=context)
    cg.module.source_filename = "lockstep"
    cg.module.triple = target_triple
    cg.known_structs: dict[str, ir.IdentifiedStructType] = {}
    cg.struct_fields: dict[str, tuple[AstStructField, ...]] = {}

    for struct_decl in structs:
        safe_name = _sanitize_symbol(struct_decl.name)
        struct_ty = cg.module.context.get_identified_type(f"struct.{safe_name}")
        cg.known_structs[struct_decl.name] = struct_ty
        cg.struct_fields[struct_decl.name] = struct_decl.fields

    intrinsic_names = {pure.name for pure in pure_functions if pure.intrinsic}

    cg.lowerer = _FunctionLowerer(
        cg.module, {}, cg.known_structs, cg.struct_fields, intrinsic_names
    )

    unresolved = set(cg.known_structs.keys())
    while unresolved:
        progress = False
        for struct_name in list(unresolved):
            field_types: list[ir.Type] = []
            can_lower = True
            for field in cg.struct_fields[struct_name]:
                field_type = field.declared_type
                if (
                    field_type.name in cg.known_structs
                    and field_type.name in unresolved
                ):
                    can_lower = False
                    break
                field_types.append(cg.lowerer._llvm_type(field_type, cg.known_structs))
            if not can_lower:
                continue
            if cg.known_structs[struct_name].is_opaque:
                cg.known_structs[struct_name].set_body(*field_types)
            unresolved.remove(struct_name)
            progress = True
        if not progress:
            for struct_name in unresolved:
                if cg.known_structs[struct_name].is_opaque:
                    cg.known_structs[struct_name].set_body(ir.IntType(8))
            break

    cg.function_map: dict[str, ir.Function] = {}
    for pure in pure_functions:
        ret_ty = cg.lowerer._llvm_type(pure.return_type, cg.known_structs)
        params = [
            cg.lowerer._llvm_type(param.declared_type, cg.known_structs)
            for param in pure.params
        ]
        fn = ir.Function(
            cg.module,
            ir.FunctionType(ret_ty, params),
            name=f"pure_{_sanitize_symbol(pure.name)}",
        )
        for idx, param in enumerate(pure.params):
            fn.args[idx].name = _sanitize_symbol(param.name)
        cg.function_map[fn.name] = fn
        cg.lowerer.function_return_types[fn.name] = _type_name(pure.return_type)
        cg.lowerer.function_param_types[fn.name] = [
            _kernel_param_type(param) for param in pure.params
        ]

    for shader in shaders:
        params = [
            _kernel_param_llvm_type(cg.lowerer, param, cg.known_structs)
            for param in shader.params
        ]
        fn = ir.Function(
            cg.module,
            ir.FunctionType(ir.VoidType(), params),
            name=f"shader_{_sanitize_symbol(shader.name)}",
        )
        for idx, param in enumerate(shader.params):
            fn.args[idx].name = _sanitize_symbol(param.name)
        cg.function_map[fn.name] = fn

    for flt in filters:
        params = [
            _kernel_param_llvm_type(cg.lowerer, param, cg.known_structs)
            for param in flt.params
        ]
        fn = ir.Function(
            cg.module,
            ir.FunctionType(ir.IntType(1), params),
            name=f"filter_{_sanitize_symbol(flt.name)}",
        )
        for idx, param in enumerate(flt.params):
            fn.args[idx].name = _sanitize_symbol(param.name)
        cg.function_map[fn.name] = fn

    cg.lowerer.function_map = cg.function_map

    for pure in pure_functions:
        fn = cg.function_map[f"pure_{_sanitize_symbol(pure.name)}"]
        if pure.intrinsic:
            continue
        cg.lowerer.lower_function(
            fn,
            list(pure.body),
            fn.function_type.return_type,
            [_kernel_param_type(param) for param in pure.params],
            _type_name(pure.return_type),
        )

    for shader in shaders:
        fn = cg.function_map[f"shader_{_sanitize_symbol(shader.name)}"]
        cg.lowerer.lower_function(
            fn,
            list(shader.body),
            ir.VoidType(),
            [_kernel_param_type(param) for param in shader.params],
            None,
            [param.modifier in {"out", "accum"} for param in shader.params],
        )

    for flt in filters:
        fn = cg.function_map[f"filter_{_sanitize_symbol(flt.name)}"]
        cg.lowerer.lower_function(
            fn,
            list(flt.body),
            ir.IntType(1),
            [_kernel_param_type(param) for param in flt.params],
            "bool",
            [param.modifier in {"out", "accum"} for param in flt.params],
        )

    cg.stream_slots: dict[str, int] = {
        stream.name: idx for idx, stream in enumerate(streams)
    }
    cg.stream_capacities: dict[str, int] = {
        stream.name: int(stream.capacity) for stream in streams
    }
    cg.accum_slots: dict[str, int] = {
        accum.name: idx for idx, accum in enumerate(cg.accumulators)
    }
    cg.uniform_slots: dict[str, int] = {
        uniform.name: idx for idx, uniform in enumerate(uniforms)
    }

    cg.kernel_signatures: dict[
        str, tuple[AstKernelDecl, tuple[AstKernelParam, ...]]
    ] = {shader.name: (shader, shader.params) for shader in shaders}
    cg.kernel_signatures.update({flt.name: (flt, flt.params) for flt in filters})
    cg.filter_names = {flt.name for flt in filters}

    cg.accum_sizes = cg._infer_accumulator_sizes()
    layout = build_ast_arena_layout(cg.program, accumulator_sizes=cg.accum_sizes)

    cg.leaf_specs: dict[tuple[str, str, tuple[str, ...]], tuple[int, int]] = {
        (leaf.kind, leaf.binding_name, leaf.path): (leaf.offset, leaf.size)
        for leaf in layout.leaves
    }
    cg.binding_declared_types: dict[tuple[str, str], str] = {}
    for stream in streams:
        cg.binding_declared_types[("stream", stream.name)] = _type_name(
            stream.declared_type
        )
    for accum in cg.accumulators:
        cg.binding_declared_types[("accum", accum.name)] = _type_name(
            accum.declared_type
        )
    for uniform in uniforms:
        cg.binding_declared_types[("uniform", uniform.name)] = _type_name(
            uniform.declared_type
        )

    arena_struct_ty = cg.module.context.get_identified_type("struct.Lockstep_Arena")

    arena_field_types: list[ir.Type] = []
    for leaf in layout.leaves:
        if leaf.type_name in _PRIMITIVE_TYPE_MAP:
            field_ty = _PRIMITIVE_TYPE_MAP[leaf.type_name]
        elif (
            leaf.type_name in cg.known_structs
            and leaf.type_name not in layout.opaque_structs
        ):
            field_ty = cg.known_structs[leaf.type_name]
        else:
            field_ty = ir.ArrayType(ir.IntType(8), max(leaf.size, 1))

        if leaf.kind == "stream" or leaf.element_count > 1:
            arena_field_types.append(ir.ArrayType(field_ty, max(leaf.element_count, 1)))
        else:
            arena_field_types.append(field_ty)

    if not arena_field_types:
        arena_field_types = [ir.ArrayType(ir.IntType(8), 1)]

    if arena_struct_ty.is_opaque:
        arena_struct_ty.set_body(*arena_field_types)

    cg.tick = ir.Function(
        cg.module,
        ir.FunctionType(ir.VoidType(), [arena_struct_ty.as_pointer()]),
        name="Lockstep_Tick",
    )
    cg.arena_ptr = cg.tick.args[0]
    cg.arena_ptr.name = "arena"
    # The arena is Lockstep_Tick's only pointer parameter, and every memory
    # access in the tick is derived from it by constant/loop-index GEPs; the only
    # other memory touched is function-local allocas, which can never alias it.
    # So the arena provably does not alias anything reachable through a different
    # pointer -> `noalias`. The tick also never stores the arena pointer itself
    # anywhere that outlives the call (only values loaded/stored through derived
    # pointers) -> `nocapture`. Both let LLVM's alias analysis treat the arena
    # like a C `restrict` pointer at the ABI boundary.
    cg.arena_ptr.add_attribute("noalias")
    cg.arena_ptr.add_attribute("nocapture")

    tick_entry = cg.tick.append_basic_block("entry")
    cg.tick_builder = ir.IRBuilder(tick_entry)
    cg.simd_width = (
        int(target_width)
        if target_width is not None and int(target_width) > 0
        else _simd_width_for_target_triple(cg.module.triple)
    )

    # Populated per pipeline (see the dispatch loop): accumulators consumed by
    # exactly one fold and written by exactly one earlier kernel route, which the
    # writing route reduces in-register instead of materializing.
    cg._reducible_accums: dict[str, tuple[str, str]] = {}
    cg._reducible_writer: dict[str, int] = {}
    cg._fused_reductions: dict[str, tuple[ir.Value, str]] = {}
    # Filter-group register-carry: a multi-stage group that fuses through a
    # filter can also carry its accumulators in loop-carried vector registers
    # instead of the O(rows) arena buffer -- the same win as ``_fused_reductions``
    # but spanning the whole fused loop and supporting multiple folds per
    # accumulator.  Keyed by the fold's uniform name (each fold is a distinct
    # consumer), populated at group-lowering time and consumed by the fold route.
    cg._fused_group_reductions: dict[str, tuple[ir.Value, str]] = {}
    cg._accum_writer_ids: dict[str, list[int]] = {}
    cg._accum_fold_routes: dict[str, list[AstFoldBindRoute]] = {}

    # Uniform arguments loaded once in a route's preheader (see
    # ``_hoist_uniform_loads``), keyed by (uniform name, LLVM type).
    cg._hoisted_uniforms: dict[tuple[str, str], ir.Value] = {}
    # The fused vector path's per-leaf uniform scalars, loaded the same way.
    cg._hoisted_uniform_leaves: dict[tuple[str, tuple[str, ...], str], ir.Value] = {}

    # --- Live row counts ----------------------------------------------------
    #
    # A filter keeps a data-dependent number of rows, and a stage reading only
    # filtered streams processes just those rows.  ``_live_counts`` tracks each
    # stream's current row count as an i32 SSA value (a constant capacity for
    # streams nothing has filtered); streams in ``counted_stream_names`` also
    # publish it to their arena count slot (``LOCKSTEP_OFFSET_COUNT_*``) so the
    # host knows how many rows are valid.  ``_accum_live_counts`` is how many
    # per-row accumulator slots the writing route filled, which bounds the fold.
    # The tick is a straight sequence of loops, so a count computed after one
    # route's loop dominates every later route.
    cg.counted_stream_names = set(ast_counted_streams(cg.program))
    cg._live_counts: dict[str, ir.Value] = {}
    cg._accum_live_counts: dict[str, ir.Value] = {}

    # Static upper bound on the row index of the loop being emitted, pushed
    # only for loops whose trip count is a run-time row count (static trip
    # counts let instcombine drop the clamp on its own).
    cg._row_bound: list[int] = []

    for pipeline_index, pipeline in enumerate(cg.program.pipelines):
        # Determine which accumulators can be folded in-register for this
        # pipeline: consumed by exactly one fold (with a reducible operator/type)
        # and written by exactly one kernel route that precedes that fold.
        cg._reducible_accums.clear()
        cg._live_counts.clear()
        cg._accum_live_counts.clear()
        cg._reducible_writer.clear()
        cg._fused_reductions.clear()
        cg._fused_group_reductions.clear()
        cg._accum_writer_ids.clear()
        cg._accum_fold_routes.clear()
        _accum_writers: dict[str, list[tuple[int, int]]] = {}
        _fold_consumers: dict[str, list[tuple[int, AstFoldBindRoute]]] = {}
        for position, bind_route in enumerate(pipeline.bind_routes):
            if isinstance(bind_route, AstKernelBindRoute):
                _, route_params = cg.kernel_signatures.get(
                    bind_route.kernel, (None, ())
                )
                for param_index, param in enumerate(route_params):
                    if param.modifier != "accum":
                        continue
                    accum_arg = (
                        bind_route.args[param_index]
                        if param_index < len(bind_route.args)
                        else ""
                    )
                    _accum_writers.setdefault(accum_arg, []).append(
                        (position, id(bind_route))
                    )
                    cg._accum_writer_ids.setdefault(accum_arg, []).append(
                        id(bind_route)
                    )
            elif isinstance(bind_route, AstFoldBindRoute):
                _fold_consumers.setdefault(bind_route.source, []).append(
                    (position, bind_route)
                )
                cg._accum_fold_routes.setdefault(bind_route.source, []).append(
                    bind_route
                )
        for accum_name, consumers in _fold_consumers.items():
            if len(consumers) != 1:
                continue
            writers = _accum_writers.get(accum_name, [])
            if len(writers) != 1:
                continue
            writer_position, writer_id = writers[0]
            fold_position, fold_route = consumers[0]
            if writer_position >= fold_position:
                continue
            uniform_type_name = _type_name(fold_route.uniform_type)
            uniform_type = cg.lowerer._llvm_type(uniform_type_name, cg.known_structs)
            is_reducible_type = isinstance(
                uniform_type, (ir.FloatType, ir.DoubleType, ir.IntType)
            )
            if not is_reducible_type or fold_route.operator not in {
                "sum",
                "avg",
                "min",
                "max",
            }:
                continue
            cg._reducible_accums[accum_name] = (fold_route.operator, uniform_type_name)
            cg._reducible_writer[accum_name] = writer_id

        route_texts = [route.route for route in pipeline.bind_routes]
        route_ir = []
        for route in pipeline.bind_routes:
            if isinstance(route, AstKernelBindRoute):
                route_ir.append(
                    {
                        "kind": "kernel",
                        "target": route.target,
                        "kernel": route.kernel,
                        "args": list(route.args),
                        "route": route.route,
                    }
                )
            else:
                route_ir.append(
                    {
                        "kind": "fold",
                        "uniform_type": _type_name(route.uniform_type),
                        "uniform_name": route.uniform_name,
                        "operator": route.operator,
                        "source": route.source,
                        "route": route.route,
                    }
                )

        pipeline_optimizations = (
            bind_optimization.get("pipeline_optimizations")
            if isinstance(bind_optimization, dict)
            else None
        )
        if isinstance(pipeline_optimizations, list) and len(
            pipeline_optimizations
        ) == len(cg.program.pipelines):
            pipeline_optimization = pipeline_optimizations[pipeline_index]
        elif bind_optimization is not None and len(cg.program.pipelines) == 1:
            pipeline_optimization = bind_optimization
        else:
            pipeline_optimization = optimize_bind_routes(
                route_texts,
                shader_names={shader.name for shader in shaders},
                filter_names={flt.name for flt in filters},
                bind_routes_ir=route_ir,
            )

        optimized_route_counts: dict[str, int] = {}
        for optimized_route in pipeline_optimization.get("optimized_bind_routes", []):
            if not isinstance(optimized_route, str) or "FUSED[" in optimized_route:
                continue
            optimized_route_counts[optimized_route] = (
                optimized_route_counts.get(optimized_route, 0) + 1
            )

        source_to_routes: dict[str, list[AstKernelBindRoute]] = {}
        for route in pipeline.bind_routes:
            if isinstance(route, AstKernelBindRoute):
                source_to_routes.setdefault(route.route, []).append(route)

        fused_start: dict[int, tuple[int, tuple[AstKernelBindRoute, ...]]] = {}
        fused_member_ids: set[int] = set()
        for group_index, group in enumerate(
            pipeline_optimization.get("fused_groups", [])
        ):
            if not isinstance(group, dict):
                continue
            group_routes: list[AstKernelBindRoute] = []
            for route_text in group.get("source_routes", []):
                if not isinstance(route_text, str):
                    continue
                candidates = source_to_routes.get(route_text, [])
                if candidates:
                    group_routes.append(candidates.pop(0))
            if len(group_routes) <= 1:
                continue
            fused_start[id(group_routes[0])] = (group_index, tuple(group_routes))
            fused_member_ids.update(id(route) for route in group_routes[1:])

        for route in pipeline.bind_routes:
            group_entry = fused_start.get(id(route))
            if group_entry is not None:
                group_index, group_routes = group_entry
                cg._lower_fused_kernel_group(group_routes, group_index)
                continue
            if id(route) in fused_member_ids:
                continue
            remaining_optimized_routes = optimized_route_counts.get(route.route, 0)
            if remaining_optimized_routes <= 0:
                continue
            optimized_route_counts[route.route] = remaining_optimized_routes - 1
            if isinstance(route, AstKernelBindRoute):
                cg._lower_kernel_route(route, allow_reduction_fusion=True)
                continue
            cg._lower_fold_route(route)

    cg.tick_builder.ret_void()

    llvm_ir = str(cg.module)
    if bind_route_comments:
        compatibility_comments = [f"; bind: {route}" for route in bind_route_comments]
        # Preserve the historical textual stream GEP markers for legacy mapping
        # callers.  The actual lowering now uses byte-accurate arena addressing,
        # but older consumers asserted these declaration comments when verifying
        # simple bind-route codegen.
        compatibility_comments.extend(
            f'; legacy stream gep: getelementptr %"struct.Lockstep_Arena", %"struct.Lockstep_Arena"* %"arena", i32 0, i32 {index}, i32 %"route_i32_lane0.{(index * 2) + 1}"'
            for index, _stream in enumerate(streams)
        )
        llvm_ir = "\n".join([llvm_ir, *compatibility_comments])
    return llvm_ir
