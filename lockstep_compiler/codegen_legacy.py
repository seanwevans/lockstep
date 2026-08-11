"""Legacy entity-dict → typed-AST adapter for the code generator.

The public compiler passes an :class:`~lockstep_compiler.ast.AstProgram` into
``emit_llvm_ir``, but a number of tests and older integrations still call it with
the historical dictionary shape produced by ``ast_to_entities``. This module
keeps that compatibility isolated at the codegen boundary so the generator itself
only ever operates on the typed AST. It has no dependency on the code generator's
internals — it is a pure input normalizer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .ast import (
    AstAccumulatorDecl,
    AstFoldBindRoute,
    AstKernelBindRoute,
    AstKernelDecl,
    AstKernelParam,
    AstPipelineDecl,
    AstProgram,
    AstPureDecl,
    AstStreamDecl,
    AstStructDecl,
    AstStructField,
    AstUniformDecl,
)
from .optimizer import _parse_bind_route


def program_from_legacy_mapping(
    program: Mapping[str, Any],
) -> tuple[AstProgram, dict[str, int]]:
    """Normalize the historical dictionary-shaped codegen input into an AST.

    The public compiler now passes an :class:`AstProgram`, but a number of
    tests and integrations still call ``emit_llvm_ir`` with the older entity
    dictionary produced by ``ast_to_entities``.  Keep that compatibility at the
    boundary so codegen internals can operate on one representation.
    """

    def _items(name: str) -> Sequence[Mapping[str, Any]]:
        value = program.get(name, ())
        return (
            value if isinstance(value, Sequence) and not isinstance(value, str) else ()
        )

    structs = tuple(
        AstStructDecl(
            name=str(struct.get("name", "")),
            fields=tuple(
                AstStructField(
                    declared_type=field.get(
                        "type", field.get("declared_type", "float")
                    ),
                    name=str(field.get("name", "")),
                )
                for field in struct.get("fields", ())
                if isinstance(field, Mapping)
            ),
        )
        for struct in _items("structs")
    )

    def _params(entity: Mapping[str, Any]) -> tuple[AstKernelParam, ...]:
        return tuple(
            AstKernelParam(
                modifier=str(param.get("modifier", "in")),
                declared_type=param.get("type", param.get("declared_type", "float")),
                name=str(param.get("name", "")),
            )
            for param in entity.get("params", ())
            if isinstance(param, Mapping)
        )

    pure_functions = tuple(
        AstPureDecl(
            name=str(function.get("name", "")),
            return_type=function.get("return_type", function.get("type", "void")),
            params=_params(function),
            body=tuple(function.get("body_ast", function.get("body", ()))),
            intrinsic=bool(function.get("intrinsic", False)),
        )
        for function in _items("pure_functions")
    )
    shaders = tuple(
        AstKernelDecl(
            name=str(shader.get("name", "")),
            params=_params(shader),
            body=tuple(shader.get("body_ast", shader.get("body", ()))),
        )
        for shader in _items("shaders")
    )
    filters = tuple(
        AstKernelDecl(
            name=str(filter_decl.get("name", "")),
            params=_params(filter_decl),
            body=tuple(filter_decl.get("body_ast", filter_decl.get("body", ()))),
        )
        for filter_decl in _items("filters")
    )
    streams = tuple(
        AstStreamDecl(
            name=str(stream.get("name", "")),
            declared_type=stream.get("type", stream.get("declared_type", "float")),
            capacity=str(stream.get("capacity", 1)),
        )
        for stream in _items("streams")
    )
    accumulators = tuple(
        AstAccumulatorDecl(
            name=str(accum.get("name", "")),
            declared_type=accum.get("type", accum.get("declared_type", "float")),
        )
        for accum in _items("accumulators")
    )
    explicit_accumulator_sizes = {
        str(accum.get("name", "")): max(int(accum.get("size", 1)), 1)
        for accum in _items("accumulators")
        if "size" in accum
    }
    uniforms = tuple(
        AstUniformDecl(
            name=str(uniform.get("name", "")),
            declared_type=uniform.get("type", uniform.get("declared_type", "float")),
            initializer=uniform.get("initializer"),
        )
        for uniform in _items("uniforms")
    )

    bind_routes: list[AstKernelBindRoute | AstFoldBindRoute] = []
    for raw_route in program.get("bind_routes", ()):
        if not isinstance(raw_route, str):
            continue
        parsed_route = _parse_bind_route(raw_route)
        if parsed_route is None:
            continue
        bind_routes.append(
            AstKernelBindRoute(
                target=parsed_route.target,
                kernel=parsed_route.callee,
                args=parsed_route.args,
                route=parsed_route.raw,
            )
        )

    if program.get("bind_routes_ir"):
        bind_routes = []

    for route in _items("bind_routes_ir"):
        if route.get("kind") == "fold":
            bind_routes.append(
                AstFoldBindRoute(
                    uniform_type=route.get("uniform_type", "float"),
                    uniform_name=str(route.get("uniform_name", "")),
                    operator=str(route.get("operator", "sum")),
                    source=str(route.get("source", "")),
                    route=str(route.get("route", "")),
                )
            )
        elif route.get("kind") == "kernel":
            bind_routes.append(
                AstKernelBindRoute(
                    target=str(route.get("target", "")),
                    kernel=str(route.get("kernel", "")),
                    args=tuple(str(arg) for arg in route.get("args", ())),
                    route=str(route.get("route", "")),
                )
            )

    return (
        AstProgram(
            structs=structs,
            pure_functions=pure_functions,
            pipelines=(
                AstPipelineDecl(
                    name=str(program.get("name", "P")),
                    streams=streams,
                    accumulators=accumulators,
                    uniforms=uniforms,
                    bind_routes=tuple(bind_routes),
                ),
            ),
            shaders=shaders,
            filters=filters,
        ),
        explicit_accumulator_sizes,
    )
