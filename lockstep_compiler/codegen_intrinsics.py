"""Scalar builtin-intrinsic lowering for the LLVM backend.

This module owns the semantics of Lockstep's straight-line scalar intrinsics
(`step`, `mix`, `min`, `max`, `clamp`, `abs`, `sign`, `smoothstep`): how each one
is expanded into LLVM IR and how the supporting `llvm.*` declarations are
materialized. It is intentionally free of any `_FunctionLowerer` state — every
function takes the LLVM `builder`/`module` and an ``error`` callback explicitly —
so the intrinsic contract lives in one small, directly testable place rather than
threaded through the 3k-line code generator.

The lowering is byte-for-byte identical to the previous in-class implementation;
`tests/test_golden_ir.py` pins the emitted IR.
"""

from __future__ import annotations

from typing import Callable

from llvmlite import ir

# Raises a codegen error with the given message (never returns).
ErrorFn = Callable[[str], None]


def declare_binary_llvm_intrinsic(
    module: ir.Module, intrinsic_name: str, arg_type: ir.Type, error: ErrorFn
) -> ir.Function:
    """Declare (or reuse) a two-operand `llvm.<name>.<suffix>` intrinsic."""
    if isinstance(arg_type, ir.FloatType):
        suffix = "f32"
    elif isinstance(arg_type, ir.DoubleType):
        suffix = "f64"
    else:
        error(f"intrinsic '{intrinsic_name}' expects float or double arguments")
    llvm_name = f"llvm.{intrinsic_name}.{suffix}"
    intrinsic = module.globals.get(llvm_name)
    if intrinsic is not None:
        return intrinsic
    fn_type = ir.FunctionType(arg_type, [arg_type, arg_type])
    return ir.Function(module, fn_type, name=llvm_name)


def declare_unary_llvm_intrinsic(
    module: ir.Module, intrinsic_name: str, arg_type: ir.Type, error: ErrorFn
) -> ir.Function:
    """Declare (or reuse) a one-operand `llvm.<name>.<suffix>` intrinsic."""
    if isinstance(arg_type, ir.FloatType):
        suffix = "f32"
    elif isinstance(arg_type, ir.DoubleType):
        suffix = "f64"
    else:
        error(f"intrinsic '{intrinsic_name}' expects float or double arguments")
    llvm_name = f"llvm.{intrinsic_name}.{suffix}"
    intrinsic = module.globals.get(llvm_name)
    if intrinsic is not None:
        return intrinsic
    fn_type = ir.FunctionType(arg_type, [arg_type])
    return ir.Function(module, fn_type, name=llvm_name)


def float_intrinsic_type(name: str, args: list[ir.Value], error: ErrorFn) -> ir.Type:
    """Validate that every argument is the same float/double type and return it."""
    if not args or not all(
        isinstance(arg.type, (ir.FloatType, ir.DoubleType)) for arg in args
    ):
        error(f"intrinsic '{name}' expects float arguments")
    first_type = args[0].type
    if not all(arg.type == first_type for arg in args):
        error(f"intrinsic '{name}' expects matching float arguments")
    return first_type


def lower_intrinsic_call(
    builder: ir.IRBuilder,
    module: ir.Module,
    name: str,
    args: list[ir.Value],
    error: ErrorFn,
) -> ir.Value | None:
    """Lower a scalar builtin intrinsic call, or return ``None`` if ``name`` is
    not a recognized intrinsic (so the caller can fall through to user calls)."""
    if name == "step" and len(args) == 2:
        arg_type = float_intrinsic_type(name, args, error)
        edge, x_val = args
        cmp_result = builder.fcmp_ordered(">=", x_val, edge, name="step_cmp")
        return builder.uitofp(cmp_result, arg_type, name="step")

    if name == "mix" and len(args) == 3:
        arg_type = float_intrinsic_type(name, args, error)
        a, b, t = args
        one_minus_t = builder.fsub(
            ir.Constant(arg_type, 1.0), t, name="mix_one_minus_t"
        )
        return builder.fadd(
            builder.fmul(a, one_minus_t),
            builder.fmul(b, t),
            name="mix",
        )

    if name == "max" and len(args) == 2:
        arg_type = float_intrinsic_type(name, args, error)
        maxnum = declare_binary_llvm_intrinsic(module, "maxnum", arg_type, error)
        return builder.call(maxnum, args, name="max")

    if name == "min" and len(args) == 2:
        arg_type = float_intrinsic_type(name, args, error)
        minnum = declare_binary_llvm_intrinsic(module, "minnum", arg_type, error)
        return builder.call(minnum, args, name="min")

    if name == "clamp" and len(args) == 3:
        arg_type = float_intrinsic_type(name, args, error)
        x_val, min_value, max_value = args
        maxnum = declare_binary_llvm_intrinsic(module, "maxnum", arg_type, error)
        minnum = declare_binary_llvm_intrinsic(module, "minnum", arg_type, error)
        clamped_min = builder.call(maxnum, [x_val, min_value], name="clamp_min")
        return builder.call(minnum, [clamped_min, max_value], name="clamp")

    if name == "abs" and len(args) == 1:
        arg_type = float_intrinsic_type(name, args, error)
        fabs = declare_unary_llvm_intrinsic(module, "fabs", arg_type, error)
        return builder.call(fabs, args, name="abs")

    if name == "sign" and len(args) == 1:
        arg_type = float_intrinsic_type(name, args, error)
        x = args[0]
        zero = ir.Constant(arg_type, 0.0)
        pos = builder.fcmp_ordered(">", x, zero, name="sign_pos")
        neg = builder.fcmp_ordered("<", x, zero, name="sign_neg")
        pos_f = builder.uitofp(pos, arg_type, name="sign_pos_f")
        neg_f = builder.uitofp(neg, arg_type, name="sign_neg_f")
        return builder.fsub(pos_f, neg_f, name="sign")

    if name == "smoothstep" and len(args) == 3:
        arg_type = float_intrinsic_type(name, args, error)
        edge0, edge1, x = args
        # t = clamp((x - edge0) / (edge1 - edge0), 0, 1)
        diff = builder.fsub(x, edge0, name="ss_diff")
        range_val = builder.fsub(edge1, edge0, name="ss_range")
        t_raw = builder.fdiv(diff, range_val, name="ss_t_raw")
        maxnum = declare_binary_llvm_intrinsic(module, "maxnum", arg_type, error)
        minnum = declare_binary_llvm_intrinsic(module, "minnum", arg_type, error)
        t_clamped = builder.call(
            maxnum, [t_raw, ir.Constant(arg_type, 0.0)], name="ss_clamp_lo"
        )
        t = builder.call(minnum, [t_clamped, ir.Constant(arg_type, 1.0)], name="ss_t")
        # result = t * t * (3 - 2 * t)
        two_t = builder.fmul(ir.Constant(arg_type, 2.0), t, name="ss_2t")
        three_minus_2t = builder.fsub(
            ir.Constant(arg_type, 3.0), two_t, name="ss_3m2t"
        )
        t_sq = builder.fmul(t, t, name="ss_tsq")
        return builder.fmul(t_sq, three_minus_2t, name="smoothstep")

    return None
