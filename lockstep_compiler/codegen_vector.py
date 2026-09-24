"""SIMD value helpers and SoA vector loads/stores for the fused-vector path.

Part of the ``Lockstep_Tick`` code generator: ``codegen._TickCodegen`` combines
this mixin with the others and holds the per-tick state they share through
``self`` (the arena pointer, IR builder, arena layout, and lowering caches).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llvmlite import ir

from .ast import (
    AstAssignStmt,
    AstExprBinary,
    AstExprCall,
    AstExprCast,
    AstExprLiteral,
    AstExprUnary,
    AstExprVar,
    AstStatement,
    AstType,
    AstVarDeclStmt,
)
from .codegen_lowerer import CodegenError, _PRIMITIVE_TYPE_MAP, _type_name
from .utils import sanitize_symbol as _sanitize_symbol

if TYPE_CHECKING:
    from .codegen import _TickCodegen


class _FusedVectorLowerer:
    def __init__(self, cg: "_TickCodegen"):
        self.cg = cg
        self.values: dict[tuple[str, ...], ir.Value] = {}
        self.types: dict[tuple[str, ...], str] = {}

    @staticmethod
    def _key(path: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        if not path:
            raise CodegenError("empty vector variable path")
        return (_sanitize_symbol(path[0]), *tuple(path[1:]))

    def _leaf_fields(
        self, type_name: AstType | str, prefix: tuple[str, ...] = ()
    ) -> dict[tuple[str, ...], tuple[ir.Type, str]]:
        leaves = self.cg._vectorizable_leaf_fields(type_name, prefix)
        if leaves is None:
            raise CodegenError(f"type '{type_name}' cannot be SIMD-vectorized")
        return leaves

    def define_slot(self, name: str, type_name: AstType | str) -> None:
        root_key = self._key((name,))
        declared_type_name = _type_name(type_name)
        self.types[root_key] = declared_type_name
        for rel_path, (scalar_ty, leaf_type_name) in self._leaf_fields(
            declared_type_name
        ).items():
            key = root_key + rel_path
            self.types[key] = leaf_type_name
            vector_ty = self.cg._vector_type_for_scalar(scalar_ty)
            if vector_ty is None:
                raise CodegenError(f"type '{leaf_type_name}' cannot be SIMD-vectorized")
            self.values[key] = ir.Constant(vector_ty, None)

    def set_slot_leaf(
        self, name: str, rel_path: tuple[str, ...], value: ir.Value
    ) -> None:
        self.values[self._key((name,)) + rel_path] = value

    def slot_leaf_values(self, name: str) -> dict[tuple[str, ...], ir.Value]:
        root_key = self._key((name,))
        return {
            key[len(root_key) :]: value
            for key, value in self.values.items()
            if key[: len(root_key)] == root_key and key != root_key
        }

    def _declared_vector_type(self, type_name: AstType | str) -> ir.VectorType:
        scalar_ty = self.cg.lowerer._llvm_type(type_name, self.cg.known_structs)
        vector_ty = self.cg._vector_type_for_scalar(scalar_ty)
        if vector_ty is None:
            raise CodegenError(f"type '{type_name}' cannot be SIMD-vectorized")
        return vector_ty

    def _path_type(self, path: tuple[str, ...]) -> str | None:
        key = self._key(path)
        if key in self.types:
            return self.types[key]
        if len(path) == 1:
            return None
        root_type = self.types.get(self._key((path[0],)))
        if root_type is None:
            return None
        current_type = root_type
        for field_name in path[1:]:
            current_ty = self.cg.lowerer._llvm_type(current_type, self.cg.known_structs)
            field_info = self.cg._field_index_and_type(current_ty, field_name)
            if field_info is None:
                return None
            _, _, current_type = field_info
        self.types[key] = current_type
        return current_type

    @staticmethod
    def _promote_scalar_type_names(
        left_type: str | None, right_type: str | None
    ) -> str | None:
        if left_type is None or right_type is None:
            return left_type or right_type
        if left_type == right_type:
            return left_type

        numeric_rank = {"int": 0, "uint": 1, "float": 2, "double": 3}
        if left_type in numeric_rank and right_type in numeric_rank:
            return max((left_type, right_type), key=numeric_rank.__getitem__)
        return None

    def _infer_binary_operand_type(self, node: AstExprBinary) -> str | None:
        left_type = self._infer_expr_type(node.left)
        right_type = self._infer_expr_type(node.right)
        op = node.op
        if op in {"&&", "||"}:
            return "bool" if left_type == "bool" and right_type == "bool" else None
        if op in {"<<", ">>"}:
            return left_type if left_type in {"int", "uint"} else None
        if op in {"+", "-", "*", "/", "%", "<", "<=", ">", ">=", "==", "!="}:
            return self._promote_scalar_type_names(left_type, right_type)
        if op in {"&", "|", "^"}:
            if left_type in {"int", "uint"} and right_type in {"int", "uint"}:
                return self._promote_scalar_type_names(left_type, right_type)
            return None
        return None

    def _infer_expr_type(self, node) -> str | None:
        if isinstance(node, AstExprLiteral):
            return node.kind if node.kind in _PRIMITIVE_TYPE_MAP else None
        if isinstance(node, AstExprVar):
            return self._path_type(node.path)
        if isinstance(node, AstExprCast):
            return _type_name(node.target_type)
        if isinstance(node, AstExprUnary):
            return self._infer_expr_type(node.operand)
        if isinstance(node, AstExprCall):
            if node.name in {"int", "uint", "float", "double", "bool"}:
                return node.name
            if node.name == "select" and len(node.args) == 3:
                return self._infer_expr_type(node.args[1])
            if node.name in {
                "step",
                "mix",
                "min",
                "max",
                "clamp",
                "abs",
                "sign",
                "smoothstep",
            }:
                return "float"
            return None
        if isinstance(node, AstExprBinary):
            operand_type = self._infer_binary_operand_type(node)
            return (
                "bool"
                if node.op in {"<", "<=", ">", ">=", "==", "!=", "&&", "||"}
                and operand_type is not None
                else operand_type
            )
        return None

    def _literal(self, node: AstExprLiteral) -> ir.Value:
        if node.kind == "float":
            scalar = ir.Constant(ir.FloatType(), float(node.value))
        elif node.kind == "double":
            scalar = ir.Constant(ir.DoubleType(), float(node.value))
        elif node.kind in {"int", "uint"}:
            scalar = ir.Constant(ir.IntType(32), int(node.value))
        elif node.kind == "bool":
            scalar = ir.Constant(ir.IntType(1), int(node.value == "true"))
        else:
            raise CodegenError("strings cannot be SIMD-vectorized")
        return self.cg._splat_to_vector(
            scalar, ir.VectorType(scalar.type, self.cg.simd_width)
        )

    def _load_var(self, node: AstExprVar) -> ir.Value:
        key = self._key(node.path)
        if key not in self.values:
            if (
                len(node.path) == 1
                and self._path_type(node.path) in self.cg.struct_fields
            ):
                raise CodegenError(
                    f"whole-struct value '{node.path[0]}' cannot be used as "
                    "a SIMD scalar expression"
                )
            raise CodegenError(f"undefined vector variable '{'.'.join(node.path)}'")
        return self.values[key]

    def _numeric_unary_minus(self, value: ir.Value) -> ir.Value:
        zero = ir.Constant(value.type, None)
        if isinstance(value.type.element, (ir.FloatType, ir.DoubleType)):
            return self.cg.tick_builder.fsub(zero, value, name="fused_neg")
        if isinstance(value.type.element, ir.IntType):
            return self.cg.tick_builder.sub(zero, value, name="fused_neg")
        raise CodegenError(f"unary '-' is unsupported for type '{value.type}'")

    def _binary(
        self, op: str, lhs: ir.Value, rhs: ir.Value, type_name: str | None
    ) -> ir.Value:
        if isinstance(lhs.type, ir.VectorType) and not isinstance(
            rhs.type, ir.VectorType
        ):
            rhs = self.cg._splat_to_vector(rhs, lhs.type)
        elif isinstance(rhs.type, ir.VectorType) and not isinstance(
            lhs.type, ir.VectorType
        ):
            lhs = self.cg._splat_to_vector(lhs, rhs.type)
        if lhs.type != rhs.type:
            raise CodegenError(
                f"operator '{op}' requires matching vector operand types, got '{lhs.type}' and '{rhs.type}'"
            )
        elem_ty = lhs.type.element
        if op in {"+", "-", "*", "/", "%"}:
            if isinstance(elem_ty, (ir.FloatType, ir.DoubleType)):
                return {
                    "+": self.cg.tick_builder.fadd,
                    "-": self.cg.tick_builder.fsub,
                    "*": self.cg.tick_builder.fmul,
                    "/": self.cg.tick_builder.fdiv,
                    "%": self.cg.tick_builder.frem,
                }[op](lhs, rhs, name="fused_math")
            if isinstance(elem_ty, ir.IntType):
                return {
                    "+": self.cg.tick_builder.add,
                    "-": self.cg.tick_builder.sub,
                    "*": self.cg.tick_builder.mul,
                    "/": (
                        self.cg.tick_builder.udiv
                        if type_name == "uint"
                        else self.cg.tick_builder.sdiv
                    ),
                    "%": (
                        self.cg.tick_builder.urem
                        if type_name == "uint"
                        else self.cg.tick_builder.srem
                    ),
                }[op](lhs, rhs, name="fused_math")
        if op in {"&", "|", "^", "<<", ">>"} and isinstance(elem_ty, ir.IntType):
            if op == "&":
                return self.cg.tick_builder.and_(lhs, rhs, name="fused_and")
            if op == "|":
                return self.cg.tick_builder.or_(lhs, rhs, name="fused_or")
            if op == "^":
                return self.cg.tick_builder.xor(lhs, rhs, name="fused_xor")
            if op == "<<":
                return self.cg.tick_builder.shl(lhs, rhs, name="fused_shl")
            return (
                self.cg.tick_builder.lshr
                if type_name == "uint"
                else self.cg.tick_builder.ashr
            )(lhs, rhs, name="fused_shr")
        if op in {"<", "<=", ">", ">=", "==", "!="}:
            rel_map = {
                "<": "<",
                "<=": "<=",
                ">": ">",
                ">=": ">=",
                "==": "==",
                "!=": "!=",
            }
            if isinstance(elem_ty, (ir.FloatType, ir.DoubleType)):
                return self.cg.tick_builder.fcmp_ordered(
                    rel_map[op], lhs, rhs, name="fused_cmp"
                )
            if isinstance(elem_ty, ir.IntType):
                cmp_op = (
                    self.cg.tick_builder.icmp_unsigned
                    if type_name == "uint"
                    else self.cg.tick_builder.icmp_signed
                )
                return cmp_op(rel_map[op], lhs, rhs, name="fused_cmp")
        if op == "&&" or op == "||":
            return (
                self.cg.tick_builder.and_ if op == "&&" else self.cg.tick_builder.or_
            )(lhs, rhs, name="fused_bool")
        raise CodegenError(f"unsupported vector binary operator '{op}'")

    def _call(self, name: str, args: list[ir.Value]) -> ir.Value:
        if name in {"int", "uint", "float", "double", "bool"} and len(args) == 1:
            return self.cg._coerce_vector_value(
                args[0], self._declared_vector_type(name), name
            )
        if name == "select" and len(args) == 3:
            return self.cg.tick_builder.select(
                args[0], args[1], args[2], name="fused_select"
            )
        if name == "mix" and len(args) == 3:
            a, b, t = args
            one = self.cg._splat_to_vector(ir.Constant(ir.FloatType(), 1.0), a.type)
            return self.cg.tick_builder.fadd(
                self.cg.tick_builder.fmul(
                    a, self.cg.tick_builder.fsub(one, t, name="fused_mix_omt")
                ),
                self.cg.tick_builder.fmul(b, t),
                name="fused_mix",
            )
        if name == "step" and len(args) == 2:
            edge, x_val = args
            cmp_result = self.cg.tick_builder.fcmp_ordered(
                ">=", x_val, edge, name="fused_step_cmp"
            )
            return self.cg.tick_builder.uitofp(
                cmp_result, x_val.type, name="fused_step"
            )
        if name in {"min", "max"} and len(args) == 2:
            lhs, rhs = args
            pred = self.cg.tick_builder.fcmp_ordered(
                "<" if name == "min" else ">", lhs, rhs, name=f"fused_{name}_cmp"
            )
            return self.cg.tick_builder.select(pred, lhs, rhs, name=f"fused_{name}")
        if name == "clamp" and len(args) == 3:
            x_val, lo, hi = args
            lower = self._call("max", [x_val, lo])
            return self._call("min", [lower, hi])
        if name == "abs" and len(args) == 1:
            x_val = args[0]
            zero = ir.Constant(x_val.type, None)
            neg = self.cg.tick_builder.fsub(zero, x_val, name="fused_abs_neg")
            pred = self.cg.tick_builder.fcmp_ordered(
                "<", x_val, zero, name="fused_abs_cmp"
            )
            return self.cg.tick_builder.select(pred, neg, x_val, name="fused_abs")
        if name == "sign" and len(args) == 1:
            x_val = args[0]
            zero = ir.Constant(x_val.type, None)
            one = self.cg._splat_to_vector(ir.Constant(ir.FloatType(), 1.0), x_val.type)
            neg_one = self.cg._splat_to_vector(
                ir.Constant(ir.FloatType(), -1.0), x_val.type
            )
            pos = self.cg.tick_builder.fcmp_ordered(
                ">", x_val, zero, name="fused_sign_pos"
            )
            neg = self.cg.tick_builder.fcmp_ordered(
                "<", x_val, zero, name="fused_sign_neg"
            )
            return self.cg.tick_builder.select(
                pos,
                one,
                self.cg.tick_builder.select(neg, neg_one, zero),
                name="fused_sign",
            )
        if name == "smoothstep" and len(args) == 3:
            edge0, edge1, x_val = args
            zero = self.cg._splat_to_vector(
                ir.Constant(ir.FloatType(), 0.0), x_val.type
            )
            one = self.cg._splat_to_vector(ir.Constant(ir.FloatType(), 1.0), x_val.type)
            two = self.cg._splat_to_vector(ir.Constant(ir.FloatType(), 2.0), x_val.type)
            three = self.cg._splat_to_vector(
                ir.Constant(ir.FloatType(), 3.0), x_val.type
            )
            span = self.cg.tick_builder.fsub(edge1, edge0, name="fused_ss_range")
            t_raw = self.cg.tick_builder.fdiv(
                self.cg.tick_builder.fsub(x_val, edge0, name="fused_ss_diff"),
                span,
                name="fused_ss_raw",
            )
            # Degenerate edges (edge0 == edge1) give t = 0, like the scalar
            # intrinsic and the simulator, instead of clamping +-inf/NaN.
            degenerate = self.cg.tick_builder.fcmp_ordered(
                "==", span, zero, name="fused_ss_degenerate"
            )
            t_raw = self.cg.tick_builder.select(
                degenerate, zero, t_raw, name="fused_ss_raw_safe"
            )
            t = self._call("clamp", [t_raw, zero, one])
            return self.cg.tick_builder.fmul(
                self.cg.tick_builder.fmul(t, t, name="fused_ss_tsq"),
                self.cg.tick_builder.fsub(
                    three, self.cg.tick_builder.fmul(two, t), name="fused_ss_poly"
                ),
                name="fused_smoothstep",
            )
        raise CodegenError(f"unsupported vector call '{name}'")

    def lower_expr(self, node) -> ir.Value:
        if isinstance(node, AstExprLiteral):
            return self._literal(node)
        if isinstance(node, AstExprVar):
            return self._load_var(node)
        if isinstance(node, AstExprUnary):
            operand = self.lower_expr(node.operand)
            if node.op == "-":
                return self._numeric_unary_minus(operand)
            return self.cg.tick_builder.not_(operand, name="fused_not")
        if isinstance(node, AstExprCall):
            return self._call(node.name, [self.lower_expr(arg) for arg in node.args])
        if isinstance(node, AstExprCast):
            return self.cg._coerce_vector_value(
                self.lower_expr(node.value),
                self._declared_vector_type(node.target_type),
                _type_name(node.target_type),
                self._infer_expr_type(node.value),
            )
        if isinstance(node, AstExprBinary):
            lhs = self.lower_expr(node.left)
            rhs = self.lower_expr(node.right)
            expr_type_name = self._infer_expr_type(node.left) or self._infer_expr_type(
                node.right
            )
            operand_type_name = self._infer_binary_operand_type(node) or expr_type_name
            if operand_type_name in _PRIMITIVE_TYPE_MAP:
                vector_ty = self._declared_vector_type(operand_type_name)
                lhs_type_name = self._infer_expr_type(node.left)
                rhs_type_name = self._infer_expr_type(node.right)
                lhs = self.cg._coerce_vector_value(
                    lhs, vector_ty, operand_type_name, lhs_type_name
                )
                rhs = self.cg._coerce_vector_value(
                    rhs, vector_ty, operand_type_name, rhs_type_name
                )
            return self._binary(node.op, lhs, rhs, operand_type_name)
        raise CodegenError(
            f"unsupported vector expression node '{type(node).__name__}'"
        )

    def lower_statement(self, statement: AstStatement) -> None:
        if isinstance(statement, AstAssignStmt):
            key = self._key(statement.target)
            if key not in self.values:
                raise CodegenError(
                    "undefined vector assignment target "
                    f"'{'.'.join(statement.target)}'"
                )
            target_ty = self.values[key].type
            type_name = self.types.get(key)
            self.values[key] = self.cg._coerce_vector_value(
                self.lower_expr(statement.value), target_ty, type_name
            )
            return
        if isinstance(statement, AstVarDeclStmt):
            declared_type = (
                _type_name(statement.declared_type)
                if statement.declared_type
                else "float"
            )
            self.define_slot(statement.name, declared_type)
            key = self._key((statement.name,))
            if statement.initializer is not None:
                if key not in self.values:
                    raise CodegenError(
                        f"whole-struct initializer for '{statement.name}' "
                        "cannot be SIMD-vectorized"
                    )
                vector_ty = self._declared_vector_type(declared_type)
                self.values[key] = self.cg._coerce_vector_value(
                    self.lower_expr(statement.initializer), vector_ty, declared_type
                )
            return
        raise CodegenError(
            f"unsupported vector statement node '{type(statement).__name__}'"
        )


class VectorMemoryMixin:
    """SIMD value helpers and SoA vector loads/stores for the fused-vector path."""

    def _vector_type_for_scalar(self, scalar_type: ir.Type) -> ir.VectorType | None:
        if isinstance(
            scalar_type, (ir.FloatType, ir.DoubleType, ir.IntType)
        ) and not isinstance(scalar_type, ir.VectorType):
            return ir.VectorType(scalar_type, self.simd_width)
        return None

    def _struct_name_for_type(self, llvm_type: ir.Type) -> str | None:
        for name, known_ty in self.known_structs.items():
            if llvm_type is known_ty:
                return name
        return None

    def _field_index_and_type(
        self, llvm_type: ir.Type, field_name: str
    ) -> tuple[int, ir.Type, str] | None:
        struct_name = self._struct_name_for_type(llvm_type)
        if struct_name is None:
            return None
        fields = self.struct_fields.get(struct_name, ())
        for index, field in enumerate(fields):
            if field.name == field_name:
                declared_type_name = _type_name(field.declared_type)
                return (
                    index,
                    self.lowerer._llvm_type(declared_type_name, self.known_structs),
                    declared_type_name,
                )
        return None

    def _vectorizable_leaf_fields(
        self, type_name: AstType | str, prefix: tuple[str, ...] = ()
    ) -> dict[tuple[str, ...], tuple[ir.Type, str]] | None:
        declared_type_name = _type_name(type_name)
        llvm_type = self.lowerer._llvm_type(declared_type_name, self.known_structs)
        if self._vector_type_for_scalar(llvm_type) is not None:
            return {prefix: (llvm_type, declared_type_name)}
        if not isinstance(llvm_type, ir.IdentifiedStructType):
            return None
        fields = self.struct_fields.get(declared_type_name)
        if fields is None:
            return None
        leaves: dict[tuple[str, ...], tuple[ir.Type, str]] = {}
        for field in fields:
            child = self._vectorizable_leaf_fields(
                field.declared_type, prefix + (field.name,)
            )
            if child is None:
                return None
            leaves.update(child)
        return leaves

    def _splat_to_vector(
        self,
        value: ir.Value,
        vector_ty: ir.VectorType,
        type_name: str | None = None,
        source_type_name: str | None = None,
    ) -> ir.Value:
        if value.type == vector_ty:
            return value
        if value.type != vector_ty.element:
            value = self.lowerer._coerce_value_to_type(
                value, vector_ty.element, type_name, source_type_name
            )
        seed = self.tick_builder.insert_element(
            ir.Constant(vector_ty, ir.Undefined), value, ir.Constant(ir.IntType(32), 0)
        )
        mask = ir.Constant(
            ir.VectorType(ir.IntType(32), self.simd_width), [0] * self.simd_width
        )
        return self.tick_builder.shuffle_vector(
            seed, ir.Constant(vector_ty, ir.Undefined), mask, name="fused_splat"
        )

    def _coerce_vector_value(
        self,
        value: ir.Value,
        target_ty: ir.Type,
        type_name: str | None = None,
        source_type_name: str | None = None,
    ) -> ir.Value:
        if value.type == target_ty:
            return value
        if isinstance(target_ty, ir.VectorType):
            if not isinstance(value.type, ir.VectorType):
                return self._splat_to_vector(
                    value, target_ty, type_name, source_type_name
                )
            source_elem = value.type.element
            target_elem = target_ty.element
            if source_elem == target_elem:
                return value
            if isinstance(target_elem, ir.IntType) and isinstance(
                source_elem, ir.IntType
            ):
                if source_elem.width < target_elem.width:
                    return (
                        self.tick_builder.zext(value, target_ty)
                        if source_elem.width == 1
                        else self.tick_builder.sext(value, target_ty)
                    )
                if source_elem.width > target_elem.width:
                    if target_elem.width == 1:
                        return self.tick_builder.icmp_signed(
                            "!=",
                            value,
                            ir.Constant(value.type, None),
                            name="fused_int_to_bool",
                        )
                    return self.tick_builder.trunc(value, target_ty)
            if isinstance(target_elem, (ir.FloatType, ir.DoubleType)) and isinstance(
                source_elem, ir.IntType
            ):
                # ``bool`` lanes are ``i1``: a signed conversion maps true to -1.
                if source_type_name == "uint" or source_elem.width == 1:
                    return self.tick_builder.uitofp(value, target_ty)
                return self.tick_builder.sitofp(value, target_ty)
            if isinstance(target_elem, ir.IntType) and isinstance(
                source_elem, (ir.FloatType, ir.DoubleType)
            ):
                if target_elem.width == 1:
                    return self.tick_builder.fcmp_unordered(
                        "!=",
                        value,
                        ir.Constant(value.type, None),
                        name="fused_float_to_bool",
                    )
                return (
                    self.tick_builder.fptoui(value, target_ty)
                    if type_name == "uint"
                    else self.tick_builder.fptosi(value, target_ty)
                )
            if isinstance(target_elem, ir.FloatType) and isinstance(
                source_elem, ir.DoubleType
            ):
                return self.tick_builder.fptrunc(value, target_ty)
            if isinstance(target_elem, ir.DoubleType) and isinstance(
                source_elem, ir.FloatType
            ):
                return self.tick_builder.fpext(value, target_ty)
        if isinstance(value.type, ir.VectorType):
            raise CodegenError(
                f"cannot coerce vector value of type '{value.type}' to '{target_ty}'"
            )
        return self.lowerer._coerce_value_to_type(
            value, target_ty, type_name, source_type_name
        )

    def _vector_lane_indices(self, current: ir.Value) -> ir.Value:
        vector_ty = ir.VectorType(ir.IntType(32), self.simd_width)
        lanes = ir.Constant(vector_ty, list(range(self.simd_width)))
        return self.tick_builder.add(
            self._splat_to_vector(current, vector_ty), lanes, name="fused_lane_indices"
        )

    def _binding_capacity(self, kind: str, name: str) -> int:
        if kind == "accum":
            return int(self.accum_sizes.get(name, 0))
        return int(self.stream_capacities.get(name, 0))

    def _stream_has_contiguous_vector_chunk(
        self, name: str, chunk_trip_count: int, kind: str = "stream"
    ) -> bool:
        return self._binding_capacity(kind, name) >= chunk_trip_count

    def _stream_vector_ptr(
        self, name: str, scalar_ty: ir.Type, current: ir.Value, kind: str = "stream"
    ) -> ir.Value | None:
        scalar_ptr = self._load_tick_param_ptr(kind, name, scalar_ty, current)
        if scalar_ptr is None:
            return None
        vector_ty = ir.VectorType(scalar_ty, self.simd_width)
        return self.tick_builder.bitcast(
            scalar_ptr,
            vector_ty.as_pointer(),
            name=f"fused_{_sanitize_symbol(name)}_vector_ptr",
        )

    def _leaf_memory_scalar(self, scalar_ty: ir.Type) -> ir.Type:
        # ``bool`` is an ``i1`` value but occupies one whole byte per row in the
        # arena.  A contiguous ``<N x i1>`` load would be bit-packed (N lanes in
        # one byte) and misread the column, so bool leaves are moved as ``<N x
        # i8>`` and trunc/zext'd at the register boundary.  Every other primitive
        # leaf already matches its byte stride, so its memory type is itself.
        if isinstance(scalar_ty, ir.IntType) and scalar_ty.width == 1:
            return ir.IntType(8)
        return scalar_ty

    def _leaf_supports_contiguous_vector(self, scalar_ty: ir.Type) -> bool:
        # A contiguous ``<N x T>`` memory op over an SoA column is valid when the
        # element's packed layout matches the column's per-row byte stride.  That
        # holds for float/double and byte-or-wider integers directly, and for
        # ``i1`` (bool) via the ``i8`` memory type in ``_leaf_memory_scalar``.
        if isinstance(scalar_ty, ir.IntType):
            return scalar_ty.width == 1 or scalar_ty.width >= 8
        return isinstance(scalar_ty, (ir.FloatType, ir.DoubleType))

    def _leaf_vector_ptr(
        self,
        name: str,
        rel_path: tuple[str, ...],
        scalar_ty: ir.Type,
        current: ir.Value,
        kind: str = "stream",
    ) -> ir.Value | None:
        scalar_ptr = self._leaf_ptr(kind, name, rel_path, scalar_ty, current)
        if scalar_ptr is None:
            return None
        vector_ty = ir.VectorType(scalar_ty, self.simd_width)
        return self.tick_builder.bitcast(
            scalar_ptr,
            vector_ty.as_pointer(),
            name=(
                f"fused_{_sanitize_symbol(name)}_" f"{'_'.join(rel_path)}_vector_ptr"
            ),
        )

    def _leaf_contiguous_vector_load(
        self,
        name: str,
        rel_path: tuple[str, ...],
        scalar_ty: ir.Type,
        current: ir.Value,
        chunk_trip_count: int,
        kind: str,
    ) -> ir.Value | None:
        # Load ``simd_width`` consecutive rows of one SoA leaf column as a single
        # packed vector, or ``None`` when that leaf/chunk can't use the contiguous
        # path (caller falls back to lane-by-lane gather).
        if not (
            self._leaf_supports_contiguous_vector(scalar_ty)
            and self._stream_has_contiguous_vector_chunk(name, chunk_trip_count, kind)
        ):
            return None
        mem_ty = self._leaf_memory_scalar(scalar_ty)
        vector_ptr = self._leaf_vector_ptr(name, rel_path, mem_ty, current, kind)
        if vector_ptr is None:
            return None
        vector_load = self.tick_builder.load(
            vector_ptr,
            name=f"fused_{_sanitize_symbol(name)}_{'_'.join(rel_path)}_vector",
        )
        vector_load.align = 1
        if mem_ty is scalar_ty:
            return vector_load
        return self.tick_builder.trunc(
            vector_load,
            ir.VectorType(scalar_ty, self.simd_width),
            name=f"fused_{_sanitize_symbol(name)}_{'_'.join(rel_path)}_trunc",
        )

    def _leaf_contiguous_vector_store(
        self,
        name: str,
        rel_path: tuple[str, ...],
        value: ir.Value,
        current: ir.Value,
        chunk_trip_count: int,
        kind: str,
    ) -> bool:
        # Store a packed leaf vector back over ``simd_width`` consecutive rows.
        # Returns ``False`` when the contiguous path does not apply so the caller
        # can scatter lane-by-lane instead.
        scalar_ty = value.type.element
        if not (
            self._leaf_supports_contiguous_vector(scalar_ty)
            and self._stream_has_contiguous_vector_chunk(name, chunk_trip_count, kind)
        ):
            return False
        mem_ty = self._leaf_memory_scalar(scalar_ty)
        vector_ptr = self._leaf_vector_ptr(name, rel_path, mem_ty, current, kind)
        if vector_ptr is None:
            return False
        store_value = value
        if mem_ty is not scalar_ty:
            store_value = self.tick_builder.zext(
                value,
                ir.VectorType(mem_ty, self.simd_width),
                name=f"fused_store_{_sanitize_symbol(name)}_{'_'.join(rel_path)}_zext",
            )
        store_inst = self.tick_builder.store(store_value, vector_ptr)
        store_inst.align = 1
        return True

    def _load_stream_vector(
        self,
        name: str,
        scalar_ty: ir.Type,
        current: ir.Value,
        chunk_trip_count: int,
        kind: str = "stream",
    ) -> ir.Value:
        vector_ty = ir.VectorType(scalar_ty, self.simd_width)
        if self._stream_has_contiguous_vector_chunk(name, chunk_trip_count, kind):
            vector_ptr = self._stream_vector_ptr(name, scalar_ty, current, kind)
            if vector_ptr is not None:
                vector_load = self.tick_builder.load(
                    vector_ptr, name=f"fused_{_sanitize_symbol(name)}_vector"
                )
                vector_load.align = 1
                return vector_load

        lane_indices = self._vector_lane_indices(current)
        result = ir.Constant(vector_ty, ir.Undefined)
        for lane in range(self.simd_width):
            lane_index = self.tick_builder.extract_element(
                lane_indices,
                ir.Constant(ir.IntType(32), lane),
                name=f"fused_load_lane_{lane}_idx",
            )
            clamped = self._clamped_stream_index(name, lane_index, kind)
            lane_value = self._load_tick_param(kind, name, scalar_ty, clamped)
            result = self.tick_builder.insert_element(
                result,
                lane_value,
                ir.Constant(ir.IntType(32), lane),
                name=f"fused_{_sanitize_symbol(name)}_lane_{lane}",
            )
        return result

    def _store_stream_vector(
        self,
        name: str,
        value: ir.Value,
        current: ir.Value,
        chunk_trip_count: int,
        kind: str = "stream",
    ) -> None:
        scalar_ty = value.type.element
        if self._stream_has_contiguous_vector_chunk(name, chunk_trip_count, kind):
            vector_ptr = self._stream_vector_ptr(name, scalar_ty, current, kind)
            if vector_ptr is not None:
                vector_store = self.tick_builder.store(value, vector_ptr)
                vector_store.align = 1
                return

        lane_indices = self._vector_lane_indices(current)
        for lane in range(self.simd_width):
            lane_index = self.tick_builder.extract_element(
                lane_indices,
                ir.Constant(ir.IntType(32), lane),
                name=f"fused_store_lane_{lane}_idx",
            )
            clamped = self._clamped_stream_index(name, lane_index, kind)
            ptr = self._load_tick_param_ptr(kind, name, scalar_ty, clamped)
            if ptr is not None:
                lane_value = self.tick_builder.extract_element(
                    value,
                    ir.Constant(ir.IntType(32), lane),
                    name=f"fused_store_lane_{lane}",
                )
                self.tick_builder.store(lane_value, ptr)

    def _load_stream_binding_vectors(
        self,
        name: str,
        type_name: str,
        current: ir.Value,
        chunk_trip_count: int,
        kind: str = "stream",
    ) -> dict[tuple[str, ...], ir.Value]:
        leaves = self._vectorizable_leaf_fields(type_name)
        if leaves is None:
            raise CodegenError(f"type '{type_name}' cannot be SIMD-vectorized")
        loaded: dict[tuple[str, ...], ir.Value] = {}
        for rel_path, (scalar_ty, leaf_type_name) in leaves.items():
            if not rel_path:
                loaded[rel_path] = self._load_stream_vector(
                    name, scalar_ty, current, chunk_trip_count, kind
                )
                continue
            vector_ty = ir.VectorType(scalar_ty, self.simd_width)
            contiguous = self._leaf_contiguous_vector_load(
                name, rel_path, scalar_ty, current, chunk_trip_count, kind
            )
            if contiguous is not None:
                loaded[rel_path] = contiguous
                continue
            lane_indices = self._vector_lane_indices(current)
            result = ir.Constant(vector_ty, ir.Undefined)
            for lane in range(self.simd_width):
                lane_index = self.tick_builder.extract_element(
                    lane_indices,
                    ir.Constant(ir.IntType(32), lane),
                    name=f"fused_load_lane_{lane}_idx",
                )
                clamped = self._clamped_stream_index(name, lane_index, kind)
                lane_value = self._load_value(
                    kind,
                    name,
                    scalar_ty,
                    leaf_type_name,
                    rel_path,
                    clamped,
                )
                result = self.tick_builder.insert_element(
                    result,
                    lane_value,
                    ir.Constant(ir.IntType(32), lane),
                    name=(
                        f"fused_{_sanitize_symbol(name)}_"
                        f"{'_'.join(rel_path)}_lane_{lane}"
                    ),
                )
            loaded[rel_path] = result
        return loaded

    def _store_stream_binding_vectors(
        self,
        name: str,
        type_name: str,
        values: dict[tuple[str, ...], ir.Value],
        current: ir.Value,
        chunk_trip_count: int,
        kind: str = "stream",
    ) -> None:
        leaves = self._vectorizable_leaf_fields(type_name)
        if leaves is None:
            raise CodegenError(f"type '{type_name}' cannot be SIMD-vectorized")
        for rel_path, (_, leaf_type_name) in leaves.items():
            value = values.get(rel_path)
            if value is None:
                continue
            if not rel_path:
                self._store_stream_vector(name, value, current, chunk_trip_count, kind)
                continue
            if self._leaf_contiguous_vector_store(
                name, rel_path, value, current, chunk_trip_count, kind
            ):
                continue
            lane_indices = self._vector_lane_indices(current)
            for lane in range(self.simd_width):
                lane_index = self.tick_builder.extract_element(
                    lane_indices,
                    ir.Constant(ir.IntType(32), lane),
                    name=f"fused_store_lane_{lane}_idx",
                )
                clamped = self._clamped_stream_index(name, lane_index, kind)
                lane_value = self.tick_builder.extract_element(
                    value,
                    ir.Constant(ir.IntType(32), lane),
                    name=(
                        f"fused_store_{_sanitize_symbol(name)}_"
                        f"{'_'.join(rel_path)}_lane_{lane}"
                    ),
                )
                self._store_value(
                    kind,
                    name,
                    lane_value,
                    leaf_type_name,
                    rel_path,
                    clamped,
                )

    def _compress_store_binding_vectors(
        self,
        name: str,
        type_name: str,
        values: dict[tuple[str, ...], ir.Value],
        write_index: ir.Value,
        mask: ir.Value,
    ) -> None:
        # Pack the kept lanes of each SoA leaf column contiguously at
        # ``write_index`` (``vpcompress`` on AVX-512; LLVM expands it elsewhere).
        leaves = self._vectorizable_leaf_fields(type_name)
        if leaves is None:
            raise CodegenError(f"type '{type_name}' cannot be SIMD-vectorized")
        for rel_path in leaves:
            value = values.get(rel_path)
            if value is None:
                continue
            scalar_ty = value.type.element
            mem_ty = self._leaf_memory_scalar(scalar_ty)
            if mem_ty is not scalar_ty:
                value = self.tick_builder.zext(
                    value,
                    ir.VectorType(mem_ty, self.simd_width),
                    name=f"fused_compress_{_sanitize_symbol(name)}_zext",
                )
            ptr = self._leaf_ptr("stream", name, rel_path, mem_ty, write_index)
            if ptr is None:
                continue
            vector_ty = ir.VectorType(mem_ty, self.simd_width)
            compress = self._get_vector_reduce_intrinsic(
                f"llvm.masked.compressstore.v{self.simd_width}{mem_ty.intrinsic_name}",
                ir.VoidType(),
                [
                    vector_ty,
                    mem_ty.as_pointer(),
                    ir.VectorType(ir.IntType(1), self.simd_width),
                ],
            )
            self.tick_builder.call(compress, [value, ptr, mask])
