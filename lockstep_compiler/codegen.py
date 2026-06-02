from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from llvmlite import ir

from .ast import (
    AstAccumulatorDecl,
    AstAssignStmt,
    AstExprBinary,
    AstExprCall,
    AstExprCast,
    AstExprLiteral,
    AstExprUnary,
    AstExprVar,
    AstFoldBindRoute,
    AstKernelBindRoute,
    AstKernelDecl,
    AstKernelParam,
    AstPipelineDecl,
    AstProgram,
    AstPureDecl,
    AstReturnStmt,
    AstStatement,
    AstStreamDecl,
    AstStructDecl,
    AstStructField,
    AstType,
    AstUniformDecl,
    AstVarDeclStmt,
    _normalize_type,
)
from .arena_layout import build_ast_arena_layout
from .optimizer import optimize_bind_routes
from .utils import sanitize_symbol as _sanitize_symbol

_PRIMITIVE_TYPE_MAP: dict[str, ir.Type] = {
    "bool": ir.IntType(1),
    "int": ir.IntType(32),
    "uint": ir.IntType(32),
    "float": ir.FloatType(),
    "double": ir.DoubleType(),
    "string": ir.IntType(8).as_pointer(),
}


class CodegenError(RuntimeError):
    """Raised when IR lowering encounters an invalid or unsupported expression."""


def _type_name(value: AstType | str) -> str:
    return str(value) if isinstance(value, AstType) else value


class _FunctionLowerer:
    def __init__(
        self,
        module: ir.Module,
        function_map: dict[str, ir.Function],
        known_structs: dict[str, ir.IdentifiedStructType] | None = None,
        struct_fields: dict[str, tuple[AstStructField, ...]] | None = None,
        intrinsic_names: set[str] | None = None,
    ):
        self.module = module
        self.function_map = function_map
        self.known_structs = known_structs or {}
        self.struct_fields = struct_fields or {}
        self.intrinsic_names = intrinsic_names or set()
        self.builder: ir.IRBuilder | None = None
        self.locals: dict[str, ir.AllocaInstr] = {}
        self.local_types: dict[str, str] = {}
        self.local_indirections: dict[str, bool] = {}
        self.function_return_types: dict[str, str] = {}
        self.function_param_types: dict[str, list[str | None]] = {}
        self._string_literal_counter = 0

    def _compiler_error(self, message: str) -> None:
        raise CodegenError(message)

    def _declare_llvm_intrinsic(
        self, intrinsic_name: str, arg_type: ir.Type
    ) -> ir.Function:
        if isinstance(arg_type, ir.FloatType):
            suffix = "f32"
        elif isinstance(arg_type, ir.DoubleType):
            suffix = "f64"
        else:
            self._compiler_error(
                f"intrinsic '{intrinsic_name}' expects float or double arguments"
            )
        llvm_name = f"llvm.{intrinsic_name}.{suffix}"
        intrinsic = self.module.globals.get(llvm_name)
        if intrinsic is not None:
            return intrinsic
        fn_type = ir.FunctionType(arg_type, [arg_type, arg_type])
        return ir.Function(self.module, fn_type, name=llvm_name)

    def _lower_intrinsic_call(self, name: str, args: list[ir.Value]) -> ir.Value | None:
        if name == "step" and len(args) == 2:
            if not all(isinstance(arg.type, ir.FloatType) for arg in args):
                self._compiler_error("intrinsic 'step' expects float arguments")
            edge, x_val = args
            cmp_result = self.builder.fcmp_ordered(">=", x_val, edge, name="step_cmp")
            return self.builder.uitofp(cmp_result, ir.FloatType(), name="step")

        if name == "mix" and len(args) == 3:
            if not all(isinstance(arg.type, ir.FloatType) for arg in args):
                self._compiler_error("intrinsic 'mix' expects float arguments")
            a, b, t = args
            one_minus_t = self.builder.fsub(
                ir.Constant(ir.FloatType(), 1.0), t, name="mix_one_minus_t"
            )
            return self.builder.fadd(
                self.builder.fmul(a, one_minus_t),
                self.builder.fmul(b, t),
                name="mix",
            )

        if name == "max" and len(args) == 2:
            if not all(isinstance(arg.type, ir.FloatType) for arg in args):
                self._compiler_error("intrinsic 'max' expects float arguments")
            maxnum = self._declare_llvm_intrinsic("maxnum", ir.FloatType())
            return self.builder.call(maxnum, args, name="max")

        if name == "min" and len(args) == 2:
            if not all(isinstance(arg.type, ir.FloatType) for arg in args):
                self._compiler_error("intrinsic 'min' expects float arguments")
            minnum = self._declare_llvm_intrinsic("minnum", ir.FloatType())
            return self.builder.call(minnum, args, name="min")

        if name == "clamp" and len(args) == 3:
            if not all(isinstance(arg.type, ir.FloatType) for arg in args):
                self._compiler_error("intrinsic 'clamp' expects float arguments")
            x_val, min_value, max_value = args
            maxnum = self._declare_llvm_intrinsic("maxnum", ir.FloatType())
            minnum = self._declare_llvm_intrinsic("minnum", ir.FloatType())
            clamped_min = self.builder.call(
                maxnum, [x_val, min_value], name="clamp_min"
            )
            return self.builder.call(minnum, [clamped_min, max_value], name="clamp")

        if name == "abs" and len(args) == 1:
            if not isinstance(args[0].type, ir.FloatType):
                self._compiler_error("intrinsic 'abs' expects a float argument")
            llvm_name = "llvm.fabs.f32"
            fabs = self.module.globals.get(llvm_name)
            if fabs is None:
                fabs = ir.Function(
                    self.module,
                    ir.FunctionType(ir.FloatType(), [ir.FloatType()]),
                    name=llvm_name,
                )
            return self.builder.call(fabs, args, name="abs")

        if name == "sign" and len(args) == 1:
            if not isinstance(args[0].type, ir.FloatType):
                self._compiler_error("intrinsic 'sign' expects a float argument")
            x = args[0]
            zero = ir.Constant(ir.FloatType(), 0.0)
            pos = self.builder.fcmp_ordered(">", x, zero, name="sign_pos")
            neg = self.builder.fcmp_ordered("<", x, zero, name="sign_neg")
            pos_f = self.builder.uitofp(pos, ir.FloatType(), name="sign_pos_f")
            neg_f = self.builder.uitofp(neg, ir.FloatType(), name="sign_neg_f")
            return self.builder.fsub(pos_f, neg_f, name="sign")

        if name == "smoothstep" and len(args) == 3:
            if not all(isinstance(arg.type, ir.FloatType) for arg in args):
                self._compiler_error("intrinsic 'smoothstep' expects float arguments")
            edge0, edge1, x = args
            # t = clamp((x - edge0) / (edge1 - edge0), 0, 1)
            diff = self.builder.fsub(x, edge0, name="ss_diff")
            range_val = self.builder.fsub(edge1, edge0, name="ss_range")
            t_raw = self.builder.fdiv(diff, range_val, name="ss_t_raw")
            maxnum = self._declare_llvm_intrinsic("maxnum", ir.FloatType())
            minnum = self._declare_llvm_intrinsic("minnum", ir.FloatType())
            t_clamped = self.builder.call(
                maxnum, [t_raw, ir.Constant(ir.FloatType(), 0.0)], name="ss_clamp_lo"
            )
            t = self.builder.call(
                minnum, [t_clamped, ir.Constant(ir.FloatType(), 1.0)], name="ss_t"
            )
            # result = t * t * (3 - 2 * t)
            two_t = self.builder.fmul(ir.Constant(ir.FloatType(), 2.0), t, name="ss_2t")
            three_minus_2t = self.builder.fsub(
                ir.Constant(ir.FloatType(), 3.0), two_t, name="ss_3m2t"
            )
            t_sq = self.builder.fmul(t, t, name="ss_tsq")
            return self.builder.fmul(t_sq, three_minus_2t, name="smoothstep")

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
        fields = self.struct_fields.get(struct_name, [])
        for index, field in enumerate(fields):
            if field.name == field_name:
                declared_type_name = _type_name(field.declared_type)
                return (
                    index,
                    self._llvm_type(declared_type_name, self.known_structs),
                    declared_type_name,
                )
        return None

    def _coerce_value_to_type(
        self,
        value: ir.Value,
        target_type: ir.Type,
        target_type_name: str | None = None,
    ) -> ir.Value:
        if value.type == target_type:
            return value

        if isinstance(target_type, ir.IntType) and isinstance(value.type, ir.IntType):
            if value.type.width < target_type.width:
                if value.type.width == 1:
                    return self.builder.zext(value, target_type)
                return self.builder.sext(value, target_type)
            if value.type.width > target_type.width:
                return self.builder.trunc(value, target_type)

        if isinstance(target_type, (ir.FloatType, ir.DoubleType)) and isinstance(
            value.type, ir.IntType
        ):
            return self.builder.sitofp(value, target_type)

        if isinstance(target_type, ir.IntType) and isinstance(
            value.type, (ir.FloatType, ir.DoubleType)
        ):
            if target_type_name == "uint":
                return self.builder.fptoui(value, target_type)
            return self.builder.fptosi(value, target_type)

        if isinstance(target_type, ir.FloatType) and isinstance(
            value.type, ir.DoubleType
        ):
            return self.builder.fptrunc(value, target_type)

        if isinstance(target_type, ir.DoubleType) and isinstance(
            value.type, ir.FloatType
        ):
            return self.builder.fpext(value, target_type)

        self._compiler_error(
            f"cannot coerce value of type '{value.type}' to '{target_type}'"
        )

    def _extract_field_path(self, value: ir.Value, path: list[str]) -> ir.Value:
        current = value
        for field_name in path:
            field_info = self._field_index_and_type(current.type, field_name)
            if field_info is None:
                self._compiler_error(
                    f"unknown struct field '{field_name}' in access path"
                )
            index, _, _ = field_info
            current = self.builder.extract_value(
                current, index, name=f"{field_name}_field"
            )
        return current

    def _insert_field_path(
        self, aggregate: ir.Value, path: list[str], value: ir.Value
    ) -> ir.Value:
        field_info = self._field_index_and_type(aggregate.type, path[0])
        if field_info is None:
            self._compiler_error(f"unknown struct field '{path[0]}' in assignment path")
        index, field_type, field_type_name = field_info
        if len(path) == 1:
            coerced = self._coerce_value_to_type(value, field_type, field_type_name)
            return self.builder.insert_value(
                aggregate, coerced, index, name=f"set_{path[0]}"
            )

        nested = self.builder.extract_value(aggregate, index, name=f"load_{path[0]}")
        updated_nested = self._insert_field_path(nested, path[1:], value)
        return self.builder.insert_value(
            aggregate, updated_nested, index, name=f"set_{path[0]}"
        )

    def _llvm_type(
        self,
        type_name: AstType | str,
        known_structs: dict[str, ir.IdentifiedStructType],
    ) -> ir.Type:
        ast_type = _normalize_type(type_name)
        if any(suffix.kind == "template" for suffix in ast_type.suffixes):
            return ir.IntType(8).as_pointer()

        if ast_type.name in _PRIMITIVE_TYPE_MAP:
            llvm_type = _PRIMITIVE_TYPE_MAP[ast_type.name]
        elif ast_type.name in known_structs:
            llvm_type = known_structs[ast_type.name]
        else:
            llvm_type = ir.IntType(8).as_pointer()

        for suffix in reversed(ast_type.suffixes):
            if suffix.kind != "array":
                continue
            try:
                element_count = int(suffix.size or "0")
            except ValueError:
                self._compiler_error(
                    f"array type '{ast_type}' has a non-integer element count"
                )
            if element_count < 0:
                self._compiler_error(
                    f"array type '{ast_type}' has a negative element count"
                )
            llvm_type = ir.ArrayType(llvm_type, element_count)
        return llvm_type

    def _emit_numeric_unary_minus(self, value: ir.Value) -> ir.Value:
        if isinstance(value.type, (ir.FloatType, ir.DoubleType)):
            return self.builder.fsub(ir.Constant(value.type, 0.0), value)
        if isinstance(value.type, ir.IntType):
            return self.builder.sub(ir.Constant(value.type, 0), value)
        self._compiler_error(f"unary '-' is unsupported for type '{value.type}'")

    def _emit_numeric_binary(
        self, op: str, lhs: ir.Value, rhs: ir.Value, *, type_name: str | None = None
    ) -> ir.Value:
        if lhs.type != rhs.type:
            self._compiler_error(
                f"operator '{op}' requires matching operand types, got '{lhs.type}' and '{rhs.type}'"
            )

        if isinstance(lhs.type, (ir.FloatType, ir.DoubleType)):
            return {
                "+": self.builder.fadd,
                "-": self.builder.fsub,
                "*": self.builder.fmul,
                "/": self.builder.fdiv,
                "%": self.builder.frem,
            }[op](lhs, rhs)

        if isinstance(lhs.type, ir.IntType):
            is_unsigned = type_name == "uint"
            return {
                "+": self.builder.add,
                "-": self.builder.sub,
                "*": self.builder.mul,
                "/": self.builder.udiv if is_unsigned else self.builder.sdiv,
                "%": self.builder.urem if is_unsigned else self.builder.srem,
            }[op](lhs, rhs)

        self._compiler_error(f"operator '{op}' is unsupported for type '{lhs.type}'")

    @staticmethod
    def _decode_string_literal(token_text: str) -> str:
        if len(token_text) >= 2 and token_text[0] == '"' and token_text[-1] == '"':
            token_text = token_text[1:-1]
        return bytes(token_text, "utf-8").decode("unicode_escape")

    def _lower_string_literal(self, token_text: str) -> ir.Value:
        decoded = self._decode_string_literal(token_text)
        payload = decoded.encode("utf-8") + b"\x00"
        array_type = ir.ArrayType(ir.IntType(8), len(payload))
        global_name = f".str.{self._string_literal_counter}"
        self._string_literal_counter += 1
        global_value = ir.GlobalVariable(self.module, array_type, name=global_name)
        global_value.global_constant = True
        global_value.linkage = "private"
        global_value.unnamed_addr = True
        global_value.initializer = ir.Constant(array_type, bytearray(payload))
        return self.builder.gep(
            global_value,
            [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)],
            inbounds=True,
            name="strptr",
        )

    def _emit_relational_compare(
        self, op: str, lhs: ir.Value, rhs: ir.Value, *, type_name: str | None = None
    ) -> ir.Value:
        if lhs.type != rhs.type:
            self._compiler_error(
                f"comparison '{op}' requires matching operand types, got '{lhs.type}' and '{rhs.type}'"
            )

        rel_map = {"<": "<", "<=": "<=", ">": ">", ">=": ">=", "==": "==", "!=": "!="}
        if isinstance(lhs.type, (ir.FloatType, ir.DoubleType)):
            return self.builder.fcmp_ordered(rel_map[op], lhs, rhs)
        if isinstance(lhs.type, ir.IntType):
            if type_name == "uint":
                return self.builder.icmp_unsigned(rel_map[op], lhs, rhs)
            return self.builder.icmp_signed(rel_map[op], lhs, rhs)
        self._compiler_error(f"comparison '{op}' is unsupported for type '{lhs.type}'")

    def _resolve_field_declared_type(
        self, base_type: str, field_path: list[str]
    ) -> str | None:
        current_type = base_type
        for field_name in field_path:
            fields = self.struct_fields.get(current_type)
            if fields is None:
                return None
            field_type = None
            for field in fields:
                if field.name == field_name:
                    field_type = _type_name(field.declared_type)
                    break
            if field_type is None:
                return None
            current_type = field_type
        return current_type

    @staticmethod
    def _promote_scalar_type_names(
        left_type: str | None, right_type: str | None
    ) -> str | None:
        if left_type is None or right_type is None:
            return left_type or right_type
        if left_type == right_type:
            return left_type
        if left_type in {"int", "uint"} and right_type in {"int", "uint"}:
            return "uint" if "uint" in {left_type, right_type} else "int"
        return None

    def _infer_binary_operand_type(self, node: AstExprBinary) -> str | None:
        left_type = self._infer_expr_type(node.left)
        right_type = self._infer_expr_type(node.right)
        op = node.op

        if op in {"&&", "||"}:
            return "bool" if left_type == "bool" and right_type == "bool" else None
        if op in {"<<", ">>"}:
            if left_type in {"int", "uint"} and (
                right_type in {"int", "uint"} or right_type is None
            ):
                return left_type
            return None
        if op in {"+", "-", "*", "/", "%", "&", "|", "^"}:
            return self._promote_scalar_type_names(left_type, right_type)
        if op in {"<", "<=", ">", ">=", "==", "!="}:
            return self._promote_scalar_type_names(left_type, right_type)
        return None

    def _infer_binary_result_type(self, node: AstExprBinary) -> str | None:
        operand_type = self._infer_binary_operand_type(node)
        if node.op in {"&&", "||"}:
            return "bool" if operand_type == "bool" else None
        if node.op in {"<", "<=", ">", ">=", "==", "!="}:
            return "bool" if operand_type is not None else None
        return operand_type

    def _infer_expr_type(
        self,
        node: (
            AstExprLiteral
            | AstExprVar
            | AstExprUnary
            | AstExprBinary
            | AstExprCall
            | AstExprCast
        ),
    ) -> str | None:
        if isinstance(node, AstExprLiteral):
            if node.kind in {"float", "int", "bool", "double", "uint", "string"}:
                return node.kind
            return None
        if isinstance(node, AstExprVar):
            key = _sanitize_symbol(node.path[0])
            base_type = self.local_types.get(key)
            if base_type is None:
                return None
            if len(node.path) == 1:
                return base_type
            return self._resolve_field_declared_type(base_type, node.path[1:])
        if isinstance(node, AstExprCast):
            return _type_name(node.target_type)
        if isinstance(node, AstExprUnary):
            return self._infer_expr_type(node.operand)
        if isinstance(node, AstExprCall):
            if node.name in {"int", "uint", "float", "double", "bool"}:
                return node.name
            return self.function_return_types.get(f"pure_{_sanitize_symbol(node.name)}")
        if isinstance(node, AstExprBinary):
            return self._infer_binary_result_type(node)
        return None

    def _load_var(self, name: str) -> ir.Value:
        parts = name.split(".")
        base_key = _sanitize_symbol(parts[0])
        if base_key in self.locals:
            base_slot = self.locals[base_key]
            base_value = self.builder.load(base_slot, name=f"{base_key}_val")
            if self.local_indirections.get(base_key):
                base_value = self.builder.load(base_value, name=f"{base_key}_ref")
            if len(parts) == 1:
                return base_value
            return self._extract_field_path(base_value, parts[1:])
        self._compiler_error(f"undefined variable '{parts[0]}'")

    def _lower_call(self, name: str, args: list[ir.Value]) -> ir.Value:
        if name == "select":
            if len(args) != 3:
                self._compiler_error(
                    f"built-in 'select' expects 3 argument(s), got {len(args)}"
                )
            condition, when_true, when_false = args
            if not isinstance(condition.type, ir.IntType) or condition.type.width != 1:
                self._compiler_error("built-in 'select' expects a bool condition")
            if when_true.type != when_false.type:
                self._compiler_error(
                    "built-in 'select' expects matching true/false value types"
                )
            return self.builder.select(condition, when_true, when_false, name="select")

        if name in self.intrinsic_names:
            lowered_intrinsic = self._lower_intrinsic_call(name, args)
            if lowered_intrinsic is not None:
                return lowered_intrinsic
        callee = self.function_map.get(f"pure_{_sanitize_symbol(name)}")
        if callee is not None:
            if len(args) != len(callee.args):
                self._compiler_error(
                    f"function '{name}' expects {len(callee.args)} argument(s), got {len(args)}"
                )
            param_type_names = self.function_param_types.get(
                callee.name, [None] * len(args)
            )
            coerced = [
                self._coerce_value_to_type(arg, param.type, param_type_name)
                for arg, param, param_type_name in zip(
                    args, callee.args, param_type_names
                )
            ]
            return self.builder.call(callee, coerced, name=f"call_{name}")
        self._compiler_error(f"unknown function '{name}'")

    def _lower_binary_op(
        self, op: str, lhs: ir.Value, rhs: ir.Value, *, type_name: str | None = None
    ) -> ir.Value:
        if op in {"+", "-", "*", "/", "%"}:
            return self._emit_numeric_binary(op, lhs, rhs, type_name=type_name)
        if (
            op in {"&", "|", "^", "<<", ">>"}
            and isinstance(lhs.type, ir.IntType)
            and lhs.type == rhs.type
        ):
            if op == "&":
                return self.builder.and_(lhs, rhs)
            if op == "|":
                return self.builder.or_(lhs, rhs)
            if op == "^":
                return self.builder.xor(lhs, rhs)
            if op == "<<":
                return self.builder.shl(lhs, rhs)
            if type_name == "uint":
                return self.builder.lshr(lhs, rhs)
            return self.builder.ashr(lhs, rhs)
        if op in {"<", "<=", ">", ">=", "==", "!="}:
            return self._emit_relational_compare(op, lhs, rhs, type_name=type_name)
        if op == "&&":
            if (
                not isinstance(lhs.type, ir.IntType)
                or lhs.type.width != 1
                or lhs.type != rhs.type
            ):
                self._compiler_error("operator '&&' expects matching boolean operands")
            return self.builder.and_(lhs, rhs)
        if op == "||":
            if (
                not isinstance(lhs.type, ir.IntType)
                or lhs.type.width != 1
                or lhs.type != rhs.type
            ):
                self._compiler_error("operator '||' expects matching boolean operands")
            return self.builder.or_(lhs, rhs)
        self._compiler_error(f"unsupported binary operator '{op}'")

    def _lower_expr(
        self,
        node: (
            AstExprLiteral
            | AstExprVar
            | AstExprUnary
            | AstExprBinary
            | AstExprCall
            | AstExprCast
        ),
    ):

        if isinstance(node, AstExprLiteral):
            if node.kind == "float":
                return ir.Constant(ir.FloatType(), float(node.value))
            if node.kind == "double":
                return ir.Constant(ir.DoubleType(), float(node.value))
            if node.kind in {"int", "uint"}:
                return ir.Constant(ir.IntType(32), int(node.value))
            if node.kind == "string":
                return self._lower_string_literal(node.value)
            return ir.Constant(ir.IntType(1), int(node.value == "true"))
        if isinstance(node, AstExprVar):
            return self._load_var(".".join(node.path))
        if isinstance(node, AstExprUnary):
            operand = self._lower_expr(node.operand)
            if node.op == "-":
                return self._emit_numeric_unary_minus(operand)
            return self.builder.not_(operand)
        if isinstance(node, AstExprCall):
            return self._lower_call(
                node.name, [self._lower_expr(arg) for arg in node.args]
            )
        if isinstance(node, AstExprCast):
            target_type = self._llvm_type(
                _type_name(node.target_type), self.known_structs
            )
            return self._coerce_value_to_type(
                self._lower_expr(node.value), target_type, _type_name(node.target_type)
            )
        if not isinstance(node, AstExprBinary):
            self._compiler_error(f"unsupported expression node '{type(node).__name__}'")
        lhs, rhs = self._lower_expr(node.left), self._lower_expr(node.right)
        expr_type_name = self._infer_binary_operand_type(node)
        if expr_type_name in _PRIMITIVE_TYPE_MAP:
            operand_type = self._llvm_type(expr_type_name, self.known_structs)
            lhs = self._coerce_value_to_type(lhs, operand_type, expr_type_name)
            rhs = self._coerce_value_to_type(rhs, operand_type, expr_type_name)
        return self._lower_binary_op(node.op, lhs, rhs, type_name=expr_type_name)

    def _lower_assignment(self, target_name: str, value: ir.Value):
        base_name, *field_path = target_name.split(".")
        key = _sanitize_symbol(base_name)
        if key not in self.locals:
            self._compiler_error(f"undefined variable '{base_name}' in assignment")
        if not field_path:
            if self.local_indirections.get(key):
                ref_ptr = self.builder.load(self.locals[key], name=f"{key}_ptr")
                slot_type = ref_ptr.type.pointee
                self.builder.store(
                    self._coerce_value_to_type(
                        value, slot_type, self.local_types.get(key)
                    ),
                    ref_ptr,
                )
            else:
                slot_type = self.locals[key].type.pointee
                self.builder.store(
                    self._coerce_value_to_type(
                        value, slot_type, self.local_types.get(key)
                    ),
                    self.locals[key],
                )
            return

        if self.local_indirections.get(key):
            ref_ptr = self.builder.load(self.locals[key], name=f"{key}_ptr")
            current = self.builder.load(ref_ptr, name=f"{key}_ref")
        else:
            current = self.builder.load(self.locals[key], name=f"{key}_val")
        updated = self._insert_field_path(current, field_path, value)
        if self.local_indirections.get(key):
            ref_ptr = self.builder.load(self.locals[key], name=f"{key}_ptr")
            self.builder.store(updated, ref_ptr)
        else:
            self.builder.store(updated, self.locals[key])

    def _lower_statement(
        self,
        statement: AstStatement,
        return_type: ir.Type,
        return_type_name: str | None = None,
    ):
        if isinstance(statement, AstReturnStmt):
            value = self._lower_expr(statement.value)
            if isinstance(return_type, ir.VoidType):
                self.builder.ret_void()
            else:
                self.builder.ret(
                    self._coerce_value_to_type(value, return_type, return_type_name)
                )
            return

        if isinstance(statement, AstAssignStmt):
            self._lower_assignment(
                ".".join(statement.target), self._lower_expr(statement.value)
            )
            return

        if isinstance(statement, AstVarDeclStmt):
            key = _sanitize_symbol(statement.name)
            declared_type = (
                _type_name(statement.declared_type)
                if statement.declared_type
                else "float"
            )
            llvm_type = self._llvm_type(declared_type, self.known_structs)
            self.local_types[key] = declared_type
            if key not in self.locals:
                slot = self.builder.alloca(llvm_type, name=key)
                self.locals[key] = slot
                self.builder.store(ir.Constant(llvm_type, None), slot)
            if statement.initializer is not None:
                value = self._lower_expr(statement.initializer)
                slot_type = self.locals[key].type.pointee
                self.builder.store(
                    self._coerce_value_to_type(
                        value, slot_type, self.local_types.get(key)
                    ),
                    self.locals[key],
                )
            return

        self._compiler_error(f"unsupported statement node '{type(statement).__name__}'")

    def lower_function(
        self,
        fn: ir.Function,
        statements: list[AstStatement],
        return_type: ir.Type,
        param_type_names: list[str] | None = None,
        return_type_name: str | None = None,
        param_by_ref: list[bool] | None = None,
    ):
        block = fn.append_basic_block("entry")
        self.builder = ir.IRBuilder(block)
        self.locals = {}
        self.local_types = {}
        self.local_indirections = {}
        for idx, arg in enumerate(fn.args):
            key = _sanitize_symbol(arg.name)
            slot = self.builder.alloca(arg.type, name=key)
            self.builder.store(arg, slot)
            self.locals[key] = slot
            is_by_ref = (
                bool(param_by_ref[idx])
                if param_by_ref is not None and idx < len(param_by_ref)
                else False
            )
            self.local_indirections[key] = is_by_ref
            if param_type_names is not None and idx < len(param_type_names):
                self.local_types[key] = param_type_names[idx]

        for statement in statements:
            if self.builder.block.is_terminated:
                break
            self._lower_statement(statement, return_type, return_type_name)

        if not self.builder.block.is_terminated:
            if isinstance(return_type, ir.VoidType):
                self.builder.ret_void()
            elif isinstance(return_type, ir.FloatType):
                self.builder.ret(ir.Constant(ir.FloatType(), 0.0))
            else:
                self.builder.ret(ir.Constant(return_type, None))


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


def _program_from_legacy_mapping(
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


def emit_llvm_ir(
    program: AstProgram | Mapping[str, Any],
    *,
    target_width: int | None = None,
    bind_optimization: dict[str, object] | None = None,
) -> str:
    """Generate LLVM IR from the typed Lockstep AST."""

    explicit_accumulator_sizes: dict[str, int] = {}
    if isinstance(program, Mapping):
        program, explicit_accumulator_sizes = _program_from_legacy_mapping(program)
    if not isinstance(program, AstProgram):
        raise TypeError("emit_llvm_ir expects an AstProgram")

    structs = program.structs
    shaders = program.shaders
    filters = program.filters
    pure_functions = program.pure_functions
    streams = _pipeline_streams(program)
    accumulators = _pipeline_accumulators(program)
    uniforms = _pipeline_uniforms(program)

    context = ir.Context()
    module = ir.Module(name="lockstep", context=context)
    module.source_filename = "lockstep"
    module.triple = "x86_64-unknown-linux-gnu"
    known_structs: dict[str, ir.IdentifiedStructType] = {}
    struct_fields: dict[str, tuple[AstStructField, ...]] = {}

    for struct_decl in structs:
        safe_name = _sanitize_symbol(struct_decl.name)
        struct_ty = module.context.get_identified_type(f"struct.{safe_name}")
        known_structs[struct_decl.name] = struct_ty
        struct_fields[struct_decl.name] = struct_decl.fields

    intrinsic_names = {pure.name for pure in pure_functions if pure.intrinsic}

    lowerer = _FunctionLowerer(
        module, {}, known_structs, struct_fields, intrinsic_names
    )

    unresolved = set(known_structs.keys())
    while unresolved:
        progress = False
        for struct_name in list(unresolved):
            field_types: list[ir.Type] = []
            can_lower = True
            for field in struct_fields[struct_name]:
                field_type = field.declared_type
                if field_type.name in known_structs and field_type.name in unresolved:
                    can_lower = False
                    break
                field_types.append(lowerer._llvm_type(field_type, known_structs))
            if not can_lower:
                continue
            if known_structs[struct_name].is_opaque:
                known_structs[struct_name].set_body(*field_types)
            unresolved.remove(struct_name)
            progress = True
        if not progress:
            for struct_name in unresolved:
                if known_structs[struct_name].is_opaque:
                    known_structs[struct_name].set_body(ir.IntType(8))
            break

    function_map: dict[str, ir.Function] = {}
    for pure in pure_functions:
        ret_ty = lowerer._llvm_type(pure.return_type, known_structs)
        params = [
            lowerer._llvm_type(param.declared_type, known_structs)
            for param in pure.params
        ]
        fn = ir.Function(
            module,
            ir.FunctionType(ret_ty, params),
            name=f"pure_{_sanitize_symbol(pure.name)}",
        )
        for idx, param in enumerate(pure.params):
            fn.args[idx].name = _sanitize_symbol(param.name)
        function_map[fn.name] = fn
        lowerer.function_return_types[fn.name] = _type_name(pure.return_type)
        lowerer.function_param_types[fn.name] = [
            _kernel_param_type(param) for param in pure.params
        ]

    for shader in shaders:
        params = [
            _kernel_param_llvm_type(lowerer, param, known_structs)
            for param in shader.params
        ]
        fn = ir.Function(
            module,
            ir.FunctionType(ir.VoidType(), params),
            name=f"shader_{_sanitize_symbol(shader.name)}",
        )
        for idx, param in enumerate(shader.params):
            fn.args[idx].name = _sanitize_symbol(param.name)
        function_map[fn.name] = fn

    for flt in filters:
        params = [
            _kernel_param_llvm_type(lowerer, param, known_structs)
            for param in flt.params
        ]
        fn = ir.Function(
            module,
            ir.FunctionType(ir.VoidType(), params),
            name=f"filter_{_sanitize_symbol(flt.name)}",
        )
        for idx, param in enumerate(flt.params):
            fn.args[idx].name = _sanitize_symbol(param.name)
        function_map[fn.name] = fn

    lowerer.function_map = function_map

    for pure in pure_functions:
        fn = function_map[f"pure_{_sanitize_symbol(pure.name)}"]
        if pure.intrinsic:
            continue
        lowerer.lower_function(
            fn,
            list(pure.body),
            fn.function_type.return_type,
            [_kernel_param_type(param) for param in pure.params],
            _type_name(pure.return_type),
        )

    for shader in shaders:
        fn = function_map[f"shader_{_sanitize_symbol(shader.name)}"]
        lowerer.lower_function(
            fn,
            list(shader.body),
            ir.VoidType(),
            [_kernel_param_type(param) for param in shader.params],
            None,
            [param.modifier in {"out", "accum"} for param in shader.params],
        )

    for flt in filters:
        fn = function_map[f"filter_{_sanitize_symbol(flt.name)}"]
        lowerer.lower_function(
            fn,
            list(flt.body),
            ir.VoidType(),
            [_kernel_param_type(param) for param in flt.params],
            None,
            [param.modifier in {"out", "accum"} for param in flt.params],
        )

    stream_slots: dict[str, int] = {
        stream.name: idx for idx, stream in enumerate(streams)
    }
    stream_capacities: dict[str, int] = {
        stream.name: int(stream.capacity) for stream in streams
    }
    accum_slots: dict[str, int] = {
        accum.name: idx for idx, accum in enumerate(accumulators)
    }
    uniform_slots: dict[str, int] = {
        uniform.name: idx for idx, uniform in enumerate(uniforms)
    }

    kernel_signatures: dict[str, tuple[AstKernelDecl, tuple[AstKernelParam, ...]]] = {
        shader.name: (shader, shader.params) for shader in shaders
    }
    kernel_signatures.update({flt.name: (flt, flt.params) for flt in filters})

    def _infer_accumulator_sizes() -> dict[str, int]:
        inferred_sizes = {accum.name: 1 for accum in accumulators}
        for name, size in explicit_accumulator_sizes.items():
            if name in inferred_sizes:
                inferred_sizes[name] = max(inferred_sizes[name], size)
        for pipeline in program.pipelines:
            pipeline_stream_capacities = {
                stream.name: int(stream.capacity) for stream in pipeline.streams
            }
            pipeline_accumulators = {accum.name for accum in pipeline.accumulators}
            for route in pipeline.bind_routes:
                if not isinstance(route, AstKernelBindRoute):
                    continue
                signature = kernel_signatures.get(route.kernel)
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

    accum_sizes = _infer_accumulator_sizes()
    layout = build_ast_arena_layout(program, accumulator_sizes=accum_sizes)

    leaf_specs: dict[tuple[str, str, tuple[str, ...]], tuple[int, int]] = {
        (leaf.kind, leaf.binding_name, leaf.path): (leaf.offset, leaf.size)
        for leaf in layout.leaves
    }
    binding_declared_types: dict[tuple[str, str], str] = {}
    for stream in streams:
        binding_declared_types[("stream", stream.name)] = _type_name(
            stream.declared_type
        )
    for accum in accumulators:
        binding_declared_types[("accum", accum.name)] = _type_name(accum.declared_type)
    for uniform in uniforms:
        binding_declared_types[("uniform", uniform.name)] = _type_name(
            uniform.declared_type
        )

    arena_struct_ty = module.context.get_identified_type("struct.Lockstep_Arena")

    arena_field_types: list[ir.Type] = []
    for leaf in layout.leaves:
        if leaf.type_name in _PRIMITIVE_TYPE_MAP:
            field_ty = _PRIMITIVE_TYPE_MAP[leaf.type_name]
        elif (
            leaf.type_name in known_structs
            and leaf.type_name not in layout.opaque_structs
        ):
            field_ty = known_structs[leaf.type_name]
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

    tick = ir.Function(
        module,
        ir.FunctionType(ir.VoidType(), [arena_struct_ty.as_pointer()]),
        name="Lockstep_Tick",
    )
    arena_ptr = tick.args[0]
    arena_ptr.name = "arena"

    tick_entry = tick.append_basic_block("entry")
    tick_builder = ir.IRBuilder(tick_entry)
    simd_width = (
        int(target_width)
        if target_width is not None and int(target_width) > 0
        else _simd_width_for_target_triple(module.triple)
    )

    def _zero_value(llvm_type: ir.Type) -> ir.Value:
        if isinstance(llvm_type, ir.VoidType):
            return ir.Constant(ir.IntType(32), 0)
        return ir.Constant(llvm_type, None)

    def _leaf_ptr(
        kind: str,
        name: str,
        path: tuple[str, ...],
        leaf_type: ir.Type,
        loop_index_reg: ir.Value | None = None,
    ) -> ir.Value | None:
        leaf_key = (kind, name, path)
        if leaf_key not in leaf_specs:
            return None
        leaf_offset, leaf_size = leaf_specs[leaf_key]
        byte_offset: ir.Value = ir.Constant(ir.IntType(32), leaf_offset)
        if kind in {"stream", "accum"} and loop_index_reg is not None:
            index_type = loop_index_reg.type
            if isinstance(index_type, ir.IntType) and index_type != byte_offset.type:
                byte_offset = ir.Constant(index_type, leaf_offset)
            stride = ir.Constant(byte_offset.type, leaf_size)
            scaled_index = tick_builder.mul(
                loop_index_reg,
                stride,
                name=f"{kind}_{_sanitize_symbol(name)}_byte_index",
            )
            byte_offset = tick_builder.add(
                byte_offset,
                scaled_index,
                name=f"{kind}_{_sanitize_symbol(name)}_byte_offset",
            )

        # Address leaves with raw byte arithmetic.  The arena struct layout is
        # still useful for ABI/type descriptions, but using it as the GEP base
        # would make LLVM scale offsets by the selected field type instead of
        # treating ``leaf_offset`` as an exact byte position.
        arena_byte_ptr_type = ir.IntType(8).as_pointer()
        bytes_ptr = tick_builder.bitcast(
            arena_ptr,
            arena_byte_ptr_type,
            name=f"{kind}_{_sanitize_symbol(name)}_arena_bytes",
        )
        leaf_byte_addr = tick_builder.gep(
            bytes_ptr,
            [byte_offset],
            name=f"{kind}_{_sanitize_symbol(name)}_{'_'.join(path) if path else 'value'}_byte_ptr",
        )
        typed_ptr = tick_builder.bitcast(leaf_byte_addr, leaf_type.as_pointer())
        if typed_ptr.type.pointee != leaf_type:
            return None
        return typed_ptr

    def _load_value(
        kind: str,
        name: str,
        value_type: ir.Type,
        declared_type: str,
        path: tuple[str, ...] = (),
        element_index: ir.Value | None = None,
    ) -> ir.Value:
        if declared_type in struct_fields and isinstance(
            value_type, ir.IdentifiedStructType
        ):
            aggregate = ir.Constant(value_type, ir.Undefined)
            fields = struct_fields.get(declared_type, ())
            for index, element_type in enumerate(value_type.elements):
                field = fields[index] if index < len(fields) else None
                child_name = field.name if field is not None else f"field{index}"
                child_type = (
                    _type_name(field.declared_type) if field is not None else "float"
                )
                field_value = _load_value(
                    kind,
                    name,
                    element_type,
                    child_type,
                    path + (child_name,),
                    element_index,
                )
                aggregate = tick_builder.insert_value(aggregate, field_value, index)
            return aggregate

        ptr = _leaf_ptr(kind, name, path, value_type, element_index)
        if ptr is None:
            return _zero_value(value_type)
        return tick_builder.load(ptr, name=f"{kind}_{_sanitize_symbol(name)}_val")

    def _store_value(
        kind: str,
        name: str,
        value: ir.Value,
        declared_type: str,
        path: tuple[str, ...] = (),
        element_index: ir.Value | None = None,
    ):
        value_type = value.type
        if declared_type in struct_fields and isinstance(
            value_type, ir.IdentifiedStructType
        ):
            fields = struct_fields.get(declared_type, ())
            for index, element_type in enumerate(value_type.elements):
                part = tick_builder.extract_value(value, index)
                field = fields[index] if index < len(fields) else None
                child_name = field.name if field is not None else f"field{index}"
                child_type = (
                    _type_name(field.declared_type) if field is not None else "float"
                )
                _store_value(
                    kind, name, part, child_type, path + (child_name,), element_index
                )
            return
        ptr = _leaf_ptr(kind, name, path, value_type, element_index)
        if ptr is None:
            return
        tick_builder.store(value, ptr)

    def _load_tick_param(
        kind: str,
        name: str,
        field_type: ir.Type,
        element_index: ir.Value | None = None,
    ) -> ir.Value:
        declared_type = binding_declared_types.get((kind, name), "float")
        return _load_value(
            kind, name, field_type, declared_type, element_index=element_index
        )

    def _load_tick_param_ptr(
        kind: str,
        name: str,
        field_type: ir.Type,
        element_index: ir.Value | None = None,
    ) -> ir.Value:
        return _leaf_ptr(kind, name, (), field_type, element_index)

    def _get_vector_reduce_intrinsic(
        name: str, ret_ty: ir.Type, arg_tys: list[ir.Type]
    ) -> ir.Function:
        fn_ty = ir.FunctionType(ret_ty, arg_tys)
        intrinsic = module.globals.get(name)
        if intrinsic is None:
            intrinsic = ir.Function(module, fn_ty, name=name)
        return intrinsic

    def _vector_i32_splat(value: ir.Value, width: int = 4) -> ir.Value:
        vec_ty = ir.VectorType(ir.IntType(32), width)
        seed = tick_builder.insert_element(
            ir.Constant(vec_ty, ir.Undefined), value, ir.Constant(ir.IntType(32), 0)
        )
        mask_ty = ir.VectorType(ir.IntType(32), width)
        mask = ir.Constant(mask_ty, [0] * width)
        return tick_builder.shuffle_vector(
            seed, ir.Constant(vec_ty, ir.Undefined), mask, name="route_i32_splat"
        )

    def _vector_i32_extract_lane0(vector: ir.Value) -> ir.Value:
        return tick_builder.extract_element(
            vector, ir.Constant(ir.IntType(32), 0), name="route_i32_lane0"
        )

    def _vector_i32_max(lhs: ir.Value, rhs: ir.Value, width: int = 4) -> ir.Value:
        lhs_vec = _vector_i32_splat(lhs, width)
        rhs_vec = _vector_i32_splat(rhs, width)
        pred = tick_builder.icmp_signed(">", lhs_vec, rhs_vec, name="route_vec_max_cmp")
        merged = tick_builder.select(pred, lhs_vec, rhs_vec, name="route_vec_max")
        return _vector_i32_extract_lane0(merged)

    def _vector_i32_min(lhs: ir.Value, rhs: ir.Value, width: int = 4) -> ir.Value:
        lhs_vec = _vector_i32_splat(lhs, width)
        rhs_vec = _vector_i32_splat(rhs, width)
        pred = tick_builder.icmp_signed("<", lhs_vec, rhs_vec, name="route_vec_min_cmp")
        merged = tick_builder.select(pred, lhs_vec, rhs_vec, name="route_vec_min")
        return _vector_i32_extract_lane0(merged)

    def _vector_i32_clamp(value: ir.Value, lo: ir.Value, hi: ir.Value) -> ir.Value:
        return _vector_i32_min(_vector_i32_max(value, lo), hi)

    def _reduce_fold(
        operator: str, source_name: str, uniform_type: ir.Type
    ) -> ir.Value:
        lane_count = max(int(accum_sizes.get(source_name, 1)), 1)
        vector_ty = ir.VectorType(uniform_type, simd_width)

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
            return _zero_value(uniform_type)

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
        vector_accumulator = ir.Constant(vector_ty, [identity_value] * simd_width)

        def _load_accum_chunk_vector(chunk_vector_ptr: ir.Value) -> ir.Value:
            # The strip-mined fold consumes a contiguous SIMD-width chunk of the
            # accumulator.  The strip loop carries a vector pointer induction
            # variable, so the loop body performs one uniform vector load and
            # one vector-strided pointer increment instead of rebuilding scalar
            # byte offsets independently for every SIMD lane.
            return tick_builder.load(
                chunk_vector_ptr, name=f"fold_{_sanitize_symbol(source_name)}_chunk"
            )

        def _insert_accum_chunk_lane(
            vector_value: ir.Value, lane: int, element_index: ir.Value
        ) -> ir.Value:
            lane_value = _load_tick_param(
                "accum",
                source_name,
                uniform_type,
                element_index,
            )
            return tick_builder.insert_element(
                vector_value,
                lane_value,
                ir.Constant(ir.IntType(32), lane),
                name=f"fold_lane_{lane}",
            )

        def _combine_vectors(lhs: ir.Value, rhs: ir.Value, name: str) -> ir.Value:
            if operator in {"sum", "avg"}:
                if is_float:
                    return tick_builder.fadd(lhs, rhs, name=name)
                return tick_builder.add(lhs, rhs, name=name)
            if operator == "min":
                if is_float:
                    predicate = tick_builder.fcmp_ordered(
                        "<", rhs, lhs, name=f"{name}_cmp"
                    )
                else:
                    predicate = tick_builder.icmp_signed(
                        "<", rhs, lhs, name=f"{name}_cmp"
                    )
                return tick_builder.select(predicate, rhs, lhs, name=name)
            if operator == "max":
                if is_float:
                    predicate = tick_builder.fcmp_ordered(
                        ">", rhs, lhs, name=f"{name}_cmp"
                    )
                else:
                    predicate = tick_builder.icmp_signed(
                        ">", rhs, lhs, name=f"{name}_cmp"
                    )
                return tick_builder.select(predicate, rhs, lhs, name=name)
            return lhs

        # Strip-mine accumulator buffers that are wider than the active hardware
        # vector width.  Each loop iteration folds one vector-sized block into a
        # lane-wise partial accumulator; the horizontal reduction runs only once
        # after the strip loop and any scalar tail have been merged.
        full_chunk_limit = (lane_count // simd_width) * simd_width
        if full_chunk_limit:
            first_chunk_ptr = _leaf_ptr("accum", source_name, (), uniform_type)
            if first_chunk_ptr is not None:
                first_chunk_vector_ptr = tick_builder.bitcast(
                    first_chunk_ptr,
                    vector_ty.as_pointer(),
                    name=f"fold_{_sanitize_symbol(source_name)}_chunk_ptr",
                )
                preheader_block = tick_builder.block
                loop_cond = tick.append_basic_block(
                    f"fold_{_sanitize_symbol(source_name)}_strip_cond"
                )
                loop_body = tick.append_basic_block(
                    f"fold_{_sanitize_symbol(source_name)}_strip_body"
                )
                loop_exit = tick.append_basic_block(
                    f"fold_{_sanitize_symbol(source_name)}_strip_exit"
                )

                tick_builder.branch(loop_cond)
                tick_builder.position_at_end(loop_cond)
                loop_index = tick_builder.phi(ir.IntType(32), name="fold_index")
                loop_chunk_ptr = tick_builder.phi(
                    first_chunk_vector_ptr.type, name="fold_chunk_ptr"
                )
                loop_accumulator = tick_builder.phi(vector_ty, name="fold_vector_acc")
                loop_index.add_incoming(
                    ir.Constant(ir.IntType(32), 0), preheader_block
                )
                loop_chunk_ptr.add_incoming(first_chunk_vector_ptr, preheader_block)
                loop_accumulator.add_incoming(vector_accumulator, preheader_block)
                in_full_chunks = tick_builder.icmp_unsigned(
                    "<",
                    loop_index,
                    ir.Constant(ir.IntType(32), full_chunk_limit),
                    name="fold_has_full_chunk",
                )
                tick_builder.cbranch(in_full_chunks, loop_body, loop_exit)

                tick_builder.position_at_end(loop_body)
                chunk_vector = _load_accum_chunk_vector(loop_chunk_ptr)
                next_accumulator = _combine_vectors(
                    loop_accumulator, chunk_vector, "fold_vector_next"
                )
                next_index = tick_builder.add(
                    loop_index,
                    ir.Constant(ir.IntType(32), simd_width),
                    name="fold_index_next",
                )
                next_chunk_ptr = tick_builder.gep(
                    loop_chunk_ptr,
                    [ir.Constant(ir.IntType(32), 1)],
                    name="fold_chunk_ptr_next",
                )
                tick_builder.branch(loop_cond)
                loop_index.add_incoming(next_index, tick_builder.block)
                loop_chunk_ptr.add_incoming(next_chunk_ptr, tick_builder.block)
                loop_accumulator.add_incoming(next_accumulator, tick_builder.block)

                tick_builder.position_at_end(loop_exit)
                vector_accumulator = loop_accumulator

        tail_count = lane_count - full_chunk_limit
        if tail_count:
            tail_vector = ir.Constant(vector_ty, [identity_value] * simd_width)
            for lane in range(tail_count):
                tail_vector = _insert_accum_chunk_lane(
                    tail_vector,
                    lane,
                    ir.Constant(ir.IntType(32), full_chunk_limit + lane),
                )
            vector_accumulator = _combine_vectors(
                vector_accumulator, tail_vector, "fold_tail_acc"
            )

        intrinsic_name = f"llvm.vector.reduce.{intrinsic_suffix}.v{simd_width}{uniform_type.intrinsic_name}"
        # fadd requires a starting accumulator argument
        needs_start_value = is_float and operator in {"sum", "avg"}
        if needs_start_value:
            intrinsic = _get_vector_reduce_intrinsic(
                intrinsic_name, uniform_type, [uniform_type, vector_ty]
            )
            reduced = tick_builder.call(
                intrinsic,
                [ir.Constant(uniform_type, 0.0), vector_accumulator],
                name="fold_reduce",
            )
        else:
            intrinsic = _get_vector_reduce_intrinsic(
                intrinsic_name, uniform_type, [vector_ty]
            )
            reduced = tick_builder.call(
                intrinsic, [vector_accumulator], name="fold_reduce"
            )

        if is_float:
            reduced.fastmath.add("fast")

        if operator == "avg":
            if is_float:
                reduced = tick_builder.fdiv(
                    reduced,
                    ir.Constant(uniform_type, float(lane_count)),
                    name="fold_avg",
                )
            else:
                reduced = tick_builder.sdiv(
                    reduced, ir.Constant(uniform_type, lane_count), name="fold_avg"
                )

        return reduced

    def _kernel_function_and_params(
        kernel_name: str,
    ) -> tuple[ir.Function | None, tuple[AstKernelParam, ...]]:
        callee = function_map.get(
            f"shader_{_sanitize_symbol(kernel_name)}"
        ) or function_map.get(f"filter_{_sanitize_symbol(kernel_name)}")
        signature = kernel_signatures.get(kernel_name)
        return callee, signature[1] if signature is not None else ()

    def _kernel_route_trip_count(route: AstKernelBindRoute) -> int:
        callee, params = _kernel_function_and_params(route.kernel)
        if callee is None:
            raise CodegenError(
                f"undefined shader/filter '{route.kernel}' in bind route: {route.route}"
            )
        trip_count = 0
        for index, arg_name in enumerate(route.args):
            if index >= len(params):
                break
            if params[index].modifier == "in" and arg_name in stream_capacities:
                trip_count = max(trip_count, stream_capacities[arg_name])
        if route.target in stream_capacities:
            trip_count = max(trip_count, stream_capacities[route.target])
        return max(trip_count, 1)

    def _clamped_stream_index(name: str, current: ir.Value) -> ir.Value:
        raw_capacity = stream_capacities.get(name, 0)
        safe_capacity = max(int(raw_capacity), 1)
        max_index = ir.Constant(ir.IntType(32), safe_capacity - 1)
        return _vector_i32_clamp(current, ir.Constant(ir.IntType(32), 0), max_index)

    def _route_arg_value(
        *,
        arg_name: str,
        param: ir.Argument,
        modifier: str | None,
        current: ir.Value,
        local_slots: dict[str, ir.AllocaInstr] | None = None,
    ) -> ir.Value:
        local_slots = local_slots or {}
        if modifier == "in" and arg_name in local_slots:
            return tick_builder.load(
                local_slots[arg_name], name=f"fused_{_sanitize_symbol(arg_name)}"
            )
        if modifier == "out" and arg_name in local_slots:
            return local_slots[arg_name]
        if modifier in {"in", "out"} and arg_name in stream_slots:
            clamped_index = _clamped_stream_index(arg_name, current)
            if modifier == "out":
                return _load_tick_param_ptr(
                    "stream", arg_name, param.type.pointee, clamped_index
                )
            return _load_tick_param("stream", arg_name, param.type, clamped_index)
        if modifier == "accum" and arg_name in accum_slots:
            return _load_tick_param_ptr("accum", arg_name, param.type.pointee, current)
        if modifier == "uniform" and arg_name in uniform_slots:
            return _load_tick_param("uniform", arg_name, param.type)
        return _zero_value(param.type)

    def _emit_kernel_call(
        route: AstKernelBindRoute,
        current: ir.Value,
        *,
        local_slots: dict[str, ir.AllocaInstr] | None = None,
    ) -> None:
        callee, params = _kernel_function_and_params(route.kernel)
        if callee is None:
            return
        call_args = []
        for index, param in enumerate(callee.args):
            arg_name = route.args[index] if index < len(route.args) else ""
            modifier = params[index].modifier if index < len(params) else None
            call_args.append(
                _route_arg_value(
                    arg_name=arg_name,
                    param=param,
                    modifier=modifier,
                    current=current,
                    local_slots=local_slots,
                )
            )
        tick_builder.call(callee, call_args)

    def _lower_kernel_route(route: AstKernelBindRoute):
        trip_count = _kernel_route_trip_count(route)
        kernel_name = route.kernel

        index_ptr = tick_builder.alloca(
            ir.IntType(32), name=f"{_sanitize_symbol(kernel_name)}_idx"
        )
        tick_builder.store(ir.Constant(ir.IntType(32), 0), index_ptr)

        loop_cond = tick.append_basic_block(
            f"route_{_sanitize_symbol(kernel_name)}_cond"
        )
        loop_body = tick.append_basic_block(
            f"route_{_sanitize_symbol(kernel_name)}_body"
        )
        loop_exit = tick.append_basic_block(
            f"route_{_sanitize_symbol(kernel_name)}_exit"
        )
        tick_builder.branch(loop_cond)

        tick_builder.position_at_end(loop_cond)
        current = tick_builder.load(index_ptr, name="idx")
        cond = tick_builder.icmp_signed(
            "<", current, ir.Constant(ir.IntType(32), trip_count), name="route_active"
        )
        tick_builder.cbranch(cond, loop_body, loop_exit)

        tick_builder.position_at_end(loop_body)
        _emit_kernel_call(route, current)
        next_index = tick_builder.add(
            current, ir.Constant(ir.IntType(32), 1), name="idx_next"
        )
        tick_builder.store(next_index, index_ptr)
        tick_builder.branch(loop_cond)

        tick_builder.position_at_end(loop_exit)

    def _vector_type_for_scalar(scalar_type: ir.Type) -> ir.VectorType | None:
        if isinstance(
            scalar_type, (ir.FloatType, ir.DoubleType, ir.IntType)
        ) and not isinstance(scalar_type, ir.VectorType):
            return ir.VectorType(scalar_type, simd_width)
        return None

    def _splat_to_vector(
        value: ir.Value, vector_ty: ir.VectorType, type_name: str | None = None
    ) -> ir.Value:
        if value.type == vector_ty:
            return value
        if value.type != vector_ty.element:
            value = lowerer._coerce_value_to_type(value, vector_ty.element, type_name)
        seed = tick_builder.insert_element(
            ir.Constant(vector_ty, ir.Undefined), value, ir.Constant(ir.IntType(32), 0)
        )
        mask = ir.Constant(ir.VectorType(ir.IntType(32), simd_width), [0] * simd_width)
        return tick_builder.shuffle_vector(
            seed, ir.Constant(vector_ty, ir.Undefined), mask, name="fused_splat"
        )

    def _coerce_vector_value(
        value: ir.Value, target_ty: ir.Type, type_name: str | None = None
    ) -> ir.Value:
        if value.type == target_ty:
            return value
        if isinstance(target_ty, ir.VectorType):
            if not isinstance(value.type, ir.VectorType):
                return _splat_to_vector(value, target_ty, type_name)
            source_elem = value.type.element
            target_elem = target_ty.element
            if source_elem == target_elem:
                return value
            if isinstance(target_elem, ir.IntType) and isinstance(
                source_elem, ir.IntType
            ):
                if source_elem.width < target_elem.width:
                    return (
                        tick_builder.zext(value, target_ty)
                        if source_elem.width == 1
                        else tick_builder.sext(value, target_ty)
                    )
                if source_elem.width > target_elem.width:
                    return tick_builder.trunc(value, target_ty)
            if isinstance(target_elem, (ir.FloatType, ir.DoubleType)) and isinstance(
                source_elem, ir.IntType
            ):
                return tick_builder.sitofp(value, target_ty)
            if isinstance(target_elem, ir.IntType) and isinstance(
                source_elem, (ir.FloatType, ir.DoubleType)
            ):
                return (
                    tick_builder.fptoui(value, target_ty)
                    if type_name == "uint"
                    else tick_builder.fptosi(value, target_ty)
                )
            if isinstance(target_elem, ir.FloatType) and isinstance(
                source_elem, ir.DoubleType
            ):
                return tick_builder.fptrunc(value, target_ty)
            if isinstance(target_elem, ir.DoubleType) and isinstance(
                source_elem, ir.FloatType
            ):
                return tick_builder.fpext(value, target_ty)
        if isinstance(value.type, ir.VectorType):
            raise CodegenError(
                f"cannot coerce vector value of type '{value.type}' to '{target_ty}'"
            )
        return lowerer._coerce_value_to_type(value, target_ty, type_name)

    def _can_vectorize_fused_group(routes: tuple[AstKernelBindRoute, ...]) -> bool:
        supported_statements = (AstAssignStmt, AstVarDeclStmt)
        supported_exprs = (
            AstExprLiteral,
            AstExprVar,
            AstExprUnary,
            AstExprBinary,
            AstExprCast,
            AstExprCall,
        )

        def expr_supported(expr) -> bool:
            if not isinstance(expr, supported_exprs):
                return False
            if isinstance(expr, (AstExprLiteral, AstExprVar)):
                return True
            if isinstance(expr, AstExprUnary):
                return expr_supported(expr.operand)
            if isinstance(expr, AstExprBinary):
                return expr_supported(expr.left) and expr_supported(expr.right)
            if isinstance(expr, AstExprCast):
                return expr_supported(expr.value)
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
            signature = kernel_signatures.get(route.kernel)
            if signature is None:
                return False
            kernel_decl, params = signature
            for param in params:
                param_ty = lowerer._llvm_type(param.declared_type, known_structs)
                if _vector_type_for_scalar(param_ty) is None:
                    return False
                if param.modifier == "accum":
                    return False
            for statement in kernel_decl.body:
                if not isinstance(statement, supported_statements):
                    return False
                if isinstance(statement, AstAssignStmt):
                    if len(statement.target) != 1 or not expr_supported(
                        statement.value
                    ):
                        return False
                elif isinstance(statement, AstVarDeclStmt):
                    declared = statement.declared_type or AstType("float")
                    if (
                        _vector_type_for_scalar(
                            lowerer._llvm_type(declared, known_structs)
                        )
                        is None
                    ):
                        return False
                    if statement.initializer is not None and not expr_supported(
                        statement.initializer
                    ):
                        return False
        return True

    class _FusedVectorLowerer:
        def __init__(self):
            self.values: dict[str, ir.Value] = {}
            self.types: dict[str, str] = {}

        def _declared_vector_type(self, type_name: AstType | str) -> ir.VectorType:
            scalar_ty = lowerer._llvm_type(type_name, known_structs)
            vector_ty = _vector_type_for_scalar(scalar_ty)
            if vector_ty is None:
                raise CodegenError(f"type '{type_name}' cannot be SIMD-vectorized")
            return vector_ty

        @staticmethod
        def _promote_scalar_type_names(
            left_type: str | None, right_type: str | None
        ) -> str | None:
            if left_type is None or right_type is None:
                return left_type or right_type
            if left_type == right_type:
                return left_type
            if left_type in {"int", "uint"} and right_type in {"int", "uint"}:
                return "uint" if "uint" in {left_type, right_type} else "int"
            return None

        def _infer_binary_operand_type(self, node: AstExprBinary) -> str | None:
            left_type = self._infer_expr_type(node.left)
            right_type = self._infer_expr_type(node.right)
            op = node.op
            if op in {"&&", "||"}:
                return "bool" if left_type == "bool" and right_type == "bool" else None
            if op in {"<<", ">>"}:
                return left_type if left_type in {"int", "uint"} else None
            if op in {
                "+",
                "-",
                "*",
                "/",
                "%",
                "&",
                "|",
                "^",
                "<",
                "<=",
                ">",
                ">=",
                "==",
                "!=",
            }:
                return self._promote_scalar_type_names(left_type, right_type)
            return None

        def _infer_expr_type(self, node) -> str | None:
            if isinstance(node, AstExprLiteral):
                return node.kind if node.kind in _PRIMITIVE_TYPE_MAP else None
            if isinstance(node, AstExprVar):
                return self.types.get(_sanitize_symbol(node.path[0]))
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
            return _splat_to_vector(scalar, ir.VectorType(scalar.type, simd_width))

        def _load_var(self, node: AstExprVar) -> ir.Value:
            if len(node.path) != 1:
                raise CodegenError("struct field access cannot be SIMD-vectorized")
            key = _sanitize_symbol(node.path[0])
            if key not in self.values:
                raise CodegenError(f"undefined vector variable '{node.path[0]}'")
            return self.values[key]

        def _numeric_unary_minus(self, value: ir.Value) -> ir.Value:
            zero = ir.Constant(value.type, None)
            if isinstance(value.type.element, (ir.FloatType, ir.DoubleType)):
                return tick_builder.fsub(zero, value, name="fused_neg")
            if isinstance(value.type.element, ir.IntType):
                return tick_builder.sub(zero, value, name="fused_neg")
            raise CodegenError(f"unary '-' is unsupported for type '{value.type}'")

        def _binary(
            self, op: str, lhs: ir.Value, rhs: ir.Value, type_name: str | None
        ) -> ir.Value:
            if isinstance(lhs.type, ir.VectorType) and not isinstance(
                rhs.type, ir.VectorType
            ):
                rhs = _splat_to_vector(rhs, lhs.type)
            elif isinstance(rhs.type, ir.VectorType) and not isinstance(
                lhs.type, ir.VectorType
            ):
                lhs = _splat_to_vector(lhs, rhs.type)
            if lhs.type != rhs.type:
                raise CodegenError(
                    f"operator '{op}' requires matching vector operand types, got '{lhs.type}' and '{rhs.type}'"
                )
            elem_ty = lhs.type.element
            if op in {"+", "-", "*", "/", "%"}:
                if isinstance(elem_ty, (ir.FloatType, ir.DoubleType)):
                    return {
                        "+": tick_builder.fadd,
                        "-": tick_builder.fsub,
                        "*": tick_builder.fmul,
                        "/": tick_builder.fdiv,
                        "%": tick_builder.frem,
                    }[op](lhs, rhs, name="fused_math")
                if isinstance(elem_ty, ir.IntType):
                    return {
                        "+": tick_builder.add,
                        "-": tick_builder.sub,
                        "*": tick_builder.mul,
                        "/": (
                            tick_builder.udiv
                            if type_name == "uint"
                            else tick_builder.sdiv
                        ),
                        "%": (
                            tick_builder.urem
                            if type_name == "uint"
                            else tick_builder.srem
                        ),
                    }[op](lhs, rhs, name="fused_math")
            if op in {"&", "|", "^", "<<", ">>"} and isinstance(elem_ty, ir.IntType):
                if op == "&":
                    return tick_builder.and_(lhs, rhs, name="fused_and")
                if op == "|":
                    return tick_builder.or_(lhs, rhs, name="fused_or")
                if op == "^":
                    return tick_builder.xor(lhs, rhs, name="fused_xor")
                if op == "<<":
                    return tick_builder.shl(lhs, rhs, name="fused_shl")
                return (
                    tick_builder.lshr if type_name == "uint" else tick_builder.ashr
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
                    return tick_builder.fcmp_ordered(
                        rel_map[op], lhs, rhs, name="fused_cmp"
                    )
                if isinstance(elem_ty, ir.IntType):
                    cmp_op = (
                        tick_builder.icmp_unsigned
                        if type_name == "uint"
                        else tick_builder.icmp_signed
                    )
                    return cmp_op(rel_map[op], lhs, rhs, name="fused_cmp")
            if op == "&&" or op == "||":
                return (tick_builder.and_ if op == "&&" else tick_builder.or_)(
                    lhs, rhs, name="fused_bool"
                )
            raise CodegenError(f"unsupported vector binary operator '{op}'")

        def _call(self, name: str, args: list[ir.Value]) -> ir.Value:
            if name in {"int", "uint", "float", "double", "bool"} and len(args) == 1:
                return _coerce_vector_value(
                    args[0], self._declared_vector_type(name), name
                )
            if name == "select" and len(args) == 3:
                return tick_builder.select(
                    args[0], args[1], args[2], name="fused_select"
                )
            if name == "mix" and len(args) == 3:
                a, b, t = args
                one = _splat_to_vector(ir.Constant(ir.FloatType(), 1.0), a.type)
                return tick_builder.fadd(
                    tick_builder.fmul(
                        a, tick_builder.fsub(one, t, name="fused_mix_omt")
                    ),
                    tick_builder.fmul(b, t),
                    name="fused_mix",
                )
            if name == "step" and len(args) == 2:
                edge, x_val = args
                cmp_result = tick_builder.fcmp_ordered(
                    ">=", x_val, edge, name="fused_step_cmp"
                )
                return tick_builder.uitofp(cmp_result, x_val.type, name="fused_step")
            if name in {"min", "max"} and len(args) == 2:
                lhs, rhs = args
                pred = tick_builder.fcmp_ordered(
                    "<" if name == "min" else ">", lhs, rhs, name=f"fused_{name}_cmp"
                )
                return tick_builder.select(pred, lhs, rhs, name=f"fused_{name}")
            if name == "clamp" and len(args) == 3:
                x_val, lo, hi = args
                lower = self._call("max", [x_val, lo])
                return self._call("min", [lower, hi])
            if name == "abs" and len(args) == 1:
                x_val = args[0]
                zero = ir.Constant(x_val.type, None)
                neg = tick_builder.fsub(zero, x_val, name="fused_abs_neg")
                pred = tick_builder.fcmp_ordered("<", x_val, zero, name="fused_abs_cmp")
                return tick_builder.select(pred, neg, x_val, name="fused_abs")
            if name == "sign" and len(args) == 1:
                x_val = args[0]
                zero = ir.Constant(x_val.type, None)
                one = _splat_to_vector(ir.Constant(ir.FloatType(), 1.0), x_val.type)
                neg_one = _splat_to_vector(
                    ir.Constant(ir.FloatType(), -1.0), x_val.type
                )
                pos = tick_builder.fcmp_ordered(">", x_val, zero, name="fused_sign_pos")
                neg = tick_builder.fcmp_ordered("<", x_val, zero, name="fused_sign_neg")
                return tick_builder.select(
                    pos, one, tick_builder.select(neg, neg_one, zero), name="fused_sign"
                )
            if name == "smoothstep" and len(args) == 3:
                edge0, edge1, x_val = args
                zero = _splat_to_vector(ir.Constant(ir.FloatType(), 0.0), x_val.type)
                one = _splat_to_vector(ir.Constant(ir.FloatType(), 1.0), x_val.type)
                two = _splat_to_vector(ir.Constant(ir.FloatType(), 2.0), x_val.type)
                three = _splat_to_vector(ir.Constant(ir.FloatType(), 3.0), x_val.type)
                t_raw = tick_builder.fdiv(
                    tick_builder.fsub(x_val, edge0, name="fused_ss_diff"),
                    tick_builder.fsub(edge1, edge0, name="fused_ss_range"),
                    name="fused_ss_raw",
                )
                t = self._call("clamp", [t_raw, zero, one])
                return tick_builder.fmul(
                    tick_builder.fmul(t, t, name="fused_ss_tsq"),
                    tick_builder.fsub(
                        three, tick_builder.fmul(two, t), name="fused_ss_poly"
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
                return tick_builder.not_(operand, name="fused_not")
            if isinstance(node, AstExprCall):
                return self._call(
                    node.name, [self.lower_expr(arg) for arg in node.args]
                )
            if isinstance(node, AstExprCast):
                return _coerce_vector_value(
                    self.lower_expr(node.value),
                    self._declared_vector_type(node.target_type),
                    _type_name(node.target_type),
                )
            if isinstance(node, AstExprBinary):
                lhs = self.lower_expr(node.left)
                rhs = self.lower_expr(node.right)
                expr_type_name = self._infer_expr_type(
                    node.left
                ) or self._infer_expr_type(node.right)
                operand_type_name = (
                    self._infer_binary_operand_type(node) or expr_type_name
                )
                if operand_type_name in _PRIMITIVE_TYPE_MAP:
                    vector_ty = self._declared_vector_type(operand_type_name)
                    lhs = _coerce_vector_value(lhs, vector_ty, operand_type_name)
                    rhs = _coerce_vector_value(rhs, vector_ty, operand_type_name)
                return self._binary(node.op, lhs, rhs, operand_type_name)
            raise CodegenError(
                f"unsupported vector expression node '{type(node).__name__}'"
            )

        def lower_statement(self, statement: AstStatement) -> None:
            if isinstance(statement, AstAssignStmt):
                if len(statement.target) != 1:
                    raise CodegenError("field assignment cannot be SIMD-vectorized")
                key = _sanitize_symbol(statement.target[0])
                target_ty = self.values[key].type
                type_name = self.types.get(key)
                self.values[key] = _coerce_vector_value(
                    self.lower_expr(statement.value), target_ty, type_name
                )
                return
            if isinstance(statement, AstVarDeclStmt):
                key = _sanitize_symbol(statement.name)
                declared_type = (
                    _type_name(statement.declared_type)
                    if statement.declared_type
                    else "float"
                )
                vector_ty = self._declared_vector_type(declared_type)
                self.types[key] = declared_type
                self.values[key] = ir.Constant(vector_ty, None)
                if statement.initializer is not None:
                    self.values[key] = _coerce_vector_value(
                        self.lower_expr(statement.initializer), vector_ty, declared_type
                    )
                return
            raise CodegenError(
                f"unsupported vector statement node '{type(statement).__name__}'"
            )

    def _vector_lane_indices(current: ir.Value) -> ir.Value:
        vector_ty = ir.VectorType(ir.IntType(32), simd_width)
        lanes = ir.Constant(vector_ty, list(range(simd_width)))
        return tick_builder.add(
            _splat_to_vector(current, vector_ty), lanes, name="fused_lane_indices"
        )

    def _load_stream_vector(
        name: str, scalar_ty: ir.Type, lane_indices: ir.Value
    ) -> ir.Value:
        vector_ty = ir.VectorType(scalar_ty, simd_width)
        result = ir.Constant(vector_ty, ir.Undefined)
        for lane in range(simd_width):
            lane_index = tick_builder.extract_element(
                lane_indices,
                ir.Constant(ir.IntType(32), lane),
                name=f"fused_load_lane_{lane}_idx",
            )
            clamped = _clamped_stream_index(name, lane_index)
            lane_value = _load_tick_param("stream", name, scalar_ty, clamped)
            result = tick_builder.insert_element(
                result,
                lane_value,
                ir.Constant(ir.IntType(32), lane),
                name=f"fused_{_sanitize_symbol(name)}_lane_{lane}",
            )
        return result

    def _store_stream_vector(
        name: str, value: ir.Value, lane_indices: ir.Value
    ) -> None:
        for lane in range(simd_width):
            lane_index = tick_builder.extract_element(
                lane_indices,
                ir.Constant(ir.IntType(32), lane),
                name=f"fused_store_lane_{lane}_idx",
            )
            clamped = _clamped_stream_index(name, lane_index)
            ptr = _load_tick_param_ptr("stream", name, value.type.element, clamped)
            if ptr is not None:
                lane_value = tick_builder.extract_element(
                    value,
                    ir.Constant(ir.IntType(32), lane),
                    name=f"fused_store_lane_{lane}",
                )
                tick_builder.store(lane_value, ptr)

    def _emit_vector_fused_chunk(
        routes: tuple[AstKernelBindRoute, ...],
        current: ir.Value,
        eliminated_targets: set[str],
    ) -> None:
        lane_indices = _vector_lane_indices(current)
        route_values: dict[str, ir.Value] = {}
        for route in routes:
            signature = kernel_signatures[route.kernel]
            kernel_decl, params = signature
            vector_lowerer = _FusedVectorLowerer()
            out_params: list[tuple[str, str]] = []
            for index, param in enumerate(params):
                arg_name = route.args[index] if index < len(route.args) else ""
                key = _sanitize_symbol(param.name)
                scalar_ty = lowerer._llvm_type(param.declared_type, known_structs)
                vector_ty = ir.VectorType(scalar_ty, simd_width)
                vector_lowerer.types[key] = _type_name(param.declared_type)
                if param.modifier == "in" and arg_name in route_values:
                    vector_lowerer.values[key] = route_values[arg_name]
                elif param.modifier == "in" and arg_name in stream_slots:
                    vector_lowerer.values[key] = _load_stream_vector(
                        arg_name, scalar_ty, lane_indices
                    )
                elif param.modifier == "uniform" and arg_name in uniform_slots:
                    scalar = _load_tick_param("uniform", arg_name, scalar_ty)
                    vector_lowerer.values[key] = _splat_to_vector(scalar, vector_ty)
                elif param.modifier == "out":
                    vector_lowerer.values[key] = ir.Constant(vector_ty, None)
                    out_params.append((arg_name, key))
                else:
                    vector_lowerer.values[key] = ir.Constant(vector_ty, None)
            for statement in kernel_decl.body:
                vector_lowerer.lower_statement(statement)
            for arg_name, key in out_params:
                value = vector_lowerer.values[key]
                if arg_name in eliminated_targets:
                    route_values[arg_name] = value
                elif arg_name in stream_slots:
                    _store_stream_vector(arg_name, value, lane_indices)

    def _lower_fused_kernel_group(
        routes: tuple[AstKernelBindRoute, ...], group_index: int
    ):
        if not routes:
            return
        trip_count = max(_kernel_route_trip_count(route) for route in routes)
        eliminated_targets = {route.target for route in routes[:-1]}
        if not eliminated_targets:
            _lower_kernel_route(routes[0])
            return
        if not _can_vectorize_fused_group(routes):
            for route in routes:
                _lower_kernel_route(route)
            return

        full_trip_count = (trip_count // simd_width) * simd_width
        index_ptr = tick_builder.alloca(ir.IntType(32), name=f"fused_{group_index}_idx")
        tick_builder.store(ir.Constant(ir.IntType(32), 0), index_ptr)

        loop_cond = tick.append_basic_block(f"fused_{group_index}_cond")
        loop_body = tick.append_basic_block(f"fused_{group_index}_body")
        loop_exit = tick.append_basic_block(f"fused_{group_index}_exit")
        tick_builder.branch(loop_cond)

        tick_builder.position_at_end(loop_cond)
        current = tick_builder.load(index_ptr, name="fused_idx")
        cond = tick_builder.icmp_signed(
            "<",
            current,
            ir.Constant(ir.IntType(32), full_trip_count),
            name="fused_vector_active",
        )
        tick_builder.cbranch(cond, loop_body, loop_exit)

        tick_builder.position_at_end(loop_body)
        _emit_vector_fused_chunk(routes, current, eliminated_targets)
        next_index = tick_builder.add(
            current, ir.Constant(ir.IntType(32), simd_width), name="fused_idx_next"
        )
        tick_builder.store(next_index, index_ptr)
        tick_builder.branch(loop_cond)

        tick_builder.position_at_end(loop_exit)

        if full_trip_count < trip_count:
            tail_index_ptr = tick_builder.alloca(
                ir.IntType(32), name=f"fused_{group_index}_tail_idx"
            )
            tick_builder.store(
                ir.Constant(ir.IntType(32), full_trip_count), tail_index_ptr
            )
            tail_cond = tick.append_basic_block(f"fused_{group_index}_tail_cond")
            tail_body = tick.append_basic_block(f"fused_{group_index}_tail_body")
            tail_exit = tick.append_basic_block(f"fused_{group_index}_tail_exit")
            tick_builder.branch(tail_cond)
            tick_builder.position_at_end(tail_cond)
            tail_current = tick_builder.load(tail_index_ptr, name="fused_tail_idx")
            tail_active = tick_builder.icmp_signed(
                "<",
                tail_current,
                ir.Constant(ir.IntType(32), trip_count),
                name="fused_tail_active",
            )
            tick_builder.cbranch(tail_active, tail_body, tail_exit)
            tick_builder.position_at_end(tail_body)
            local_slots: dict[str, ir.AllocaInstr] = {}
            for route in routes:
                callee, params = _kernel_function_and_params(route.kernel)
                if callee is None:
                    continue
                for index, param in enumerate(callee.args):
                    if index >= len(route.args) or index >= len(params):
                        continue
                    if (
                        params[index].modifier == "out"
                        and route.args[index] in eliminated_targets
                        and route.args[index] not in local_slots
                        and hasattr(param.type, "pointee")
                    ):
                        slot_name = _sanitize_symbol(route.args[index])
                        local_slots[route.args[index]] = tick_builder.alloca(
                            param.type.pointee, name=f"fused_{slot_name}_tail_slot"
                        )
                _emit_kernel_call(route, tail_current, local_slots=local_slots)
            tail_next = tick_builder.add(
                tail_current, ir.Constant(ir.IntType(32), 1), name="fused_tail_next"
            )
            tick_builder.store(tail_next, tail_index_ptr)
            tick_builder.branch(tail_cond)
            tick_builder.position_at_end(tail_exit)

    def _lower_fold_route(route: AstFoldBindRoute) -> None:
        source_name = route.source
        uniform_name = route.uniform_name
        if source_name not in accum_slots or uniform_name not in uniform_slots:
            return
        uniform_type_name = _type_name(route.uniform_type)
        uniform_type = lowerer._llvm_type(uniform_type_name, known_structs)
        reduced = _reduce_fold(route.operator, source_name, uniform_type)
        _store_value("uniform", uniform_name, reduced, uniform_type_name)

    for pipeline_index, pipeline in enumerate(program.pipelines):
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
        ) == len(program.pipelines):
            pipeline_optimization = pipeline_optimizations[pipeline_index]
        elif bind_optimization is not None and len(program.pipelines) == 1:
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
                _lower_fused_kernel_group(group_routes, group_index)
                continue
            if id(route) in fused_member_ids:
                continue
            remaining_optimized_routes = optimized_route_counts.get(route.route, 0)
            if remaining_optimized_routes <= 0:
                continue
            optimized_route_counts[route.route] = remaining_optimized_routes - 1
            if isinstance(route, AstKernelBindRoute):
                _lower_kernel_route(route)
                continue
            _lower_fold_route(route)

    tick_builder.ret_void()

    return str(module)
