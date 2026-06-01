from __future__ import annotations

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
    AstProgram,
    AstReturnStmt,
    AstStatement,
    AstStreamDecl,
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
    ) -> tuple[int, ir.Type] | None:
        struct_name = self._struct_name_for_type(llvm_type)
        if struct_name is None:
            return None
        fields = self.struct_fields.get(struct_name, [])
        for index, field in enumerate(fields):
            if field.name == field_name:
                return index, self._llvm_type(
                    _type_name(field.declared_type), self.known_structs
                )
        return None

    def _coerce_value_to_type(self, value: ir.Value, target_type: ir.Type) -> ir.Value:
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
            index, _ = field_info
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
        index, field_type = field_info
        if len(path) == 1:
            coerced = self._coerce_value_to_type(value, field_type)
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
        if node.op in {"&&", "||"}:
            return "bool" if left_type == "bool" and right_type == "bool" else None
        if node.op in {"<<", ">>"}:
            return left_type
        return self._promote_scalar_type_names(left_type, right_type)

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
            if node.op in {"&&", "||"}:
                return "bool"
            if node.op in {"<", "<=", ">", ">=", "==", "!="}:
                return "bool"
            return self._infer_binary_operand_type(node)
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
            coerced = [
                self._coerce_value_to_type(arg, param.type)
                for arg, param in zip(args, callee.args)
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
            return self._coerce_value_to_type(self._lower_expr(node.value), target_type)
        if not isinstance(node, AstExprBinary):
            self._compiler_error(f"unsupported expression node '{type(node).__name__}'")
        lhs, rhs = self._lower_expr(node.left), self._lower_expr(node.right)
        expr_type_name = self._infer_binary_operand_type(node)
        if expr_type_name in _PRIMITIVE_TYPE_MAP:
            operand_type = self._llvm_type(expr_type_name, self.known_structs)
            lhs = self._coerce_value_to_type(lhs, operand_type)
            rhs = self._coerce_value_to_type(rhs, operand_type)
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
                    self._coerce_value_to_type(value, slot_type), ref_ptr
                )
            else:
                slot_type = self.locals[key].type.pointee
                self.builder.store(
                    self._coerce_value_to_type(value, slot_type), self.locals[key]
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

    def _lower_statement(self, statement: AstStatement, return_type: ir.Type):
        if isinstance(statement, AstReturnStmt):
            value = self._lower_expr(statement.value)
            if isinstance(return_type, ir.VoidType):
                self.builder.ret_void()
            else:
                self.builder.ret(self._coerce_value_to_type(value, return_type))
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
                    self._coerce_value_to_type(value, slot_type), self.locals[key]
                )
            return

        self._compiler_error(f"unsupported statement node '{type(statement).__name__}'")

    def lower_function(
        self,
        fn: ir.Function,
        statements: list[AstStatement],
        return_type: ir.Type,
        param_type_names: list[str] | None = None,
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
            self._lower_statement(statement, return_type)

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
    program: AstProgram,
    *,
    target_width: int | None = None,
    bind_optimization: dict[str, object] | None = None,
) -> str:
    """Generate LLVM IR from the typed Lockstep AST."""

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
        )

    for shader in shaders:
        fn = function_map[f"shader_{_sanitize_symbol(shader.name)}"]
        lowerer.lower_function(
            fn,
            list(shader.body),
            ir.VoidType(),
            [_kernel_param_type(param) for param in shader.params],
            [param.modifier in {"out", "accum"} for param in shader.params],
        )

    for flt in filters:
        fn = function_map[f"filter_{_sanitize_symbol(flt.name)}"]
        lowerer.lower_function(
            fn,
            list(flt.body),
            ir.VoidType(),
            [_kernel_param_type(param) for param in flt.params],
            [param.modifier in {"out", "accum"} for param in flt.params],
        )

    layout = build_ast_arena_layout(program)

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

    leaf_specs: dict[tuple[str, str, tuple[str, ...]], tuple[int, int]] = {
        (leaf.kind, leaf.binding_name, leaf.path): (leaf.offset, leaf.size)
        for leaf in layout.leaves
    }
    accum_sizes: dict[str, int] = {accum.name: 1 for accum in accumulators}
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

    kernel_signatures: dict[str, tuple[AstKernelDecl, tuple[AstKernelParam, ...]]] = {
        shader.name: (shader, shader.params) for shader in shaders
    }
    kernel_signatures.update({flt.name: (flt, flt.params) for flt in filters})

    arena_struct_ty = module.context.get_identified_type("struct.Lockstep_Arena")

    arena_field_types: list[ir.Type] = []
    leaf_field_indices: dict[tuple[str, str, tuple[str, ...]], int] = {}
    leaf_field_is_array: dict[tuple[str, str, tuple[str, ...]], bool] = {}
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

        leaf_key = (leaf.kind, leaf.binding_name, leaf.path)
        leaf_field_indices[leaf_key] = len(arena_field_types)
        if leaf.element_count > 1:
            arena_field_types.append(ir.ArrayType(field_ty, max(leaf.element_count, 1)))
            leaf_field_is_array[leaf_key] = True
        else:
            arena_field_types.append(field_ty)
            leaf_field_is_array[leaf_key] = False

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
        field_index = leaf_field_indices.get(leaf_key)
        if field_index is None:
            return None

        gep_indices = [
            ir.Constant(ir.IntType(32), 0),
            ir.Constant(ir.IntType(32), field_index),
        ]
        if leaf_field_is_array.get(leaf_key, False):
            element_index = (
                loop_index_reg
                if kind in {"stream", "accum"} and loop_index_reg is not None
                else ir.Constant(ir.IntType(32), 0)
            )
            gep_indices.append(element_index)

        raw_ptr = tick_builder.gep(
            arena_ptr,
            gep_indices,
            name=f"{kind}_{_sanitize_symbol(name)}_{'_'.join(path) if path else 'value'}_ptr",
        )
        if raw_ptr.type.pointee == leaf_type:
            return raw_ptr
        typed_ptr = tick_builder.bitcast(raw_ptr, leaf_type.as_pointer())
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

        full_chunk_limit = (lane_count // simd_width) * simd_width
        if full_chunk_limit:
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
            loop_accumulator = tick_builder.phi(vector_ty, name="fold_vector_acc")
            loop_index.add_incoming(ir.Constant(ir.IntType(32), 0), preheader_block)
            loop_accumulator.add_incoming(vector_accumulator, preheader_block)
            in_full_chunks = tick_builder.icmp_unsigned(
                "<",
                loop_index,
                ir.Constant(ir.IntType(32), full_chunk_limit),
                name="fold_has_full_chunk",
            )
            tick_builder.cbranch(in_full_chunks, loop_body, loop_exit)

            tick_builder.position_at_end(loop_body)
            chunk_vector = ir.Constant(vector_ty, ir.Undefined)
            for lane in range(simd_width):
                element_index = tick_builder.add(
                    loop_index,
                    ir.Constant(ir.IntType(32), lane),
                    name=f"fold_elem_{lane}",
                )
                chunk_vector = _insert_accum_chunk_lane(
                    chunk_vector, lane, element_index
                )
            next_accumulator = _combine_vectors(
                loop_accumulator, chunk_vector, "fold_vector_next"
            )
            next_index = tick_builder.add(
                loop_index,
                ir.Constant(ir.IntType(32), simd_width),
                name="fold_index_next",
            )
            tick_builder.branch(loop_cond)
            loop_index.add_incoming(next_index, tick_builder.block)
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
        _, params = _kernel_function_and_params(route.kernel)
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
            return _load_tick_param_ptr("accum", arg_name, param.type.pointee)
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

        index_ptr = tick_builder.alloca(
            ir.IntType(32), name=f"fused_{group_index}_idx"
        )
        tick_builder.store(ir.Constant(ir.IntType(32), 0), index_ptr)

        loop_cond = tick.append_basic_block(f"fused_{group_index}_cond")
        loop_body = tick.append_basic_block(f"fused_{group_index}_body")
        loop_exit = tick.append_basic_block(f"fused_{group_index}_exit")
        tick_builder.branch(loop_cond)

        tick_builder.position_at_end(loop_cond)
        current = tick_builder.load(index_ptr, name="fused_idx")
        cond = tick_builder.icmp_signed(
            "<", current, ir.Constant(ir.IntType(32), trip_count), name="fused_active"
        )
        tick_builder.cbranch(cond, loop_body, loop_exit)

        tick_builder.position_at_end(loop_body)

        def _emit_fused_lane(lane_index: ir.Value) -> None:
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
                            param.type.pointee, name=f"fused_{slot_name}_slot"
                        )
                _emit_kernel_call(route, lane_index, local_slots=local_slots)

        _emit_fused_lane(current)
        for lane in range(1, simd_width):
            lane_index = tick_builder.add(
                current,
                ir.Constant(ir.IntType(32), lane),
                name=f"fused_lane_{lane}_idx",
            )
            lane_active = tick_builder.icmp_signed(
                "<",
                lane_index,
                ir.Constant(ir.IntType(32), trip_count),
                name=f"fused_lane_{lane}_active",
            )
            lane_body = tick.append_basic_block(f"fused_{group_index}_lane_{lane}")
            lane_next = tick.append_basic_block(
                f"fused_{group_index}_lane_{lane}_next"
            )
            tick_builder.cbranch(lane_active, lane_body, lane_next)
            tick_builder.position_at_end(lane_body)
            _emit_fused_lane(lane_index)
            tick_builder.branch(lane_next)
            tick_builder.position_at_end(lane_next)

        next_index = tick_builder.add(
            current, ir.Constant(ir.IntType(32), simd_width), name="fused_idx_next"
        )
        tick_builder.store(next_index, index_ptr)
        tick_builder.branch(loop_cond)

        tick_builder.position_at_end(loop_exit)

    def _lower_fold_route(route: AstFoldBindRoute) -> None:
        source_name = route.source
        uniform_name = route.uniform_name
        if source_name not in accum_slots or uniform_name not in uniform_slots:
            return
        uniform_type_name = _type_name(route.uniform_type)
        uniform_type = lowerer._llvm_type(uniform_type_name, known_structs)
        reduced = _reduce_fold(route.operator, source_name, uniform_type)
        _store_value("uniform", uniform_name, reduced, uniform_type_name)

    for pipeline in program.pipelines:
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

        if bind_optimization is not None and len(program.pipelines) == 1:
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
