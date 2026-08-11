"""Per-kernel scalar IR lowering for the code generator.

Houses the shared low-level codegen primitives -- the primitive-type map, the
`CodegenError` exception, and the `_type_name` helper -- together with
`_FunctionLowerer`, the visitor that lowers a single kernel/pure body's
statements and expressions to LLVM IR. Splitting it out of `codegen.py` keeps
the pipeline/tick emission in `emit_llvm_ir` separate from the expression-level
lowering. This module does not import from `codegen.py`, so there is no cycle;
`codegen.py` imports these names back.
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
    AstReturnStmt,
    AstStatement,
    AstStructField,
    AstType,
    AstVarDeclStmt,
    _normalize_type,
)
from .codegen_intrinsics import lower_intrinsic_call
from .utils import decode_escape_sequences
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

    def _lower_intrinsic_call(self, name: str, args: list[ir.Value]) -> ir.Value | None:
        # Scalar builtin-intrinsic lowering lives in `codegen_intrinsics`; this
        # method just supplies the current builder/module and error callback.
        return lower_intrinsic_call(
            self.builder, self.module, name, args, self._compiler_error
        )

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
        source_type_name: str | None = None,
    ) -> ir.Value:
        if value.type == target_type:
            return value

        if isinstance(target_type, ir.IntType) and isinstance(value.type, ir.IntType):
            if value.type.width < target_type.width:
                if value.type.width == 1:
                    return self.builder.zext(value, target_type)
                return self.builder.sext(value, target_type)
            if value.type.width > target_type.width:
                if target_type.width == 1:
                    return self.builder.icmp_signed(
                        "!=",
                        value,
                        ir.Constant(value.type, 0),
                        name="int_to_bool",
                    )
                return self.builder.trunc(value, target_type)

        if isinstance(target_type, (ir.FloatType, ir.DoubleType)) and isinstance(
            value.type, ir.IntType
        ):
            if source_type_name == "uint":
                return self.builder.uitofp(value, target_type)
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

    def _array_index_and_type(
        self, llvm_type: ir.Type, field_name: str
    ) -> tuple[int, ir.Type, str | None] | None:
        if not isinstance(llvm_type, ir.ArrayType):
            return None
        try:
            index = int(field_name)
        except ValueError:
            return None
        if index < 0 or index >= llvm_type.count:
            self._compiler_error(
                f"array index '{field_name}' is out of bounds for access path"
            )
        return index, llvm_type.element, None

    def _aggregate_index_and_type(
        self, llvm_type: ir.Type, field_name: str
    ) -> tuple[int, ir.Type, str | None] | None:
        array_info = self._array_index_and_type(llvm_type, field_name)
        if array_info is not None:
            return array_info
        return self._field_index_and_type(llvm_type, field_name)

    def _extract_field_path(self, value: ir.Value, path: list[str]) -> ir.Value:
        current = value
        for field_name in path:
            field_info = self._aggregate_index_and_type(current.type, field_name)
            if field_info is None:
                self._compiler_error(
                    f"unknown aggregate field or array index '{field_name}' in access path"
                )
            index, _, _ = field_info
            current = self.builder.extract_value(
                current, index, name=f"{field_name}_field"
            )
        return current

    def _insert_field_path(
        self, aggregate: ir.Value, path: list[str], value: ir.Value
    ) -> ir.Value:
        field_info = self._aggregate_index_and_type(aggregate.type, path[0])
        if field_info is None:
            self._compiler_error(
                f"unknown aggregate field or array index '{path[0]}' in assignment path"
            )
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
        return decode_escape_sequences(token_text)

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
            if left_type in {"int", "uint"} and (
                right_type in {"int", "uint"} or right_type is None
            ):
                return left_type
            return None
        if op in {"+", "-", "*", "/", "%"}:
            return self._promote_scalar_type_names(left_type, right_type)
        if op in {"&", "|", "^"}:
            if left_type in {"int", "uint"} and right_type in {"int", "uint"}:
                return self._promote_scalar_type_names(left_type, right_type)
            return None
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
        self._compiler_error(f"unsupported binary operator '{op}'")

    def _validate_logical_operand(self, op: str, value: ir.Value) -> ir.Value:
        if not isinstance(value.type, ir.IntType) or value.type.width != 1:
            self._compiler_error(f"operator '{op}' expects boolean operands")
        return value

    def _lower_logical_binary_op(self, node: AstExprBinary) -> ir.Value:
        expr_type_name = self._infer_binary_operand_type(node)
        bool_ty = ir.IntType(1)
        lhs = self._validate_logical_operand(node.op, self._lower_expr(node.left))
        if expr_type_name == "bool":
            lhs = self._coerce_value_to_type(lhs, bool_ty, "bool")

        lhs_block = self.builder.block
        rhs_block = lhs_block.function.append_basic_block(
            f"logic_{'and' if node.op == '&&' else 'or'}_rhs"
        )
        merge_block = lhs_block.function.append_basic_block(
            f"logic_{'and' if node.op == '&&' else 'or'}_merge"
        )

        if node.op == "&&":
            short_value = ir.Constant(bool_ty, 0)
            self.builder.cbranch(lhs, rhs_block, merge_block)
        else:
            short_value = ir.Constant(bool_ty, 1)
            self.builder.cbranch(lhs, merge_block, rhs_block)

        self.builder.position_at_end(rhs_block)
        rhs = self._validate_logical_operand(node.op, self._lower_expr(node.right))
        if expr_type_name == "bool":
            rhs = self._coerce_value_to_type(rhs, bool_ty, "bool")
        rhs_value_block = self.builder.block
        rhs_reaches_merge = not rhs_value_block.is_terminated
        if rhs_reaches_merge:
            self.builder.branch(merge_block)

        self.builder.position_at_end(merge_block)
        result = self.builder.phi(bool_ty, name="logic_result")
        result.add_incoming(short_value, lhs_block)
        if rhs_reaches_merge:
            result.add_incoming(rhs, rhs_value_block)
        return result

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
                self._lower_expr(node.value),
                target_type,
                _type_name(node.target_type),
                self._infer_expr_type(node.value),
            )
        if not isinstance(node, AstExprBinary):
            self._compiler_error(f"unsupported expression node '{type(node).__name__}'")
        if node.op in {"&&", "||"}:
            return self._lower_logical_binary_op(node)
        lhs, rhs = self._lower_expr(node.left), self._lower_expr(node.right)
        expr_type_name = self._infer_binary_operand_type(node)
        if expr_type_name in _PRIMITIVE_TYPE_MAP:
            operand_type = self._llvm_type(expr_type_name, self.known_structs)
            lhs_type_name = self._infer_expr_type(node.left)
            rhs_type_name = self._infer_expr_type(node.right)
            lhs = self._coerce_value_to_type(
                lhs, operand_type, expr_type_name, lhs_type_name
            )
            rhs = self._coerce_value_to_type(
                rhs, operand_type, expr_type_name, rhs_type_name
            )
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
            elif isinstance(return_type, ir.IntType) and return_type.width == 1:
                # Filters default to keeping the row when control reaches the end
                # without an explicit return, matching simulator semantics.
                self.builder.ret(ir.Constant(return_type, 1))
            elif isinstance(return_type, ir.FloatType):
                self.builder.ret(ir.Constant(ir.FloatType(), 0.0))
            else:
                self.builder.ret(ir.Constant(return_type, None))


