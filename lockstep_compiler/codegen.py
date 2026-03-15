from __future__ import annotations

from typing import Any

from llvmlite import ir

from .ast import (
    AstAssignStmt,
    AstExprBinary,
    AstExprCall,
    AstExprCast,
    AstExprLiteral,
    AstExprUnary,
    AstExprVar,
    AstProgram,
    AstReturnStmt,
    AstStatement,
    AstType,
    AstVarDeclStmt,
    ast_to_entities,
)
from .utils import sanitize_symbol as _sanitize_symbol


_PRIMITIVE_TYPE_MAP: dict[str, ir.Type] = {
    "bool": ir.IntType(1),
    "int": ir.IntType(32),
    "uint": ir.IntType(32),
    "float": ir.FloatType(),
    "double": ir.DoubleType(),
}


class CodegenError(RuntimeError):
    """Raised when IR lowering encounters an invalid or unsupported expression."""


def _type_name(value: AstType | str) -> str:
    return value.name if isinstance(value, AstType) else value


class _FunctionLowerer:
    def __init__(
        self,
        module: ir.Module,
        function_map: dict[str, ir.Function],
        known_structs: dict[str, ir.IdentifiedStructType] | None = None,
        struct_fields: dict[str, list[dict[str, str]]] | None = None,
        intrinsic_names: set[str] | None = None,
    ):
        self.module = module
        self.function_map = function_map
        self.known_structs = known_structs or {}
        self.struct_fields = struct_fields or {}
        self.intrinsic_names = intrinsic_names or set()
        self.builder: ir.IRBuilder | None = None
        self.locals: dict[str, ir.AllocaInstr] = {}

    def _compiler_error(self, message: str) -> None:
        raise CodegenError(message)

    def _declare_llvm_intrinsic(
        self, intrinsic_name: str, arg_type: ir.Type
    ) -> ir.Function:
        llvm_name = f"llvm.{intrinsic_name}.f32"
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
            if field.get("name") == field_name:
                return index, self._llvm_type(
                    field.get("type", "float"), self.known_structs
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

        if isinstance(target_type, ir.FloatType) and isinstance(value.type, ir.IntType):
            return self.builder.sitofp(value, target_type)

        if isinstance(target_type, ir.IntType) and isinstance(value.type, ir.FloatType):
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
        self, type_name: str, known_structs: dict[str, ir.IdentifiedStructType]
    ) -> ir.Type:
        if type_name in _PRIMITIVE_TYPE_MAP:
            return _PRIMITIVE_TYPE_MAP[type_name]
        if type_name in known_structs:
            return known_structs[type_name]
        return ir.IntType(8).as_pointer()

    def _emit_numeric_unary_minus(self, value: ir.Value) -> ir.Value:
        if isinstance(value.type, ir.FloatType):
            return self.builder.fsub(ir.Constant(value.type, 0.0), value)
        if isinstance(value.type, ir.IntType):
            return self.builder.sub(ir.Constant(value.type, 0), value)
        self._compiler_error(f"unary '-' is unsupported for type '{value.type}'")

    def _emit_numeric_binary(self, op: str, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        if lhs.type != rhs.type:
            self._compiler_error(
                f"operator '{op}' requires matching operand types, got '{lhs.type}' and '{rhs.type}'"
            )

        if isinstance(lhs.type, ir.FloatType):
            return {
                "+": self.builder.fadd,
                "-": self.builder.fsub,
                "*": self.builder.fmul,
                "/": self.builder.fdiv,
                "%": self.builder.frem,
            }[op](lhs, rhs)

        if isinstance(lhs.type, ir.IntType):
            return {
                "+": self.builder.add,
                "-": self.builder.sub,
                "*": self.builder.mul,
                "/": self.builder.sdiv,
                "%": self.builder.srem,
            }[op](lhs, rhs)

        self._compiler_error(f"operator '{op}' is unsupported for type '{lhs.type}'")

    def _emit_relational_compare(
        self, op: str, lhs: ir.Value, rhs: ir.Value
    ) -> ir.Value:
        if lhs.type != rhs.type:
            self._compiler_error(
                f"comparison '{op}' requires matching operand types, got '{lhs.type}' and '{rhs.type}'"
            )

        rel_map = {"<": "<", "<=": "<=", ">": ">", ">=": ">=", "==": "==", "!=": "!="}
        if isinstance(lhs.type, ir.FloatType):
            return self.builder.fcmp_ordered(rel_map[op], lhs, rhs)
        if isinstance(lhs.type, ir.IntType):
            return self.builder.icmp_signed(rel_map[op], lhs, rhs)
        self._compiler_error(f"comparison '{op}' is unsupported for type '{lhs.type}'")

    def _load_var(self, name: str) -> ir.Value:
        parts = name.split(".")
        base_key = _sanitize_symbol(parts[0])
        if base_key in self.locals:
            base_value = self.builder.load(
                self.locals[base_key], name=f"{base_key}_val"
            )
            if len(parts) == 1:
                return base_value
            return self._extract_field_path(base_value, parts[1:])
        self._compiler_error(f"undefined variable '{parts[0]}'")

    def _lower_call(self, name: str, args: list[ir.Value]) -> ir.Value:
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

    def _lower_binary_op(self, op: str, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        if op in {"+", "-", "*", "/", "%"}:
            return self._emit_numeric_binary(op, lhs, rhs)
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
            return self._emit_relational_compare(op, lhs, rhs)
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
            if node.kind == "int":
                return ir.Constant(ir.IntType(32), int(node.value))
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
        return self._lower_binary_op(node.op, lhs, rhs)

    def _lower_assignment(self, target_name: str, value: ir.Value):
        base_name, *field_path = target_name.split(".")
        key = _sanitize_symbol(base_name)
        if key not in self.locals:
            slot = self.builder.alloca(value.type, name=key)
            self.locals[key] = slot
        if not field_path:
            slot_type = self.locals[key].type.pointee
            self.builder.store(
                self._coerce_value_to_type(value, slot_type), self.locals[key]
            )
            return

        current = self.builder.load(self.locals[key], name=f"{key}_val")
        updated = self._insert_field_path(current, field_path, value)
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
        self, fn: ir.Function, statements: list[AstStatement], return_type: ir.Type
    ):
        block = fn.append_basic_block("entry")
        self.builder = ir.IRBuilder(block)
        self.locals = {}
        for arg in fn.args:
            slot = self.builder.alloca(arg.type, name=_sanitize_symbol(arg.name))
            self.builder.store(arg, slot)
            self.locals[_sanitize_symbol(arg.name)] = slot

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


def _normalize_codegen_input(
    program_or_entities: AstProgram | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(program_or_entities, AstProgram):
        return ast_to_entities(program_or_entities)
    return program_or_entities


def _ast_body_for(entity: dict[str, Any], *, entity_kind: str) -> list[AstStatement]:
    body_ast = entity.get("body_ast")
    if body_ast is None:
        name = entity.get("name", "<unknown>")
        raise CodegenError(
            f"{entity_kind} '{name}' is missing body_ast; string-based bodies are no longer supported"
        )
    return body_ast


def emit_llvm_ir(program_or_entities: AstProgram | dict[str, Any]) -> str:
    """Generate LLVM IR using llvmlite lowering for pure/kernels."""

    entities = _normalize_codegen_input(program_or_entities)

    structs = entities.get("structs", [])
    shaders = entities.get("shaders", [])
    filters = entities.get("filters", [])
    pure_functions = entities.get("pure_functions", [])
    streams = entities.get("streams", [])
    accumulators = entities.get("accumulators", [])
    uniforms = entities.get("uniforms", [])
    bind_routes = entities.get("bind_routes", [])
    bind_routes_ir = entities.get("bind_routes_ir", [])

    module = ir.Module(name="lockstep")
    module.source_filename = "lockstep"
    known_structs: dict[str, ir.IdentifiedStructType] = {}
    struct_fields: dict[str, list[dict[str, str]]] = {}

    normalized_structs: list[dict[str, Any]] = []
    for struct_decl in structs:
        if isinstance(struct_decl, str):
            normalized_structs.append({"name": struct_decl, "fields": []})
        elif isinstance(struct_decl, dict) and struct_decl.get("name"):
            fields = (
                struct_decl.get("fields")
                if isinstance(struct_decl.get("fields"), list)
                else []
            )
            normalized_structs.append({"name": struct_decl["name"], "fields": fields})

    for struct_decl in normalized_structs:
        struct_name = struct_decl["name"]
        safe_name = _sanitize_symbol(struct_name)
        struct_ty = module.context.get_identified_type(f"struct.{safe_name}")
        known_structs[struct_name] = struct_ty
        struct_fields[struct_name] = struct_decl["fields"]

    intrinsic_names = {
        pure.get("name")
        for pure in pure_functions
        if isinstance(pure, dict) and pure.get("intrinsic") and pure.get("name")
    }

    lowerer = _FunctionLowerer(
        module, {}, known_structs, struct_fields, intrinsic_names
    )

    unresolved = set(known_structs.keys())
    while unresolved:
        progress = False
        for struct_name in list(unresolved):
            field_types: list[ir.Type] = []
            can_lower = True
            for field in struct_fields.get(struct_name, []):
                field_type_name = field.get("type", "float")
                if field_type_name in known_structs and field_type_name in unresolved:
                    can_lower = False
                    break
                field_types.append(lowerer._llvm_type(field_type_name, known_structs))
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
        ret_ty = lowerer._llvm_type(pure.get("return_type", "float"), known_structs)
        params = [
            lowerer._llvm_type(param.get("type", "float"), known_structs)
            for param in pure.get("params", [])
        ]
        fn = ir.Function(
            module,
            ir.FunctionType(ret_ty, params),
            name=f"pure_{_sanitize_symbol(pure['name'])}",
        )
        for idx, param in enumerate(pure.get("params", [])):
            fn.args[idx].name = _sanitize_symbol(param.get("name", f"arg{idx}"))
        function_map[fn.name] = fn

    for shader in shaders:
        params = [
            lowerer._llvm_type(param.get("type", "float"), known_structs)
            for param in shader.get("params", [])
        ]
        fn = ir.Function(
            module,
            ir.FunctionType(ir.VoidType(), params),
            name=f"shader_{_sanitize_symbol(shader['name'])}",
        )
        for idx, param in enumerate(shader.get("params", [])):
            fn.args[idx].name = _sanitize_symbol(param.get("name", f"arg{idx}"))
        function_map[fn.name] = fn

    for flt in filters:
        params = [
            lowerer._llvm_type(param.get("type", "float"), known_structs)
            for param in flt.get("params", [])
        ]
        fn = ir.Function(
            module,
            ir.FunctionType(ir.VoidType(), params),
            name=f"filter_{_sanitize_symbol(flt['name'])}",
        )
        for idx, param in enumerate(flt.get("params", [])):
            fn.args[idx].name = _sanitize_symbol(param.get("name", f"arg{idx}"))
        function_map[fn.name] = fn

    lowerer.function_map = function_map

    for pure in pure_functions:
        fn = function_map[f"pure_{_sanitize_symbol(pure['name'])}"]
        if pure.get("intrinsic"):
            continue
        lowerer.lower_function(
            fn,
            _ast_body_for(pure, entity_kind="pure function"),
            fn.function_type.return_type,
        )

    for shader in shaders:
        fn = function_map[f"shader_{_sanitize_symbol(shader['name'])}"]
        lowerer.lower_function(
            fn, _ast_body_for(shader, entity_kind="shader"), ir.VoidType()
        )

    for flt in filters:
        fn = function_map[f"filter_{_sanitize_symbol(flt['name'])}"]
        lowerer.lower_function(
            fn, _ast_body_for(flt, entity_kind="filter"), ir.VoidType()
        )

    arena_fields: list[tuple[str, str, ir.Type]] = []
    stream_slots: dict[str, int] = {}
    stream_capacities: dict[str, int] = {}
    for stream in streams:
        stream_slots[stream["name"]] = len(arena_fields)
        arena_fields.append(
            (
                "stream",
                stream["name"],
                lowerer._llvm_type(stream["type"], known_structs),
            )
        )
        stream_capacities[stream["name"]] = int(stream.get("capacity", 0))
    accum_slots: dict[str, int] = {}
    for accum in accumulators:
        accum_slots[accum["name"]] = len(arena_fields)
        arena_fields.append(
            ("accum", accum["name"], lowerer._llvm_type(accum["type"], known_structs))
        )
    uniform_slots: dict[str, int] = {}
    for uniform in uniforms:
        uniform_slots[uniform["name"]] = len(arena_fields)
        arena_fields.append(
            (
                "uniform",
                uniform["name"],
                lowerer._llvm_type(uniform["type"], known_structs),
            )
        )

    kernel_signatures = {
        shader["name"]: {"kind": "shader", "params": shader.get("params", [])}
        for shader in shaders
    }
    kernel_signatures.update(
        {
            flt["name"]: {"kind": "filter", "params": flt.get("params", [])}
            for flt in filters
        }
    )

    tick_arg_specs: list[tuple[str, str, ir.Type, bool]] = []
    for stream in streams:
        tick_arg_specs.append(
            (
                "stream",
                stream["name"],
                lowerer._llvm_type(stream["type"], known_structs),
                True,
            )
        )
    for accum in accumulators:
        tick_arg_specs.append(
            (
                "accum",
                accum["name"],
                lowerer._llvm_type(accum["type"], known_structs),
                True,
            )
        )
    for uniform in uniforms:
        tick_arg_specs.append(
            (
                "uniform",
                uniform["name"],
                lowerer._llvm_type(uniform["type"], known_structs),
                False,
            )
        )

    tick_param_types = [
        field_type.as_pointer() for _, _, field_type, _ in tick_arg_specs
    ]
    tick = ir.Function(
        module, ir.FunctionType(ir.VoidType(), tick_param_types), name="Lockstep_Tick"
    )
    tick_param_ptrs: dict[tuple[str, str], ir.Argument] = {}
    for arg, (kind, name, _, apply_noalias) in zip(tick.args, tick_arg_specs):
        arg.name = f"{kind}_{_sanitize_symbol(name)}"
        if apply_noalias:
            arg.add_attribute("noalias")
        tick_param_ptrs[(kind, name)] = arg

    tick_entry = tick.append_basic_block("entry")
    tick_builder = ir.IRBuilder(tick_entry)
    simd_width = 8

    def _zero_value(llvm_type: ir.Type) -> ir.Value:
        if isinstance(llvm_type, ir.VoidType):
            return ir.Constant(ir.IntType(32), 0)
        return ir.Constant(llvm_type, None)

    def _load_tick_param(kind: str, name: str, field_type: ir.Type) -> ir.Value:
        ptr = tick_param_ptrs.get((kind, name))
        if ptr is None:
            return _zero_value(field_type)
        return tick_builder.load(ptr, name=f"{kind}_{_sanitize_symbol(name)}_val")

    def _store_arena_slot(field_index: int, value: ir.Value):
        if field_index < 0 or field_index >= len(arena_fields):
            return
        kind, name, _ = arena_fields[field_index]
        ptr = tick_param_ptrs.get((kind, name))
        if ptr is None:
            return
        slot_ptr = tick_builder.alloca(value.type, name=f"arena_slot_{field_index}")
        tick_builder.store(value, slot_ptr)
        tick_builder.store(tick_builder.load(slot_ptr), ptr)

    def _build_vector_splat(value: ir.Value, width: int) -> ir.Value:
        vec_ty = ir.VectorType(value.type, width)
        seed = tick_builder.insert_element(
            ir.Constant(vec_ty, ir.Undefined), value, ir.Constant(ir.IntType(32), 0)
        )
        mask_ty = ir.VectorType(ir.IntType(32), width)
        mask = ir.Constant(mask_ty, [0] * width)
        return tick_builder.shuffle_vector(
            seed, ir.Constant(vec_ty, ir.Undefined), mask, name="fold_splat"
        )

    def _get_vector_reduce_intrinsic(
        name: str, ret_ty: ir.Type, arg_tys: list[ir.Type]
    ) -> ir.Function:
        fn_ty = ir.FunctionType(ret_ty, arg_tys)
        intrinsic = module.globals.get(name)
        if intrinsic is None:
            intrinsic = ir.Function(module, fn_ty, name=name)
        return intrinsic

    def _reduce_fold(
        operator: str, accum_value: ir.Value, uniform_type: ir.Type
    ) -> ir.Value:
        if accum_value.type != uniform_type:
            return _zero_value(uniform_type)

        vector_value = _build_vector_splat(accum_value, simd_width)
        vector_ty = vector_value.type

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

        intrinsic_name = f"llvm.vector.reduce.{intrinsic_suffix}.v{simd_width}{uniform_type.intrinsic_name}"
        # fadd requires a starting accumulator argument
        needs_start_value = is_float and operator in {"sum", "avg"}
        if needs_start_value:
            intrinsic = _get_vector_reduce_intrinsic(
                intrinsic_name, uniform_type, [uniform_type, vector_ty]
            )
            reduced = tick_builder.call(
                intrinsic,
                [ir.Constant(uniform_type, 0.0), vector_value],
                name="fold_reduce",
            )
        else:
            intrinsic = _get_vector_reduce_intrinsic(
                intrinsic_name, uniform_type, [vector_ty]
            )
            reduced = tick_builder.call(intrinsic, [vector_value], name="fold_reduce")

        if is_float:
            reduced.fastmath.add("fast")

        if operator == "avg":
            if is_float:
                reduced = tick_builder.fdiv(
                    reduced,
                    ir.Constant(uniform_type, float(simd_width)),
                    name="fold_avg",
                )
            else:
                reduced = tick_builder.sdiv(
                    reduced, ir.Constant(uniform_type, simd_width), name="fold_avg"
                )

        return reduced

    def _lower_kernel_route(route: dict[str, Any]):
        kernel_name = str(route.get("kernel", ""))
        callee = function_map.get(
            f"shader_{_sanitize_symbol(kernel_name)}"
        ) or function_map.get(f"filter_{_sanitize_symbol(kernel_name)}")
        if callee is None:
            return

        signature = kernel_signatures.get(kernel_name, {})
        params = signature.get("params", []) if isinstance(signature, dict) else []
        arg_names = route.get("args") if isinstance(route.get("args"), list) else []

        trip_count = 0
        for index, arg_name in enumerate(arg_names):
            if index >= len(params):
                break
            modifier = params[index].get("modifier")
            if modifier == "in" and arg_name in stream_capacities:
                trip_count = max(trip_count, stream_capacities[arg_name])
        target = route.get("target")
        if isinstance(target, str) and target in stream_capacities:
            trip_count = max(trip_count, stream_capacities[target])
        if trip_count <= 0:
            trip_count = 1

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
        call_args = []
        for index, param in enumerate(callee.args):
            arg_name = arg_names[index] if index < len(arg_names) else ""
            modifier = params[index].get("modifier") if index < len(params) else None
            value = None
            if modifier in {"in", "out"} and arg_name in stream_slots:
                value = _load_tick_param("stream", arg_name, param.type)
            elif modifier == "accum" and arg_name in accum_slots:
                value = _load_tick_param("accum", arg_name, param.type)
            elif modifier == "uniform" and arg_name in uniform_slots:
                value = _load_tick_param("uniform", arg_name, param.type)
            if value is None:
                value = _zero_value(param.type)
            call_args.append(value)

        tick_builder.call(callee, call_args)
        next_index = tick_builder.add(
            current, ir.Constant(ir.IntType(32), 1), name="idx_next"
        )
        tick_builder.store(next_index, index_ptr)
        tick_builder.branch(loop_cond)

        tick_builder.position_at_end(loop_exit)

    if bind_routes_ir:
        for route in bind_routes_ir:
            if route.get("kind") == "kernel":
                _lower_kernel_route(route)
                continue
            if route.get("kind") == "fold":
                source_name = str(route.get("source", ""))
                uniform_name = str(route.get("uniform_name", ""))
                operator = str(route.get("operator", ""))
                source_slot = accum_slots.get(source_name)
                uniform_slot = uniform_slots.get(uniform_name)
                if source_slot is None or uniform_slot is None:
                    continue
                uniform_type_name = str(route.get("uniform_type", "float"))
                uniform_type = lowerer._llvm_type(uniform_type_name, known_structs)
                accum_kind, accum_name, _ = arena_fields[source_slot]
                accum_value = _load_tick_param(accum_kind, accum_name, uniform_type)
                reduced = _reduce_fold(operator, accum_value, uniform_type)
                _store_arena_slot(uniform_slot, reduced)
                continue
            asm_ty = ir.FunctionType(ir.VoidType(), [])
            escaped = (
                str(route.get("route", route)).replace("\\", "\\\\").replace('"', '\\"')
            )
            asm = ir.InlineAsm(asm_ty, f"; bind: {escaped}", "", side_effect=True)
            tick_builder.call(asm, [])
    else:
        for route in bind_routes:
            asm_ty = ir.FunctionType(ir.VoidType(), [])
            escaped = str(route).replace("\\", "\\\\").replace('"', '\\"')
            asm = ir.InlineAsm(asm_ty, f"; bind: {escaped}", "", side_effect=True)
            tick_builder.call(asm, [])
    tick_builder.ret_void()

    return str(module)
