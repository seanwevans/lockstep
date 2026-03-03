from __future__ import annotations

from typing import Any

import functools

from llvmlite import ir

from .ast import (
    AstAssignStmt,
    AstBinaryExpr,
    AstBoolExpr,
    AstCallExpr,
    AstFloatExpr,
    AstIntExpr,
    AstKernelDecl,
    AstLValueExpr,
    AstProgram,
    AstPureDecl,
    AstReturnStmt,
    AstStatement,
    AstUnaryExpr,
    AstVarDeclStmt,
    ast_to_entities,
)


@functools.lru_cache(maxsize=256)
def _parse_statement_text(statement: str) -> AstStatement | None:
    statement = statement.strip()
    if not statement:
        return None
    if statement.startswith("return") and not statement.startswith("return "):
        statement = f"return {statement[len("return") :]}"
    from antlr4 import CommonTokenStream, InputStream
    from generated.parser.LockstepLexer import LockstepLexer
    from generated.parser.LockstepParser import LockstepParser
    from generated.parser.LockstepVisitor import LockstepVisitor
    from .ast import build_program_ast

    source = f"pure float __tmp() {{ {statement} }}"
    parser = LockstepParser(CommonTokenStream(LockstepLexer(InputStream(source))))
    tree = parser.program()
    program = build_program_ast(tree, LockstepVisitor)
    if not program.pure_functions:
        return None
    body = program.pure_functions[0].body
    return body[0] if body else None


_PRIMITIVE_TYPE_MAP: dict[str, ir.Type] = {
    "bool": ir.IntType(1),
    "int": ir.IntType(32),
    "uint": ir.IntType(32),
    "float": ir.FloatType(),
    "double": ir.DoubleType(),
}


def _sanitize_symbol(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)


class _FunctionLowerer:
    def __init__(
        self,
        module: ir.Module,
        function_map: dict[str, ir.Function],
        known_structs: dict[str, ir.IdentifiedStructType] | None = None,
        struct_fields: dict[str, list[dict[str, str]]] | None = None,
    ):
        self.module = module
        self.function_map = function_map
        self.known_structs = known_structs or {}
        self.struct_fields = struct_fields or {}
        self.builder: ir.IRBuilder | None = None
        self.locals: dict[str, ir.AllocaInstr] = {}

    def _struct_name_for_type(self, llvm_type: ir.Type) -> str | None:
        for name, known_ty in self.known_structs.items():
            if llvm_type is known_ty:
                return name
        return None

    def _field_index_and_type(self, llvm_type: ir.Type, field_name: str) -> tuple[int, ir.Type] | None:
        struct_name = self._struct_name_for_type(llvm_type)
        if struct_name is None:
            return None
        fields = self.struct_fields.get(struct_name, [])
        for index, field in enumerate(fields):
            if field.get("name") == field_name:
                return index, self._llvm_type(field.get("type", "float"), self.known_structs)
        return None

    def _coerce_value_to_type(self, value: ir.Value, target_type: ir.Type) -> ir.Value:
        if value.type == target_type:
            return value
        if isinstance(target_type, ir.IntType) and isinstance(value.type, ir.IntType):
            if value.type.width < target_type.width:
                return self.builder.sext(value, target_type)
            if value.type.width > target_type.width:
                return self.builder.trunc(value, target_type)
        raise TypeError(f"cannot coerce {value.type} to {target_type}")

    def _extract_field_path(self, value: ir.Value, path: list[str]) -> ir.Value:
        current = value
        for field_name in path:
            field_info = self._field_index_and_type(current.type, field_name)
            if field_info is None:
                return ir.Constant(ir.FloatType(), 0.0)
            index, _ = field_info
            current = self.builder.extract_value(current, index, name=f"{field_name}_field")
        return current

    def _insert_field_path(self, aggregate: ir.Value, path: list[str], value: ir.Value) -> ir.Value:
        field_info = self._field_index_and_type(aggregate.type, path[0])
        if field_info is None:
            return aggregate
        index, field_type = field_info
        if len(path) == 1:
            coerced = self._coerce_value_to_type(value, field_type)
            return self.builder.insert_value(aggregate, coerced, index, name=f"set_{path[0]}")

        nested = self.builder.extract_value(aggregate, index, name=f"load_{path[0]}")
        updated_nested = self._insert_field_path(nested, path[1:], value)
        return self.builder.insert_value(aggregate, updated_nested, index, name=f"set_{path[0]}")

    def _llvm_type(self, type_name: str, known_structs: dict[str, ir.IdentifiedStructType]) -> ir.Type:
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
        raise TypeError(f"unary '-' requires numeric operand, got {value.type}")

    def _emit_numeric_binary(self, op: str, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        if lhs.type != rhs.type:
            raise TypeError(f"type mismatch for '{op}': {lhs.type} vs {rhs.type}")

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

        raise TypeError(f"operator '{op}' requires numeric operands, got {lhs.type}")

    def _emit_relational_compare(self, op: str, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        if lhs.type != rhs.type:
            raise TypeError(f"type mismatch for '{op}': {lhs.type} vs {rhs.type}")

        rel_map = {"<": "<", "<=": "<=", ">": ">", ">=": ">=", "==": "==", "!=": "!="}
        if isinstance(lhs.type, ir.FloatType):
            return self.builder.fcmp_ordered(rel_map[op], lhs, rhs)
        if isinstance(lhs.type, ir.IntType):
            return self.builder.icmp_signed(rel_map[op], lhs, rhs)
        raise TypeError(f"operator '{op}' requires comparable operands, got {lhs.type}")

    def _load_var(self, name_or_parts: str | tuple[str, ...]) -> ir.Value:
        parts = name_or_parts.split(".") if isinstance(name_or_parts, str) else list(name_or_parts)
        base_key = _sanitize_symbol(parts[0])
        if base_key in self.locals:
            base_value = self.builder.load(self.locals[base_key], name=f"{base_key}_val")
            if len(parts) == 1:
                return base_value
            return self._extract_field_path(base_value, parts[1:])
        return ir.Constant(ir.FloatType(), 0.0)

    def _lower_expr(self, expr):
        if isinstance(expr, str):
            raise TypeError("string expressions are not supported in typed lowering")
        if isinstance(expr, AstFloatExpr):
            return ir.Constant(ir.FloatType(), expr.value)
        if isinstance(expr, AstIntExpr):
            return ir.Constant(ir.IntType(32), expr.value)
        if isinstance(expr, AstBoolExpr):
            return ir.Constant(ir.IntType(1), int(expr.value))
        if isinstance(expr, AstLValueExpr):
            return self._load_var(expr.lvalue.parts)
        if isinstance(expr, AstUnaryExpr):
            op, operand = expr.operator, self._lower_expr(expr.operand)
            if op == "-":
                return self._emit_numeric_unary_minus(operand)
            return self.builder.not_(operand)
        if isinstance(expr, AstCallExpr):
            name = expr.name
            args = [self._lower_expr(arg) for arg in expr.args]
            if name == "mix" and len(args) == 3:
                if not all(isinstance(arg.type, ir.FloatType) for arg in args):
                    raise TypeError("mix expects float arguments")
                a, b, t = args
                one_minus_t = self.builder.fsub(ir.Constant(ir.FloatType(), 1.0), t, name="mix_one_minus_t")
                return self.builder.fadd(self.builder.fmul(a, one_minus_t), self.builder.fmul(b, t), name="mix")
            if name == "step" and len(args) == 2:
                if not all(isinstance(arg.type, ir.FloatType) for arg in args):
                    raise TypeError("step expects float arguments")
                edge = args[0]
                x_val = args[1]
                cmp_result = self.builder.fcmp_ordered(">=", x_val, edge, name="step_cmp")
                return self.builder.uitofp(cmp_result, ir.FloatType(), name="step")
            callee = self.function_map.get(f"pure_{_sanitize_symbol(name)}")
            if callee is not None:
                coerced = [self._coerce_value_to_type(arg, param.type) for arg, param in zip(args, callee.args)]
                return self.builder.call(callee, coerced, name=f"call_{name}")
            return ir.Constant(ir.FloatType(), 0.0)

        if not isinstance(expr, AstBinaryExpr):
            return ir.Constant(ir.FloatType(), 0.0)
        op, lhs, rhs = expr.operator, self._lower_expr(expr.left), self._lower_expr(expr.right)
        if op in {"+", "-", "*", "/", "%"}:
            return self._emit_numeric_binary(op, lhs, rhs)
        if op in {"<", "<=", ">", ">=", "==", "!="}:
            return self._emit_relational_compare(op, lhs, rhs)
        if op == "&&":
            return self.builder.and_(lhs, rhs)
        if op == "||":
            return self.builder.or_(lhs, rhs)
        return ir.Constant(ir.FloatType(), 0.0)

    def _lower_statement(self, statement: AstStatement | str, return_type: ir.Type):
        if isinstance(statement, str):
            parsed_statement = _parse_statement_text(statement)
            if parsed_statement is None:
                return
            statement = parsed_statement
        if isinstance(statement, AstReturnStmt):
            value = self._lower_expr(statement.value)
            if isinstance(return_type, ir.VoidType):
                self.builder.ret_void()
            else:
                self.builder.ret(self._coerce_value_to_type(value, return_type))
            return

        if isinstance(statement, AstAssignStmt):
            base_name, *field_path = statement.target.parts
            key = _sanitize_symbol(base_name)
            value = self._lower_expr(statement.value)
            if key not in self.locals:
                slot = self.builder.alloca(value.type, name=key)
                self.locals[key] = slot
            if not field_path:
                slot_type = self.locals[key].type.pointee
                self.builder.store(self._coerce_value_to_type(value, slot_type), self.locals[key])
                return

            current = self.builder.load(self.locals[key], name=f"{key}_val")
            updated = self._insert_field_path(current, field_path, value)
            self.builder.store(updated, self.locals[key])
            return

        if isinstance(statement, AstVarDeclStmt):
            key = _sanitize_symbol(statement.name.replace(".", "_"))
            initializer = self._lower_expr(statement.initializer) if statement.initializer is not None else None
            if key not in self.locals:
                if statement.declared_type is not None:
                    llvm_type = self._llvm_type(statement.declared_type, self.known_structs)
                elif initializer is not None:
                    llvm_type = initializer.type
                else:
                    llvm_type = ir.FloatType()
                slot = self.builder.alloca(llvm_type, name=key)
                self.locals[key] = slot
            if initializer is None:
                self.builder.store(ir.Constant(self.locals[key].type.pointee, None), self.locals[key])
            else:
                coerced = self._coerce_value_to_type(initializer, self.locals[key].type.pointee)
                self.builder.store(coerced, self.locals[key])

    def lower_function(self, fn: ir.Function, statements: list[AstStatement | str], return_type: ir.Type):
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


def _normalize_codegen_input(program_or_entities: AstProgram | dict[str, Any]) -> tuple[AstProgram | None, dict[str, Any]]:
    if isinstance(program_or_entities, AstProgram):
        return program_or_entities, ast_to_entities(program_or_entities)
    return None, program_or_entities


def emit_llvm_ir(program_or_entities: AstProgram | dict[str, Any]) -> str:
    """Generate LLVM IR using llvmlite lowering for pure/kernels."""

    typed_program, entities = _normalize_codegen_input(program_or_entities)

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
            fields = struct_decl.get("fields") if isinstance(struct_decl.get("fields"), list) else []
            normalized_structs.append({"name": struct_decl["name"], "fields": fields})

    for struct_decl in normalized_structs:
        struct_name = struct_decl["name"]
        safe_name = _sanitize_symbol(struct_name)
        struct_ty = module.context.get_identified_type(f"struct.{safe_name}")
        known_structs[struct_name] = struct_ty
        struct_fields[struct_name] = struct_decl["fields"]

    lowerer = _FunctionLowerer(module, {}, known_structs, struct_fields)

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
        params = [lowerer._llvm_type(param.get("type", "float"), known_structs) for param in pure.get("params", [])]
        fn = ir.Function(module, ir.FunctionType(ret_ty, params), name=f"pure_{_sanitize_symbol(pure['name'])}")
        for idx, param in enumerate(pure.get("params", [])):
            fn.args[idx].name = _sanitize_symbol(param.get("name", f"arg{idx}"))
        function_map[fn.name] = fn

    for shader in shaders:
        params = [lowerer._llvm_type(param.get("type", "float"), known_structs) for param in shader.get("params", [])]
        fn = ir.Function(module, ir.FunctionType(ir.VoidType(), params), name=f"shader_{_sanitize_symbol(shader['name'])}")
        for idx, param in enumerate(shader.get("params", [])):
            fn.args[idx].name = _sanitize_symbol(param.get("name", f"arg{idx}"))
        function_map[fn.name] = fn

    for flt in filters:
        params = [lowerer._llvm_type(param.get("type", "float"), known_structs) for param in flt.get("params", [])]
        fn = ir.Function(module, ir.FunctionType(ir.VoidType(), params), name=f"filter_{_sanitize_symbol(flt['name'])}")
        for idx, param in enumerate(flt.get("params", [])):
            fn.args[idx].name = _sanitize_symbol(param.get("name", f"arg{idx}"))
        function_map[fn.name] = fn

    lowerer.function_map = function_map

    typed_pure_by_name: dict[str, AstPureDecl] = {}
    typed_shaders_by_name: dict[str, AstKernelDecl] = {}
    typed_filters_by_name: dict[str, AstKernelDecl] = {}
    if typed_program is not None:
        typed_pure_by_name = {decl.name: decl for decl in typed_program.pure_functions}
        typed_shaders_by_name = {decl.name: decl for decl in typed_program.shaders}
        typed_filters_by_name = {decl.name: decl for decl in typed_program.filters}

    for pure in pure_functions:
        fn = function_map[f"pure_{_sanitize_symbol(pure['name'])}"]
        typed_decl = typed_pure_by_name.get(pure["name"])
        lowerer.lower_function(fn, list(typed_decl.body) if typed_decl else pure.get("body", []), fn.function_type.return_type)

    for shader in shaders:
        fn = function_map[f"shader_{_sanitize_symbol(shader['name'])}"]
        typed_decl = typed_shaders_by_name.get(shader["name"])
        lowerer.lower_function(fn, list(typed_decl.body) if typed_decl else shader.get("body", []), ir.VoidType())

    for flt in filters:
        fn = function_map[f"filter_{_sanitize_symbol(flt['name'])}"]
        typed_decl = typed_filters_by_name.get(flt["name"])
        lowerer.lower_function(fn, list(typed_decl.body) if typed_decl else flt.get("body", []), ir.VoidType())

    stream_globals: dict[str, ir.GlobalVariable] = {}
    stream_capacities: dict[str, int] = {}
    for stream in streams:
        gv = ir.GlobalVariable(module, lowerer._llvm_type(stream["type"], known_structs), name=f"stream_{_sanitize_symbol(stream['name'])}")
        gv.linkage = "external"
        stream_globals[stream["name"]] = gv
        stream_capacities[stream["name"]] = int(stream.get("capacity", 0))
    accum_globals: dict[str, ir.GlobalVariable] = {}
    for accum in accumulators:
        gv = ir.GlobalVariable(module, lowerer._llvm_type(accum["type"], known_structs), name=f"accum_{_sanitize_symbol(accum['name'])}")
        gv.linkage = "external"
        accum_globals[accum["name"]] = gv
    uniform_globals: dict[str, ir.GlobalVariable] = {}
    for uniform in uniforms:
        gv = ir.GlobalVariable(module, lowerer._llvm_type(uniform["type"], known_structs), name=f"uniform_{_sanitize_symbol(uniform['name'])}")
        gv.linkage = "external"
        uniform_globals[uniform["name"]] = gv

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

    tick = ir.Function(module, ir.FunctionType(ir.VoidType(), []), name="Lockstep_Tick")
    tick_entry = tick.append_basic_block("entry")
    tick_builder = ir.IRBuilder(tick_entry)

    def _zero_value(llvm_type: ir.Type) -> ir.Value:
        if isinstance(llvm_type, ir.VoidType):
            return ir.Constant(ir.IntType(32), 0)
        return ir.Constant(llvm_type, None)

    def _lower_kernel_route(route: dict[str, Any]):
        kernel_name = str(route.get("kernel", ""))
        callee = function_map.get(f"shader_{_sanitize_symbol(kernel_name)}") or function_map.get(
            f"filter_{_sanitize_symbol(kernel_name)}"
        )
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

        index_ptr = tick_builder.alloca(ir.IntType(32), name=f"{_sanitize_symbol(kernel_name)}_idx")
        tick_builder.store(ir.Constant(ir.IntType(32), 0), index_ptr)

        loop_cond = tick.append_basic_block(f"route_{_sanitize_symbol(kernel_name)}_cond")
        loop_body = tick.append_basic_block(f"route_{_sanitize_symbol(kernel_name)}_body")
        loop_exit = tick.append_basic_block(f"route_{_sanitize_symbol(kernel_name)}_exit")
        tick_builder.branch(loop_cond)

        tick_builder.position_at_end(loop_cond)
        current = tick_builder.load(index_ptr, name="idx")
        cond = tick_builder.icmp_signed("<", current, ir.Constant(ir.IntType(32), trip_count), name="route_active")
        tick_builder.cbranch(cond, loop_body, loop_exit)

        tick_builder.position_at_end(loop_body)
        call_args = []
        for index, param in enumerate(callee.args):
            arg_name = arg_names[index] if index < len(arg_names) else ""
            modifier = params[index].get("modifier") if index < len(params) else None
            value = None
            if modifier in {"in", "out"} and arg_name in stream_globals:
                value = tick_builder.load(stream_globals[arg_name])
            elif modifier == "accum" and arg_name in accum_globals:
                value = tick_builder.load(accum_globals[arg_name])
            elif modifier == "uniform" and arg_name in uniform_globals:
                value = tick_builder.load(uniform_globals[arg_name])
            if value is None:
                value = _zero_value(param.type)
            call_args.append(value)

        tick_builder.call(callee, call_args)
        next_index = tick_builder.add(current, ir.Constant(ir.IntType(32), 1), name="idx_next")
        tick_builder.store(next_index, index_ptr)
        tick_builder.branch(loop_cond)

        tick_builder.position_at_end(loop_exit)

    if bind_routes_ir:
        for route in bind_routes_ir:
            if route.get("kind") == "kernel":
                _lower_kernel_route(route)
                continue
            asm_ty = ir.FunctionType(ir.VoidType(), [])
            escaped = str(route.get("route", route)).replace("\\", "\\\\").replace('"', '\\"')
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
