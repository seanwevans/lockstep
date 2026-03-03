from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from llvmlite import ir


_PRIMITIVE_TYPE_MAP: dict[str, ir.Type] = {
    "bool": ir.IntType(1),
    "int": ir.IntType(32),
    "uint": ir.IntType(32),
    "float": ir.FloatType(),
    "double": ir.DoubleType(),
}


def _sanitize_symbol(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)


def _tokenize_expr(expr: str) -> list[str]:
    token_pattern = r"\s*(==|!=|<=|>=|&&|\|\||[()+\-*/%,<>!]|[A-Za-z_][A-Za-z0-9_\.]*|\d+\.\d+|\d+|true|false)"
    return [token for token in re.findall(token_pattern, expr) if token.strip()]


@dataclass
class _ExprParser:
    tokens: list[str]
    index: int = 0

    def _peek(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def _take(self, token: str | None = None) -> str:
        current = self._peek()
        if current is None:
            raise ValueError("unexpected end of expression")
        if token is not None and current != token:
            raise ValueError(f"expected '{token}' but got '{current}'")
        self.index += 1
        return current

    def parse(self):
        return self._parse_or()

    def _parse_or(self):
        node = self._parse_and()
        while self._peek() == "||":
            op = self._take()
            node = ("bin", op, node, self._parse_and())
        return node

    def _parse_and(self):
        node = self._parse_equality()
        while self._peek() == "&&":
            op = self._take()
            node = ("bin", op, node, self._parse_equality())
        return node

    def _parse_equality(self):
        node = self._parse_rel()
        while self._peek() in {"==", "!="}:
            op = self._take()
            node = ("bin", op, node, self._parse_rel())
        return node

    def _parse_rel(self):
        node = self._parse_add()
        while self._peek() in {"<", "<=", ">", ">="}:
            op = self._take()
            node = ("bin", op, node, self._parse_add())
        return node

    def _parse_add(self):
        node = self._parse_mul()
        while self._peek() in {"+", "-"}:
            op = self._take()
            node = ("bin", op, node, self._parse_mul())
        return node

    def _parse_mul(self):
        node = self._parse_unary()
        while self._peek() in {"*", "/", "%"}:
            op = self._take()
            node = ("bin", op, node, self._parse_unary())
        return node

    def _parse_unary(self):
        if self._peek() in {"-", "!"}:
            op = self._take()
            return ("un", op, self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self):
        token = self._peek()
        if token == "(":
            self._take("(")
            node = self._parse_or()
            self._take(")")
            return node
        token = self._take()
        if re.match(r"\d+\.\d+", token):
            return ("float", float(token))
        if re.match(r"\d+", token):
            return ("int", int(token))
        if token in {"true", "false"}:
            return ("bool", token == "true")
        if self._peek() == "(":
            self._take("(")
            args = []
            if self._peek() != ")":
                while True:
                    args.append(self._parse_or())
                    if self._peek() != ",":
                        break
                    self._take(",")
            self._take(")")
            return ("call", token, args)
        return ("var", token)


class _FunctionLowerer:
    def __init__(self, module: ir.Module, function_map: dict[str, ir.Function]):
        self.module = module
        self.function_map = function_map
        self.builder: ir.IRBuilder | None = None
        self.locals: dict[str, ir.AllocaInstr] = {}

    def _llvm_type(self, type_name: str, known_structs: dict[str, ir.IdentifiedStructType]) -> ir.Type:
        if type_name in _PRIMITIVE_TYPE_MAP:
            return _PRIMITIVE_TYPE_MAP[type_name]
        if type_name in known_structs:
            return known_structs[type_name]
        return ir.IntType(8).as_pointer()

    def _coerce_float(self, value: ir.Value) -> ir.Value:
        if isinstance(value.type, ir.IntType) and value.type.width == 1:
            return self.builder.uitofp(value, ir.FloatType())
        if isinstance(value.type, ir.IntType):
            return self.builder.sitofp(value, ir.FloatType())
        return value

    def _load_var(self, name: str) -> ir.Value:
        key = _sanitize_symbol(name.replace(".", "_"))
        if key in self.locals:
            return self.builder.load(self.locals[key], name=f"{key}_val")
        return ir.Constant(ir.FloatType(), 0.0)

    def _parse_expr(self, expr: str):
        parser = _ExprParser(_tokenize_expr(expr))
        return parser.parse()

    def _lower_expr(self, node):
        kind = node[0]
        if kind == "float":
            return ir.Constant(ir.FloatType(), node[1])
        if kind == "int":
            return ir.Constant(ir.IntType(32), node[1])
        if kind == "bool":
            return ir.Constant(ir.IntType(1), int(node[1]))
        if kind == "var":
            return self._load_var(node[1])
        if kind == "un":
            op, operand = node[1], self._lower_expr(node[2])
            if op == "-":
                return self.builder.fsub(ir.Constant(ir.FloatType(), 0.0), self._coerce_float(operand))
            return self.builder.not_(operand)
        if kind == "call":
            name = node[1]
            args = [self._lower_expr(arg) for arg in node[2]]
            if name == "mix" and len(args) == 3:
                a, b, t = (self._coerce_float(arg) for arg in args)
                one_minus_t = self.builder.fsub(ir.Constant(ir.FloatType(), 1.0), t, name="mix_one_minus_t")
                return self.builder.fadd(self.builder.fmul(a, one_minus_t), self.builder.fmul(b, t), name="mix")
            if name == "step" and len(args) == 2:
                edge = self._coerce_float(args[0])
                x_val = self._coerce_float(args[1])
                cmp_result = self.builder.fcmp_ordered(">=", x_val, edge, name="step_cmp")
                return self.builder.uitofp(cmp_result, ir.FloatType(), name="step")
            callee = self.function_map.get(f"pure_{_sanitize_symbol(name)}")
            if callee is not None:
                coerced = [self._coerce_float(arg) if isinstance(param.type, ir.FloatType) else arg for arg, param in zip(args, callee.args)]
                return self.builder.call(callee, coerced, name=f"call_{name}")
            return ir.Constant(ir.FloatType(), 0.0)

        op, lhs, rhs = node[1], self._lower_expr(node[2]), self._lower_expr(node[3])
        if op in {"+", "-", "*", "/", "%"}:
            lval, rval = self._coerce_float(lhs), self._coerce_float(rhs)
            return {
                "+": self.builder.fadd,
                "-": self.builder.fsub,
                "*": self.builder.fmul,
                "/": self.builder.fdiv,
                "%": self.builder.frem,
            }[op](lval, rval)
        if op in {"<", "<=", ">", ">=", "==", "!="}:
            rel_map = {"<": "<", "<=": "<=", ">": ">", ">=": ">=", "==": "==", "!=": "!="}
            return self.builder.fcmp_ordered(rel_map[op], self._coerce_float(lhs), self._coerce_float(rhs))
        if op == "&&":
            return self.builder.and_(lhs, rhs)
        if op == "||":
            return self.builder.or_(lhs, rhs)
        return ir.Constant(ir.FloatType(), 0.0)

    def _lower_statement(self, statement: str, return_type: ir.Type):
        statement = statement.strip()
        if statement.startswith("return"):
            expr = statement[len("return") :].strip().rstrip(";")
            value = self._lower_expr(self._parse_expr(expr))
            if isinstance(return_type, ir.VoidType):
                self.builder.ret_void()
            else:
                self.builder.ret(value)
            return

        if "=" in statement:
            lhs, rhs = statement.rstrip(";").split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            lhs_parts = lhs.split()
            name = lhs_parts[-1]
            key = _sanitize_symbol(name.replace(".", "_"))
            value = self._lower_expr(self._parse_expr(rhs))
            if key not in self.locals:
                slot = self.builder.alloca(value.type, name=key)
                self.locals[key] = slot
            self.builder.store(value, self.locals[key])
            return

        if statement.endswith(";"):
            maybe_decl = statement[:-1].split()
            if maybe_decl:
                key = _sanitize_symbol(maybe_decl[-1].replace(".", "_"))
                if key not in self.locals:
                    slot = self.builder.alloca(ir.FloatType(), name=key)
                    self.locals[key] = slot
                    self.builder.store(ir.Constant(ir.FloatType(), 0.0), slot)

    def lower_function(self, fn: ir.Function, statements: list[str], return_type: ir.Type):
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


def emit_llvm_ir(entities: dict[str, Any]) -> str:
    """Generate LLVM IR using llvmlite lowering for pure/kernels."""

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

    for struct_name in structs:
        safe_name = _sanitize_symbol(struct_name)
        struct_ty = module.context.get_identified_type(f"struct.{safe_name}")
        struct_ty.set_body(ir.IntType(8))
        known_structs[struct_name] = struct_ty

    lowerer = _FunctionLowerer(module, {})

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

    for pure in pure_functions:
        fn = function_map[f"pure_{_sanitize_symbol(pure['name'])}"]
        lowerer.lower_function(fn, pure.get("body", []), fn.function_type.return_type)

    for shader in shaders:
        fn = function_map[f"shader_{_sanitize_symbol(shader['name'])}"]
        lowerer.lower_function(fn, shader.get("body", []), ir.VoidType())

    for flt in filters:
        fn = function_map[f"filter_{_sanitize_symbol(flt['name'])}"]
        lowerer.lower_function(fn, flt.get("body", []), ir.VoidType())

    globals_by_name: dict[str, ir.GlobalVariable] = {}

    for stream in streams:
        gv = ir.GlobalVariable(module, lowerer._llvm_type(stream["type"], known_structs), name=f"stream_{_sanitize_symbol(stream['name'])}")
        gv.linkage = "external"
        globals_by_name[stream["name"]] = gv
    for accum in accumulators:
        gv = ir.GlobalVariable(module, lowerer._llvm_type(accum["type"], known_structs), name=f"accum_{_sanitize_symbol(accum['name'])}")
        gv.linkage = "external"
        globals_by_name[accum["name"]] = gv
    for uniform in uniforms:
        gv = ir.GlobalVariable(module, lowerer._llvm_type(uniform["type"], known_structs), name=f"uniform_{_sanitize_symbol(uniform['name'])}")
        gv.linkage = "external"
        globals_by_name[uniform["name"]] = gv

    kernel_params = {
        shader["name"]: shader.get("params", [])
        for shader in shaders
    }
    kernel_params.update({flt["name"]: flt.get("params", []) for flt in filters})

    tick = ir.Function(module, ir.FunctionType(ir.VoidType(), []), name="Lockstep_Tick")
    tick_entry = tick.append_basic_block("entry")
    tick_builder = ir.IRBuilder(tick_entry)
    if bind_routes_ir:
        for route in bind_routes_ir:
            if route.get("kind") == "kernel":
                kernel_name = str(route.get("kernel", ""))
                callee = function_map.get(f"shader_{_sanitize_symbol(kernel_name)}") or function_map.get(
                    f"filter_{_sanitize_symbol(kernel_name)}"
                )
                if callee is None:
                    continue

                args = route.get("args", []) if isinstance(route.get("args", []), list) else []
                params = kernel_params.get(kernel_name, [])
                target_name = str(route.get("target", ""))
                call_args: list[ir.Value] = []
                for idx, param in enumerate(callee.args):
                    arg_name = args[idx] if idx < len(args) else None
                    source_name = arg_name if isinstance(arg_name, str) and arg_name else None
                    if source_name is None and idx < len(params) and params[idx].get("modifier") == "out":
                        source_name = target_name

                    global_var = globals_by_name.get(source_name) if source_name else None
                    if global_var is not None and global_var.type.pointee == param.type:
                        call_args.append(tick_builder.load(global_var, name=f"load_{_sanitize_symbol(source_name)}"))
                    else:
                        call_args.append(ir.Constant(param.type, None))
                tick_builder.call(callee, call_args)
                continue

            if route.get("kind") == "fold":
                source_name = route.get("source")
                uniform_name = route.get("uniform_name")
                source_var = globals_by_name.get(source_name) if isinstance(source_name, str) else None
                target_var = globals_by_name.get(uniform_name) if isinstance(uniform_name, str) else None
                if source_var is None or target_var is None or source_var.type != target_var.type:
                    continue
                folded_value = tick_builder.load(source_var, name=f"fold_{_sanitize_symbol(source_name)}")
                tick_builder.store(folded_value, target_var)
    else:
        for route in bind_routes:
            asm_ty = ir.FunctionType(ir.VoidType(), [])
            escaped = str(route).replace("\\", "\\\\").replace('"', '\\"')
            asm = ir.InlineAsm(asm_ty, f"; bind: {escaped}", "", side_effect=True)
            tick_builder.call(asm, [])
    tick_builder.ret_void()

    return str(module)
