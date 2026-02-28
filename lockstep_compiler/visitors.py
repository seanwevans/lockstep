from typing import Any

from .models import (
    LockstepDiagnostic,
    SemanticKernelParam,
    SemanticStructField,
    SemanticSymbol,
)


def build_debug_visitor(base_visitor_cls):
    class LockstepDebugVisitor(base_visitor_cls):
        """Walks the Parse Tree and extracts the pipeline architecture."""

        def __init__(self, verbose: bool = True, *, normalize_bind_routes: bool = False):
            self.verbose = verbose
            self.normalize_bind_routes = normalize_bind_routes
            self.structs = []
            self.shaders = []
            self.filters = []
            self.pure_functions = []
            self.streams = []
            self.accumulators = []
            self.uniforms = []
            self.bind_routes = []
            self.diagnostics: list[LockstepDiagnostic] = []
            self._seen_structs = set()
            self._seen_shaders = set()
            self._seen_filters = set()
            self._seen_pure_functions = set()
            self._seen_streams: set[str] = set()
            self._seen_accumulators: set[str] = set()
            self._seen_uniforms: set[str] = set()

        def _print(self, message: str):
            if self.verbose:
                print(message)

        def _line_col(self, ctx) -> tuple[int, int]:
            token = getattr(ctx, "start", None)
            return (
                getattr(token, "line", 0),
                getattr(token, "column", 0),
            )

        def visitProgram(self, ctx):
            self._print("=== LOCKSTEP COMPILER FRONTEND ===")
            self._print("Parsing program...\n")
            return self.visitChildren(ctx)

        def visitStructDecl(self, ctx):
            name = ctx.ID().getText()
            line, column = self._line_col(ctx)
            if name in self._seen_structs:
                self.diagnostics.append(
                    LockstepDiagnostic(
                        severity="warning",
                        code="LCK201",
                        message=f"Struct '{name}' is redeclared.",
                        line=line,
                        column=column,
                        hint="Rename or remove duplicate struct declarations.",
                    )
                )
            self._seen_structs.add(name)
            self.structs.append(name)
            self._print(f"[Struct] Discovered: {name}")
            return self.visitChildren(ctx)

        def visitPureDecl(self, ctx):
            name = ctx.ID().getText()
            ret_type = ctx.typeName().getText()
            line, column = self._line_col(ctx)
            if name in self._seen_pure_functions:
                self.diagnostics.append(
                    LockstepDiagnostic(
                        severity="warning",
                        code="LCK205",
                        message=f"Pure function '{name}' is redeclared.",
                        line=line,
                        column=column,
                        hint="Rename or remove duplicate pure function declarations.",
                    )
                )
            self._seen_pure_functions.add(name)
            self.pure_functions.append({"name": name, "return_type": ret_type})
            self._print(f"[Pure Function] {name} -> {ret_type}")
            return self.visitChildren(ctx)

        def visitFilterDecl(self, ctx):
            name = ctx.ID().getText()
            line, column = self._line_col(ctx)
            if name in self._seen_filters:
                self.diagnostics.append(
                    LockstepDiagnostic(
                        severity="warning",
                        code="LCK206",
                        message=f"Filter '{name}' is redeclared.",
                        line=line,
                        column=column,
                        hint="Rename or remove duplicate filter declarations.",
                    )
                )
            self._seen_filters.add(name)

            params = []
            self._print(f"\n[Filter Kernel] {name}")
            if ctx.paramList():
                for param in ctx.paramList().param():
                    modifier = param.getChild(0).getText()
                    p_type = param.typeName().getText()
                    p_name = param.ID().getText()
                    params.append({"modifier": modifier, "type": p_type, "name": p_name})
                    self._print(f"  └─ Param: ({modifier}) {p_type} {p_name}")
            self.filters.append({"name": name, "params": params})
            return self.visitChildren(ctx)

        def visitShaderDecl(self, ctx):
            name = ctx.ID().getText()
            params = []
            line, column = self._line_col(ctx)
            if name in self._seen_shaders:
                self.diagnostics.append(
                    LockstepDiagnostic(
                        severity="warning",
                        code="LCK202",
                        message=f"Shader '{name}' is redeclared.",
                        line=line,
                        column=column,
                        hint="Rename or remove duplicate shader declarations.",
                    )
                )
            self._seen_shaders.add(name)
            self._print(f"\n[Shader Kernel] {name}")
            if ctx.paramList():
                for param in ctx.paramList().param():
                    modifier = param.getChild(0).getText()
                    p_type = param.typeName().getText()
                    p_name = param.ID().getText()
                    params.append({"modifier": modifier, "type": p_type, "name": p_name})
                    self._print(f"  └─ Param: ({modifier}) {p_type} {p_name}")
            self.shaders.append({"name": name, "params": params})
            return self.visitChildren(ctx)

        def visitPipelineDecl(self, ctx):
            name = ctx.ID().getText()
            self._seen_streams = set()
            self._seen_accumulators = set()
            self._seen_uniforms = set()
            self._print(f"\n[Pipeline Topology] {name}")
            return self.visitChildren(ctx)

        def visitStreamDecl(self, ctx):
            s_type = ctx.typeName().getText()
            capacity = ctx.INT().getText()
            name = ctx.ID().getText()
            line, column = self._line_col(ctx)
            if name in self._seen_streams:
                self.diagnostics.append(
                    LockstepDiagnostic(
                        severity="warning",
                        code="LCK203",
                        message=f"Stream '{name}' is redeclared.",
                        line=line,
                        column=column,
                        hint="Each stream in a pipeline should have a unique name.",
                    )
                )
            self._seen_streams.add(name)
            self.streams.append({"name": name, "type": s_type, "capacity": capacity})
            self._print(f"  └─ Stream: {name} <{s_type}, {capacity}>")
            return self.visitChildren(ctx)

        def visitAccumDecl(self, ctx):
            a_type = ctx.typeName().getText()
            name = ctx.ID().getText()
            line, column = self._line_col(ctx)
            if name in self._seen_accumulators:
                self.diagnostics.append(
                    LockstepDiagnostic(
                        severity="warning",
                        code="LCK204",
                        message=f"Accumulator '{name}' is redeclared.",
                        line=line,
                        column=column,
                        hint="Each accumulator in a pipeline should have a unique name.",
                    )
                )
            self._seen_accumulators.add(name)
            self.accumulators.append({"name": name, "type": a_type})
            self._print(f"  └─ Accumulator: {name} <{a_type}>")
            return self.visitChildren(ctx)

        def visitUniformDecl(self, ctx):
            u_type = ctx.typeName().getText()
            name = ctx.ID().getText()
            line, column = self._line_col(ctx)
            if name in self._seen_uniforms:
                self.diagnostics.append(
                    LockstepDiagnostic(
                        severity="warning",
                        code="LCK207",
                        message=f"Uniform '{name}' is redeclared.",
                        line=line,
                        column=column,
                        hint="Each uniform in a pipeline should have a unique name.",
                    )
                )
            self._seen_uniforms.add(name)

            initializer = None
            if ctx.expr():
                initializer = ctx.expr().getText()
            self.uniforms.append({"name": name, "type": u_type, "initializer": initializer})
            self._print(f"  └─ Uniform: {name} <{u_type}>")
            return self.visitChildren(ctx)

        def visitBindBlock(self, ctx):
            self._print("  └─ Routing:")
            bind_statements = ctx.bindStmt()
            line, column = self._line_col(ctx)
            if not bind_statements:
                self.diagnostics.append(
                    LockstepDiagnostic(
                        severity="info",
                        code="LCK101",
                        message="Bind block is empty; pipeline has no executable routes.",
                        line=line,
                        column=column,
                        hint="Add at least one binding statement in the bind block.",
                    )
                )
            for stmt in bind_statements:
                route = stmt.getText()
                if self.normalize_bind_routes:
                    route = " ".join(route.split())
                self.bind_routes.append(route)
                self._print(f"       {route}")
            return self.visitChildren(ctx)

    return LockstepDebugVisitor


def build_semantic_validator(base_visitor_cls):
    class LockstepSemanticValidator(base_visitor_cls):
        """Runs semantic checks on a parsed Lockstep program."""

        def __init__(self):
            self.diagnostics: list[LockstepDiagnostic] = []
            self.scopes: list[dict[str, SemanticSymbol]] = []
            self.shaders: dict[str, list[SemanticKernelParam]] = {}
            self.filters: dict[str, list[SemanticKernelParam]] = {}
            self.structs: dict[str, dict[str, SemanticStructField]] = {}

        def _line_col(self, ctx) -> tuple[int, int]:
            token = getattr(ctx, "start", None)
            return (getattr(token, "line", 0), getattr(token, "column", 0))

        def _add_diagnostic(self, *, severity: str, code: str, message: str, ctx, hint: str | None = None):
            line, column = self._line_col(ctx)
            self.diagnostics.append(LockstepDiagnostic(severity, code, message, line, column, hint))

        def _push_scope(self):
            self.scopes.append({})

        def _pop_scope(self):
            if self.scopes:
                self.scopes.pop()

        def _declare(self, name: str, declared_type: str, ctx, *, duplicate_code: str, kind: str = "symbol"):
            if not self.scopes:
                self._push_scope()
            current_scope = self.scopes[-1]
            if name in current_scope:
                self._add_diagnostic(
                    severity="error",
                    code=duplicate_code,
                    message=f"Duplicate declaration for '{name}' in the same scope.",
                    ctx=ctx,
                    hint="Rename one declaration or move it to a different scope.",
                )
                return False
            current_scope[name] = SemanticSymbol(name=name, declared_type=declared_type, kind=kind)
            return True

        def _lookup(self, name: str) -> SemanticSymbol | None:
            for scope in reversed(self.scopes):
                if name in scope:
                    return scope[name]
            return None

        def _declared_in_current_scope(self, name: str) -> bool:
            return bool(self.scopes and name in self.scopes[-1])

        def _record_kernel_signature(self, ctx, target: dict[str, list[SemanticKernelParam]]):
            name = ctx.ID().getText()
            if name in target:
                self._add_diagnostic(
                    severity="error",
                    code="LCK307",
                    message=f"Duplicate shader/filter declaration for '{name}'.",
                    ctx=ctx,
                    hint="Rename one declaration to avoid symbol collisions.",
                )
            params = []
            if ctx.paramList():
                for param in ctx.paramList().param():
                    params.append(
                        SemanticKernelParam(
                            name=param.ID().getText(),
                            declared_type=param.typeName().getText(),
                            modifier=param.getChild(0).getText(),
                        )
                    )
            target[name] = params
            return name, params

        def _check_expression_identifier(self, name: str, ctx):
            if self._lookup(name) is None:
                self._add_diagnostic(
                    severity="error",
                    code="LCK301",
                    message=f"Undefined identifier '{name}'.",
                    ctx=ctx,
                    hint="Declare the identifier in scope before using it.",
                )
                return None
            symbol = self._lookup(name)
            return symbol.declared_type if symbol else None

        def _collect_id_tokens(self, ctx):
            id_tokens = ctx.ID()
            if isinstance(id_tokens, list):
                return id_tokens

            tokens = []
            index = 0
            while True:
                try:
                    token = ctx.ID(index)
                except Exception:
                    break
                if token is None:
                    break
                tokens.append(token)
                index += 1
            return tokens or [id_tokens]

        def _resolve_lvalue_type(self, ctx):
            id_tokens = self._collect_id_tokens(ctx)
            if not id_tokens:
                return None

            root_identifier = id_tokens[0].getText()
            current_type = self._check_expression_identifier(root_identifier, ctx)
            if current_type is None:
                return None

            for field_token in id_tokens[1:]:
                field_name = field_token.getText()
                struct_fields = self.structs.get(current_type)
                if struct_fields is None:
                    self._add_diagnostic(
                        severity="error",
                        code="LCK302",
                        message=(
                            f"Cannot access field '{field_name}' on non-struct type "
                            f"'{current_type}'."
                        ),
                        ctx=ctx,
                        hint="Use field access only on values declared as struct types.",
                    )
                    return None
                if field_name not in struct_fields:
                    self._add_diagnostic(
                        severity="error",
                        code="LCK302",
                        message=f"Struct '{current_type}' has no field '{field_name}'.",
                        ctx=ctx,
                        hint="Use one of the fields declared on this struct.",
                    )
                    return None
                current_type = struct_fields[field_name].declared_type

            return current_type

        def _type_check_bind_call(self, ctx, target_name: str, callee_name: str, arg_names):
            modifier_to_kind = {
                "in": "stream",
                "out": "stream",
                "uniform": "uniform",
                "accum": "accumulator",
            }

            kernel = self.shaders.get(callee_name) or self.filters.get(callee_name)
            if kernel is None:
                self._add_diagnostic(
                    severity="error",
                    code="LCK303",
                    message=f"Undefined shader/filter '{callee_name}' in bind statement.",
                    ctx=ctx,
                    hint="Declare the shader/filter before using it in bind.",
                )
                return

            expected_arity = len(kernel)
            actual_arity = len(arg_names)
            if expected_arity != actual_arity:
                self._add_diagnostic(
                    severity="error",
                    code="LCK304",
                    message=(
                        f"Invocation of '{callee_name}' expects {expected_arity} argument(s), "
                        f"but got {actual_arity}."
                    ),
                    ctx=ctx,
                    hint="Match bind arguments to the shader/filter parameter list.",
                )
                return

            target_symbol = self._lookup(target_name)
            if target_symbol is None:
                self._add_diagnostic(
                    severity="error",
                    code="LCK301",
                    message=f"Undefined identifier '{target_name}'.",
                    ctx=ctx,
                    hint="Declare pipeline streams/accumulators/uniforms before binding.",
                )
            else:
                output_params = [param for param in kernel if param.modifier == "out"]
                if not output_params:
                    self._add_diagnostic(
                        severity="error",
                        code="LCK309",
                        message=(
                            f"Bind target '{target_name}' is assigned from '{callee_name}', "
                            "but the kernel has no out parameter."
                        ),
                        ctx=ctx,
                        hint="Bind only kernels that declare an out parameter.",
                    )
                else:
                    expected_output = output_params[0]
                    expected_output_kind = modifier_to_kind[expected_output.modifier]
                    if target_symbol.kind != expected_output_kind:
                        self._add_diagnostic(
                            severity="error",
                            code="LCK309",
                            message=(
                                f"Bind target '{target_name}' for '{callee_name}' must be a "
                                f"{expected_output_kind} for out parameter '{expected_output.name}', "
                                f"got {target_symbol.kind}."
                            ),
                            ctx=ctx,
                            hint="Route kernel outputs to a stream-compatible bind target.",
                        )
                    if target_symbol.declared_type != expected_output.declared_type:
                        self._add_diagnostic(
                            severity="error",
                            code="LCK305",
                            message=(
                                f"Type mismatch for bind target '{target_name}' in '{callee_name}': "
                                f"expected {expected_output.declared_type}, got {target_symbol.declared_type}."
                            ),
                            ctx=ctx,
                            hint="Align bind target type with the kernel out parameter type.",
                        )

            for arg_name, expected in zip(arg_names, kernel):
                actual_symbol = self._lookup(arg_name)
                if actual_symbol is None:
                    self._add_diagnostic(
                        severity="error",
                        code="LCK301",
                        message=f"Undefined identifier '{arg_name}'.",
                        ctx=ctx,
                        hint="Declare pipeline symbols before passing them to bind.",
                    )
                    continue

                expected_kind = modifier_to_kind.get(expected.modifier)
                if expected_kind is not None and actual_symbol.kind != expected_kind:
                    self._add_diagnostic(
                        severity="error",
                        code="LCK308",
                        message=(
                            f"Modifier mismatch for argument '{arg_name}' in '{callee_name}': "
                            f"parameter '{expected.name}' requires {expected.modifier} "
                            f"({expected_kind}), got {actual_symbol.kind}."
                        ),
                        ctx=ctx,
                        hint="Pass a symbol with the kind required by the parameter modifier.",
                    )

                actual_type = actual_symbol.declared_type
                if actual_type != expected.declared_type:
                    self._add_diagnostic(
                        severity="error",
                        code="LCK305",
                        message=(
                            f"Type mismatch for argument '{arg_name}' in '{callee_name}': "
                            f"expected {expected.declared_type}, got {actual_type}."
                        ),
                        ctx=ctx,
                        hint="Align argument types with the shader/filter signature.",
                    )

        def visitProgram(self, ctx):
            self._push_scope()
            result = self.visitChildren(ctx)
            self._pop_scope()
            return result

        def visitShaderDecl(self, ctx):
            _name, params = self._record_kernel_signature(ctx, self.shaders)
            self._push_scope()
            for param in params:
                self._declare(
                    param.name,
                    param.declared_type,
                    ctx,
                    duplicate_code="LCK306",
                    kind=f"param:{param.modifier}",
                )
            result = self.visitChildren(ctx)
            self._pop_scope()
            return result

        def visitStructDecl(self, ctx):
            struct_name = ctx.ID().getText()
            fields = {}
            seen_field_names: set[str] = set()
            for member in ctx.structMember() or []:
                field_name = member.ID().getText()
                if field_name in seen_field_names:
                    self._add_diagnostic(
                        severity="error",
                        code="LCK311",
                        message=(
                            f"Struct '{struct_name}' has duplicate field declaration "
                            f"'{field_name}'."
                        ),
                        ctx=member,
                        hint="Rename or remove duplicate struct member declarations.",
                    )
                    continue
                seen_field_names.add(field_name)
                fields[field_name] = SemanticStructField(name=field_name, declared_type=member.typeName().getText())
            self.structs[struct_name] = fields
            return self.visitChildren(ctx)

        def visitFilterDecl(self, ctx):
            _name, params = self._record_kernel_signature(ctx, self.filters)
            self._push_scope()
            for param in params:
                self._declare(
                    param.name,
                    param.declared_type,
                    ctx,
                    duplicate_code="LCK306",
                    kind=f"param:{param.modifier}",
                )
            result = self.visitChildren(ctx)
            self._pop_scope()
            return result

        def visitVarDecl(self, ctx):
            self._declare(ctx.ID().getText(), ctx.typeName().getText(), ctx, duplicate_code="LCK306", kind="local")
            return self.visitChildren(ctx)

        def visitPipelineDecl(self, ctx):
            self._push_scope()
            result = self.visitChildren(ctx)
            self._pop_scope()
            return result

        def visitStreamDecl(self, ctx):
            self._declare(ctx.ID().getText(), ctx.typeName().getText(), ctx, duplicate_code="LCK306", kind="stream")
            return self.visitChildren(ctx)

        def visitAccumDecl(self, ctx):
            self._declare(ctx.ID().getText(), ctx.typeName().getText(), ctx, duplicate_code="LCK306", kind="accumulator")
            return self.visitChildren(ctx)

        def visitUniformDecl(self, ctx):
            self._declare(ctx.ID().getText(), ctx.typeName().getText(), ctx, duplicate_code="LCK306", kind="uniform")
            return self.visitChildren(ctx)

        def visitBindStmt(self, ctx):
            id_tokens = ctx.ID()
            if ctx.argList() is not None:
                target_name = id_tokens[0].getText()
                callee_name = id_tokens[1].getText()
                arg_names = [token.getText() for token in id_tokens[2:]]
                self._type_check_bind_call(ctx, target_name, callee_name, arg_names)
                return self.visitChildren(ctx)

            fold_target = id_tokens[0].getText()
            fold_operator = ctx.foldOperator().getText()
            fold_source = id_tokens[1].getText()
            declared_type = ctx.typeName().getText()

            self._declare(
                fold_target,
                declared_type,
                ctx,
                duplicate_code="LCK306",
                kind="uniform",
            )

            fold_source_symbol = self._lookup(fold_source)
            if fold_source_symbol is None:
                self._add_diagnostic(
                    severity="error",
                    code="LCK401",
                    message=f"Fold source accumulator '{fold_source}' is undefined.",
                    ctx=ctx,
                    hint="Declare an accumulator and use it as the fold source.",
                )
            elif fold_source_symbol.kind != "accumulator":
                self._add_diagnostic(
                    severity="error",
                    code="LCK403",
                    message=(
                        f"Fold source '{fold_source}' must reference an accumulator, "
                        f"got {fold_source_symbol.kind}."
                    ),
                    ctx=ctx,
                    hint="Use an accumulator as the input to fold.",
                )
            elif fold_operator not in {"sum", "avg", "min", "max"}:
                self._add_diagnostic(
                    severity="error",
                    code="LCK402",
                    message=f"Unsupported fold operator '{fold_operator}'.",
                    ctx=ctx,
                    hint="Use a valid fold operator such as sum, avg, min, or max.",
                )
            elif fold_source_symbol.declared_type != declared_type:
                self._add_diagnostic(
                    severity="error",
                    code="LCK404",
                    message=(
                        f"Fold target '{fold_target}' has type {declared_type}, but fold source "
                        f"'{fold_source}' has accumulator type {fold_source_symbol.declared_type}."
                    ),
                    ctx=ctx,
                    hint="Match the folded uniform type to the accumulator type.",
                )

            return self.visitChildren(ctx)

        def visitPrimaryExpr(self, ctx):
            if ctx.ID():
                if ctx.exprList() is None:
                    self._check_expression_identifier(ctx.ID().getText(), ctx)
                return self.visitChildren(ctx)
            return self.visitChildren(ctx)

        def visitLvalue(self, ctx):
            self._resolve_lvalue_type(ctx)
            return self.visitChildren(ctx)

        def validate(self, tree):
            self.visit(tree)
            return self.diagnostics

    return LockstepSemanticValidator


def validate_semantics(parse_tree: Any, visitor_cls) -> list[LockstepDiagnostic]:
    validator = build_semantic_validator(visitor_cls)()
    return validator.validate(parse_tree)
