from difflib import get_close_matches
from types import SimpleNamespace
from typing import Any

from .ast import (
    AstExpr,
    AstExprBinary,
    AstExprCall,
    AstExprCast,
    AstExprLiteral,
    AstAssignStmt,
    AstBindRoute,
    AstFoldBindRoute,
    AstKernelDecl,
    AstKernelBindRoute,
    AstPureDecl,
    AstReturnStmt,
    AstExprUnary,
    AstExprVar,
    AstStatement,
    AstVarDeclStmt,
)
from .prelude import load_intrinsics

from .ast import AstProgram
from .models import (
    LockstepDiagnostic,
    SemanticKernelParam,
    SemanticPipelineResource,
    SemanticPureFunctionContext,
    SemanticPureFunctionSignature,
    SemanticStructField,
    SemanticSymbol,
)
from .type_analysis import validate_type_name
from .visitor_common import SEMANTIC_DIAGNOSTIC_CODES, Scope, ScopedSymbolData


def build_semantic_validator(base_visitor_cls):
    class LockstepSemanticValidator(base_visitor_cls):
        """Runs semantic checks on a parsed Lockstep program."""

        def __init__(self):
            self.diagnostics: list[LockstepDiagnostic] = []
            self.scopes: list[Scope] = []
            self.shaders: dict[str, list[SemanticKernelParam]] = {}
            self.filters: dict[str, list[SemanticKernelParam]] = {}
            self.pure_functions: dict[str, SemanticPureFunctionSignature] = {
                name: SemanticPureFunctionSignature(
                    return_type=signature.return_type,
                    params=tuple(
                        SemanticKernelParam(
                            name=param.name,
                            declared_type=param.type_name,
                            modifier="value",
                        )
                        for param in signature.params
                    ),
                    intrinsic=True,
                )
                for name, signature in load_intrinsics().items()
            }
            self.structs: dict[str, dict[str, SemanticStructField]] = {}
            self._primitive_types = {"int", "uint", "float", "double", "bool", "string"}
            self._current_pure_function: SemanticPureFunctionContext | None = None
            self._pipeline_resource_stack: list[dict[str, SemanticPipelineResource]] = (
                []
            )
            self._pipeline_bind_usage_stack: list[set[str]] = []

        def _line_col(self, ctx) -> tuple[int, int]:
            token = getattr(ctx, "start", None)
            return (getattr(token, "line", 0), getattr(token, "column", 0))

        def _ctx_from_location(self, location):
            return SimpleNamespace(
                start=SimpleNamespace(
                    line=getattr(location, "line", 0),
                    column=getattr(location, "column", 0),
                )
            )

        def _add_diagnostic(
            self,
            *,
            severity: str,
            code: str,
            message: str,
            ctx,
            hint: str | None = None,
        ):
            line, column = self._line_col(ctx)
            self.diagnostics.append(
                LockstepDiagnostic(severity, code, message, line, column, hint)
            )

        def _push_scope(self):
            self.scopes.append(Scope())

        def _pop_scope(self):
            if self.scopes:
                scope = self.scopes[-1]
                for name, scoped_symbol_data in scope.symbols.items():
                    symbol = scoped_symbol_data.symbol
                    if symbol.kind != "local":
                        continue
                    if scoped_symbol_data.usage_count > 0:
                        continue
                    self._add_diagnostic(
                        severity="warning",
                        code=SEMANTIC_DIAGNOSTIC_CODES["unused_symbol"],
                        message=f"Local variable '{name}' is declared but never used.",
                        ctx=scoped_symbol_data.declaration_ctx,
                        hint="Remove unused variables or use them in an expression/assignment.",
                    )
                self.scopes.pop()

        def _declare(
            self,
            name: str,
            declared_type: str,
            ctx,
            *,
            duplicate_code: str,
            kind: str = "symbol",
            assigned: bool = True,
        ):
            if not self.scopes:
                self._push_scope()
            current_scope = self.scopes[-1]
            if name in current_scope.symbols:
                self._add_diagnostic(
                    severity="error",
                    code=duplicate_code,
                    message=f"Duplicate declaration for '{name}' in the same scope.",
                    ctx=ctx,
                    hint="Rename one declaration or move it to a different scope.",
                )
                return False
            current_scope.symbols[name] = ScopedSymbolData(
                symbol=SemanticSymbol(
                    name=name, declared_type=declared_type, kind=kind
                ),
                usage_count=0,
                declaration_ctx=ctx,
                is_assigned=assigned,
            )
            return True

        def _set_symbol_assigned(self, name: str):
            for scope in reversed(self.scopes):
                if name in scope.symbols:
                    scope.symbols[name].is_assigned = True
                    return

        def _is_symbol_assigned(self, name: str) -> bool:
            for scope in reversed(self.scopes):
                if name in scope.symbols:
                    return scope.symbols[name].is_assigned
            return False

        def _mark_symbol_used(self, name: str):
            for scope in reversed(self.scopes):
                if name in scope.symbols:
                    scope.symbols[name].usage_count += 1
                    return

        def _lookup(self, name: str) -> SemanticSymbol | None:
            for scope in reversed(self.scopes):
                if name in scope.symbols:
                    return scope.symbols[name].symbol
            return None

        def _declared_in_current_scope(self, name: str) -> bool:
            return bool(self.scopes and name in self.scopes[-1].symbols)

        def _known_types(self) -> set[str]:
            return self._primitive_types | set(self.structs.keys())

        def _validate_declared_type(self, type_name: str, ctx, code: str) -> bool:
            valid, issues = validate_type_name(type_name, self._known_types())
            if not valid and not issues:
                self._add_diagnostic(
                    severity="error",
                    code=code,
                    message=f"Invalid declared type syntax '{type_name}'.",
                    ctx=ctx,
                    hint="Use forms like T, T[4], Ctor<T>, or Ctor<T,4>.",
                )
                return False

            all_known = True
            for issue in issues:
                hint = (
                    f"Unknown type '{issue.referenced_type}'. Use a primitive ({', '.join(sorted(self._primitive_types))}) "
                    "or declare a struct with this name before using it."
                )
                if issue.suggestions:
                    hint = f"Did you mean {', '.join(issue.suggestions)}? {hint}"

                self._add_diagnostic(
                    severity="error",
                    code=code,
                    message=f"Unknown declared type '{issue.referenced_type}' in '{type_name}'.",
                    ctx=ctx,
                    hint=hint,
                )
                all_known = False
            return all_known

        def _record_kernel_signature(
            self, ctx, target: dict[str, list[SemanticKernelParam]]
        ):
            name = ctx.ID().getText()
            params = []
            if ctx.paramList():
                for param in ctx.paramList().param():
                    self._validate_declared_type(
                        param.typeName().getText(),
                        param.typeName(),
                        "LCK310",
                    )
                    params.append(
                        SemanticKernelParam(
                            name=param.ID().getText(),
                            declared_type=param.typeName().getText(),
                            modifier=param.getChild(0).getText(),
                        )
                    )

            if name in target:
                self._add_diagnostic(
                    severity="error",
                    code=SEMANTIC_DIAGNOSTIC_CODES["duplicate_kernel_declaration"],
                    message=f"Duplicate shader/filter declaration for '{name}'.",
                    ctx=ctx,
                    hint="Rename one declaration to avoid symbol collisions.",
                )
                return name, params

            target[name] = params
            return name, params

        def _check_expression_identifier(
            self, name: str, ctx, *, require_assigned: bool = True
        ):
            symbol = self._lookup(name)
            if symbol is None:
                self._add_diagnostic(
                    severity="error",
                    code=SEMANTIC_DIAGNOSTIC_CODES["undefined_identifier"],
                    message=f"Undefined identifier '{name}'.",
                    ctx=ctx,
                    hint="Declare the identifier in scope before using it.",
                )
                return None
            if (
                require_assigned
                and symbol.kind == "local"
                and not self._is_symbol_assigned(name)
            ):
                self._add_diagnostic(
                    severity="error",
                    code=SEMANTIC_DIAGNOSTIC_CODES["use_before_definition"],
                    message=f"Local variable '{name}' is used before it is assigned.",
                    ctx=ctx,
                    hint="Assign a value to the variable before reading it.",
                )
                return None
            self._mark_symbol_used(name)
            return symbol.declared_type

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

        def _resolve_lvalue_type(self, ctx, *, for_write: bool = False):
            id_tokens = self._collect_id_tokens(ctx)
            if not id_tokens:
                return None

            root_identifier = id_tokens[0].getText()
            current_type = self._check_expression_identifier(
                root_identifier,
                ctx,
                require_assigned=not for_write,
            )
            if current_type is None:
                return None

            for field_token in id_tokens[1:]:
                field_name = field_token.getText()
                struct_fields = self.structs.get(current_type)
                if struct_fields is None:
                    self._add_diagnostic(
                        severity="error",
                        code=SEMANTIC_DIAGNOSTIC_CODES[
                            "invalid_field_access_non_struct"
                        ],
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
                        code=SEMANTIC_DIAGNOSTIC_CODES[
                            "invalid_field_access_unknown_field"
                        ],
                        message=f"Struct '{current_type}' has no field '{field_name}'.",
                        ctx=ctx,
                        hint="Use one of the fields declared on this struct.",
                    )
                    return None
                current_type = struct_fields[field_name].declared_type

            return current_type

        def _type_check_bind_call(
            self, ctx, target_name: str, callee_name: str, arg_names
        ):
            modifier_to_kind = {
                "in": "stream",
                "out": "stream",
                "uniform": "uniform",
                "accum": "accumulator",
            }

            def bind_fixit(action: str) -> str:
                return f"Fix-it: {action}."

            known_kernels = sorted(set(self.shaders.keys()) | set(self.filters.keys()))
            kernel_suggestions = get_close_matches(
                callee_name, known_kernels, n=2, cutoff=0.3
            )

            kernel = self.shaders.get(callee_name) or self.filters.get(callee_name)
            if kernel is None:
                hint = "Declare the shader/filter before using it in bind."
                if kernel_suggestions:
                    suggestion = kernel_suggestions[0]
                    fix = bind_fixit(f"replace '{callee_name}' with '{suggestion}'")
                    hint = f"Did you mean '{suggestion}'? {fix} {hint}"
                self._add_diagnostic(
                    severity="error",
                    code=SEMANTIC_DIAGNOSTIC_CODES["bind_unknown_target"],
                    message=f"Undefined shader/filter '{callee_name}' in bind statement.",
                    ctx=ctx,
                    hint=hint,
                )
                return

            expected_arity = len(kernel)
            actual_arity = len(arg_names)
            if expected_arity != actual_arity:
                parameter_names = [param.name for param in kernel]
                call_example = (
                    f"{target_name} = {callee_name}({', '.join(parameter_names)});"
                )
                self._add_diagnostic(
                    severity="error",
                    code=SEMANTIC_DIAGNOSTIC_CODES["bind_argument_count_mismatch"],
                    message=(
                        f"Invocation of '{callee_name}' expects {expected_arity} argument(s), "
                        f"but got {actual_arity}."
                    ),
                    ctx=ctx,
                    hint=(
                        bind_fixit(f"use the full signature `{call_example}`") + " "
                        "Match bind arguments to the shader/filter parameter list."
                    ),
                )
                return

            out_param_index = next(
                (
                    index
                    for index, param in enumerate(kernel)
                    if param.modifier == "out"
                ),
                None,
            )
            out_param = kernel[out_param_index] if out_param_index is not None else None

            target_symbol = self._lookup(target_name)
            if target_symbol is None:
                self._add_diagnostic(
                    severity="error",
                    code=SEMANTIC_DIAGNOSTIC_CODES["undefined_identifier"],
                    message=f"Undefined identifier '{target_name}'.",
                    ctx=ctx,
                    hint="Declare pipeline streams/accumulators/uniforms before binding.",
                )
            else:
                if out_param is None:
                    self._add_diagnostic(
                        severity="error",
                        code=SEMANTIC_DIAGNOSTIC_CODES[
                            "bind_output_target_kind_mismatch"
                        ],
                        message=(
                            f"Bind target '{target_name}' is assigned from '{callee_name}', "
                            "but the kernel has no out parameter."
                        ),
                        ctx=ctx,
                        hint="Bind only kernels that declare an out parameter.",
                    )
                else:
                    out_arg_name = arg_names[out_param_index]
                    if out_arg_name != target_name:
                        self._add_diagnostic(
                            severity="error",
                            code=SEMANTIC_DIAGNOSTIC_CODES[
                                "bind_output_symbol_mismatch"
                            ],
                            message=(
                                f"Bind target '{target_name}' must match out argument "
                                f"'{out_arg_name}' for parameter '{out_param.name}' in '{callee_name}'."
                            ),
                            ctx=ctx,
                            hint=(
                                bind_fixit(
                                    f"change the bind target to '{out_arg_name}' or the out argument to '{target_name}'"
                                )
                                + " "
                                "Use the same symbol for assignment target and out argument."
                            ),
                        )
                self._mark_symbol_used(target_name)
                if self._pipeline_bind_usage_stack:
                    self._pipeline_bind_usage_stack[-1].add(target_name)
                if out_param is not None:
                    expected_output_kind = modifier_to_kind[out_param.modifier]
                    if target_symbol.kind != expected_output_kind:
                        self._add_diagnostic(
                            severity="error",
                            code=SEMANTIC_DIAGNOSTIC_CODES[
                                "bind_output_target_kind_mismatch"
                            ],
                            message=(
                                f"Bind target '{target_name}' for '{callee_name}' must be a "
                                f"{expected_output_kind} for out parameter '{out_param.name}', "
                                f"got {target_symbol.kind}."
                            ),
                            ctx=ctx,
                            hint=(
                                bind_fixit(
                                    f"bind output to a '{expected_output_kind}' symbol"
                                )
                                + " "
                                "Route kernel outputs to a stream-compatible bind target."
                            ),
                        )
                    if target_symbol.declared_type != out_param.declared_type:
                        self._add_diagnostic(
                            severity="error",
                            code=SEMANTIC_DIAGNOSTIC_CODES["bind_type_mismatch"],
                            message=(
                                f"Type mismatch for bind target '{target_name}' in '{callee_name}': "
                                f"expected {out_param.declared_type}, got {target_symbol.declared_type}."
                            ),
                            ctx=ctx,
                            hint="Align bind target type with the kernel out parameter type.",
                        )

            for arg_name, expected in zip(arg_names, kernel):
                actual_symbol = self._lookup(arg_name)
                if actual_symbol is None:
                    self._add_diagnostic(
                        severity="error",
                        code=SEMANTIC_DIAGNOSTIC_CODES["undefined_identifier"],
                        message=f"Undefined identifier '{arg_name}'.",
                        ctx=ctx,
                        hint=(
                            bind_fixit(f"declare `{arg_name}` in the pipeline block")
                            + " "
                            "Declare pipeline symbols before passing them to bind."
                        ),
                    )
                    continue
                self._mark_symbol_used(arg_name)
                if self._pipeline_bind_usage_stack:
                    self._pipeline_bind_usage_stack[-1].add(arg_name)

                expected_kind = modifier_to_kind.get(expected.modifier)
                if expected_kind is not None and actual_symbol.kind != expected_kind:
                    self._add_diagnostic(
                        severity="error",
                        code=SEMANTIC_DIAGNOSTIC_CODES["bind_modifier_mismatch"],
                        message=(
                            f"Modifier mismatch for argument '{arg_name}' in '{callee_name}': "
                            f"parameter '{expected.name}' requires {expected.modifier} "
                            f"({expected_kind}), got {actual_symbol.kind}."
                        ),
                        ctx=ctx,
                        hint=(
                            bind_fixit(
                                f"replace `{arg_name}` with a {expected_kind} symbol"
                            )
                            + " "
                            "Pass a symbol with the kind required by the parameter modifier."
                        ),
                    )

                actual_type = actual_symbol.declared_type
                if actual_type != expected.declared_type:
                    self._add_diagnostic(
                        severity="error",
                        code=SEMANTIC_DIAGNOSTIC_CODES["bind_type_mismatch"],
                        message=(
                            f"Type mismatch for argument '{arg_name}' in '{callee_name}': "
                            f"expected {expected.declared_type}, got {actual_type}."
                        ),
                        ctx=ctx,
                        hint="Align argument types with the shader/filter signature.",
                    )

        def visitProgram(self, ctx):
            typed_ast = getattr(self, "typed_ast", None)
            if isinstance(typed_ast, AstProgram):
                for struct in typed_ast.structs:
                    struct_ctx = self._ctx_from_location(struct.location)
                    if struct.name in self.structs:
                        self._add_diagnostic(
                            severity="error",
                            code=SEMANTIC_DIAGNOSTIC_CODES["duplicate_declaration"],
                            message=f"Duplicate struct declaration for '{struct.name}'.",
                            ctx=struct_ctx,
                            hint="Rename one struct declaration to keep type names unique.",
                        )
                        continue
                    fields = {}
                    seen_field_names: set[str] = set()
                    for field in struct.fields:
                        field_ctx = self._ctx_from_location(field.location)
                        if field.name in seen_field_names:
                            self._add_diagnostic(
                                severity="error",
                                code=SEMANTIC_DIAGNOSTIC_CODES[
                                    "duplicate_struct_field"
                                ],
                                message=(
                                    f"Struct '{struct.name}' has duplicate field declaration "
                                    f"'{field.name}'."
                                ),
                                ctx=field_ctx,
                                hint="Rename or remove duplicate struct member declarations.",
                            )
                            continue
                        seen_field_names.add(field.name)
                        fields[field.name] = SemanticStructField(
                            name=field.name, declared_type=str(field.declared_type)
                        )
                    self.structs[struct.name] = fields

                for shader in typed_ast.shaders:
                    shader_ctx = self._ctx_from_location(shader.location)
                    params = [
                        SemanticKernelParam(
                            name=param.name,
                            declared_type=str(param.declared_type),
                            modifier=param.modifier,
                        )
                        for param in shader.params
                    ]
                    if shader.name in self.shaders:
                        self._add_diagnostic(
                            severity="error",
                            code=SEMANTIC_DIAGNOSTIC_CODES[
                                "duplicate_kernel_declaration"
                            ],
                            message=f"Duplicate shader/filter declaration for '{shader.name}'.",
                            ctx=shader_ctx,
                            hint="Rename one declaration to avoid symbol collisions.",
                        )
                    else:
                        self.shaders[shader.name] = params

                for filter_decl in typed_ast.filters:
                    filter_ctx = self._ctx_from_location(filter_decl.location)
                    params = [
                        SemanticKernelParam(
                            name=param.name,
                            declared_type=str(param.declared_type),
                            modifier=param.modifier,
                        )
                        for param in filter_decl.params
                    ]
                    if filter_decl.name in self.filters:
                        self._add_diagnostic(
                            severity="error",
                            code=SEMANTIC_DIAGNOSTIC_CODES[
                                "duplicate_kernel_declaration"
                            ],
                            message=f"Duplicate shader/filter declaration for '{filter_decl.name}'.",
                            ctx=filter_ctx,
                            hint="Rename one declaration to avoid symbol collisions.",
                        )
                    else:
                        self.filters[filter_decl.name] = params
            self._push_scope()
            result = self.visitChildren(ctx)
            self._pop_scope()
            return result

        def visitShaderDecl(self, ctx):
            if isinstance(getattr(self, "typed_ast", None), AstProgram):
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
            else:
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
            if isinstance(getattr(self, "typed_ast", None), AstProgram):
                return self.visitChildren(ctx)
            struct_name = ctx.ID().getText()
            if struct_name in self.structs:
                self._add_diagnostic(
                    severity="error",
                    code=SEMANTIC_DIAGNOSTIC_CODES["duplicate_declaration"],
                    message=f"Duplicate struct declaration for '{struct_name}'.",
                    ctx=ctx,
                    hint="Rename one struct declaration to keep type names unique.",
                )
                return self.visitChildren(ctx)

            fields = {}
            seen_field_names: set[str] = set()
            for member in ctx.structMember() or []:
                field_name = member.ID().getText()
                if field_name in seen_field_names:
                    self._add_diagnostic(
                        severity="error",
                        code=SEMANTIC_DIAGNOSTIC_CODES["duplicate_struct_field"],
                        message=(
                            f"Struct '{struct_name}' has duplicate field declaration "
                            f"'{field_name}'."
                        ),
                        ctx=member,
                        hint="Rename or remove duplicate struct member declarations.",
                    )
                    continue
                seen_field_names.add(field_name)
                fields[field_name] = SemanticStructField(
                    name=field_name, declared_type=member.typeName().getText()
                )
            self.structs[struct_name] = fields
            return self.visitChildren(ctx)

        def visitFilterDecl(self, ctx):
            if isinstance(getattr(self, "typed_ast", None), AstProgram):
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
            else:
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

        def visitPureDecl(self, ctx):
            name = ctx.ID().getText()
            return_type = ctx.typeName().getText()
            self._validate_declared_type(return_type, ctx.typeName(), "LCK310")
            params = []
            if ctx.pureParamList() is not None:
                param_list = ctx.pureParamList()
                param_types = param_list.typeName()
                param_names = param_list.ID()
                for index, param_name in enumerate(param_names):
                    self._validate_declared_type(
                        param_types[index].getText(),
                        param_types[index],
                        "LCK310",
                    )
                    params.append(
                        SemanticKernelParam(
                            name=param_name.getText(),
                            declared_type=param_types[index].getText(),
                            modifier="value",
                        )
                    )

            self.pure_functions[name] = SemanticPureFunctionSignature(
                return_type=return_type,
                params=tuple(params),
            )

            statements = []
            if hasattr(ctx, "statement") and callable(ctx.statement):
                statement_nodes = ctx.statement() or []
                if isinstance(statement_nodes, list):
                    statements = statement_nodes
                else:
                    statements = [statement_nodes]

            # Lightweight test doubles may expose a direct returnStmt() on the
            # pure declaration context instead of nesting it under statement().
            # Mirror parser behavior by treating that as a single statement.
            if (
                not statements
                and hasattr(ctx, "returnStmt")
                and callable(ctx.returnStmt)
                and ctx.returnStmt() is not None
            ):
                statements = [ctx]

            return_statements: list[tuple[int, Any]] = []
            for index, statement in enumerate(statements):
                if not hasattr(statement, "returnStmt") or not callable(
                    statement.returnStmt
                ):
                    continue
                return_stmt = statement.returnStmt()
                if return_stmt is not None:
                    return_statements.append((index, return_stmt))

            if not return_statements:
                self._add_diagnostic(
                    severity="error",
                    code=SEMANTIC_DIAGNOSTIC_CODES["pure_missing_return"],
                    message=f"Pure function '{name}' must include a return statement.",
                    ctx=ctx,
                    hint="Add a return statement that produces a value matching the declared return type.",
                )
            else:
                if len(return_statements) > 1:
                    self._add_diagnostic(
                        severity="warning",
                        code=SEMANTIC_DIAGNOSTIC_CODES["pure_multiple_returns"],
                        message=(
                            f"Pure function '{name}' contains multiple return statements; "
                            "only the first return is reachable in straight-line semantics."
                        ),
                        ctx=return_statements[1][1],
                        hint="Keep a single terminal return to avoid dead code and ambiguous intent.",
                    )

                first_return_index = return_statements[0][0]
                for unreachable_stmt in statements[first_return_index + 1 :]:
                    self._add_diagnostic(
                        severity="warning",
                        code=SEMANTIC_DIAGNOSTIC_CODES["pure_unreachable_after_return"],
                        message=(
                            f"Unreachable statement in pure function '{name}' after return statement."
                        ),
                        ctx=unreachable_stmt,
                        hint="Remove or move statements before the return.",
                    )

            self._push_scope()
            previous_pure_function = self._current_pure_function
            self._current_pure_function = SemanticPureFunctionContext(
                name=name,
                return_type=return_type,
            )
            for param in params:
                self._declare(
                    param.name,
                    param.declared_type,
                    ctx,
                    duplicate_code="LCK306",
                    kind=f"param:{param.modifier}",
                )
            result = self.visitChildren(ctx)
            self._current_pure_function = previous_pure_function
            self._pop_scope()
            return result

        def _resolve_expr_type(self, expr):
            if expr is None:
                return None
            if not isinstance(expr, AstExpr):
                return None
            return self._resolve_ast_expr_type(expr)

        def _lookup_path(self, path: tuple[str, ...]) -> str | None:
            if not path:
                return None
            symbol = self._lookup(path[0])
            if symbol is None:
                return None
            current_type = symbol.declared_type
            for field_name in path[1:]:
                fields = self.structs.get(current_type)
                if fields is None or field_name not in fields:
                    return None
                current_type = fields[field_name].declared_type
            return current_type

        def _resolve_ast_expr_type(self, expr: AstExpr) -> str | None:
            numeric_types = {"int", "uint", "float", "double"}

            def _resolve_binary(expr_binary: AstExprBinary) -> str | None:
                left_type = self._resolve_ast_expr_type(expr_binary.left)
                right_type = self._resolve_ast_expr_type(expr_binary.right)
                op = expr_binary.op

                if left_type is None or right_type is None:
                    return None
                if op in {"+", "-", "*", "/"}:
                    if left_type in numeric_types and right_type == left_type:
                        return left_type
                    return None
                if op in {"&", "|", "^"}:
                    if left_type in {"int", "uint"} and right_type == left_type:
                        return left_type
                    return None
                if op in {"&&", "||"}:
                    if left_type == "bool" and right_type == "bool":
                        return "bool"
                    return None
                if op in {"==", "!=", "<", "<=", ">", ">="}:
                    if left_type == right_type:
                        return "bool"
                    return None
                if op in {"<<", ">>"}:
                    if left_type in {"int", "uint"} and right_type in {"int", "uint"}:
                        return left_type
                    return None
                return None

            def _resolve_call(expr_call: AstExprCall) -> str | None:
                if expr_call.name in {"int", "float", "double", "uint", "bool"}:
                    return expr_call.name if len(expr_call.args) == 1 else None
                if expr_call.name == "select":
                    if len(expr_call.args) != 3:
                        return None
                    condition_type = self._resolve_ast_expr_type(expr_call.args[0])
                    true_type = self._resolve_ast_expr_type(expr_call.args[1])
                    false_type = self._resolve_ast_expr_type(expr_call.args[2])
                    if condition_type == "bool" and true_type == false_type:
                        return true_type
                    return None
                signature = self.pure_functions.get(expr_call.name)
                return signature.return_type if signature is not None else None

            type_dispatch = {
                AstExprLiteral: lambda node: node.kind,
                AstExprVar: lambda node: self._lookup_path(node.path),
                AstExprUnary: lambda node: (
                    "bool"
                    if node.op == "!"
                    else (
                        self._resolve_ast_expr_type(node.operand)
                        if (
                            node.op != "-"
                            or self._resolve_ast_expr_type(node.operand)
                            in numeric_types
                        )
                        else None
                    )
                ),
                AstExprBinary: _resolve_binary,
                AstExprCall: _resolve_call,
                AstExprCast: lambda node: node.target_type.name,
            }

            for expr_type, resolver in type_dispatch.items():
                if isinstance(expr, expr_type):
                    return resolver(expr)
            return None

        def _type_check_pure_call(self, ctx):
            callee_name = ctx.ID().getText()
            signature = self.pure_functions.get(callee_name)
            if signature is None:
                self._add_diagnostic(
                    severity="error",
                    code=SEMANTIC_DIAGNOSTIC_CODES["pure_unknown_function"],
                    message=f"Undefined pure function '{callee_name}'.",
                    ctx=ctx,
                    hint="Declare the pure function before calling it.",
                )
                return

            expected_params = signature.params
            actual_args = []
            if ctx.exprList() is not None:
                actual_args = ctx.exprList().expr()

            if len(actual_args) != len(expected_params):
                self._add_diagnostic(
                    severity="error",
                    code=SEMANTIC_DIAGNOSTIC_CODES["pure_argument_count_mismatch"],
                    message=(
                        f"Pure function '{callee_name}' expects {len(expected_params)} argument(s), "
                        f"but got {len(actual_args)}."
                    ),
                    ctx=ctx,
                    hint="Pass the exact number of arguments declared in the pure function signature.",
                )
                return

            for index, (arg_expr, expected) in enumerate(
                zip(actual_args, expected_params), start=1
            ):
                actual_type = self._resolve_expr_type(arg_expr)
                if actual_type != expected.declared_type:
                    resolved_actual = (
                        actual_type if actual_type is not None else "<unresolved>"
                    )
                    self._add_diagnostic(
                        severity="error",
                        code=SEMANTIC_DIAGNOSTIC_CODES["pure_argument_type_mismatch"],
                        message=(
                            f"Type mismatch for argument {index} in pure call '{callee_name}': "
                            f"expected {expected.declared_type}, got {resolved_actual}."
                        ),
                        ctx=ctx,
                        hint="Ensure each argument type matches the pure function parameter type.",
                    )

        def visitVarDecl(self, ctx):
            type_ctx = (
                ctx.typeName()
                if hasattr(ctx, "typeName") and callable(ctx.typeName)
                else None
            )
            declared_type = type_ctx.getText() if type_ctx is not None else None
            symbol_name = ctx.ID().getText()
            if declared_type is not None:
                self._validate_declared_type(declared_type, type_ctx, "LCK310")

            if declared_type is None:
                existing_symbol = self._lookup(symbol_name)
                if existing_symbol is not None:
                    has_initializer = (
                        hasattr(ctx, "expr")
                        and callable(ctx.expr)
                        and ctx.expr() is not None
                    )
                    if has_initializer:
                        initializer_type = self._resolve_expr_type(ctx.expr())
                        if (
                            initializer_type is not None
                            and existing_symbol.declared_type != initializer_type
                        ):
                            self._add_diagnostic(
                                severity="error",
                                code=SEMANTIC_DIAGNOSTIC_CODES[
                                    "assignment_type_mismatch"
                                ],
                                message=(
                                    "Type mismatch in assignment: "
                                    f"left-hand side expects {existing_symbol.declared_type}, got {initializer_type}."
                                ),
                                ctx=ctx,
                                hint="Assign expressions whose type matches the lvalue declaration.",
                            )
                    self._mark_symbol_used(symbol_name)
                    return self.visitChildren(ctx)

            has_initializer = (
                hasattr(ctx, "expr") and callable(ctx.expr) and ctx.expr() is not None
            )
            initializer_type = (
                self._resolve_expr_type(ctx.expr()) if has_initializer else None
            )

            if declared_type is None:
                if initializer_type is None:
                    self._add_diagnostic(
                        severity="error",
                        code=SEMANTIC_DIAGNOSTIC_CODES["cannot_infer_type"],
                        message=(
                            f"Cannot infer type for local variable '{symbol_name}' without a typed initializer."
                        ),
                        ctx=ctx,
                        hint="Provide an explicit type or initialize with an expression whose type can be resolved.",
                    )
                    return self.visitChildren(ctx)
                declared_type = initializer_type

            self._declare(
                symbol_name,
                declared_type,
                ctx,
                duplicate_code="LCK306",
                kind="local",
                assigned=has_initializer,
            )

            if has_initializer:
                if initializer_type is not None and initializer_type != declared_type:
                    self._add_diagnostic(
                        severity="error",
                        code=SEMANTIC_DIAGNOSTIC_CODES["var_initializer_type_mismatch"],
                        message=(
                            f"Type mismatch in initializer for '{ctx.ID().getText()}': "
                            f"expected {declared_type}, got {initializer_type}."
                        ),
                        ctx=ctx,
                        hint="Use an initializer expression with the same type as the declared variable.",
                    )
            return self.visitChildren(ctx)

        def visitAssignStmt(self, ctx):
            lvalue_ctx = (
                ctx.lvalue()
                if hasattr(ctx, "lvalue") and callable(ctx.lvalue)
                else None
            )
            expr_ctx = (
                ctx.expr() if hasattr(ctx, "expr") and callable(ctx.expr) else None
            )

            lvalue_type = (
                self._resolve_lvalue_type(lvalue_ctx, for_write=True)
                if lvalue_ctx is not None
                else None
            )
            expr_type = self._resolve_expr_type(expr_ctx)

            if (
                lvalue_type is not None
                and expr_type is not None
                and lvalue_type != expr_type
            ):
                self._add_diagnostic(
                    severity="error",
                    code=SEMANTIC_DIAGNOSTIC_CODES["assignment_type_mismatch"],
                    message=(
                        "Type mismatch in assignment: "
                        f"left-hand side expects {lvalue_type}, got {expr_type}."
                    ),
                    ctx=ctx,
                    hint="Assign expressions whose type matches the lvalue declaration.",
                )

            if lvalue_ctx is not None:
                id_tokens = self._collect_id_tokens(lvalue_ctx)
                if id_tokens:
                    self._set_symbol_assigned(id_tokens[0].getText())

            return self.visitChildren(ctx)

        def visitReturnStmt(self, ctx):
            if self._current_pure_function is None:
                return self.visitChildren(ctx)

            expected_type = self._current_pure_function.return_type
            return_expr = (
                ctx.expr() if hasattr(ctx, "expr") and callable(ctx.expr) else None
            )
            actual_type = self._resolve_expr_type(return_expr)
            if actual_type is not None and actual_type != expected_type:
                self._add_diagnostic(
                    severity="error",
                    code=SEMANTIC_DIAGNOSTIC_CODES["pure_return_type_mismatch"],
                    message=(
                        f"Return type mismatch in pure function '{self._current_pure_function.name}': "
                        f"expected {expected_type}, got {actual_type}."
                    ),
                    ctx=ctx,
                    hint="Return an expression whose type matches the pure function return type.",
                )

            return self.visitChildren(ctx)

        def visitPipelineDecl(self, ctx):
            self._pipeline_resource_stack.append({})
            self._pipeline_bind_usage_stack.append(set())
            self._push_scope()
            result = self.visitChildren(ctx)
            self._pop_scope()

            declared_resources = self._pipeline_resource_stack.pop()
            bind_used_resources = self._pipeline_bind_usage_stack.pop()
            for resource_name, resource in declared_resources.items():
                if resource_name in bind_used_resources:
                    continue
                self._add_diagnostic(
                    severity="warning",
                    code=SEMANTIC_DIAGNOSTIC_CODES["unbound_pipeline_resource"],
                    message=(
                        f"Pipeline {resource.kind} '{resource_name}' is declared but not used in the bind block."
                    ),
                    ctx=resource.declaration_ctx,
                    hint="Reference every declared stream/accumulator in at least one bind statement.",
                )
            return result

        def visitStreamDecl(self, ctx):
            self._validate_declared_type(
                ctx.typeName().getText(), ctx.typeName(), "LCK310"
            )
            self._declare(
                ctx.ID().getText(),
                ctx.typeName().getText(),
                ctx,
                duplicate_code="LCK306",
                kind="stream",
            )
            if self._pipeline_resource_stack:
                self._pipeline_resource_stack[-1][ctx.ID().getText()] = (
                    SemanticPipelineResource(
                        kind="stream",
                        declaration_ctx=ctx,
                    )
                )
            return self.visitChildren(ctx)

        def visitAccumDecl(self, ctx):
            self._validate_declared_type(
                ctx.typeName().getText(), ctx.typeName(), "LCK310"
            )
            self._declare(
                ctx.ID().getText(),
                ctx.typeName().getText(),
                ctx,
                duplicate_code="LCK306",
                kind="accumulator",
            )
            if self._pipeline_resource_stack:
                self._pipeline_resource_stack[-1][ctx.ID().getText()] = (
                    SemanticPipelineResource(
                        kind="accumulator",
                        declaration_ctx=ctx,
                    )
                )
            return self.visitChildren(ctx)

        def visitUniformDecl(self, ctx):
            declared_type = ctx.typeName().getText()
            self._validate_declared_type(declared_type, ctx.typeName(), "LCK310")
            self._declare(
                ctx.ID().getText(),
                declared_type,
                ctx,
                duplicate_code="LCK306",
                kind="uniform",
            )

            has_initializer = (
                hasattr(ctx, "expr") and callable(ctx.expr) and ctx.expr() is not None
            )
            if has_initializer:
                initializer_type = self._resolve_expr_type(ctx.expr())
                if initializer_type is not None and initializer_type != declared_type:
                    self._add_diagnostic(
                        severity="error",
                        code=SEMANTIC_DIAGNOSTIC_CODES[
                            "uniform_initializer_type_mismatch"
                        ],
                        message=(
                            f"Type mismatch in uniform initializer for '{ctx.ID().getText()}': "
                            f"expected {declared_type}, got {initializer_type}."
                        ),
                        ctx=ctx,
                        hint="Use an initializer expression with the same type as the declared uniform.",
                    )
            if self._pipeline_resource_stack:
                self._pipeline_resource_stack[-1][ctx.ID().getText()] = (
                    SemanticPipelineResource(
                        kind="uniform",
                        declaration_ctx=ctx,
                    )
                )
            return self.visitChildren(ctx)

        def visitBindStmt(self, ctx):
            id_tokens = ctx.ID()
            if ctx.argList() is not None:
                target_name = id_tokens[0].getText()
                callee_name = id_tokens[1].getText()

                arg_tokens = ctx.argList().ID()
                if isinstance(arg_tokens, list):
                    arg_names = [token.getText() for token in arg_tokens]
                elif arg_tokens is None:
                    arg_names = []
                else:
                    arg_names = [arg_tokens.getText()]

                self._type_check_bind_call(ctx, target_name, callee_name, arg_names)
                return self.visitChildren(ctx)

            fold_target = id_tokens[0].getText()
            fold_operator = ctx.foldOperator().getText()
            fold_source = id_tokens[1].getText()
            declared_type = ctx.typeName().getText()

            self._validate_declared_type(
                declared_type,
                ctx.typeName(),
                SEMANTIC_DIAGNOSTIC_CODES["unknown_declared_type"],
            )

            self._declare(
                fold_target,
                declared_type,
                ctx,
                duplicate_code=SEMANTIC_DIAGNOSTIC_CODES["duplicate_declaration"],
                kind="uniform",
            )

            if fold_operator not in {"sum", "avg", "min", "max"}:
                self._add_diagnostic(
                    severity="error",
                    code=SEMANTIC_DIAGNOSTIC_CODES["unknown_fold_operator"],
                    message=f"Unsupported fold operator '{fold_operator}'.",
                    ctx=ctx,
                    hint="Use a valid fold operator such as sum, avg, min, or max.",
                )
                return self.visitChildren(ctx)

            fold_source_symbol = self._lookup(fold_source)
            if fold_source_symbol is None:
                self._add_diagnostic(
                    severity="error",
                    code=SEMANTIC_DIAGNOSTIC_CODES["fold_unknown_source"],
                    message=f"Fold source accumulator '{fold_source}' is undefined.",
                    ctx=ctx,
                    hint="Declare an accumulator and use it as the fold source.",
                )
            elif fold_source_symbol.kind != "accumulator":
                self._add_diagnostic(
                    severity="error",
                    code=SEMANTIC_DIAGNOSTIC_CODES["fold_unknown_source"],
                    message=(
                        f"Fold source '{fold_source}' must reference an accumulator, "
                        f"got {fold_source_symbol.kind}."
                    ),
                    ctx=ctx,
                    hint="Use an accumulator as the input to fold.",
                )
            elif fold_source_symbol.declared_type != declared_type:
                self._add_diagnostic(
                    severity="error",
                    code=SEMANTIC_DIAGNOSTIC_CODES["fold_type_mismatch"],
                    message=(
                        f"Fold target '{fold_target}' has type {declared_type}, but fold source "
                        f"'{fold_source}' has accumulator type {fold_source_symbol.declared_type}."
                    ),
                    ctx=ctx,
                    hint="Match the folded uniform type to the accumulator type.",
                )
            else:
                self._mark_symbol_used(fold_source)
                if self._pipeline_bind_usage_stack:
                    self._pipeline_bind_usage_stack[-1].add(fold_source)

            return self.visitChildren(ctx)

        def visitPrimaryExpr(self, ctx):
            if ctx.ID():
                if ctx.exprList() is None:
                    self._check_expression_identifier(ctx.ID().getText(), ctx)
                else:
                    callee_name = ctx.ID().getText()
                    if callee_name in {"int", "float", "double", "uint", "bool"}:
                        args = (
                            ctx.exprList().expr() if ctx.exprList() is not None else []
                        )
                        if len(args) != 1:
                            self._add_diagnostic(
                                severity="error",
                                code=SEMANTIC_DIAGNOSTIC_CODES[
                                    "pure_argument_count_mismatch"
                                ],
                                message=(
                                    f"Cast '{callee_name}(...)' expects 1 argument, but got {len(args)}."
                                ),
                                ctx=ctx,
                                hint="Pass exactly one expression to a cast.",
                            )
                    elif callee_name == "select":
                        args = (
                            ctx.exprList().expr() if ctx.exprList() is not None else []
                        )
                        if len(args) != 3:
                            self._add_diagnostic(
                                severity="error",
                                code=SEMANTIC_DIAGNOSTIC_CODES[
                                    "pure_argument_count_mismatch"
                                ],
                                message=(
                                    f"Built-in 'select(...)' expects 3 arguments, but got {len(args)}."
                                ),
                                ctx=ctx,
                                hint="Use `select(condition, when_true, when_false)`.",
                            )
                        else:
                            condition_type = self._resolve_expr_type(args[0])
                            true_type = self._resolve_expr_type(args[1])
                            false_type = self._resolve_expr_type(args[2])
                            if condition_type is not None and condition_type != "bool":
                                self._add_diagnostic(
                                    severity="error",
                                    code=SEMANTIC_DIAGNOSTIC_CODES[
                                        "pure_argument_type_mismatch"
                                    ],
                                    message=(
                                        "Type mismatch for built-in 'select' condition: "
                                        f"expected bool, got {condition_type}."
                                    ),
                                    ctx=ctx,
                                    hint="Pass a boolean condition as the first argument.",
                                )
                            if (
                                true_type is not None
                                and false_type is not None
                                and true_type != false_type
                            ):
                                self._add_diagnostic(
                                    severity="error",
                                    code=SEMANTIC_DIAGNOSTIC_CODES[
                                        "pure_argument_type_mismatch"
                                    ],
                                    message=(
                                        "Type mismatch for built-in 'select' value arguments: "
                                        f"expected matching types, got {true_type} and {false_type}."
                                    ),
                                    ctx=ctx,
                                    hint="Pass true/false values with the same type.",
                                )
                    else:
                        self._type_check_pure_call(ctx)
                return self.visitChildren(ctx)
            return self.visitChildren(ctx)

        def visitLvalue(self, ctx):
            self._resolve_lvalue_type(ctx)
            return self.visitChildren(ctx)

        def _validate_ast_program(self, program: AstProgram):
            def _stmt_ctx(node):
                return self._ctx_from_location(getattr(node, "location", None))

            def _validate_ast_expr(expr: AstExpr):
                if isinstance(expr, AstExprLiteral):
                    return
                if isinstance(expr, AstExprVar):
                    expr_ctx = _stmt_ctx(expr)
                    self._resolve_ast_lvalue_type(expr.path, expr_ctx, for_write=False)
                    return
                if isinstance(expr, AstExprUnary):
                    _validate_ast_expr(expr.operand)
                    return
                if isinstance(expr, AstExprBinary):
                    _validate_ast_expr(expr.left)
                    _validate_ast_expr(expr.right)
                    return
                if isinstance(expr, AstExprCall):
                    call_ctx = _stmt_ctx(expr)
                    for arg in expr.args:
                        _validate_ast_expr(arg)
                    _validate_ast_call(expr, call_ctx)
                    return
                if isinstance(expr, AstExprCast):
                    cast_ctx = _stmt_ctx(expr)
                    self._validate_declared_type(
                        str(expr.target_type), cast_ctx, "LCK310"
                    )
                    _validate_ast_expr(expr.value)

            def _validate_ast_call(expr: AstExprCall, ctx):
                callee_name = expr.name
                if callee_name in {"int", "float", "double", "uint", "bool"}:
                    if len(expr.args) != 1:
                        self._add_diagnostic(
                            severity="error",
                            code=SEMANTIC_DIAGNOSTIC_CODES[
                                "pure_argument_count_mismatch"
                            ],
                            message=(
                                f"Cast '{callee_name}(...)' expects 1 argument, but got {len(expr.args)}."
                            ),
                            ctx=ctx,
                            hint="Pass exactly one expression to a cast.",
                        )
                    return

                if callee_name == "select":
                    if len(expr.args) != 3:
                        self._add_diagnostic(
                            severity="error",
                            code=SEMANTIC_DIAGNOSTIC_CODES[
                                "pure_argument_count_mismatch"
                            ],
                            message=(
                                f"Built-in 'select(...)' expects 3 arguments, but got {len(expr.args)}."
                            ),
                            ctx=ctx,
                            hint="Use `select(condition, when_true, when_false)`.",
                        )
                        return
                    condition_type = self._resolve_ast_expr_type(expr.args[0])
                    true_type = self._resolve_ast_expr_type(expr.args[1])
                    false_type = self._resolve_ast_expr_type(expr.args[2])
                    if condition_type is not None and condition_type != "bool":
                        self._add_diagnostic(
                            severity="error",
                            code=SEMANTIC_DIAGNOSTIC_CODES[
                                "pure_argument_type_mismatch"
                            ],
                            message=(
                                "Type mismatch for built-in 'select' condition: "
                                f"expected bool, got {condition_type}."
                            ),
                            ctx=ctx,
                            hint="Pass a boolean condition as the first argument.",
                        )
                    if (
                        true_type is not None
                        and false_type is not None
                        and true_type != false_type
                    ):
                        self._add_diagnostic(
                            severity="error",
                            code=SEMANTIC_DIAGNOSTIC_CODES[
                                "pure_argument_type_mismatch"
                            ],
                            message=(
                                "Type mismatch for built-in 'select' value arguments: "
                                f"expected matching types, got {true_type} and {false_type}."
                            ),
                            ctx=ctx,
                            hint="Pass true/false values with the same type.",
                        )
                    return

                signature = self.pure_functions.get(callee_name)
                if signature is None:
                    self._add_diagnostic(
                        severity="error",
                        code=SEMANTIC_DIAGNOSTIC_CODES["pure_unknown_function"],
                        message=f"Undefined pure function '{callee_name}'.",
                        ctx=ctx,
                        hint="Declare the pure function before calling it.",
                    )
                    return

                expected_params = signature.params
                if len(expr.args) != len(expected_params):
                    self._add_diagnostic(
                        severity="error",
                        code=SEMANTIC_DIAGNOSTIC_CODES["pure_argument_count_mismatch"],
                        message=(
                            f"Pure function '{callee_name}' expects {len(expected_params)} argument(s), "
                            f"but got {len(expr.args)}."
                        ),
                        ctx=ctx,
                        hint="Pass the exact number of arguments declared in the pure function signature.",
                    )
                    return

                for index, (arg_expr, expected) in enumerate(
                    zip(expr.args, expected_params), start=1
                ):
                    actual_type = self._resolve_ast_expr_type(arg_expr)
                    if actual_type != expected.declared_type:
                        resolved_actual = (
                            actual_type if actual_type is not None else "<unresolved>"
                        )
                        self._add_diagnostic(
                            severity="error",
                            code=SEMANTIC_DIAGNOSTIC_CODES[
                                "pure_argument_type_mismatch"
                            ],
                            message=(
                                f"Type mismatch for argument {index} in pure call '{callee_name}': "
                                f"expected {expected.declared_type}, got {resolved_actual}."
                            ),
                            ctx=ctx,
                            hint="Ensure each argument type matches the pure function parameter type.",
                        )

            def _validate_ast_statement(statement: AstStatement):
                stmt_ctx = _stmt_ctx(statement)
                if isinstance(statement, AstVarDeclStmt):
                    declared_type_name = (
                        str(statement.declared_type)
                        if statement.declared_type
                        else None
                    )
                    if declared_type_name is not None:
                        self._validate_declared_type(
                            declared_type_name, stmt_ctx, "LCK310"
                        )

                    initializer_type = None
                    if statement.initializer is not None:
                        _validate_ast_expr(statement.initializer)
                        initializer_type = self._resolve_ast_expr_type(
                            statement.initializer
                        )

                    if declared_type_name is None:
                        if initializer_type is None:
                            self._add_diagnostic(
                                severity="error",
                                code=SEMANTIC_DIAGNOSTIC_CODES["cannot_infer_type"],
                                message=(
                                    f"Cannot infer type for local variable '{statement.name}' without a typed initializer."
                                ),
                                ctx=stmt_ctx,
                                hint="Provide an explicit type or initialize with an expression whose type can be resolved.",
                            )
                            return
                        declared_type_name = initializer_type

                    self._declare(
                        statement.name,
                        declared_type_name,
                        stmt_ctx,
                        duplicate_code="LCK306",
                        kind="local",
                        assigned=statement.initializer is not None,
                    )

                    if (
                        initializer_type is not None
                        and initializer_type != declared_type_name
                    ):
                        self._add_diagnostic(
                            severity="error",
                            code=SEMANTIC_DIAGNOSTIC_CODES[
                                "var_initializer_type_mismatch"
                            ],
                            message=(
                                f"Type mismatch in initializer for '{statement.name}': "
                                f"expected {declared_type_name}, got {initializer_type}."
                            ),
                            ctx=stmt_ctx,
                            hint="Use an initializer expression with the same type as the declared variable.",
                        )
                    return

                if isinstance(statement, AstAssignStmt):
                    lvalue_type = self._resolve_ast_lvalue_type(
                        statement.target, stmt_ctx, for_write=True
                    )
                    _validate_ast_expr(statement.value)
                    expr_type = self._resolve_ast_expr_type(statement.value)
                    if (
                        lvalue_type is not None
                        and expr_type is not None
                        and lvalue_type != expr_type
                    ):
                        self._add_diagnostic(
                            severity="error",
                            code=SEMANTIC_DIAGNOSTIC_CODES["assignment_type_mismatch"],
                            message=(
                                "Type mismatch in assignment: "
                                f"left-hand side expects {lvalue_type}, got {expr_type}."
                            ),
                            ctx=stmt_ctx,
                            hint="Assign expressions whose type matches the lvalue declaration.",
                        )
                    if statement.target:
                        self._set_symbol_assigned(statement.target[0])
                    return

                if isinstance(statement, AstReturnStmt):
                    _validate_ast_expr(statement.value)
                    if self._current_pure_function is None:
                        return
                    expected_type = self._current_pure_function.return_type
                    actual_type = self._resolve_ast_expr_type(statement.value)
                    if actual_type is not None and actual_type != expected_type:
                        self._add_diagnostic(
                            severity="error",
                            code=SEMANTIC_DIAGNOSTIC_CODES["pure_return_type_mismatch"],
                            message=(
                                f"Return type mismatch in pure function '{self._current_pure_function.name}': "
                                f"expected {expected_type}, got {actual_type}."
                            ),
                            ctx=stmt_ctx,
                            hint="Return an expression whose type matches the pure function return type.",
                        )

            def _check_pure_return_shape(pure: AstPureDecl):
                pure_ctx = _stmt_ctx(pure)
                return_statements = [
                    (index, statement)
                    for index, statement in enumerate(pure.body)
                    if isinstance(statement, AstReturnStmt)
                ]
                if not return_statements:
                    self._add_diagnostic(
                        severity="error",
                        code=SEMANTIC_DIAGNOSTIC_CODES["pure_missing_return"],
                        message=f"Pure function '{pure.name}' must include a return statement.",
                        ctx=pure_ctx,
                        hint="Add a return statement that produces a value matching the declared return type.",
                    )
                    return
                if len(return_statements) > 1:
                    self._add_diagnostic(
                        severity="warning",
                        code=SEMANTIC_DIAGNOSTIC_CODES["pure_multiple_returns"],
                        message=(
                            f"Pure function '{pure.name}' contains multiple return statements; "
                            "only the first return is reachable in straight-line semantics."
                        ),
                        ctx=_stmt_ctx(return_statements[1][1]),
                        hint="Keep a single terminal return to avoid dead code and ambiguous intent.",
                    )
                first_return_index = return_statements[0][0]
                for unreachable_stmt in pure.body[first_return_index + 1 :]:
                    self._add_diagnostic(
                        severity="warning",
                        code=SEMANTIC_DIAGNOSTIC_CODES["pure_unreachable_after_return"],
                        message=(
                            f"Unreachable statement in pure function '{pure.name}' after return statement."
                        ),
                        ctx=_stmt_ctx(unreachable_stmt),
                        hint="Remove or move statements before the return.",
                    )

            def _validate_ast_function_body(
                params: tuple[SemanticKernelParam, ...] | list[SemanticKernelParam],
                body: tuple[AstStatement, ...],
                *,
                current_pure_function: SemanticPureFunctionContext | None = None,
            ):
                self._push_scope()
                previous_pure_function = self._current_pure_function
                self._current_pure_function = current_pure_function
                for param in params:
                    self._declare(
                        param.name,
                        param.declared_type,
                        self._ctx_from_location(getattr(param, "location", None)),
                        duplicate_code="LCK306",
                        kind=f"param:{param.modifier}",
                    )
                for statement in body:
                    _validate_ast_statement(statement)
                self._current_pure_function = previous_pure_function
                self._pop_scope()

            def _validate_ast_fold_route(route: AstFoldBindRoute):
                ctx = _stmt_ctx(route)
                declared_type = str(route.uniform_type)
                self._validate_declared_type(
                    declared_type,
                    ctx,
                    SEMANTIC_DIAGNOSTIC_CODES["unknown_declared_type"],
                )
                self._declare(
                    route.uniform_name,
                    declared_type,
                    ctx,
                    duplicate_code=SEMANTIC_DIAGNOSTIC_CODES["duplicate_declaration"],
                    kind="uniform",
                )
                if route.operator not in {"sum", "avg", "min", "max"}:
                    self._add_diagnostic(
                        severity="error",
                        code=SEMANTIC_DIAGNOSTIC_CODES["unknown_fold_operator"],
                        message=f"Unsupported fold operator '{route.operator}'.",
                        ctx=ctx,
                        hint="Use a valid fold operator such as sum, avg, min, or max.",
                    )
                    return

                fold_source_symbol = self._lookup(route.source)
                if fold_source_symbol is None:
                    self._add_diagnostic(
                        severity="error",
                        code=SEMANTIC_DIAGNOSTIC_CODES["fold_unknown_source"],
                        message=f"Fold source accumulator '{route.source}' is undefined.",
                        ctx=ctx,
                        hint="Declare an accumulator and use it as the fold source.",
                    )
                elif fold_source_symbol.kind != "accumulator":
                    self._add_diagnostic(
                        severity="error",
                        code=SEMANTIC_DIAGNOSTIC_CODES["fold_unknown_source"],
                        message=(
                            f"Fold source '{route.source}' must reference an accumulator, "
                            f"got {fold_source_symbol.kind}."
                        ),
                        ctx=ctx,
                        hint="Use an accumulator as the input to fold.",
                    )
                elif fold_source_symbol.declared_type != declared_type:
                    self._add_diagnostic(
                        severity="error",
                        code=SEMANTIC_DIAGNOSTIC_CODES["fold_type_mismatch"],
                        message=(
                            f"Fold target '{route.uniform_name}' has type {declared_type}, but fold source "
                            f"'{route.source}' has accumulator type {fold_source_symbol.declared_type}."
                        ),
                        ctx=ctx,
                        hint="Match the folded uniform type to the accumulator type.",
                    )
                else:
                    self._mark_symbol_used(route.source)
                    if self._pipeline_bind_usage_stack:
                        self._pipeline_bind_usage_stack[-1].add(route.source)

            self._push_scope()
            for struct in program.structs:
                struct_ctx = self._ctx_from_location(struct.location)
                if struct.name in self.structs:
                    self._add_diagnostic(
                        severity="error",
                        code=SEMANTIC_DIAGNOSTIC_CODES["duplicate_declaration"],
                        message=f"Duplicate struct declaration for '{struct.name}'.",
                        ctx=struct_ctx,
                        hint="Rename one struct declaration to keep type names unique.",
                    )
                    continue
                fields = {}
                for field in struct.fields:
                    field_ctx = self._ctx_from_location(field.location)
                    self._validate_declared_type(
                        str(field.declared_type), field_ctx, "LCK310"
                    )
                    if field.name in fields:
                        self._add_diagnostic(
                            severity="error",
                            code=SEMANTIC_DIAGNOSTIC_CODES["duplicate_struct_field"],
                            message=(
                                f"Struct '{struct.name}' has duplicate field declaration '{field.name}'."
                            ),
                            ctx=field_ctx,
                            hint="Rename or remove duplicate struct member declarations.",
                        )
                        continue
                    fields[field.name] = SemanticStructField(
                        name=field.name, declared_type=str(field.declared_type)
                    )
                self.structs[struct.name] = fields

            for collection, target in (
                (program.shaders, self.shaders),
                (program.filters, self.filters),
            ):
                for kernel in collection:
                    kctx = self._ctx_from_location(kernel.location)
                    params = []
                    for param in kernel.params:
                        declared_type = str(param.declared_type)
                        self._validate_declared_type(declared_type, kctx, "LCK310")
                        params.append(
                            SemanticKernelParam(
                                name=param.name,
                                declared_type=declared_type,
                                modifier=param.modifier,
                            )
                        )
                    if kernel.name in target:
                        self._add_diagnostic(
                            severity="error",
                            code=SEMANTIC_DIAGNOSTIC_CODES[
                                "duplicate_kernel_declaration"
                            ],
                            message=f"Duplicate shader/filter declaration for '{kernel.name}'.",
                            ctx=kctx,
                            hint="Rename one declaration to avoid symbol collisions.",
                        )
                    else:
                        target[kernel.name] = params

            for pure in program.pure_functions:
                pctx = self._ctx_from_location(pure.location)
                self._validate_declared_type(str(pure.return_type), pctx, "LCK310")
                pure_params = []
                for param in pure.params:
                    declared_type = str(param.declared_type)
                    self._validate_declared_type(declared_type, pctx, "LCK310")
                    pure_params.append(
                        SemanticKernelParam(
                            name=param.name,
                            declared_type=declared_type,
                            modifier="value",
                        )
                    )
                self.pure_functions[pure.name] = SemanticPureFunctionSignature(
                    return_type=str(pure.return_type),
                    params=tuple(pure_params),
                    intrinsic=pure.intrinsic,
                )

            for kernel in (*program.shaders, *program.filters):
                kernel_params = tuple(
                    SemanticKernelParam(
                        name=param.name,
                        declared_type=str(param.declared_type),
                        modifier=param.modifier,
                    )
                    for param in kernel.params
                )
                _validate_ast_function_body(kernel_params, kernel.body)

            for pure in program.pure_functions:
                if pure.intrinsic:
                    continue
                _check_pure_return_shape(pure)
                pure_params = tuple(
                    SemanticKernelParam(
                        name=param.name,
                        declared_type=str(param.declared_type),
                        modifier="value",
                    )
                    for param in pure.params
                )
                _validate_ast_function_body(
                    pure_params,
                    pure.body,
                    current_pure_function=SemanticPureFunctionContext(
                        name=pure.name,
                        return_type=str(pure.return_type),
                    ),
                )

            for pipeline in program.pipelines:
                self._push_scope()
                self._pipeline_resource_stack.append({})
                self._pipeline_bind_usage_stack.append(set())
                for stream in pipeline.streams:
                    ctx = self._ctx_from_location(stream.location)
                    declared_type = str(stream.declared_type)
                    self._validate_declared_type(declared_type, ctx, "LCK310")
                    self._declare(
                        stream.name,
                        declared_type,
                        ctx,
                        duplicate_code="LCK306",
                        kind="stream",
                    )
                    self._pipeline_resource_stack[-1][stream.name] = (
                        SemanticPipelineResource(kind="stream", declaration_ctx=ctx)
                    )
                for accum in pipeline.accumulators:
                    ctx = self._ctx_from_location(accum.location)
                    declared_type = str(accum.declared_type)
                    self._validate_declared_type(declared_type, ctx, "LCK310")
                    self._declare(
                        accum.name,
                        declared_type,
                        ctx,
                        duplicate_code="LCK306",
                        kind="accumulator",
                    )
                    self._pipeline_resource_stack[-1][accum.name] = (
                        SemanticPipelineResource(
                            kind="accumulator", declaration_ctx=ctx
                        )
                    )
                for uniform in pipeline.uniforms:
                    ctx = self._ctx_from_location(uniform.location)
                    declared_type = str(uniform.declared_type)
                    self._validate_declared_type(declared_type, ctx, "LCK310")
                    self._declare(
                        uniform.name,
                        declared_type,
                        ctx,
                        duplicate_code="LCK306",
                        kind="uniform",
                    )
                    self._pipeline_resource_stack[-1][uniform.name] = (
                        SemanticPipelineResource(kind="uniform", declaration_ctx=ctx)
                    )

                for route in pipeline.bind_routes:
                    if isinstance(route, AstKernelBindRoute):
                        self._type_check_bind_call(
                            _stmt_ctx(route), route.target, route.kernel, route.args
                        )
                    elif isinstance(route, AstFoldBindRoute):
                        _validate_ast_fold_route(route)

                declared_resources = self._pipeline_resource_stack.pop()
                bind_used_resources = self._pipeline_bind_usage_stack.pop()
                for resource_name, resource in declared_resources.items():
                    if resource_name in bind_used_resources:
                        continue
                    self._add_diagnostic(
                        severity="warning",
                        code=SEMANTIC_DIAGNOSTIC_CODES["unbound_pipeline_resource"],
                        message=(
                            f"Pipeline {resource.kind} '{resource_name}' is declared but not used in the bind block."
                        ),
                        ctx=resource.declaration_ctx,
                        hint="Reference every declared stream/accumulator in at least one bind statement.",
                    )
                self._pop_scope()

            self._pop_scope()

        def _resolve_ast_lvalue_type(
            self, path: tuple[str, ...], ctx, *, for_write: bool = False
        ):
            if not path:
                return None

            root_identifier = path[0]
            current_type = self._check_expression_identifier(
                root_identifier,
                ctx,
                require_assigned=not for_write,
            )
            if current_type is None:
                return None

            for field_name in path[1:]:
                struct_fields = self.structs.get(current_type)
                if struct_fields is None:
                    self._add_diagnostic(
                        severity="error",
                        code=SEMANTIC_DIAGNOSTIC_CODES[
                            "invalid_field_access_non_struct"
                        ],
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
                        code=SEMANTIC_DIAGNOSTIC_CODES[
                            "invalid_field_access_unknown_field"
                        ],
                        message=f"Struct '{current_type}' has no field '{field_name}'.",
                        ctx=ctx,
                        hint="Use one of the fields declared on this struct.",
                    )
                    return None
                current_type = struct_fields[field_name].declared_type

            return current_type

        def validate(self, tree):
            typed_ast = getattr(self, "typed_ast", None)
            if not isinstance(typed_ast, AstProgram):
                return self.diagnostics
            self._validate_ast_program(typed_ast)
            return self.diagnostics

    return LockstepSemanticValidator


def validate_semantics(
    parse_tree: Any, visitor_cls, *, typed_ast=None
) -> list[LockstepDiagnostic]:
    validator = build_semantic_validator(visitor_cls)()
    validator.typed_ast = typed_ast
    return validator.validate(parse_tree)
