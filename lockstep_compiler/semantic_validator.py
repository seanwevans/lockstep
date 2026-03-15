from difflib import get_close_matches
from typing import Any

from .prelude import load_intrinsics

from .models import (
    LockstepDiagnostic,
    ParsedTypeArraySuffix,
    ParsedTypeGenericSuffix,
    ParsedTypeName,
    SemanticKernelParam,
    SemanticPipelineResource,
    SemanticPureFunctionContext,
    SemanticPureFunctionSignature,
    SemanticStructField,
    SemanticSymbol,
)
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
            self._primitive_types = {"int", "float", "bool", "string"}
            self._current_pure_function: SemanticPureFunctionContext | None = None
            self._pipeline_resource_stack: list[dict[str, SemanticPipelineResource]] = []
            self._pipeline_bind_usage_stack: list[set[str]] = []

        def _line_col(self, ctx) -> tuple[int, int]:
            token = getattr(ctx, "start", None)
            return (getattr(token, "line", 0), getattr(token, "column", 0))

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

        def _consume_identifier(
            self, type_name: str, index: int
        ) -> tuple[str | None, int]:
            if index >= len(type_name) or not (
                type_name[index].isalpha() or type_name[index] == "_"
            ):
                return None, index

            start = index
            index += 1
            while index < len(type_name) and (
                type_name[index].isalnum() or type_name[index] == "_"
            ):
                index += 1
            return type_name[start:index], index

        def _consume_digits(self, type_name: str, index: int) -> tuple[str | None, int]:
            if index >= len(type_name) or not type_name[index].isdigit():
                return None, index

            start = index
            index += 1
            while index < len(type_name) and type_name[index].isdigit():
                index += 1
            return type_name[start:index], index

        def _parse_type_name(
            self, type_name: str, index: int = 0
        ) -> tuple[ParsedTypeName | None, int]:
            base_name, index = self._consume_identifier(type_name, index)
            if base_name is None:
                return None, index

            suffixes: list[ParsedTypeArraySuffix | ParsedTypeGenericSuffix] = []
            while index < len(type_name):
                if type_name[index] == "[":
                    index += 1
                    length, index = self._consume_digits(type_name, index)
                    if (
                        length is None
                        or index >= len(type_name)
                        or type_name[index] != "]"
                    ):
                        return None, index
                    suffixes.append(ParsedTypeArraySuffix(size=int(length)))
                    index += 1
                    continue

                if type_name[index] == "<":
                    index += 1
                    inner_type, index = self._parse_type_name(type_name, index)
                    if inner_type is None:
                        return None, index

                    arity = None
                    if index < len(type_name) and type_name[index] == ",":
                        index += 1
                        digits, index = self._consume_digits(type_name, index)
                        if digits is None:
                            return None, index
                        arity = int(digits)

                    if index >= len(type_name) or type_name[index] != ">":
                        return None, index
                    suffixes.append(
                        ParsedTypeGenericSuffix(type_name=inner_type, arity=arity)
                    )
                    index += 1
                    continue

                break

            return ParsedTypeName(base=base_name, inner=tuple(suffixes)), index

        def _collect_referenced_type_names(
            self, parsed_type: ParsedTypeName
        ) -> list[str]:
            has_generic_suffix = any(
                isinstance(suffix, ParsedTypeGenericSuffix)
                for suffix in parsed_type.inner
            )
            referenced_names = [] if has_generic_suffix else [parsed_type.base]
            for suffix in parsed_type.inner:
                if isinstance(suffix, ParsedTypeGenericSuffix):
                    referenced_names.extend(
                        self._collect_referenced_type_names(suffix.type_name)
                    )
            return referenced_names

        def _validate_declared_type(self, type_name: str, ctx, code: str) -> bool:
            parsed_type, index = self._parse_type_name(type_name)
            if parsed_type is None or index != len(type_name):
                self._add_diagnostic(
                    severity="error",
                    code=code,
                    message=f"Invalid declared type syntax '{type_name}'.",
                    ctx=ctx,
                    hint="Use forms like T, T[4], Ctor<T>, or Ctor<T,4>.",
                )
                return False

            known_types = self._known_types()
            referenced_types = self._collect_referenced_type_names(parsed_type)
            all_known = True
            for referenced_type in referenced_types:
                if referenced_type in known_types:
                    continue

                suggestions = get_close_matches(
                    referenced_type, sorted(known_types), n=2, cutoff=0.6
                )
                hint = (
                    f"Unknown type '{referenced_type}'. Use a primitive ({', '.join(sorted(self._primitive_types))}) "
                    "or declare a struct with this name before using it."
                )
                if suggestions:
                    hint = f"Did you mean {', '.join(suggestions)}? {hint}"

                self._add_diagnostic(
                    severity="error",
                    code=code,
                    message=f"Unknown declared type '{referenced_type}' in '{type_name}'.",
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

        def _resolve_expr_type(self, ctx):
            if ctx is None:
                return None

            def _as_list(value):
                if value is None:
                    return []
                return value if isinstance(value, list) else [value]

            def _is_cast_function_name(name: str) -> bool:
                return name in {"int", "float", "double", "uint", "bool"}

            def _child_text(index: int) -> str | None:
                if not hasattr(ctx, "getChild"):
                    return None
                try:
                    child = ctx.getChild(index)
                except Exception:
                    return None
                return (
                    child.getText()
                    if child is not None and hasattr(child, "getText")
                    else None
                )

            def _report_operand_error(
                operator: str, expected: str, actual_types: list[str | None]
            ):
                rendered_types = ", ".join(
                    type_name if type_name is not None else "<unresolved>"
                    for type_name in actual_types
                )
                self._add_diagnostic(
                    severity="error",
                    code=SEMANTIC_DIAGNOSTIC_CODES["invalid_operand_types"],
                    message=(
                        f"Operator '{operator}' expects {expected} operand type(s), "
                        f"but got [{rendered_types}]."
                    ),
                    ctx=ctx,
                    hint="Adjust operand types so they match the operator semantics.",
                )

            def _resolve_numeric_sequence(operator: str, operand_contexts: list[Any]):
                operand_types = [
                    self._resolve_expr_type(operand_ctx)
                    for operand_ctx in operand_contexts
                ]
                known = [
                    operand_type
                    for operand_type in operand_types
                    if operand_type is not None
                ]
                if any(operand_type not in {"int", "float"} for operand_type in known):
                    _report_operand_error(operator, "numeric", operand_types)
                    return None
                if len(known) != len(operand_contexts):
                    return None
                if "int" in known and "float" in known:
                    self._add_diagnostic(
                        severity="error",
                        code=SEMANTIC_DIAGNOSTIC_CODES["implicit_numeric_widening"],
                        message=(
                            f"Operator '{operator}' mixes int and float operands without an explicit cast."
                        ),
                        ctx=ctx,
                        hint="Use explicit casts so numeric widening is intentional and target-compatible.",
                    )
                    return None
                return known[0] if known else None

            def _resolve_boolean_sequence(operator: str, operand_contexts: list[Any]):
                operand_types = [
                    self._resolve_expr_type(operand_ctx)
                    for operand_ctx in operand_contexts
                ]
                known = [
                    operand_type
                    for operand_type in operand_types
                    if operand_type is not None
                ]
                if any(operand_type != "bool" for operand_type in known):
                    _report_operand_error(operator, "bool", operand_types)
                    return None
                if len(known) != len(operand_contexts):
                    return None
                return "bool"

            def _resolve_int_sequence(operator: str, operand_contexts: list[Any]):
                operand_types = [
                    self._resolve_expr_type(operand_ctx)
                    for operand_ctx in operand_contexts
                ]
                known = [
                    operand_type
                    for operand_type in operand_types
                    if operand_type is not None
                ]
                if any(operand_type != "int" for operand_type in known):
                    _report_operand_error(operator, "int", operand_types)
                    return None
                if len(known) != len(operand_contexts):
                    return None
                return "int"

            if hasattr(ctx, "logicalAndExpr") and callable(ctx.logicalAndExpr):
                operands = _as_list(ctx.logicalAndExpr())
                if len(operands) > 1:
                    return _resolve_boolean_sequence("||", operands)
                if len(operands) == 1:
                    return self._resolve_expr_type(operands[0])

            if hasattr(ctx, "bitwiseOrExpr") and callable(ctx.bitwiseOrExpr):
                operands = _as_list(ctx.bitwiseOrExpr())
                if len(operands) > 1:
                    return _resolve_boolean_sequence("&&", operands)
                if len(operands) == 1:
                    return self._resolve_expr_type(operands[0])

            if hasattr(ctx, "bitwiseXorExpr") and callable(ctx.bitwiseXorExpr):
                operands = _as_list(ctx.bitwiseXorExpr())
                if len(operands) > 1:
                    return _resolve_int_sequence("|", operands)
                if len(operands) == 1:
                    return self._resolve_expr_type(operands[0])

            if hasattr(ctx, "bitwiseAndExpr") and callable(ctx.bitwiseAndExpr):
                operands = _as_list(ctx.bitwiseAndExpr())
                if len(operands) > 1:
                    return _resolve_int_sequence("^", operands)
                if len(operands) == 1:
                    return self._resolve_expr_type(operands[0])

            if hasattr(ctx, "equalityExpr") and callable(ctx.equalityExpr):
                operands = _as_list(ctx.equalityExpr())
                if len(operands) > 1:
                    return _resolve_int_sequence("&", operands)
                if len(operands) == 1:
                    return self._resolve_expr_type(operands[0])

            if hasattr(ctx, "relExpr") and callable(ctx.relExpr):
                operands = _as_list(ctx.relExpr())
                if len(operands) > 1:
                    operand_types = [
                        self._resolve_expr_type(operand_ctx) for operand_ctx in operands
                    ]
                    known = [
                        operand_type
                        for operand_type in operand_types
                        if operand_type is not None
                    ]
                    if any(
                        operand_type not in {"int", "float", "bool"}
                        for operand_type in known
                    ):
                        operator = _child_text(1) or "=="
                        _report_operand_error(operator, "comparable", operand_types)
                        return None
                    if len(set(known)) > 1:
                        operator = _child_text(1) or "=="
                        _report_operand_error(operator, "matching", operand_types)
                        return None
                    if len(known) != len(operands):
                        return None
                    return "bool"
                if len(operands) == 1:
                    return self._resolve_expr_type(operands[0])

            if hasattr(ctx, "shiftExpr") and callable(ctx.shiftExpr):
                operands = _as_list(ctx.shiftExpr())
                if len(operands) > 1:
                    operator = _child_text(1) or "<"
                    operand_types = [
                        self._resolve_expr_type(operand_ctx) for operand_ctx in operands
                    ]
                    known = [
                        operand_type
                        for operand_type in operand_types
                        if operand_type is not None
                    ]
                    if any(
                        operand_type not in {"int", "float", "bool"}
                        for operand_type in known
                    ):
                        _report_operand_error(operator, "comparable", operand_types)
                        return None
                    if len(set(known)) > 1:
                        _report_operand_error(operator, "matching", operand_types)
                        return None
                    if len(known) != len(operands):
                        return None
                    return "bool"
                if len(operands) == 1:
                    return self._resolve_expr_type(operands[0])

            if hasattr(ctx, "addExpr") and callable(ctx.addExpr):
                operands = _as_list(ctx.addExpr())
                if len(operands) > 1:
                    return _resolve_int_sequence(_child_text(1) or "<<", operands)
                if len(operands) == 1:
                    return self._resolve_expr_type(operands[0])

            if hasattr(ctx, "mulExpr") and callable(ctx.mulExpr):
                operands = _as_list(ctx.mulExpr())
                if len(operands) > 1:
                    operator = _child_text(1) or "+"
                    return _resolve_numeric_sequence(operator, operands)
                if len(operands) == 1:
                    return self._resolve_expr_type(operands[0])

            if (
                hasattr(ctx, "unaryExpr")
                and callable(ctx.unaryExpr)
                and ctx.unaryExpr() is not None
            ):
                operator = _child_text(0) or ""
                operand_type = self._resolve_expr_type(ctx.unaryExpr())
                type_name_ctx = (
                    ctx.typeName()
                    if hasattr(ctx, "typeName") and callable(ctx.typeName)
                    else None
                )
                if type_name_ctx is not None and _child_text(0) == "(":
                    cast_target = type_name_ctx.getText()
                    self._validate_declared_type(
                        cast_target,
                        type_name_ctx,
                        SEMANTIC_DIAGNOSTIC_CODES["unknown_declared_type"],
                    )
                    return cast_target
                if operator == "-":
                    if operand_type is not None and operand_type not in {
                        "int",
                        "float",
                    }:
                        _report_operand_error(operator, "numeric", [operand_type])
                        return None
                    return operand_type
                if operator == "!":
                    if operand_type is not None and operand_type != "bool":
                        _report_operand_error(operator, "bool", [operand_type])
                        return None
                    return "bool" if operand_type is not None else None
                return operand_type

            if hasattr(ctx, "declared_type"):
                return ctx.declared_type

            if hasattr(ctx, "INT") and callable(ctx.INT) and ctx.INT() is not None:
                return "int"
            if (
                hasattr(ctx, "FLOAT")
                and callable(ctx.FLOAT)
                and ctx.FLOAT() is not None
            ):
                return "float"
            if hasattr(ctx, "BOOL") and callable(ctx.BOOL) and ctx.BOOL() is not None:
                return "bool"
            if (
                hasattr(ctx, "STRING")
                and callable(ctx.STRING)
                and ctx.STRING() is not None
            ):
                return "string"

            if (
                hasattr(ctx, "lvalue")
                and callable(ctx.lvalue)
                and ctx.lvalue() is not None
            ):
                return self._resolve_lvalue_type(ctx.lvalue())

            if hasattr(ctx, "ID") and callable(ctx.ID) and ctx.ID() is not None:
                if (
                    hasattr(ctx, "exprList")
                    and callable(ctx.exprList)
                    and ctx.exprList() is not None
                ):
                    function_name = ctx.ID().getText()
                    args = ctx.exprList().expr() if ctx.exprList() is not None else []
                    if _is_cast_function_name(function_name):
                        if len(args) != 1:
                            self._add_diagnostic(
                                severity="error",
                                code=SEMANTIC_DIAGNOSTIC_CODES[
                                    "pure_argument_count_mismatch"
                                ],
                                message=(
                                    f"Cast '{function_name}(...)' expects 1 argument, "
                                    f"but got {len(args)}."
                                ),
                                ctx=ctx,
                                hint="Pass exactly one expression to a cast.",
                            )
                            return None
                        return function_name
                    function_signature = self.pure_functions.get(function_name)
                    if function_signature is not None:
                        return function_signature.return_type
                    return None
                return self._check_expression_identifier(ctx.ID().getText(), ctx)

            if (
                hasattr(ctx, "primaryExpr")
                and callable(ctx.primaryExpr)
                and ctx.primaryExpr() is not None
            ):
                return self._resolve_expr_type(ctx.primaryExpr())

            if hasattr(ctx, "expr") and callable(ctx.expr):
                child_expr = ctx.expr()
                if isinstance(child_expr, list):
                    if len(child_expr) == 1:
                        return self._resolve_expr_type(child_expr[0])
                    return None
                if child_expr is not None:
                    return self._resolve_expr_type(child_expr)

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
                self._pipeline_resource_stack[-1][ctx.ID().getText()] = SemanticPipelineResource(
                    kind="stream",
                    declaration_ctx=ctx,
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
                self._pipeline_resource_stack[-1][ctx.ID().getText()] = SemanticPipelineResource(
                    kind="accumulator",
                    declaration_ctx=ctx,
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
                self._pipeline_resource_stack[-1][ctx.ID().getText()] = SemanticPipelineResource(
                    kind="uniform",
                    declaration_ctx=ctx,
                )
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
                    else:
                        self._type_check_pure_call(ctx)
                return self.visitChildren(ctx)
            return self.visitChildren(ctx)

        def visitLvalue(self, ctx):
            self._resolve_lvalue_type(ctx)
            return self.visitChildren(ctx)

        def validate(self, tree):
            self.visit(tree)
            return self.diagnostics

    return LockstepSemanticValidator


def validate_semantics(
    parse_tree: Any, visitor_cls, *, typed_ast=None
) -> list[LockstepDiagnostic]:
    validator = build_semantic_validator(visitor_cls)()
    return validator.validate(parse_tree)
