from typing import Any

from .models import LockstepDiagnostic


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
            self.bind_routes_ir = []
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
            params = []
            if hasattr(ctx, "pureParamList") and callable(ctx.pureParamList) and ctx.pureParamList():
                ids = ctx.pureParamList().ID()
                types = ctx.pureParamList().typeName()
                for param_type, param_name in zip(types, ids):
                    params.append({"type": param_type.getText(), "name": param_name.getText()})
            statements = []
            if hasattr(ctx, "statement") and callable(ctx.statement):
                statements = [statement.getText() for statement in ctx.statement()]
            self.pure_functions.append({"name": name, "return_type": ret_type, "params": params, "body": statements})
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
            statements = []
            if hasattr(ctx, "statement") and callable(ctx.statement):
                statements = [statement.getText() for statement in ctx.statement()]
            self.filters.append({"name": name, "params": params, "body": statements})
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
            statements = []
            if hasattr(ctx, "statement") and callable(ctx.statement):
                statements = [statement.getText() for statement in ctx.statement()]
            self.shaders.append({"name": name, "params": params, "body": statements})
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
                self.bind_routes_ir.append(self._build_bind_route_ir(stmt, route))
                self._print(f"       {route}")
            return self.visitChildren(ctx)

        def _build_bind_route_ir(self, stmt_ctx, route_text: str) -> dict[str, Any]:
            id_tokens_getter = getattr(stmt_ctx, "ID", None)
            id_tokens = id_tokens_getter() if callable(id_tokens_getter) else []
            fold_operator_getter = getattr(stmt_ctx, "foldOperator", None)
            fold_operator = fold_operator_getter() if callable(fold_operator_getter) else None
            if fold_operator is not None and len(id_tokens) >= 2:
                return {
                    "kind": "fold",
                    "uniform_type": stmt_ctx.typeName().getText() if getattr(stmt_ctx, "typeName", None) else "",
                    "uniform_name": id_tokens[0].getText(),
                    "operator": fold_operator.getText(),
                    "source": id_tokens[1].getText(),
                    "route": route_text,
                }

            if len(id_tokens) >= 2:
                args = [token.getText() for token in id_tokens[2:]]
                return {
                    "kind": "kernel",
                    "target": id_tokens[0].getText(),
                    "kernel": id_tokens[1].getText(),
                    "args": args,
                    "route": route_text,
                }

            return {"kind": "unknown", "route": route_text}

    return LockstepDebugVisitor
