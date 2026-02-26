from LockstepParser import LockstepParser
from LockstepVisitor import LockstepVisitor

from .diagnostics import LockstepDiagnostic


class LockstepDebugVisitor(LockstepVisitor):
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
        self._seen_streams = set()
        self._seen_accumulators = set()
        self._seen_uniforms = set()

    def _print(self, message: str):
        if self.verbose:
            print(message)

    def _line_col(self, ctx) -> tuple[int, int]:
        token = getattr(ctx, "start", None)
        return (
            getattr(token, "line", 0),
            getattr(token, "column", 0),
        )

    def visitProgram(self, ctx: LockstepParser.ProgramContext):
        self._print("=== LOCKSTEP COMPILER FRONTEND ===")
        self._print("Parsing program...\n")
        return self.visitChildren(ctx)

    def visitStructDecl(self, ctx: LockstepParser.StructDeclContext):
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

    def visitPureDecl(self, ctx: LockstepParser.PureDeclContext):
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

    def visitFilterDecl(self, ctx: LockstepParser.FilterDeclContext):
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

    def visitShaderDecl(self, ctx: LockstepParser.ShaderDeclContext):
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

    def visitPipelineDecl(self, ctx: LockstepParser.PipelineDeclContext):
        name = ctx.ID().getText()
        self._print(f"\n[Pipeline Topology] {name}")
        return self.visitChildren(ctx)

    def visitStreamDecl(self, ctx: LockstepParser.StreamDeclContext):
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

    def visitAccumDecl(self, ctx: LockstepParser.AccumDeclContext):
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

    def visitUniformDecl(self, ctx: LockstepParser.UniformDeclContext):
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

    def visitBindBlock(self, ctx: LockstepParser.BindBlockContext):
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
