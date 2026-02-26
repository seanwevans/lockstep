import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from antlr4 import InputStream, CommonTokenStream
from antlr4.error.ErrorListener import ErrorListener

PARSER_DIR = Path(__file__).parent / "generated" / "parser"
if str(PARSER_DIR) not in sys.path:
    sys.path.insert(0, str(PARSER_DIR))

from LockstepLexer import LockstepLexer
from LockstepParser import LockstepParser
from LockstepVisitor import LockstepVisitor


@dataclass
class LockstepDiagnostic:
    severity: str
    code: str
    message: str
    line: int
    column: int
    hint: str | None = None


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


@dataclass
class LockstepCompileResult:
    parse_tree: Any
    entities: dict[str, Any]
    diagnostics: list[LockstepDiagnostic] = field(default_factory=list)


class ParseErrorCollector(ErrorListener):
    """Collects syntax errors emitted by ANTLR during lex/parse."""

    def __init__(self):
        super().__init__()
        self.errors: list[LockstepDiagnostic] = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.errors.append(
            LockstepDiagnostic(
                severity="error",
                code="LCK001",
                message=msg,
                line=line,
                column=column,
                hint="Fix syntax errors before semantic analysis can continue.",
            )
        )


class LockstepCompileError(Exception):
    """Raised when the Lockstep source contains parse errors."""

    def __init__(self, errors, diagnostics=None, *, phase: str = "parse"):
        self.errors = errors
        self.diagnostics = diagnostics or []
        self.phase = phase
        super().__init__(self._format_message())

    def _format_message(self):
        count = len(self.errors)
        suffix = "" if count == 1 else "s"
        summary = f"Compilation failed with {count} {self.phase} error{suffix}."
        details = "\n".join(
            f"line {error.line}:{error.column} {error.message}" for error in self.errors
        )
        return summary if not details else f"{summary}\n{details}"


class LockstepSemanticValidator(LockstepVisitor):
    """Runs semantic checks on a parsed Lockstep program."""

    def __init__(self):
        self.diagnostics: list[LockstepDiagnostic] = []
        self.scopes: list[dict[str, dict[str, str]]] = []
        self.shaders: dict[str, list[dict[str, str]]] = {}
        self.filters: dict[str, list[dict[str, str]]] = {}
        self.current_shader_name: str | None = None

    def _line_col(self, ctx) -> tuple[int, int]:
        token = getattr(ctx, "start", None)
        return (
            getattr(token, "line", 0),
            getattr(token, "column", 0),
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
            LockstepDiagnostic(
                severity=severity,
                code=code,
                message=message,
                line=line,
                column=column,
                hint=hint,
            )
        )

    def _push_scope(self):
        self.scopes.append({})

    def _pop_scope(self):
        if self.scopes:
            self.scopes.pop()

    def _declare(
        self,
        name: str,
        declared_type: str,
        ctx,
        *,
        duplicate_code: str,
        kind: str = "symbol",
    ):
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
        current_scope[name] = {"type": declared_type, "kind": kind}
        return True

    def _lookup(self, name: str) -> dict[str, str] | None:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def _declared_in_current_scope(self, name: str) -> bool:
        return bool(self.scopes and name in self.scopes[-1])

    def _record_kernel_signature(self, ctx, target: dict[str, list[dict[str, str]]]):
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
                    {
                        "name": param.ID().getText(),
                        "type": param.typeName().getText(),
                        "modifier": param.getChild(0).getText(),
                    }
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
        return symbol["type"] if symbol else None

    def _type_check_bind_call(self, ctx, target_name: str, callee_name: str, arg_names):
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
            actual_type = actual_symbol["type"]
            if actual_type != expected["type"]:
                self._add_diagnostic(
                    severity="error",
                    code="LCK305",
                    message=(
                        f"Type mismatch for argument '{arg_name}' in '{callee_name}': "
                        f"expected {expected['type']}, got {actual_type}."
                    ),
                    ctx=ctx,
                    hint="Align argument types with the shader/filter signature.",
                )

    def visitProgram(self, ctx: LockstepParser.ProgramContext):
        self._push_scope()
        result = self.visitChildren(ctx)
        self._pop_scope()
        return result

    def visitShaderDecl(self, ctx: LockstepParser.ShaderDeclContext):
        name, params = self._record_kernel_signature(ctx, self.shaders)
        self.current_shader_name = name
        self._push_scope()
        for param in params:
            self._declare(
                param["name"],
                param["type"],
                ctx,
                duplicate_code="LCK306",
                kind=f"param:{param['modifier']}",
            )
        result = self.visitChildren(ctx)
        self._pop_scope()
        self.current_shader_name = None
        return result

    def visitFilterDecl(self, ctx: LockstepParser.FilterDeclContext):
        _name, params = self._record_kernel_signature(ctx, self.filters)
        self._push_scope()
        for param in params:
            self._declare(
                param["name"],
                param["type"],
                ctx,
                duplicate_code="LCK306",
                kind=f"param:{param['modifier']}",
            )
        result = self.visitChildren(ctx)
        self._pop_scope()
        return result

    def visitVarDecl(self, ctx: LockstepParser.VarDeclContext):
        self._declare(
            ctx.ID().getText(),
            ctx.typeName().getText(),
            ctx,
            duplicate_code="LCK306",
            kind="local",
        )
        return self.visitChildren(ctx)

    def visitPipelineDecl(self, ctx: LockstepParser.PipelineDeclContext):
        self._push_scope()
        result = self.visitChildren(ctx)
        self._pop_scope()
        return result

    def visitStreamDecl(self, ctx: LockstepParser.StreamDeclContext):
        self._declare(
            ctx.ID().getText(),
            ctx.typeName().getText(),
            ctx,
            duplicate_code="LCK306",
            kind="stream",
        )
        return self.visitChildren(ctx)

    def visitAccumDecl(self, ctx: LockstepParser.AccumDeclContext):
        self._declare(
            ctx.ID().getText(),
            ctx.typeName().getText(),
            ctx,
            duplicate_code="LCK306",
            kind="accumulator",
        )
        return self.visitChildren(ctx)

    def visitUniformDecl(self, ctx: LockstepParser.UniformDeclContext):
        self._declare(
            ctx.ID().getText(),
            ctx.typeName().getText(),
            ctx,
            duplicate_code="LCK306",
            kind="uniform",
        )
        return self.visitChildren(ctx)

    def visitBindStmt(self, ctx: LockstepParser.BindStmtContext):
        id_tokens = ctx.ID()
        if ctx.argList() is not None:
            target_name = id_tokens[0].getText()
            callee_name = id_tokens[1].getText()
            arg_names = [token.getText() for token in id_tokens[2:]]
            self._type_check_bind_call(ctx, target_name, callee_name, arg_names)
            return self.visitChildren(ctx)

        fold_target = id_tokens[0].getText()
        fold_operator = id_tokens[1].getText()
        fold_source = id_tokens[2].getText()

        if self._declared_in_current_scope(fold_target):
            self._add_diagnostic(
                severity="error",
                code="LCK306",
                message=f"Duplicate declaration for '{fold_target}' in the same scope.",
                ctx=ctx,
                hint="Rename one declaration or move it to a different scope.",
            )
        else:
            self.scopes[-1][fold_target] = {"type": ctx.typeName().getText(), "kind": "uniform"}

        fold_source_symbol = self._lookup(fold_source)
        declared_type = ctx.typeName().getText()
        if fold_source_symbol is None:
            self._add_diagnostic(
                severity="error",
                code="LCK401",
                message=f"Fold source accumulator '{fold_source}' is undefined.",
                ctx=ctx,
                hint="Declare an accumulator and use it as the fold source.",
            )
        elif fold_source_symbol["kind"] != "accumulator":
            self._add_diagnostic(
                severity="error",
                code="LCK403",
                message=(
                    f"Fold source '{fold_source}' must reference an accumulator, "
                    f"got {fold_source_symbol['kind']}."
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
        elif fold_source_symbol["type"] != declared_type:
            self._add_diagnostic(
                severity="error",
                code="LCK404",
                message=(
                    f"Fold target '{fold_target}' has type {declared_type}, but fold source "
                    f"'{fold_source}' has accumulator type {fold_source_symbol['type']}."
                ),
                ctx=ctx,
                hint="Match the folded uniform type to the accumulator type.",
            )

        return self.visitChildren(ctx)

    def visitPrimaryExpr(self, ctx: LockstepParser.PrimaryExprContext):
        if ctx.ID():
            if ctx.exprList() is None:
                self._check_expression_identifier(ctx.ID().getText(), ctx)
            return self.visitChildren(ctx)

        if ctx.lvalue():
            root_identifier = ctx.lvalue().ID(0).getText()
            self._check_expression_identifier(root_identifier, ctx)

        return self.visitChildren(ctx)

    def visitLvalue(self, ctx: LockstepParser.LvalueContext):
        root_identifier = ctx.ID(0).getText()
        self._check_expression_identifier(root_identifier, ctx)
        return self.visitChildren(ctx)

    def validate(self, tree):
        self.visit(tree)
        return self.diagnostics


def validate_semantics(parse_tree: Any) -> list[LockstepDiagnostic]:
    """Validate semantic constraints after syntactic parsing succeeds."""

    validator = LockstepSemanticValidator()
    return validator.validate(parse_tree)


def compile_lockstep(source_code: str, verbose: bool = True) -> LockstepCompileResult:
    input_stream = InputStream(source_code)
    lexer = LockstepLexer(input_stream)
    error_listener = ParseErrorCollector()
    lexer.removeErrorListeners()
    lexer.addErrorListener(error_listener)
    stream = CommonTokenStream(lexer)

    parser = LockstepParser(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(error_listener)
    tree = parser.program()

    if error_listener.errors:
        raise LockstepCompileError(error_listener.errors, diagnostics=error_listener.errors)

    semantic_diagnostics = validate_semantics(tree)
    semantic_errors = [
        diagnostic
        for diagnostic in semantic_diagnostics
        if diagnostic.severity == "error"
    ]
    if semantic_errors:
        raise LockstepCompileError(
            semantic_errors,
            diagnostics=semantic_diagnostics,
            phase="semantic",
        )

    visitor = LockstepDebugVisitor(verbose=verbose)
    visitor.visit(tree)
    return LockstepCompileResult(
        parse_tree=tree,
        entities={
            "structs": visitor.structs,
            "shaders": visitor.shaders,
            "filters": visitor.filters,
            "pure_functions": visitor.pure_functions,
            "streams": visitor.streams,
            "accumulators": visitor.accumulators,
            "uniforms": visitor.uniforms,
            "bind_routes": visitor.bind_routes,
        },
        diagnostics=[*semantic_diagnostics, *visitor.diagnostics],
    )


TEST_SOURCE = """
struct Vec3 { float x; float y; float z; };

pure Vec3 add(Vec3 a, Vec3 b) {
    Vec3 r; 
    r.x = a.x + b.x; 
    return r;
}

shader ApplyGravity(in Vec3 pos, out Vec3 new_pos, accum float energy, uniform float dt) {
    new_pos.x = pos.x;
    new_pos.y = pos.y - (9.8 * dt);
    energy = new_pos.y; 
}

pipeline Physics {
    stream<Vec3, 1000> raw_positions;
    stream<Vec3, 1000> final_positions;
    accumulator<float> total_energy;
    uniform float dt = 0.016;

    bind {
        final_positions = ApplyGravity(raw_positions, final_positions, total_energy, dt);
        uniform float sys_energy = fold sum(total_energy);
    }
}
"""


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Debug parser for Lockstep source files."
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Optional path to a Lockstep source file. Reads from stdin when omitted.",
    )
    return parser


def run_cli(argv=None, *, stdin=None, stderr=None, compiler=compile_lockstep):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    stdin = sys.stdin if stdin is None else stdin
    stderr = sys.stderr if stderr is None else stderr

    if args.path:
        source_path = Path(args.path)
        try:
            source = source_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"Unable to read '{source_path}': file not found.", file=stderr)
            return 1
        except PermissionError as err:
            reason = err.strerror or "permission denied"
            print(f"Unable to read '{source_path}': {reason}.", file=stderr)
            return 1
        except UnicodeDecodeError as err:
            print(
                f"Unable to read '{source_path}': invalid UTF-8 ({err.reason}).",
                file=stderr,
            )
            return 1
    else:
        source = stdin.read()

    try:
        compiler(source)
    except LockstepCompileError as err:
        count = len(err.errors)
        suffix = "" if count == 1 else "s"
        print(
            f"Compilation failed with {count} {err.phase} error{suffix}.",
            file=stderr,
        )
        for error in err.errors:
            print(f"line {error.line}:{error.column} {error.message}", file=stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
