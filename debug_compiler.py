import sys
from antlr4 import InputStream, CommonTokenStream

# antlr4 -Dlanguage=Python3 -visitor Lockstep.g4
from LockstepLexer import LockstepLexer
from LockstepParser import LockstepParser
from LockstepVisitor import LockstepVisitor


class LockstepDebugVisitor(LockstepVisitor):
    """Walks the Parse Tree and extracts the pipeline architecture."""

    def visitProgram(self, ctx: LockstepParser.ProgramContext):
        print("=== LOCKSTEP COMPILER FRONTEND ===")
        print("Parsing program...\n")
        return self.visitChildren(ctx)

    def visitStructDecl(self, ctx: LockstepParser.StructDeclContext):
        name = ctx.ID().getText()
        print(f"[Struct] Discovered: {name}")
        return self.visitChildren(ctx)

    def visitPureDecl(self, ctx: LockstepParser.PureDeclContext):
        name = ctx.ID().getText()
        ret_type = ctx.typeName().getText()
        print(f"[Pure Function] {name} -> {ret_type}")
        return self.visitChildren(ctx)

    def visitShaderDecl(self, ctx: LockstepParser.ShaderDeclContext):
        name = ctx.ID().getText()
        print(f"\n[Shader Kernel] {name}")
        if ctx.paramList():
            for param in ctx.paramList().param():
                modifier = param.getChild(0).getText()
                p_type = param.typeName().getText()
                p_name = param.ID().getText()
                print(f"  └─ Param: ({modifier}) {p_type} {p_name}")
        return self.visitChildren(ctx)

    def visitPipelineDecl(self, ctx: LockstepParser.PipelineDeclContext):
        name = ctx.ID().getText()
        print(f"\n[Pipeline Topology] {name}")
        return self.visitChildren(ctx)

    def visitStreamDecl(self, ctx: LockstepParser.StreamDeclContext):
        s_type = ctx.typeName().getText()
        capacity = ctx.INT().getText()
        name = ctx.ID().getText()
        print(f"  └─ Stream: {name} <{s_type}, {capacity}>")
        return self.visitChildren(ctx)

    def visitAccumDecl(self, ctx: LockstepParser.AccumDeclContext):
        a_type = ctx.typeName().getText()
        name = ctx.ID().getText()
        print(f"  └─ Accumulator: {name} <{a_type}>")
        return self.visitChildren(ctx)

    def visitBindBlock(self, ctx: LockstepParser.BindBlockContext):
        print("  └─ Routing:")
        for stmt in ctx.bindStmt():
            print(f"       {stmt.getText()}")
        return self.visitChildren(ctx)


def compile_lockstep(source_code: str):
    input_stream = InputStream(source_code)
    lexer = LockstepLexer(input_stream)
    stream = CommonTokenStream(lexer)

    parser = LockstepParser(stream)
    tree = parser.program()

    visitor = LockstepDebugVisitor()
    visitor.visit(tree)


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

if __name__ == "__main__":
    compile_lockstep(TEST_SOURCE)
