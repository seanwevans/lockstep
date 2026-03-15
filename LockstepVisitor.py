# Generated from Lockstep.g4 by ANTLR 4.7.2
from antlr4 import *

if __name__ is not None and "." in __name__:
    from .LockstepParser import LockstepParser
else:
    from LockstepParser import LockstepParser

# This class defines a complete generic visitor for a parse tree produced by LockstepParser.


class LockstepVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by LockstepParser#program.
    def visitProgram(self, ctx: LockstepParser.ProgramContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#declaration.
    def visitDeclaration(self, ctx: LockstepParser.DeclarationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#structDecl.
    def visitStructDecl(self, ctx: LockstepParser.StructDeclContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#structMember.
    def visitStructMember(self, ctx: LockstepParser.StructMemberContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#pureDecl.
    def visitPureDecl(self, ctx: LockstepParser.PureDeclContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#pureParamList.
    def visitPureParamList(self, ctx: LockstepParser.PureParamListContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#shaderDecl.
    def visitShaderDecl(self, ctx: LockstepParser.ShaderDeclContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#filterDecl.
    def visitFilterDecl(self, ctx: LockstepParser.FilterDeclContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#paramList.
    def visitParamList(self, ctx: LockstepParser.ParamListContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#param.
    def visitParam(self, ctx: LockstepParser.ParamContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#pipelineDecl.
    def visitPipelineDecl(self, ctx: LockstepParser.PipelineDeclContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#pipelineMember.
    def visitPipelineMember(self, ctx: LockstepParser.PipelineMemberContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#streamDecl.
    def visitStreamDecl(self, ctx: LockstepParser.StreamDeclContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#accumDecl.
    def visitAccumDecl(self, ctx: LockstepParser.AccumDeclContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#uniformDecl.
    def visitUniformDecl(self, ctx: LockstepParser.UniformDeclContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#bindBlock.
    def visitBindBlock(self, ctx: LockstepParser.BindBlockContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#bindStmt.
    def visitBindStmt(self, ctx: LockstepParser.BindStmtContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#argList.
    def visitArgList(self, ctx: LockstepParser.ArgListContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#statement.
    def visitStatement(self, ctx: LockstepParser.StatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#varDecl.
    def visitVarDecl(self, ctx: LockstepParser.VarDeclContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#assignStmt.
    def visitAssignStmt(self, ctx: LockstepParser.AssignStmtContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#returnStmt.
    def visitReturnStmt(self, ctx: LockstepParser.ReturnStmtContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#expr.
    def visitExpr(self, ctx: LockstepParser.ExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#exprList.
    def visitExprList(self, ctx: LockstepParser.ExprListContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#lvalue.
    def visitLvalue(self, ctx: LockstepParser.LvalueContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by LockstepParser#typeName.
    def visitTypeName(self, ctx: LockstepParser.TypeNameContext):
        return self.visitChildren(ctx)


del LockstepParser
