# Generated from Lockstep.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .LockstepParser import LockstepParser
else:
    from LockstepParser import LockstepParser

# This class defines a complete listener for a parse tree produced by LockstepParser.
class LockstepListener(ParseTreeListener):

    # Enter a parse tree produced by LockstepParser#program.
    def enterProgram(self, ctx:LockstepParser.ProgramContext):
        pass

    # Exit a parse tree produced by LockstepParser#program.
    def exitProgram(self, ctx:LockstepParser.ProgramContext):
        pass


    # Enter a parse tree produced by LockstepParser#declaration.
    def enterDeclaration(self, ctx:LockstepParser.DeclarationContext):
        pass

    # Exit a parse tree produced by LockstepParser#declaration.
    def exitDeclaration(self, ctx:LockstepParser.DeclarationContext):
        pass


    # Enter a parse tree produced by LockstepParser#structDecl.
    def enterStructDecl(self, ctx:LockstepParser.StructDeclContext):
        pass

    # Exit a parse tree produced by LockstepParser#structDecl.
    def exitStructDecl(self, ctx:LockstepParser.StructDeclContext):
        pass


    # Enter a parse tree produced by LockstepParser#structMember.
    def enterStructMember(self, ctx:LockstepParser.StructMemberContext):
        pass

    # Exit a parse tree produced by LockstepParser#structMember.
    def exitStructMember(self, ctx:LockstepParser.StructMemberContext):
        pass


    # Enter a parse tree produced by LockstepParser#pureDecl.
    def enterPureDecl(self, ctx:LockstepParser.PureDeclContext):
        pass

    # Exit a parse tree produced by LockstepParser#pureDecl.
    def exitPureDecl(self, ctx:LockstepParser.PureDeclContext):
        pass


    # Enter a parse tree produced by LockstepParser#pureParamList.
    def enterPureParamList(self, ctx:LockstepParser.PureParamListContext):
        pass

    # Exit a parse tree produced by LockstepParser#pureParamList.
    def exitPureParamList(self, ctx:LockstepParser.PureParamListContext):
        pass


    # Enter a parse tree produced by LockstepParser#shaderDecl.
    def enterShaderDecl(self, ctx:LockstepParser.ShaderDeclContext):
        pass

    # Exit a parse tree produced by LockstepParser#shaderDecl.
    def exitShaderDecl(self, ctx:LockstepParser.ShaderDeclContext):
        pass


    # Enter a parse tree produced by LockstepParser#filterDecl.
    def enterFilterDecl(self, ctx:LockstepParser.FilterDeclContext):
        pass

    # Exit a parse tree produced by LockstepParser#filterDecl.
    def exitFilterDecl(self, ctx:LockstepParser.FilterDeclContext):
        pass


    # Enter a parse tree produced by LockstepParser#paramList.
    def enterParamList(self, ctx:LockstepParser.ParamListContext):
        pass

    # Exit a parse tree produced by LockstepParser#paramList.
    def exitParamList(self, ctx:LockstepParser.ParamListContext):
        pass


    # Enter a parse tree produced by LockstepParser#param.
    def enterParam(self, ctx:LockstepParser.ParamContext):
        pass

    # Exit a parse tree produced by LockstepParser#param.
    def exitParam(self, ctx:LockstepParser.ParamContext):
        pass


    # Enter a parse tree produced by LockstepParser#pipelineDecl.
    def enterPipelineDecl(self, ctx:LockstepParser.PipelineDeclContext):
        pass

    # Exit a parse tree produced by LockstepParser#pipelineDecl.
    def exitPipelineDecl(self, ctx:LockstepParser.PipelineDeclContext):
        pass


    # Enter a parse tree produced by LockstepParser#pipelineMember.
    def enterPipelineMember(self, ctx:LockstepParser.PipelineMemberContext):
        pass

    # Exit a parse tree produced by LockstepParser#pipelineMember.
    def exitPipelineMember(self, ctx:LockstepParser.PipelineMemberContext):
        pass


    # Enter a parse tree produced by LockstepParser#streamDecl.
    def enterStreamDecl(self, ctx:LockstepParser.StreamDeclContext):
        pass

    # Exit a parse tree produced by LockstepParser#streamDecl.
    def exitStreamDecl(self, ctx:LockstepParser.StreamDeclContext):
        pass


    # Enter a parse tree produced by LockstepParser#accumDecl.
    def enterAccumDecl(self, ctx:LockstepParser.AccumDeclContext):
        pass

    # Exit a parse tree produced by LockstepParser#accumDecl.
    def exitAccumDecl(self, ctx:LockstepParser.AccumDeclContext):
        pass


    # Enter a parse tree produced by LockstepParser#uniformDecl.
    def enterUniformDecl(self, ctx:LockstepParser.UniformDeclContext):
        pass

    # Exit a parse tree produced by LockstepParser#uniformDecl.
    def exitUniformDecl(self, ctx:LockstepParser.UniformDeclContext):
        pass


    # Enter a parse tree produced by LockstepParser#bindBlock.
    def enterBindBlock(self, ctx:LockstepParser.BindBlockContext):
        pass

    # Exit a parse tree produced by LockstepParser#bindBlock.
    def exitBindBlock(self, ctx:LockstepParser.BindBlockContext):
        pass


    # Enter a parse tree produced by LockstepParser#bindStmt.
    def enterBindStmt(self, ctx:LockstepParser.BindStmtContext):
        pass

    # Exit a parse tree produced by LockstepParser#bindStmt.
    def exitBindStmt(self, ctx:LockstepParser.BindStmtContext):
        pass


    # Enter a parse tree produced by LockstepParser#foldOperator.
    def enterFoldOperator(self, ctx:LockstepParser.FoldOperatorContext):
        pass

    # Exit a parse tree produced by LockstepParser#foldOperator.
    def exitFoldOperator(self, ctx:LockstepParser.FoldOperatorContext):
        pass


    # Enter a parse tree produced by LockstepParser#argList.
    def enterArgList(self, ctx:LockstepParser.ArgListContext):
        pass

    # Exit a parse tree produced by LockstepParser#argList.
    def exitArgList(self, ctx:LockstepParser.ArgListContext):
        pass


    # Enter a parse tree produced by LockstepParser#statement.
    def enterStatement(self, ctx:LockstepParser.StatementContext):
        pass

    # Exit a parse tree produced by LockstepParser#statement.
    def exitStatement(self, ctx:LockstepParser.StatementContext):
        pass


    # Enter a parse tree produced by LockstepParser#varDecl.
    def enterVarDecl(self, ctx:LockstepParser.VarDeclContext):
        pass

    # Exit a parse tree produced by LockstepParser#varDecl.
    def exitVarDecl(self, ctx:LockstepParser.VarDeclContext):
        pass


    # Enter a parse tree produced by LockstepParser#assignStmt.
    def enterAssignStmt(self, ctx:LockstepParser.AssignStmtContext):
        pass

    # Exit a parse tree produced by LockstepParser#assignStmt.
    def exitAssignStmt(self, ctx:LockstepParser.AssignStmtContext):
        pass


    # Enter a parse tree produced by LockstepParser#returnStmt.
    def enterReturnStmt(self, ctx:LockstepParser.ReturnStmtContext):
        pass

    # Exit a parse tree produced by LockstepParser#returnStmt.
    def exitReturnStmt(self, ctx:LockstepParser.ReturnStmtContext):
        pass


    # Enter a parse tree produced by LockstepParser#expr.
    def enterExpr(self, ctx:LockstepParser.ExprContext):
        pass

    # Exit a parse tree produced by LockstepParser#expr.
    def exitExpr(self, ctx:LockstepParser.ExprContext):
        pass


    # Enter a parse tree produced by LockstepParser#logicalExpr.
    def enterLogicalExpr(self, ctx:LockstepParser.LogicalExprContext):
        pass

    # Exit a parse tree produced by LockstepParser#logicalExpr.
    def exitLogicalExpr(self, ctx:LockstepParser.LogicalExprContext):
        pass


    # Enter a parse tree produced by LockstepParser#logicalOrExpr.
    def enterLogicalOrExpr(self, ctx:LockstepParser.LogicalOrExprContext):
        pass

    # Exit a parse tree produced by LockstepParser#logicalOrExpr.
    def exitLogicalOrExpr(self, ctx:LockstepParser.LogicalOrExprContext):
        pass


    # Enter a parse tree produced by LockstepParser#logicalAndExpr.
    def enterLogicalAndExpr(self, ctx:LockstepParser.LogicalAndExprContext):
        pass

    # Exit a parse tree produced by LockstepParser#logicalAndExpr.
    def exitLogicalAndExpr(self, ctx:LockstepParser.LogicalAndExprContext):
        pass


    # Enter a parse tree produced by LockstepParser#bitwiseOrExpr.
    def enterBitwiseOrExpr(self, ctx:LockstepParser.BitwiseOrExprContext):
        pass

    # Exit a parse tree produced by LockstepParser#bitwiseOrExpr.
    def exitBitwiseOrExpr(self, ctx:LockstepParser.BitwiseOrExprContext):
        pass


    # Enter a parse tree produced by LockstepParser#bitwiseXorExpr.
    def enterBitwiseXorExpr(self, ctx:LockstepParser.BitwiseXorExprContext):
        pass

    # Exit a parse tree produced by LockstepParser#bitwiseXorExpr.
    def exitBitwiseXorExpr(self, ctx:LockstepParser.BitwiseXorExprContext):
        pass


    # Enter a parse tree produced by LockstepParser#bitwiseAndExpr.
    def enterBitwiseAndExpr(self, ctx:LockstepParser.BitwiseAndExprContext):
        pass

    # Exit a parse tree produced by LockstepParser#bitwiseAndExpr.
    def exitBitwiseAndExpr(self, ctx:LockstepParser.BitwiseAndExprContext):
        pass


    # Enter a parse tree produced by LockstepParser#equalityExpr.
    def enterEqualityExpr(self, ctx:LockstepParser.EqualityExprContext):
        pass

    # Exit a parse tree produced by LockstepParser#equalityExpr.
    def exitEqualityExpr(self, ctx:LockstepParser.EqualityExprContext):
        pass


    # Enter a parse tree produced by LockstepParser#relExpr.
    def enterRelExpr(self, ctx:LockstepParser.RelExprContext):
        pass

    # Exit a parse tree produced by LockstepParser#relExpr.
    def exitRelExpr(self, ctx:LockstepParser.RelExprContext):
        pass


    # Enter a parse tree produced by LockstepParser#shiftExpr.
    def enterShiftExpr(self, ctx:LockstepParser.ShiftExprContext):
        pass

    # Exit a parse tree produced by LockstepParser#shiftExpr.
    def exitShiftExpr(self, ctx:LockstepParser.ShiftExprContext):
        pass


    # Enter a parse tree produced by LockstepParser#addExpr.
    def enterAddExpr(self, ctx:LockstepParser.AddExprContext):
        pass

    # Exit a parse tree produced by LockstepParser#addExpr.
    def exitAddExpr(self, ctx:LockstepParser.AddExprContext):
        pass


    # Enter a parse tree produced by LockstepParser#mulExpr.
    def enterMulExpr(self, ctx:LockstepParser.MulExprContext):
        pass

    # Exit a parse tree produced by LockstepParser#mulExpr.
    def exitMulExpr(self, ctx:LockstepParser.MulExprContext):
        pass


    # Enter a parse tree produced by LockstepParser#unaryExpr.
    def enterUnaryExpr(self, ctx:LockstepParser.UnaryExprContext):
        pass

    # Exit a parse tree produced by LockstepParser#unaryExpr.
    def exitUnaryExpr(self, ctx:LockstepParser.UnaryExprContext):
        pass


    # Enter a parse tree produced by LockstepParser#primaryExpr.
    def enterPrimaryExpr(self, ctx:LockstepParser.PrimaryExprContext):
        pass

    # Exit a parse tree produced by LockstepParser#primaryExpr.
    def exitPrimaryExpr(self, ctx:LockstepParser.PrimaryExprContext):
        pass


    # Enter a parse tree produced by LockstepParser#exprList.
    def enterExprList(self, ctx:LockstepParser.ExprListContext):
        pass

    # Exit a parse tree produced by LockstepParser#exprList.
    def exitExprList(self, ctx:LockstepParser.ExprListContext):
        pass


    # Enter a parse tree produced by LockstepParser#lvalue.
    def enterLvalue(self, ctx:LockstepParser.LvalueContext):
        pass

    # Exit a parse tree produced by LockstepParser#lvalue.
    def exitLvalue(self, ctx:LockstepParser.LvalueContext):
        pass


    # Enter a parse tree produced by LockstepParser#typeName.
    def enterTypeName(self, ctx:LockstepParser.TypeNameContext):
        pass

    # Exit a parse tree produced by LockstepParser#typeName.
    def exitTypeName(self, ctx:LockstepParser.TypeNameContext):
        pass



del LockstepParser