grammar Lockstep;

program: declaration* EOF;

declaration
    : structDecl
    | pureDecl
    | shaderDecl
    | filterDecl
    | pipelineDecl
    ;

structDecl: 'struct' ID '{' structMember* '}' ';';
structMember: typeName ID ';';

pureDecl: 'pure' typeName ID '(' pureParamList? ')' '{' statement* '}';
pureParamList: typeName ID (',' typeName ID)*;

shaderDecl: 'shader' ID '(' paramList? ')' '{' statement* '}';
filterDecl: 'filter' ID '(' paramList? ')' '{' statement* '}';

paramList: param (',' param)*;
param: ('in' | 'out' | 'uniform' | 'accum') typeName ID;

pipelineDecl: 'pipeline' ID '{' pipelineMember* bindBlock '}';

pipelineMember
    : streamDecl
    | accumDecl
    | uniformDecl
    ;

streamDecl: 'stream' '<' typeName ',' INT '>' ID ';';
accumDecl: 'accumulator' '<' typeName '>' ID ';';
uniformDecl: 'uniform' typeName ID ('=' expr)? ';';

bindBlock: 'bind' '{' bindStmt* '}';

bindStmt
    : ID '=' ID '(' argList ')' ';'
    | 'uniform' typeName ID '=' 'fold' foldOperator '(' ID ')' ';'
    ;

foldOperator
    : 'sum'
    | 'avg'
    | 'min'
    | 'max'
    ;

argList: ID (',' ID)*;

statement
    : varDecl
    | assignStmt
    | returnStmt
    ;

varDecl: typeName ID ('=' expr)? ';';
assignStmt: lvalue '=' expr ';';
returnStmt: 'return' expr ';';

expr: logicalExpr;

logicalExpr: logicalOrExpr;
logicalOrExpr: logicalAndExpr ('||' logicalAndExpr)*;
logicalAndExpr: equalityExpr ('&&' equalityExpr)*;
equalityExpr: relExpr (('==' | '!=') relExpr)*;
relExpr: addExpr (('<' | '<=' | '>' | '>=') addExpr)*;
addExpr: mulExpr (('+' | '-') mulExpr)*;
mulExpr: unaryExpr (('*' | '/' | '%') unaryExpr)*;
unaryExpr: ('-' | '!') unaryExpr | primaryExpr;
primaryExpr
    : '(' expr ')'
    | ID '(' exprList? ')'
    | lvalue
    | INT
    | FLOAT
    ;

exprList: expr (',' expr)*;
lvalue: ID ('.' ID)*;
typeName: ID;

ID: [a-zA-Z_][a-zA-Z0-9_]*;
FLOAT
    : [0-9]+ '.' [0-9]* EXPONENT?
    | '.' [0-9]+ EXPONENT?
    | [0-9]+ EXPONENT
    ;
INT: [0-9]+;

fragment EXPONENT: [eE] [+-]? [0-9]+;
WS: [ \t\r\n]+ -> skip;
COMMENT: '//' ~[\r\n]* -> skip;
