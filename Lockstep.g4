grammar Lockstep;

program: declaration* EOF;

declaration
    : dependencyDecl
    | structDecl
    | pureDecl
    | shaderDecl
    | filterDecl
    | pipelineDecl
    ;

dependencyDecl
    : importDecl
    | includeDecl
    ;

importDecl: 'import' STRING ';';
includeDecl: INCLUDE_DIRECTIVE STRING ';';

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
logicalAndExpr: bitwiseOrExpr ('&&' bitwiseOrExpr)*;
bitwiseOrExpr: bitwiseXorExpr ('|' bitwiseXorExpr)*;
bitwiseXorExpr: bitwiseAndExpr ('^' bitwiseAndExpr)*;
bitwiseAndExpr: equalityExpr ('&' equalityExpr)*;
equalityExpr: relExpr (('==' | '!=') relExpr)*;
relExpr: shiftExpr (('<' | '<=' | '>' | '>=') shiftExpr)*;
shiftExpr: addExpr (('<<' | '>>') addExpr)*;
addExpr: mulExpr (('+' | '-') mulExpr)*;
mulExpr: unaryExpr (('*' | '/' | '%') unaryExpr)*;
unaryExpr
    : ('-' | '!') unaryExpr
    | '(' typeName ')' unaryExpr
    | primaryExpr
    ;
primaryExpr
    : '(' expr ')'
    | ID '(' exprList? ')'
    | lvalue
    | INT
    | FLOAT
    | BOOL
    | STRING
    ;

exprList: expr (',' expr)*;
lvalue: ID ('.' ID)*;
typeName: ID typeSuffix*;
typeSuffix
    : '[' INT ']'
    | '<' typeName genericWidth? '>'
    ;

genericWidth: ',' INT;

BOOL: 'true' | 'false';
INCLUDE_DIRECTIVE: '#include';
ID: [a-zA-Z_][a-zA-Z0-9_]*;
STRING: '"' ( ~["\\\r\n] | '\\' . )* '"';
FLOAT
    : [0-9]+ '.' [0-9]* EXPONENT?
    | '.' [0-9]+ EXPONENT?
    | [0-9]+ EXPONENT
    ;
INT: [0-9]+;

fragment EXPONENT: [eE] [+-]? [0-9]+;
WS: [ \t\r\n]+ -> skip;
COMMENT: '//' ~[\r\n]* -> channel(HIDDEN);
