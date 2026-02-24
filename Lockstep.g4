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
    | 'uniform' typeName ID '=' 'fold' ID '(' ID ')' ';'
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

expr
    : '(' expr ')'
    | ID '(' exprList? ')'
    | expr ('*' | '/' | '%') expr
    | expr ('+' | '-') expr
    | expr ('<' | '<=' | '>' | '>=') expr
    | expr ('==' | '!=') expr
    | expr ('&&' | '||') expr
    | lvalue
    | INT
    | FLOAT
    | '-' expr
    | '!' expr
    ;

exprList: expr (',' expr)*;
lvalue: ID ('.' ID)*;
typeName: ID;

ID: [a-zA-Z_][a-zA-Z0-9_]*;
INT: [0-9]+;
FLOAT: [0-9]+ '.' [0-9]+;
WS: [ \t\r\n]+ -> skip;
COMMENT: '//' ~[\r\n]* -> skip;
