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
structMember: type ID ';';

pureDecl: 'pure' type ID '(' pureParamList? ')' '{' statement* '}';
pureParamList: type ID (',' type ID)*;

shaderDecl: 'shader' ID '(' paramList? ')' '{' statement* '}';
filterDecl: 'filter' ID '(' paramList? ')' '{' statement* '}';

paramList: param (',' param)*;
param: ('in' | 'out' | 'uniform' | 'accum') type ID;

pipelineDecl: 'pipeline' ID '{' pipelineMember* bindBlock '}';

pipelineMember
    : streamDecl
    | accumDecl
    | uniformDecl
    ;

streamDecl: 'stream' '<' type ',' INT '>' ID ';';
accumDecl: 'accumulator' '<' type '>' ID ';';
uniformDecl: 'uniform' type ID ('=' expr)? ';';

bindBlock: 'bind' '{' bindStmt* '}';

bindStmt
    : ID '=' ID '(' argList ')' ';'
    | 'uniform' type ID '=' 'fold_' ID '(' ID ')' ';'
    ;

argList: ID (',' ID)*;

statement
    : varDecl
    | assignStmt
    | returnStmt
    ;

varDecl: type ID '=' expr ';';
assignStmt: lvalue '=' expr ';';
returnStmt: 'return' expr ';';

// --- Expressions ---
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
type: ID;

ID: [a-zA-Z_][a-zA-Z0-9_]*;
INT: [0-9]+;
FLOAT: [0-9]+ '.' [0-9]+;
WS: [ \t\r\n]+ -> skip;
COMMENT: '//' ~[\r\n]* -> skip;
