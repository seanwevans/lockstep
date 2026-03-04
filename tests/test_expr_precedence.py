import importlib
import pathlib
import subprocess
import sys

import pytest
from antlr4 import CommonTokenStream, InputStream


@pytest.fixture(scope="module")
def generated_parser(tmp_path_factory):
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    tmp_dir = tmp_path_factory.mktemp("antlr_generated")

    try:
        subprocess.run(
            [
                "antlr4",
                "-Dlanguage=Python3",
                "-visitor",
                "-o",
                str(tmp_dir),
                "Lockstep.g4",
            ],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"ANTLR generation unavailable: {exc}")

    sys.path.insert(0, str(tmp_dir))
    try:
        lexer_module = importlib.import_module("LockstepLexer")
        parser_module = importlib.import_module("LockstepParser")
        yield lexer_module.LockstepLexer, parser_module.LockstepParser
    finally:
        sys.path.remove(str(tmp_dir))
        for module_name in [
            "LockstepLexer",
            "LockstepParser",
            "LockstepVisitor",
            "LockstepListener",
        ]:
            sys.modules.pop(module_name, None)


def _parse_expr(text, lexer_cls, parser_cls):
    parser = parser_cls(CommonTokenStream(lexer_cls(InputStream(text))))
    tree = parser.expr()

    assert parser.getNumberOfSyntaxErrors() == 0
    assert parser.getCurrentToken().type == parser.EOF

    return tree, parser


def _first_unary(expr_ctx):
    return (
        expr_ctx.logicalExpr()
        .logicalOrExpr()
        .logicalAndExpr()[0]
        .bitwiseOrExpr()[0]
        .bitwiseXorExpr()[0]
        .bitwiseAndExpr()[0]
        .equalityExpr()[0]
        .relExpr()[0]
        .shiftExpr()[0]
        .addExpr()[0]
        .mulExpr()[0]
        .unaryExpr()[0]
    )


def test_unary_binds_tighter_than_multiplication(generated_parser):
    lexer_cls, parser_cls = generated_parser
    tree, _ = _parse_expr("-a*b", lexer_cls, parser_cls)

    add_ctx = (
        tree.logicalExpr().logicalOrExpr().logicalAndExpr()[0].bitwiseOrExpr()[0].bitwiseXorExpr()[0].bitwiseAndExpr()[0].equalityExpr()[0].relExpr()[0].shiftExpr()[0].addExpr()[0]
    )
    mul_ctx = add_ctx.mulExpr()[0]
    unary_nodes = mul_ctx.unaryExpr()

    assert len(unary_nodes) == 2
    assert unary_nodes[0].getChild(0).getText() == "-"
    assert unary_nodes[0].unaryExpr().primaryExpr().lvalue().getText() == "a"
    assert unary_nodes[1].primaryExpr().lvalue().getText() == "b"


def test_bitwise_precedence_levels(generated_parser):
    lexer_cls, parser_cls = generated_parser
    tree, _ = _parse_expr("a|b^c&d", lexer_cls, parser_cls)

    and_root = tree.logicalExpr().logicalOrExpr().logicalAndExpr()[0]
    bitor_root = and_root.bitwiseOrExpr()
    assert len(bitor_root) == 2
    assert bitor_root[0].bitwiseXorExpr()[0].bitwiseAndExpr()[0].equalityExpr()[0].relExpr()[0].shiftExpr()[0].addExpr()[0].mulExpr()[0].unaryExpr()[0].primaryExpr().lvalue().getText() == "a"


def test_shift_binds_tighter_than_relational(generated_parser):
    lexer_cls, parser_cls = generated_parser
    tree, _ = _parse_expr("a<<1<b", lexer_cls, parser_cls)

    rel_ctx = (
        tree.logicalExpr().logicalOrExpr().logicalAndExpr()[0].bitwiseOrExpr()[0].bitwiseXorExpr()[0].bitwiseAndExpr()[0].equalityExpr()[0]
    )
    assert len(rel_ctx.relExpr()) == 2


def test_boolean_literals_parse_as_primary_expressions(generated_parser):
    lexer_cls, parser_cls = generated_parser
    true_tree, _ = _parse_expr("true", lexer_cls, parser_cls)
    false_tree, _ = _parse_expr("false", lexer_cls, parser_cls)

    assert _first_unary(true_tree).primaryExpr().BOOL().getText() == "true"
    assert _first_unary(false_tree).primaryExpr().BOOL().getText() == "false"
