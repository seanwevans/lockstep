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


def _expr_levels(expr_ctx):
    logical_expr = expr_ctx.logicalExpr()
    logical_or = logical_expr.logicalOrExpr()
    return logical_or, logical_or.logicalAndExpr()


def _single_chain(expr_ctx):
    logical_or, logical_and_nodes = _expr_levels(expr_ctx)
    assert len(logical_and_nodes) == 1
    equality_nodes = logical_and_nodes[0].equalityExpr()
    assert len(equality_nodes) == 1
    rel_nodes = equality_nodes[0].relExpr()
    assert len(rel_nodes) == 1
    add_nodes = rel_nodes[0].addExpr()
    assert len(add_nodes) == 1
    return logical_or, logical_and_nodes[0], equality_nodes[0], rel_nodes[0], add_nodes[0]


def test_unary_binds_tighter_than_multiplication(generated_parser):
    lexer_cls, parser_cls = generated_parser
    tree, _ = _parse_expr("-a*b", lexer_cls, parser_cls)
    _, _, _, _, add_ctx = _single_chain(tree)

    mul_nodes = add_ctx.mulExpr()
    assert len(mul_nodes) == 1
    mul_ctx = mul_nodes[0]
    unary_nodes = mul_ctx.unaryExpr()

    assert len(unary_nodes) == 2
    assert unary_nodes[0].getChild(0).getText() == "-"
    assert unary_nodes[0].unaryExpr().primaryExpr().lvalue().getText() == "a"
    assert unary_nodes[1].primaryExpr().lvalue().getText() == "b"


def test_precedence_tree_snapshot(generated_parser):
    lexer_cls, parser_cls = generated_parser
    tree, parser = _parse_expr("a+b*c", lexer_cls, parser_cls)
    parsed = tree.toStringTree(recog=parser)

    assert parsed.startswith("(expr (logicalExpr (logicalOrExpr")


def test_unary_over_parenthesized_comparison(generated_parser):
    lexer_cls, parser_cls = generated_parser
    tree, _ = _parse_expr("!(a<b)", lexer_cls, parser_cls)
    _, _, _, _, add_ctx = _single_chain(tree)

    mul_ctx = add_ctx.mulExpr()[0]
    outer_unary = mul_ctx.unaryExpr()[0]

    assert outer_unary.getChild(0).getText() == "!"
    parenthesized = outer_unary.unaryExpr().primaryExpr().expr()

    inner_logical_or, inner_and_nodes = _expr_levels(parenthesized)
    assert len(inner_logical_or.logicalAndExpr()) == 1
    assert len(inner_and_nodes[0].equalityExpr()) == 1
    rel_nodes = inner_and_nodes[0].equalityExpr()[0].relExpr()
    assert len(rel_nodes) == 2
    assert rel_nodes[0].addExpr()[0].mulExpr()[0].unaryExpr()[0].primaryExpr().lvalue().getText() == "a"
    assert rel_nodes[1].addExpr()[0].mulExpr()[0].unaryExpr()[0].primaryExpr().lvalue().getText() == "b"


def test_multiplication_binds_tighter_than_addition(generated_parser):
    lexer_cls, parser_cls = generated_parser
    tree, _ = _parse_expr("a+b*c", lexer_cls, parser_cls)
    _, _, _, _, add_ctx = _single_chain(tree)

    mul_nodes = add_ctx.mulExpr()

    assert len(mul_nodes) == 2
    left_mul, right_mul = mul_nodes
    assert left_mul.unaryExpr()[0].primaryExpr().lvalue().getText() == "a"
    assert right_mul.unaryExpr()[0].primaryExpr().lvalue().getText() == "b"
    assert right_mul.unaryExpr()[1].primaryExpr().lvalue().getText() == "c"


def test_logical_and_binds_tighter_than_or(generated_parser):
    lexer_cls, parser_cls = generated_parser
    tree, _ = _parse_expr("a==b||c&&d", lexer_cls, parser_cls)
    logical_or, logical_and_nodes = _expr_levels(tree)

    assert len(logical_and_nodes) == 2
    left_and, right_and = logical_and_nodes
    assert "||" in logical_or.getText()

    left_equality_nodes = left_and.equalityExpr()
    right_equality_nodes = right_and.equalityExpr()
    assert len(left_equality_nodes) == 1
    assert len(right_equality_nodes) == 2

    left_lhs = left_equality_nodes[0].relExpr()[0].addExpr()[0].mulExpr()[0].unaryExpr()[0]
    left_rhs = left_equality_nodes[0].relExpr()[1].addExpr()[0].mulExpr()[0].unaryExpr()[0]
    right_lhs = right_equality_nodes[0].relExpr()[0].addExpr()[0].mulExpr()[0].unaryExpr()[0]
    right_rhs = right_equality_nodes[1].relExpr()[0].addExpr()[0].mulExpr()[0].unaryExpr()[0]

    assert left_lhs.primaryExpr().lvalue().getText() == "a"
    assert left_rhs.primaryExpr().lvalue().getText() == "b"
    assert right_lhs.primaryExpr().lvalue().getText() == "c"
    assert right_rhs.primaryExpr().lvalue().getText() == "d"
    assert "&&" in right_and.getText()
