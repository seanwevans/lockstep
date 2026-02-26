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

    return tree.toStringTree(recog=parser)


def test_unary_binds_tighter_than_multiplication(generated_parser):
    lexer_cls, parser_cls = generated_parser
    parsed = _parse_expr("-a*b", lexer_cls, parser_cls)

    assert "(mulExpr (unaryExpr - (unaryExpr (primaryExpr (lvalue a)))) *" in parsed


def test_unary_over_parenthesized_comparison(generated_parser):
    lexer_cls, parser_cls = generated_parser
    parsed = _parse_expr("!(a<b)", lexer_cls, parser_cls)

    assert "(unaryExpr ! (unaryExpr (primaryExpr ( (expr" in parsed
    assert "< (addExpr" in parsed


def test_multiplication_binds_tighter_than_addition(generated_parser):
    lexer_cls, parser_cls = generated_parser
    parsed = _parse_expr("a+b*c", lexer_cls, parser_cls)

    assert "(addExpr (mulExpr (unaryExpr (primaryExpr (lvalue a)))) +" in parsed
    assert (
        "(mulExpr (unaryExpr (primaryExpr (lvalue b))) * (unaryExpr (primaryExpr (lvalue c))))"
        in parsed
    )


def test_logical_and_binds_tighter_than_or(generated_parser):
    lexer_cls, parser_cls = generated_parser
    parsed = _parse_expr("a==b||c&&d", lexer_cls, parser_cls)

    assert "(logicalOrExpr (logicalAndExpr" in parsed
    assert "|| (logicalAndExpr" in parsed
    assert "(logicalAndExpr (equalityExpr" in parsed
    assert "&& (equalityExpr" in parsed


@pytest.mark.parametrize("literal", ["1.0", "1.", ".5", "1e3", "1.2e-3"])
def test_numeric_literal_variants_parse(generated_parser, literal):
    lexer_cls, parser_cls = generated_parser
    parsed = _parse_expr(literal, lexer_cls, parser_cls)

    assert "(primaryExpr" in parsed


def test_numeric_literals_keep_member_access_precedence(generated_parser):
    lexer_cls, parser_cls = generated_parser
    parsed = _parse_expr("obj.x + .5", lexer_cls, parser_cls)

    assert "(lvalue obj . x)" in parsed
    assert "(primaryExpr .5)" in parsed
