from antlr4 import InputStream

from generated.parser.LockstepLexer import LockstepLexer


def test_lexer_emits_comment_tokens_on_hidden_channel():
    lexer = LockstepLexer(InputStream("// hello\npipeline Main { bind { } }"))
    tokens = lexer.getAllTokens()
    comments = [token for token in tokens if token.type == LockstepLexer.COMMENT]

    assert comments
    assert comments[0].text == "// hello"
    assert comments[0].channel == 1
