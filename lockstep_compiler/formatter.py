from __future__ import annotations

from antlr4 import CommonTokenStream, InputStream


def _lex_tokens(source: str) -> list[str]:
    from generated.parser.LockstepLexer import LockstepLexer

    lexer = LockstepLexer(InputStream(source))
    stream = CommonTokenStream(lexer)
    stream.fill()
    return [token.text for token in stream.tokens if token.type != -1]


def _needs_space(previous: str, current: str) -> bool:
    if not previous:
        return False
    if current in {")", "]", "}", ",", ";", "(", "<", ">", ".", "="}:
        return False
    if previous in {"(", "[", "{", ",", "<", ".", "="}:
        return False
    if previous[-1] == ">" and (current[0].isalnum() or current[0] == "_"):
        return True
    return (previous[-1].isalnum() or previous[-1] == "_") and (
        current[0].isalnum() or current[0] == "_"
    )


def format_lockstep_source(source, *, indent="    "):
    tokens = _lex_tokens(source)
    lines = []
    current = ""
    depth = 0

    def append_token(token: str):
        nonlocal current
        if _needs_space(current, token):
            current = f"{current} {token}"
        else:
            current = f"{current}{token}"

    def flush_current():
        nonlocal current
        if current.strip():
            lines.append(f"{indent * depth}{current.strip()}")
        current = ""

    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token == "{":
            if current.strip():
                lines.append(f"{indent * depth}{current.strip()} {{")
            else:
                lines.append(f"{indent * depth}{{")
            current = ""
            depth += 1
        elif token == "}":
            flush_current()
            depth = max(depth - 1, 0)
            closing = "}"
            if index + 1 < len(tokens) and tokens[index + 1] == ";":
                closing += ";"
                index += 1
            lines.append(f"{indent * depth}{closing}")
        elif token == ";":
            append_token(token)
            flush_current()
        else:
            append_token(token)

        index += 1

    flush_current()
    return "\n".join(lines) + "\n"
