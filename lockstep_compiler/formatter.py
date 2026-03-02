def _tokenize(source):
    tokens = []
    current = []

    for char in source:
        if char in {"{", "}", ";", "\n"}:
            if current:
                token = "".join(current).strip()
                if token:
                    tokens.append(token)
                current = []
            tokens.append(char)
        else:
            current.append(char)

    if current:
        token = "".join(current).strip()
        if token:
            tokens.append(token)

    return tokens


def format_lockstep_source(source, *, indent="    "):
    tokens = _tokenize(source)
    lines = []
    current = ""
    depth = 0

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
            current = f"{current.strip()};"
            flush_current()
        elif token == "\n":
            flush_current()
        else:
            if current:
                current = f"{current} {token}"
            else:
                current = token

        index += 1

    flush_current()
    return "\n".join(lines) + "\n"
