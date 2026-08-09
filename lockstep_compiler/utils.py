def sanitize_symbol(name: str) -> str:
    """Replace non-alphanumeric, non-underscore characters with underscores."""
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)


def decode_escape_sequences(literal: str) -> str:
    """Decode C-style escape sequences in a string or path literal.

    Uses Python's ``unicode_escape`` codec, which understands the common
    escapes the grammar allows (``\\n``, ``\\t``, ``\\"``, ``\\\\`` and so on).

    Literals that contain raw backslashes which do not form a valid escape
    sequence -- most commonly absolute Windows paths such as ``C:\\Users\\...``
    -- are not valid ``unicode_escape`` input and would otherwise raise
    ``UnicodeDecodeError``. In that case the literal text is returned unchanged
    so callers keep processing (and can validate or reject the value) instead of
    the compiler crashing with an uncaught decode error.
    """
    try:
        return bytes(literal, "utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return literal
