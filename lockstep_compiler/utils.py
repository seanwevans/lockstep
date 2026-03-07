def sanitize_symbol(name: str) -> str:
    """Replace non-alphanumeric, non-underscore characters with underscores."""
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
