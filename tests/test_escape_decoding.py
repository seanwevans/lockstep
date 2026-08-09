import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lockstep_compiler.codegen import _FunctionLowerer
from lockstep_compiler.utils import decode_escape_sequences


def test_decode_escape_sequences_decodes_valid_escapes():
    assert decode_escape_sequences(r"line\nbreak") == "line\nbreak"
    assert decode_escape_sequences(r"tab\tafter") == "tab\tafter"
    assert decode_escape_sequences(r"back\\slash") == "back\\slash"


def test_decode_escape_sequences_tolerates_raw_backslashes():
    # Raw Windows paths contain backslash sequences (``\U``, ``\A``) that are not
    # valid ``unicode_escape`` input; decoding must not raise.
    windows_path = r"C:\Users\runner\AppData\outside.lock"
    assert decode_escape_sequences(windows_path) == windows_path
    assert decode_escape_sequences(r"\U0") == r"\U0"


def test_string_literal_lowering_tolerates_raw_backslashes():
    # A source string literal such as ``"C:\Users"`` previously crashed codegen
    # with an uncaught UnicodeDecodeError. The quotes are stripped and the raw
    # text is returned instead of raising.
    assert _FunctionLowerer._decode_string_literal(r'"C:\Users"') == r"C:\Users"
    assert _FunctionLowerer._decode_string_literal(r'"ok\ttab"') == "ok\ttab"
