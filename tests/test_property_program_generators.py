import pathlib
import sys

from antlr4 import CommonTokenStream, InputStream
from hypothesis import given, settings, strategies as st

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lockstep_compiler.errors import ParseErrorCollector
from generated.parser.LockstepLexer import LockstepLexer
from generated.parser.LockstepParser import LockstepParser


_IDENTIFIER_START = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
_IDENTIFIER_CONTINUE = _IDENTIFIER_START + "0123456789"


@st.composite
def identifiers(draw):
    first = draw(st.sampled_from(list(_IDENTIFIER_START)))
    rest = draw(st.text(alphabet=_IDENTIFIER_CONTINUE, min_size=0, max_size=8))
    candidate = f"{first}{rest}"
    return candidate if candidate not in {"true", "false"} else f"_{candidate}"


@st.composite
def type_names(draw):
    base = draw(st.sampled_from(["float", "int", "bool", "Vec2", "Vec3", "Particle"]))
    suffix = draw(st.sampled_from(["", "[2]", "[4]"]))
    return f"{base}{suffix}"


@st.composite
def literal_exprs(draw):
    return draw(
        st.one_of(
            st.sampled_from(["0", "1", "42", "0.0", "1.5", "true", "false"]),
            identifiers(),
        )
    )


@st.composite
def struct_decl(draw):
    name = draw(identifiers())
    field_count = draw(st.integers(min_value=1, max_value=3))
    fields = []
    for _ in range(field_count):
        fields.append(f"{draw(type_names())} {draw(identifiers())};")
    return f"struct {name} {{ {' '.join(fields)} }};"


@st.composite
def pure_decl(draw):
    ret_type = draw(type_names())
    name = draw(identifiers())
    param_count = draw(st.integers(min_value=0, max_value=2))
    params = [f"{draw(type_names())} {draw(identifiers())}" for _ in range(param_count)]
    body = [
        f"{draw(type_names())} {draw(identifiers())} = {draw(literal_exprs())};",
        f"return {draw(literal_exprs())};",
    ]
    return f"pure {ret_type} {name}({', '.join(params)}) {{ {' '.join(body)} }}"


@st.composite
def shader_decl(draw):
    name = draw(identifiers())
    param_count = draw(st.integers(min_value=0, max_value=2))
    qualifier = st.sampled_from(["in", "out", "uniform", "accum"])
    params = [
        f"{draw(qualifier)} {draw(type_names())} {draw(identifiers())}"
        for _ in range(param_count)
    ]
    return f"shader {name}({', '.join(params)}) {{ return {draw(literal_exprs())}; }}"


@st.composite
def filter_decl(draw):
    name = draw(identifiers())
    param_count = draw(st.integers(min_value=0, max_value=2))
    qualifier = st.sampled_from(["in", "out", "uniform", "accum"])
    params = [
        f"{draw(qualifier)} {draw(type_names())} {draw(identifiers())}"
        for _ in range(param_count)
    ]
    return f"filter {name}({', '.join(params)}) {{ return {draw(literal_exprs())}; }}"


@st.composite
def pipeline_decl(draw):
    name = draw(identifiers())
    stream_name = draw(identifiers())
    accum_name = draw(identifiers())
    uniform_name = draw(identifiers())
    shader_name = draw(identifiers())
    lhs_name = draw(identifiers())
    arg0 = draw(identifiers())
    arg1 = draw(identifiers())

    return (
        f"pipeline {name} {{ "
        f"stream<{draw(type_names())}, 4> {stream_name}; "
        f"accumulator<{draw(type_names())}> {accum_name}; "
        f"uniform {draw(type_names())} {uniform_name} = {draw(literal_exprs())}; "
        "bind { "
        f"{lhs_name} = {shader_name}({arg0}, {arg1}); "
        "} "
        "}"
    )


valid_programs = st.lists(
    st.one_of(
        struct_decl(), pure_decl(), shader_decl(), filter_decl(), pipeline_decl()
    ),
    min_size=1,
    max_size=4,
).map(lambda declarations: "\n".join(declarations))

invalid_programs = valid_programs.map(lambda program: f"{program}\n@")


def collect_parse_errors(source: str):
    input_stream = InputStream(source)
    lexer = LockstepLexer(input_stream)
    parser_errors = ParseErrorCollector(source_file="<property-test>")

    lexer.removeErrorListeners()
    lexer.addErrorListener(parser_errors)
    stream = CommonTokenStream(lexer)

    parser = LockstepParser(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(parser_errors)
    parser.program()

    return parser_errors.errors


@settings(max_examples=50, deadline=None)
@given(valid_programs)
def test_valid_program_generators_parse_without_errors(program):
    assert collect_parse_errors(program) == []


@settings(max_examples=50, deadline=None)
@given(invalid_programs)
def test_invalid_program_generators_fail_to_parse(program):
    assert collect_parse_errors(program)
