from antlr4 import CommonTokenStream, InputStream

from generated.parser.LockstepLexer import LockstepLexer
from lockstep_compiler.formatter import format_lockstep_source


def test_format_lockstep_source_normalizes_indentation_and_spacing():
    source = "pipeline Physics{stream<Vec3,1000> particles;bind{particles=Integrate(particles);}}"

    assert format_lockstep_source(source) == (
        "pipeline Physics {\n"
        "    stream<Vec3,1000> particles;\n"
        "    bind {\n"
        "        particles=Integrate(particles);\n"
        "    }\n"
        "}\n"
    )


def test_format_lockstep_source_preserves_struct_terminator():
    source = "struct Vec3{float x;float y;float z;};"

    assert format_lockstep_source(source) == (
        "struct Vec3 {\n" "    float x;\n" "    float y;\n" "    float z;\n" "};\n"
    )


def test_format_lockstep_source_uses_lexer_tokens_for_nested_expressions():
    source = "shader Integrate(in Vec3 p){Vec3 next=Step(p.x,p.y);return next;}"

    assert format_lockstep_source(source) == (
        "shader Integrate(in Vec3 p) {\n"
        "    Vec3 next=Step(p.x,p.y);\n"
        "    return next;\n"
        "}\n"
    )


def test_lexer_preserves_line_comments_in_token_stream():
    source = "// keep me\npipeline Main { bind { } }\n"

    lexer = LockstepLexer(InputStream(source))
    stream = CommonTokenStream(lexer)
    stream.fill()

    comments = [
        token.text
        for token in stream.tokens
        if token.text and token.text.startswith("//")
    ]
    assert comments == ["// keep me"]


def test_format_lockstep_source_preserves_trailing_and_own_line_comments():
    source = (
        "// header note\n"
        "struct Particle{\n"
        "float pos; // position\n"
        "float vel;\n"
        "};\n"
    )

    assert format_lockstep_source(source) == (
        "// header note\n"
        "struct Particle {\n"
        "    float pos;  // position\n"
        "    float vel;\n"
        "};\n"
    )


def test_format_lockstep_source_indents_comment_before_closing_brace():
    source = "pipeline Main{stream<float,4> s;bind{\n// nothing yet\n}}\n"

    assert format_lockstep_source(source) == (
        "pipeline Main {\n"
        "    stream<float,4> s;\n"
        "    bind {\n"
        "        // nothing yet\n"
        "    }\n"
        "}\n"
    )


def test_format_lockstep_source_is_idempotent_with_comments():
    source = (
        "// header note\n"
        "struct P{\n"
        "float pos; // position\n"
        "};\n"
    )

    once = format_lockstep_source(source)
    assert format_lockstep_source(once) == once


def test_format_lockstep_source_preserves_comment_only_file():
    source = "// just a comment\n// another\n"

    assert format_lockstep_source(source) == source


def test_format_lockstep_source_leaves_invalid_source_with_comments_untouched():
    # Missing bind block -> parse error; comments can't be safely re-placed, so
    # the source is returned verbatim rather than dropping the comment.
    source = "pipeline Main{ // keep me\nstream<float,4> s;\n}\n"

    assert format_lockstep_source(source) == source


def test_format_lockstep_source_formats_dependencies_and_composite_types():
    source = (
        'import"math.lock";#include"kernels.lock";'
        "pipeline Main{stream<Vec<float,4>,16> samples;bind{}}"
    )

    assert format_lockstep_source(source) == (
        'import "math.lock";\n'
        '#include "kernels.lock";\n'
        "pipeline Main {\n"
        "    stream<Vec<float,4>,16> samples;\n"
        "    bind {\n"
        "    }\n"
        "}\n"
    )


def test_format_lockstep_source_formats_fold_declarations_and_casts():
    source = (
        "shader Normalize(in Vec<float,4> p){"
        "Vec<float,4> next=(Vec<float,4>)p;return next;}"
        "pipeline Main{stream<Vec<float,4>,16> samples;"
        "uniform Vec<float,4> seed=Vec<float,4>(1,2);"
        "bind{uniform Vec<float,4> total=fold sum(samples);}}"
    )

    assert format_lockstep_source(source) == (
        "shader Normalize(in Vec<float,4> p) {\n"
        "    Vec<float,4> next=(Vec<float,4>)p;\n"
        "    return next;\n"
        "}\n"
        "pipeline Main {\n"
        "    stream<Vec<float,4>,16> samples;\n"
        "    uniform Vec<float,4> seed=Vec<float,4>(1,2);\n"
        "    bind {\n"
        "        uniform Vec<float,4> total=fold sum(samples);\n"
        "    }\n"
        "}\n"
    )
