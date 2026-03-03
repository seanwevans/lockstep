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
        "struct Vec3 {\n"
        "    float x;\n"
        "    float y;\n"
        "    float z;\n"
        "};\n"
    )


def test_format_lockstep_source_uses_lexer_tokens_for_nested_expressions():
    source = "shader Integrate(in Vec3 p){Vec3 next=Step(p.x,p.y);return next;}"

    assert format_lockstep_source(source) == (
        "shader Integrate(in Vec3 p) {\n"
        "    Vec3 next=Step(p.x,p.y);\n"
        "    return next;\n"
        "}\n"
    )
