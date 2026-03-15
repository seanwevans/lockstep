from lockstep_compiler.lsp import (
    build_analysis_context,
    build_struct_member_index,
    find_member_definition,
    provide_bind_completion_items,
    provide_hover_info,
)


SOURCE = """
struct Vec3 { float x; float y; float z; };

shader Integrate(in Vec3 pos, out Vec3 out_pos, uniform float dt) {
    out_pos.x = pos.x + dt;
}

pipeline P {
    stream<Vec3, 32> src;
    stream<Vec3, 32> dst;
    uniform float dt = 0.1;

    bind {
        dst = Integrate(src, dst, dt);
    }
}
"""


def test_build_struct_member_index_tracks_fields():
    index = build_struct_member_index(SOURCE)

    assert "Vec3" in index
    assert set(index["Vec3"]) == {"x", "y", "z"}


def test_find_member_definition_resolves_struct_field():
    target_line = 4
    target_column = 11  # out_pos.x

    definition = find_member_definition(SOURCE, target_line, target_column)

    assert definition is not None
    assert definition.struct_name == "Vec3"
    assert definition.field_name == "x"


def test_provide_bind_completion_items_includes_routes_and_kernels():
    items = provide_bind_completion_items(SOURCE, line=13, column=8)
    labels = [item["label"] for item in items]

    assert labels[0] == "dst=Integrate(src,dst,dt);"
    assert "Integrate(...)" in labels
    assert items[0]["detail"] == "Bind route template"


def test_provide_bind_completion_items_omits_bind_routes_outside_bind_block():
    items = provide_bind_completion_items(SOURCE, line=4, column=8)
    labels = [item["label"] for item in items]

    assert "dst=Integrate(src,dst,dt);" not in labels
    assert labels == ["Integrate(...)"]


def test_provide_bind_completion_items_deduplicates_and_ranks_categories():
    source = """
shader mix(in float value, out float out_value) { out_value = value; }
pure float mix(float a, float b) { return a; }

pipeline P {
    stream<float, 8> in_stream;
    stream<float, 8> out_stream;

    bind {
        out_stream = mix(in_stream, out_stream);
    }
}
"""

    items = provide_bind_completion_items(source, line=8, column=10)

    assert [item["label"] for item in items] == [
        "out_stream=mix(in_stream,out_stream);",
        "mix(...)",
    ]
    assert [item["detail"] for item in items] == [
        "Bind route template",
        "Shader/filter callable",
    ]


TRICKY_FORMAT_SOURCE = """
struct Particle {
    float mass;
    Vec3 velocity;
};

shader
Apply(
    in Particle src,
    out Particle dst,
    uniform float dt
) {
    Particle local = src;
    dst.velocity = local.velocity;
    float speed = dt;
}

filter
Post(
    in Particle src,
    out Particle dst
) {
    dst = src;
}

pure float
blend(float a, float b)
{
    return a;
}
"""



def test_build_struct_member_index_tracks_locations_from_ast():
    index = build_struct_member_index(TRICKY_FORMAT_SOURCE)

    assert index["Particle"]["mass"].line == 2
    assert index["Particle"]["velocity"].line == 3


def test_infer_and_definition_resolve_member_declared_from_body_ast():
    line = 13
    column = 8  # dst.velocity

    definition = find_member_definition(TRICKY_FORMAT_SOURCE, line, column)

    assert definition is not None
    assert definition.struct_name == "Particle"
    assert definition.field_name == "velocity"
    assert definition.line == 3


def test_hover_resolves_variable_and_callable_symbols_with_tricky_formatting():
    variable_hover = provide_hover_info(TRICKY_FORMAT_SOURCE, 13, 20)  # local.velocity
    shader_hover = provide_hover_info(TRICKY_FORMAT_SOURCE, 7, 0)  # Apply
    filter_hover = provide_hover_info(TRICKY_FORMAT_SOURCE, 18, 0)  # Post
    pure_hover = provide_hover_info(TRICKY_FORMAT_SOURCE, 26, 0)  # blend

    assert variable_hover == "(field) `Particle.velocity: Vec3`"
    assert shader_hover == "(shader) `shader Apply(...)`"
    assert filter_hover == "(filter) `filter Post(...)`"
    assert pure_hover == "(pure) `pure blend(...)`"
def test_large_source_member_definition_matches_default_and_context_path():
    struct_count = 50
    shader_count = 60
    struct_defs = [
        f"struct Vec{i} {{ float x; float y; float z; }};" for i in range(struct_count)
    ]
    shader_defs = [
        f"shader S{i}(in Vec{i % struct_count} pos, out Vec{i % struct_count} out_pos) {{ out_pos.y = pos.y; }}"
        for i in range(shader_count)
    ]
    source = "\n".join([*struct_defs, *shader_defs])

    target_line = struct_count + shader_count - 1
    target_column = shader_defs[-1].index("pos.y") + 1

    expected = find_member_definition(source, target_line, target_column)

    context = build_analysis_context(source)
    actual = find_member_definition(
        source,
        target_line,
        target_column,
        analysis_context=context,
    )

    assert expected is not None
    assert actual == expected


def test_large_source_hover_matches_default_and_context_path():
    source = "\n".join(
        [
            "struct Payload { float value; int id; };",
            *[
                f"shader Step{i}(in Payload p, out Payload out_p) {{ out_p.value = p.value; }}"
                for i in range(120)
            ],
        ]
    )

    target_line = 120
    target_column = source.splitlines()[target_line].index("p.value") + 2

    expected = provide_hover_info(source, target_line, target_column)

    context = build_analysis_context(source)
    actual = provide_hover_info(
        source,
        target_line,
        target_column,
        analysis_context=context,
    )

    assert expected == "(field) `Payload.value: float`"
    assert actual == expected
