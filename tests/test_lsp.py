from lockstep_compiler.lsp import (
    _infer_variable_types_from_entities,
    build_analysis_context,
    build_struct_member_index,
    find_definition_target,
    find_member_definition,
    provide_bind_completion_items,
    provide_hover_info,
)
from lockstep_compiler.ast import AstExprCall, AstType, AstVarDeclStmt


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


def test_member_access_position_is_exclusive_of_regex_end():
    source = """
struct Vec3 { float x; };

shader Copy(in Vec3 src, out Vec3 dst) {
    dst.x = src.x;
}
"""
    lines = source.splitlines()
    access_line = next(i for i, text in enumerate(lines) if "dst.x = src.x;" in text)
    access_text = lines[access_line]
    member_start = access_text.index("src.x")
    member_end = member_start + len("src.x")

    inside_target = find_member_definition(source, access_line, member_start + 1)
    inside_hover = provide_hover_info(source, access_line, member_start + 1)
    end_target = find_member_definition(source, access_line, member_end)

    assert inside_target is not None
    assert inside_target.struct_name == "Vec3"
    assert inside_target.field_name == "x"
    assert inside_hover == "(field) `Vec3.x: float`"
    assert end_target is None


def test_provide_bind_completion_items_includes_routes_and_kernels():
    items = provide_bind_completion_items(SOURCE, line=13, column=8)
    labels = [item["label"] for item in items]

    assert labels[0] == "dst=Integrate(src,dst,dt);"
    assert "Integrate(...)" in labels
    assert items[0]["detail"] == "Bind route template"


def test_infer_variable_types_propagates_from_pure_and_intrinsic_calls():
    entities = {
        "shaders": [
            {
                "params": [{"name": "src", "type": "float"}],
                "body_ast": [
                    AstVarDeclStmt(
                        declared_type=None,
                        name="value",
                        initializer=AstExprCall(name="gain", args=[]),
                    ),
                    AstVarDeclStmt(
                        declared_type=None,
                        name="masked",
                        initializer=AstExprCall(name="step", args=[]),
                    ),
                    AstVarDeclStmt(
                        declared_type=None,
                        name="blended",
                        initializer=AstExprCall(name="mix", args=[]),
                    ),
                    AstVarDeclStmt(
                        declared_type=AstType("float", "primitive"),
                        name="typed",
                        initializer=None,
                    ),
                ],
            }
        ],
        "filters": [],
        "pure_functions": [
            {"name": "gain", "return_type": "float", "params": []},
            {"name": "step", "return_type": "float", "params": []},
            {"name": "mix", "return_type": "float", "params": []},
        ],
    }
    inferred = _infer_variable_types_from_entities(entities)

    assert inferred["value"] == "float"
    assert inferred["masked"] == "float"
    assert inferred["blended"] == "float"


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


def test_find_definition_target_resolves_shader_filter_and_pure_function_names():
    source = """
shader Integrate(in float src, out float dst) { dst = src; }

filter Blur(in float src, out float dst) { dst = src; }

pure float
blend(float a, float b)
{
    return a;
}

pipeline P {
    stream<float, 32> src;
    stream<float, 32> dst;

    bind {
        dst = Integrate(src, dst);
        dst = Blur(src, dst);
    }
}

shader Wrap(in float src, out float dst) {
    dst = blend(src, src);
}
"""

    lines = source.splitlines()
    integrate_line = next(i for i, text in enumerate(lines) if "Integrate(src, dst)" in text)
    blur_line = next(i for i, text in enumerate(lines) if "Blur(src, dst)" in text)
    blend_line = next(i for i, text in enumerate(lines) if "blend(src, src)" in text)

    integrate_target = find_definition_target(
        source,
        integrate_line,
        lines[integrate_line].index("Integrate") + 1,
    )
    blur_target = find_definition_target(
        source,
        blur_line,
        lines[blur_line].index("Blur") + 1,
    )
    blend_target = find_definition_target(
        source,
        blend_line,
        lines[blend_line].index("blend") + 1,
    )

    assert integrate_target is not None
    assert (integrate_target.line, integrate_target.column, integrate_target.symbol) == (
        1,
        7,
        "Integrate",
    )

    assert blur_target is not None
    assert (blur_target.line, blur_target.column, blur_target.symbol) == (3, 7, "Blur")

    assert blend_target is not None
    assert (blend_target.line, blend_target.column, blend_target.symbol) == (6, 0, "blend")


def test_hover_and_definition_ignore_symbols_in_comments_and_strings():
    source = """
shader Step(in float src, out float dst) {
    // Step(src, dst);
    string msg = "Step(src, dst)";
    dst = src;
}
"""
    lines = source.splitlines()
    comment_line = next(i for i, text in enumerate(lines) if "// Step(" in text)
    string_line = next(i for i, text in enumerate(lines) if '"Step(src, dst)"' in text)

    assert provide_hover_info(source, comment_line, lines[comment_line].index("Step") + 1) is None
    assert (
        find_definition_target(
            source,
            comment_line,
            lines[comment_line].index("Step") + 1,
        )
        is None
    )
    assert provide_hover_info(source, string_line, lines[string_line].index("Step") + 1) is None
    assert (
        find_definition_target(
            source,
            string_line,
            lines[string_line].index("Step") + 1,
        )
        is None
    )


def test_definition_uses_identifier_locations_for_unusual_whitespace_layouts():
    source = """
shader
Warp
(
    in float src,
    out float dst
) {
    dst = src;
}

filter
Mix
(
    in float src,
    out float dst
) {
    dst = src;
}

pure float
fuse
(
    float a,
    float b
) {
    return a;
}

shader Use(in float src, out float dst) {
    dst = fuse(src, src);
}

pipeline P {
    stream<float, 8> src;
    stream<float, 8> dst;
    bind {
        dst = Warp(src, dst);
        dst = Mix(src, dst);
    }
}
"""
    lines = source.splitlines()
    warp_call_line = next(i for i, text in enumerate(lines) if "Warp(src, dst)" in text)
    mix_call_line = next(i for i, text in enumerate(lines) if "Mix(src, dst)" in text)
    fuse_call_line = next(i for i, text in enumerate(lines) if "fuse(src, src)" in text)

    warp_target = find_definition_target(source, warp_call_line, lines[warp_call_line].index("Warp") + 1)
    mix_target = find_definition_target(source, mix_call_line, lines[mix_call_line].index("Mix") + 1)
    fuse_target = find_definition_target(source, fuse_call_line, lines[fuse_call_line].index("fuse") + 1)

    assert warp_target is not None
    assert (warp_target.line, warp_target.column, warp_target.symbol) == (2, 0, "Warp")
    assert mix_target is not None
    assert (mix_target.line, mix_target.column, mix_target.symbol) == (11, 0, "Mix")
    assert fuse_target is not None
    assert (fuse_target.line, fuse_target.column, fuse_target.symbol) == (20, 0, "fuse")


def test_bind_completion_prefers_bind_route_metadata_with_unusual_spacing():
    source = """
shader Step(in float src, out float dst) { dst = src; }

pipeline P {
    stream<float, 4> src;
    stream<float, 4> dst;
    bind {
        dst
            =
            Step(
                src,
                dst
            );
    }
}
"""
    items = provide_bind_completion_items(source, line=7, column=8)

    assert items[0]["label"] == "dst=Step(src,dst);"
