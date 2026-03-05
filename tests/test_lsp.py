from lockstep_compiler.lsp import (
    build_struct_member_index,
    find_member_definition,
    provide_bind_completion_items,
)


SOURCE = """
struct Vec3 { float x; float y; float z; };

shader Integrate(in Vec3 pos, out Vec3 out_pos, uniform float dt) {
    out_pos.x = pos.x + dt;
}

pipeline P {
    stream<Vec3, 32> src;
    stream<Vec3, 32> dst;

    bind {
        dst = Integrate(src, dst, 0.1);
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
    items = provide_bind_completion_items(SOURCE)

    assert "dst=Integrate(src,dst,0.1);" in items
    assert "Integrate(...)" in items


SHADOWED_SOURCE = """
struct Vec3 { float x; float y; float z; };
struct Vec2 { float x; float y; };

shader Shadow(in Vec3 value) {
    Vec2 value = Vec2(0.0, 0.0);
    value.x = 1.0;
}
"""


def test_find_member_definition_prefers_nearest_shadowed_variable_scope():
    target_line = 6
    target_column = 8

    definition = find_member_definition(SHADOWED_SOURCE, target_line, target_column)

    assert definition is not None
    assert definition.struct_name == "Vec2"
    assert definition.field_name == "x"


DUPLICATE_SCOPE_SOURCE = """
struct InType { float x; };
struct StreamType { float x; };

shader Sample(in InType src) {
    src.x = 1.0;
}

pipeline P {
    stream<StreamType, 32> src;

    bind {
        src = Sample(src);
    }
}
"""


def test_find_member_definition_resolves_duplicate_identifiers_by_scope():
    shader_line = 5
    shader_column = 8

    shader_definition = find_member_definition(DUPLICATE_SCOPE_SOURCE, shader_line, shader_column)

    assert shader_definition is not None
    assert shader_definition.struct_name == "InType"
    assert shader_definition.field_name == "x"
