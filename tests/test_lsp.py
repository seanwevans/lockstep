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


def test_build_struct_member_index_ignores_braces_in_comments_and_strings():
    source = '''
struct Payload {
    float x; // this should not close the struct }
    float y;
    float z;
};

shader Use(in Payload p, out Payload out_p) {
    // string-like text with braces: "{" "}"
    out_p.z = p.z;
}
'''

    index = build_struct_member_index(source)

    assert set(index["Payload"]) == {"x", "y", "z"}
    assert index["Payload"]["z"].line == 4


def test_build_struct_member_index_handles_incomplete_struct_block_with_fallback_scanner():
    source = """
struct Good {
    float ok;
};

struct Broken {
    float missing;
"""

    index = build_struct_member_index(source)

    assert "Good" in index
    assert index["Good"]["ok"].line == 2
    assert "Broken" not in index
