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
    items = provide_bind_completion_items(SOURCE, line=12, column=8)
    labels = [item["label"] for item in items]

    assert labels[0] == "dst=Integrate(src,dst,0.1);"
    assert "Integrate(...)" in labels
    assert items[0]["detail"] == "Bind route template"


def test_provide_bind_completion_items_omits_bind_routes_outside_bind_block():
    items = provide_bind_completion_items(SOURCE, line=4, column=8)
    labels = [item["label"] for item in items]

    assert "dst=Integrate(src,dst,0.1);" not in labels
    assert labels == ["Integrate(...)"]


def test_provide_bind_completion_items_deduplicates_and_ranks_categories():
    source = """
shader mix(in float value, out float out_value) { }
pure float mix(float a, float b) { return a; }

pipeline P {
    bind {
        out = mix(in_stream, out_stream);
    }
}
"""

    items = provide_bind_completion_items(source, line=6, column=10)

    assert [item["label"] for item in items] == [
        "out=mix(in_stream,out_stream);",
        "mix(...)",
    ]
    assert [item["detail"] for item in items] == [
        "Bind route template",
        "Shader/filter callable",
    ]
