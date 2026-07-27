from lockstep_compiler import compile_lockstep
from lockstep_compiler.lsp import build_analysis_context, provide_hover_info
from lockstep_compiler.models import SemanticStructField
from lockstep_compiler.type_analysis import (
    build_struct_field_type_index,
    parse_type_name,
    validate_type_name,
)


def test_build_struct_field_type_index_accepts_strings_and_field_objects():
    from_strings = build_struct_field_type_index({"Vec": {"x": "float"}})
    from_fields = build_struct_field_type_index(
        {"Vec": {"x": SemanticStructField(name="x", declared_type="float")}}
    )

    assert from_strings == {"Vec": {"x": "float"}}
    assert from_fields == {"Vec": {"x": "float"}}


def test_parse_and_validate_type_name_shared_helpers():
    parsed, index = parse_type_name("matrix<vector<float,4>,4>")
    assert parsed is not None
    assert index == len("matrix<vector<float,4>,4>")

    valid, issues = validate_type_name("vector<Missing,4>", {"int", "float", "vector"})
    assert not valid
    assert [issue.referenced_type for issue in issues] == ["Missing"]


def test_compile_and_lsp_share_struct_field_type_resolution():
    source = """
struct Particle { float mass; };

shader Copy(in Particle src, out Particle dst) {
    dst.mass = src.mass;
}

pipeline P {
    stream<Particle, 4> src;
    stream<Particle, 4> dst;
    bind { dst = Copy(src, dst); }
}
"""
    result = compile_lockstep(source, verbose=False)
    analysis = build_analysis_context(source)

    struct_fields = next(s for s in result.entities["structs"] if s["name"] == "Particle")["fields"]
    compile_type = next(f["type"] for f in struct_fields if f["name"] == "mass")

    line = next(i for i, text in enumerate(source.splitlines()) if "src.mass" in text)
    hover = provide_hover_info(source, line, source.splitlines()[line].index("mass") + 1, analysis_context=analysis)

    assert compile_type == "float"
    assert hover == "(field) `Particle.mass: float`"
