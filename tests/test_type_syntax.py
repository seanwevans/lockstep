import pathlib
import sys
import textwrap

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lockstep_compiler import FrontendLimits, compile_lockstep
from lockstep_compiler.errors import LockstepCompileError


def test_composite_type_syntax_supports_arrays_and_generic_wrappers():
    source = """
struct Particle {
    float mass;
};

struct Payload {
    Particle[4] neighbors;
    vector<float,4> weights;
    matrix<vector<Particle,4>,4> transforms;
};

pure vector<Particle,4> passthrough(vector<Particle,4> value) {
    return value;
}

pipeline Stage {
    uniform matrix<vector<Particle,4>,4> frame;
    bind { }
}
"""

    result = compile_lockstep(source, verbose=False)

    assert all(diag.severity != "error" for diag in result.diagnostics)


def test_composite_type_syntax_rejects_unknown_nested_type_reference():
    source = """
pipeline Stage {
    uniform vector<MissingType,4> frame;
    bind { }
}
"""

    with pytest.raises(LockstepCompileError) as exc_info:
        compile_lockstep(source, verbose=False)

    assert exc_info.value.phase == "semantic"
    assert [diag.code for diag in exc_info.value.errors] == ["LCK310"]
    assert (
        "Unknown declared type 'MissingType' in 'vector<MissingType,4>'"
        in exc_info.value.errors[0].message
    )


def test_local_var_declaration_requires_explicit_type_annotation():
    source = """
shader Stage() {
    localValue;
}

pipeline Main {
    bind { }
}
"""

    with pytest.raises(LockstepCompileError) as exc_info:
        compile_lockstep(source, verbose=False)

    assert exc_info.value.phase == "parse"
    assert [diag.code for diag in exc_info.value.errors] == ["LCK001"]
    assert "no viable alternative" in exc_info.value.errors[0].message


def test_dependency_declarations_and_string_literals_compile_successfully(tmp_path):
    core_dir = tmp_path / "core"
    runtime_dir = tmp_path / "runtime"
    core_dir.mkdir()
    runtime_dir.mkdir()
    (core_dir / "math.lock").write_text("\n", encoding="utf-8")
    (runtime_dir / "platform.lock").write_text("\n", encoding="utf-8")

    source = """
import "core/math.lock";
#include "runtime/platform.lock";

pure string asset_label(string src) {
    string local = "asset://textures/noise";
    return local;
}

pipeline Main {
    uniform string title = "frame begin";
    bind { }
}
"""

    result = compile_lockstep(
        source,
        verbose=False,
        source_file=str(tmp_path / "main.lock"),
    )

    assert all(diag.severity != "error" for diag in result.diagnostics)
    assert any(uniform["type"] == "string" for uniform in result.entities["uniforms"])
    assert "private unnamed_addr constant" in result.llvm_ir
    assert "c\"asset://textures/noise\\00\"" in result.llvm_ir


def test_uint_and_double_are_supported_as_primitive_declared_types():
    source = """
pure uint advance(uint value) {
    uint next = value + uint(1);
    return next;
}

pure double amplify(double value) {
    double scaled = value * double(2.0);
    return scaled;
}

pipeline Main {
    uniform uint frame = uint(0);
    uniform double exposure = double(1.0);
    bind { }
}
"""

    result = compile_lockstep(source, verbose=False)

    assert all(diag.severity != "error" for diag in result.diagnostics)
    assert any(uniform["type"] == "uint" for uniform in result.entities["uniforms"])
    assert any(uniform["type"] == "double" for uniform in result.entities["uniforms"])
    
def test_import_dependency_file_is_prepended_and_types_are_resolved(tmp_path):
    dependency = tmp_path / "types.lock"
    dependency.write_text(
        textwrap.dedent(
            """
            struct Particle {
                float mass;
            };
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    source = f'''
import "{dependency.name}";

pipeline Main {{
    uniform Particle item;
    bind {{ }}
}}
'''

    result = compile_lockstep(source, verbose=False, source_file=str(tmp_path / "main.lock"))

    assert all(diag.severity != "error" for diag in result.diagnostics)
    assert any(struct["name"] == "Particle" for struct in result.entities["structs"])


def test_dependency_root_allows_relative_include_within_root(tmp_path, monkeypatch):
    monkeypatch.setattr("lockstep_compiler.compiler.emit_llvm_ir", lambda *_args, **_kwargs: "")
    monkeypatch.setattr("lockstep_compiler.compiler.emit_c_header", lambda *_args, **_kwargs: "")
    root = tmp_path / "project"
    deps = root / "deps"
    deps.mkdir(parents=True)
    (deps / "types.lock").write_text("struct Allowed { float x; };\n", encoding="utf-8")
    main_file = root / "main.lock"
    main_file.write_text(
        'import "deps/types.lock";\n\npipeline Main { uniform Allowed item; bind { } }\n',
        encoding="utf-8",
    )

    result = compile_lockstep(
        main_file.read_text(encoding="utf-8"),
        verbose=False,
        source_file=str(main_file),
        dependency_root=root,
    )

    assert all(diag.severity != "error" for diag in result.diagnostics)


def test_dependency_root_rejects_parent_traversal(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.lock"
    outside.write_text("struct Hidden { float x; };\n", encoding="utf-8")
    main_file = root / "main.lock"
    main_file.write_text('import "../outside.lock";\n', encoding="utf-8")

    with pytest.raises(LockstepCompileError) as exc_info:
        compile_lockstep(
            main_file.read_text(encoding="utf-8"),
            verbose=False,
            source_file=str(main_file),
            dependency_root=root,
        )

    assert [diag.code for diag in exc_info.value.errors] == ["LCK007"]
    assert exc_info.value.errors[0].source_file == str(main_file)
    assert exc_info.value.errors[0].line == 1


def test_dependency_root_rejects_absolute_include_outside_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.lock"
    outside.write_text("struct Hidden { float x; };\n", encoding="utf-8")
    main_file = root / "main.lock"
    main_file.write_text(f'import "{outside}";\n', encoding="utf-8")

    with pytest.raises(LockstepCompileError) as exc_info:
        compile_lockstep(
            main_file.read_text(encoding="utf-8"),
            verbose=False,
            source_file=str(main_file),
            dependency_root=root,
        )

    assert [diag.code for diag in exc_info.value.errors] == ["LCK007"]
    assert exc_info.value.errors[0].source_file == str(main_file)
    assert exc_info.value.errors[0].line == 1


def test_decode_dependency_path_tolerates_raw_backslashes():
    from lockstep_compiler.compiler import _decode_dependency_path

    # Absolute Windows paths contain raw backslashes (e.g. ``\\Users``) that are
    # not valid ``unicode_escape`` sequences. Decoding must not raise so the
    # dependency resolver can validate and reject the path with a diagnostic.
    windows_path = r"C:\Users\runner\AppData\outside.lock"
    assert _decode_dependency_path(windows_path) == windows_path
    # Well-formed escapes are still decoded.
    assert _decode_dependency_path(r"dir\\file.lock") == "dir\\file.lock"


def test_circular_imports_raise_compile_error(tmp_path):
    file_a = tmp_path / "a.lock"
    file_b = tmp_path / "b.lock"
    file_a.write_text('import "b.lock";\n', encoding="utf-8")
    file_b.write_text('import "a.lock";\n', encoding="utf-8")

    with pytest.raises(LockstepCompileError) as exc_info:
        compile_lockstep(file_a.read_text(encoding="utf-8"), verbose=False, source_file=str(file_a))

    assert exc_info.value.phase == "parse"
    assert [diag.code for diag in exc_info.value.errors] == ["LCK002"]
    assert "Circular dependency detected" in exc_info.value.errors[0].message


def test_malformed_utf8_dependency_raises_parse_style_error(tmp_path):
    dependency = tmp_path / "bad.lock"
    main_file = tmp_path / "main.lock"
    dependency.write_bytes(b"\xff\xfe\xfd")
    main_file.write_text(f'import "{dependency.name}";\n', encoding="utf-8")

    with pytest.raises(LockstepCompileError) as exc_info:
        compile_lockstep(
            main_file.read_text(encoding="utf-8"),
            verbose=False,
            source_file=str(main_file),
        )

    assert exc_info.value.phase == "parse"
    assert [diag.code for diag in exc_info.value.errors] == ["LCK002"]
    assert "Unable to parse dependency" in exc_info.value.errors[0].message
    assert "invalid UTF-8" in exc_info.value.errors[0].message


def test_dependency_file_count_limit_is_enforced(tmp_path):
    dep_a = tmp_path / "a.lock"
    dep_b = tmp_path / "b.lock"
    main = tmp_path / "main.lock"
    dep_a.write_text("pipeline A { bind { } }\n", encoding="utf-8")
    dep_b.write_text("pipeline B { bind { } }\n", encoding="utf-8")
    main.write_text(f'import "{dep_a.name}";\nimport "{dep_b.name}";\n', encoding="utf-8")

    with pytest.raises(LockstepCompileError) as exc_info:
        compile_lockstep(
            main.read_text(encoding="utf-8"),
            verbose=False,
            source_file=str(main),
            frontend_limits=FrontendLimits(max_dependency_files=1),
        )

    diagnostic = exc_info.value.errors[0]
    assert diagnostic.code == "LCK007"
    assert "(2 > 1)" in diagnostic.message
    assert diagnostic.line == 2


def test_dependency_total_bytes_limit_is_enforced(tmp_path):
    dep = tmp_path / "dep.lock"
    main = tmp_path / "main.lock"
    dep.write_text("pipeline Dependency { bind { } }\n", encoding="utf-8")
    main.write_text(f'import "{dep.name}";\n', encoding="utf-8")

    with pytest.raises(LockstepCompileError) as exc_info:
        compile_lockstep(
            main.read_text(encoding="utf-8"),
            verbose=False,
            source_file=str(main),
            frontend_limits=FrontendLimits(max_dependency_total_bytes=8),
        )

    diagnostic = exc_info.value.errors[0]
    assert diagnostic.code == "LCK007"
    assert "bytes > 8 bytes" in diagnostic.message
    assert diagnostic.line == 1


def test_dependency_depth_limit_is_enforced(tmp_path):
    dep2 = tmp_path / "dep2.lock"
    dep1 = tmp_path / "dep1.lock"
    main = tmp_path / "main.lock"
    dep2.write_text("pipeline Deep { bind { } }\n", encoding="utf-8")
    dep1.write_text(f'import "{dep2.name}";\n', encoding="utf-8")
    main.write_text(f'import "{dep1.name}";\n', encoding="utf-8")

    with pytest.raises(LockstepCompileError) as exc_info:
        compile_lockstep(
            main.read_text(encoding="utf-8"),
            verbose=False,
            source_file=str(main),
            frontend_limits=FrontendLimits(max_dependency_depth=1),
        )

    diagnostic = exc_info.value.errors[0]
    assert diagnostic.code == "LCK007"
    assert "(2 > 1)" in diagnostic.message
    assert diagnostic.line == 1
