import io
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lockstep_compiler.cli import run_cli


def test_run_cli_passes_library_sources_to_compiler(tmp_path):
    library = tmp_path / "math.lock"
    library.write_text("pure float identity(float v) { return v; }", encoding="utf-8")

    captured = {}

    def fake_compiler(source, *, verbose=False, library_sources=None):
        captured["source"] = source
        captured["verbose"] = verbose
        captured["library_sources"] = library_sources
        return {"streams": [], "accumulators": [], "uniforms": [], "shaders": [], "filters": [], "bind_routes": []}

    exit_code = run_cli(
        ["--lib", str(library), "--simulate"],
        stdin=io.StringIO("pipeline Main { bind { } }"),
        stdout=io.StringIO(),
        compiler=fake_compiler,
    )

    assert exit_code == 0
    assert captured["source"] == "pipeline Main { bind { } }"
    assert captured["verbose"] is False
    assert captured["library_sources"] == ["pure float identity(float v) { return v; }"]


def test_run_cli_reports_missing_library_file(tmp_path):
    missing = tmp_path / "missing.lock"
    stderr = io.StringIO()

    exit_code = run_cli(
        ["--lib", str(missing)],
        stdin=io.StringIO("pipeline Main { bind { } }"),
        stdout=io.StringIO(),
        stderr=stderr,
        compiler=lambda source: source,
    )

    assert exit_code == 1
    assert f"Unable to read '{missing}': file not found." in stderr.getvalue()
