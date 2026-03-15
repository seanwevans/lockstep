import io

from lockstep_compiler.cli import run_cli


def test_run_cli_format_outputs_formatted_source_without_compiling():
    called = {"compiler": False}

    def fake_compiler(_source):
        called["compiler"] = True

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = run_cli(
        ["--format"],
        stdin=io.StringIO("pipeline P{stream<float,4> in0;}"),
        stdout=stdout,
        stderr=stderr,
        compiler=fake_compiler,
    )

    assert exit_code == 0
    assert called["compiler"] is False
    assert stdout.getvalue() == "pipeline P {\n    stream<float,4> in0;\n}\n"
    assert stderr.getvalue() == "warning: --format currently strips comments from output.\n"
