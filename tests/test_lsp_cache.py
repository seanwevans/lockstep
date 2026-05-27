from types import SimpleNamespace

from lockstep_compiler import lsp
from lockstep_compiler.errors import LockstepCompileError


def test_compile_context_for_lsp_preserves_failure_diagnostics_and_reuses_recovery_entities(
    monkeypatch,
):
    calls: list[str] = []

    def fake_compile(source: str, verbose: bool, semantic_validator=None):
        calls.append("compile")
        if semantic_validator is None:
            diagnostic = SimpleNamespace(
                severity="error",
                code="E_PARSE",
                message="bad token",
                line=3,
                column=4,
                hint="remove token",
                source_file="test.ls",
            )
            raise LockstepCompileError([diagnostic], diagnostics=[diagnostic])
        return SimpleNamespace(entities={"structs": [{"name": "Recovered", "fields": []}]})

    monkeypatch.setattr(lsp, "compile_lockstep", fake_compile)
    compiled = lsp.compile_context_for_lsp("struct bad {")

    assert calls == ["compile", "compile"]
    assert compiled.diagnostics[0]["code"] == "E_PARSE"
    assert compiled.entities["structs"][0]["name"] == "Recovered"


def test_build_analysis_context_reuses_single_compilation_for_multiple_queries(monkeypatch):
    compile_calls = 0

    def fake_compile_context_for_lsp(source: str):
        nonlocal compile_calls
        compile_calls += 1
        return lsp.CompiledLspContext(
            entities={
                "structs": [{"name": "Vec3", "fields": [{"name": "x", "type": "float"}]}],
                "shaders": [],
                "filters": [],
                "pure_functions": [],
            },
            diagnostics=[],
        )

    monkeypatch.setattr(lsp, "compile_context_for_lsp", fake_compile_context_for_lsp)
    source = "struct Vec3 { float x; };"

    context = lsp.build_analysis_context(source)
    _ = lsp.provide_hover_info(source, 0, 7, analysis_context=context)
    _ = lsp.find_definition_target(source, 0, 7, analysis_context=context)
    _ = lsp.provide_bind_completion_items(source, analysis_context=context)

    assert compile_calls == 1
