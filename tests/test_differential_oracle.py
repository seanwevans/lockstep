"""Differential oracle: the simulator and compiled ``Lockstep_Tick`` must agree.

The v1.0 correctness bar in ``ROADMAP.md`` is that every valid program's
generated IR, compiled against the generated header, matches the simulator's
observable behavior. This module checks it three ways:

* **Random programs** (``test_generated_program_matches_simulator``): seeded
  programs from ``tests/differential/program_gen.py`` are compiled with clang,
  ticked once over full-capacity inputs, and compared with the simulator on
  every sink-stream row (exactly: the simulator models single-precision
  ``float`` and wrapping 32-bit ``int``) and every ``fold`` uniform (float
  sum/avg within a reassociation tolerance). The default is 80 seeds at SIMD
  widths 4 and 8; set ``LOCKSTEP_ORACLE_SEEDS=START:STOP`` to run a wider sweep,
  or use ``make oracle``.
* **Regressions**: a minimal program for each bug the oracle found, so each
  one stays fixed even if the generator's distribution drifts.
* **Known divergences** (strict ``xfail``): semantic gaps between the simulator
  and compiled code that need a language-design decision rather than a bug
  fix. The generator avoids these shapes; these tests document them and will
  fail (XPASS) once someone resolves one, prompting removal of the marker.

Native execution needs clang and an x86-64 Linux host (the IR hard-codes that
triple); elsewhere those tests skip. The simulator-only checks always run.
"""

from __future__ import annotations

import os

import pytest

from differential.native import toolchain_problem
from differential.oracle import case_from_source, check_case
from differential.program_gen import f32, generate_case
from lockstep_compiler.simulator import simulate_pipeline_source

_TOOLCHAIN_PROBLEM = toolchain_problem()
needs_native = pytest.mark.skipif(
    _TOOLCHAIN_PROBLEM is not None,
    reason=f"native execution unavailable: {_TOOLCHAIN_PROBLEM}",
)


def _seed_range() -> range:
    spec = os.environ.get("LOCKSTEP_ORACLE_SEEDS", "0:80")
    start, _, stop = spec.partition(":")
    return range(int(start), int(stop))


def _assert_agree(case, *, widths=(4, 8)) -> None:
    for width in widths:
        mismatches = check_case(case, target_width=width)
        assert not mismatches, (
            f"simulator and native disagree at width {width}:\n"
            + "\n".join(str(m) for m in mismatches[:20])
            + f"\n\n{case.source}"
        )


@needs_native
@pytest.mark.parametrize("seed", list(_seed_range()))
def test_generated_program_matches_simulator(seed: int) -> None:
    case = generate_case(seed)
    for width in (4, 8):
        mismatches = check_case(case, target_width=width)
        assert not mismatches, (
            f"seed {seed}, width {width}: {len(mismatches)} mismatch(es); "
            f"reproduce with `PYTHONPATH=.:tests python -m differential.oracle "
            f"{seed} --width {width}`\n"
            + "\n".join(str(m) for m in mismatches[:20])
            + f"\n\n{case.source}"
        )


def _rows(values, key="a"):
    return [{key: value} for value in values]


# --- Regressions: bugs the oracle found ------------------------------------


@needs_native
def test_bool_to_float_cast_is_one_not_minus_one() -> None:
    # ``(float) true`` lowered as ``sitofp i1`` produced -1.0.
    source = """
struct S { float a; bool b; };
shader K(in S src, out S dst) {
    dst.a = ((float) src.b) + ((float) (src.a > 0.0));
    dst.b = (bool) src.a;
}
pipeline P { stream<S, 9> s0; stream<S, 9> s1; bind { s1 = K(s0, s1); } }
"""
    rows = [{"a": float(i - 4), "b": i % 2 == 0} for i in range(9)]
    _assert_agree(case_from_source(source, {"s0": rows}, capacity=9))


@needs_native
def test_integer_division_and_remainder_truncate_toward_zero() -> None:
    # The simulator used Python's floor ``%`` and true ``/``; compiled code
    # uses C ``srem``/``sdiv``.
    source = """
struct S { int a; };
shader K(in S src, out S dst) { dst.a = (src.a % 4) * 100 + (src.a / 3); }
pipeline P { stream<S, 16> s0; stream<S, 16> s1; bind { s1 = K(s0, s1); } }
"""
    rows = _rows(range(-8, 8))
    _assert_agree(case_from_source(source, {"s0": rows}, capacity=16))


@needs_native
def test_float_arithmetic_rounds_to_single_precision() -> None:
    # The simulator computed ``float`` in double precision, so ``a - 0.3``
    # differed from compiled code in the last bits.
    source = """
struct S { float a; };
shader K(in S src, out S dst) { dst.a = (src.a - 0.3) * 0.1 + src.a / 3.0; }
pipeline P { stream<S, 8> s0; stream<S, 8> s1; bind { s1 = K(s0, s1); } }
"""
    rows = _rows([f32(1.0 / (i + 3)) for i in range(8)])
    _assert_agree(case_from_source(source, {"s0": rows}, capacity=8))


@needs_native
@pytest.mark.parametrize("capacity", [3, 16])  # scalar-tail path / vector path
def test_smoothstep_with_equal_edges_is_zero(capacity: int) -> None:
    # Both inlined lowerings divided by (edge1 - edge0) == 0 and clamped the
    # resulting inf to 1.0; the simulator and C intrinsic return 0.
    source = f"""
struct S {{ float a; }};
shader K(in S src, out S dst) {{ dst.a = smoothstep(src.a, src.a, 2.0); }}
pipeline P {{ stream<S, {capacity}> s0; stream<S, {capacity}> s1; bind {{ s1 = K(s0, s1); }} }}
"""
    rows = _rows([float(i) for i in range(capacity)])
    _assert_agree(case_from_source(source, {"s0": rows}, capacity=capacity))


@needs_native
def test_fold_over_unaligned_accumulator_does_not_fault() -> None:
    # The packed arena put ``acc`` at a byte offset that is not a multiple of
    # 32; the fold strip-mine loaded it with the vector type's natural
    # alignment (``movaps``) and segfaulted. The data-dependent filter keeps
    # the group off the fused path so the accumulator buffer is materialized.
    source = """
struct S { bool b; int v; };
shader K(in S src, out S dst, accum int acc) { dst.b = src.b; dst.v = src.v; acc = acc + src.v; }
filter F(in S src, out S dst) { dst.b = src.b; dst.v = src.v; return src.b; }
pipeline P {
    stream<S, 21> s0; stream<S, 21> s1; stream<S, 21> s2;
    accumulator<int> acc;
    bind {
        s1 = K(s0, s1, acc);
        s2 = F(s1, s2);
        uniform int total = fold max(acc);
    }
}
"""
    rows = [{"b": i % 3 == 0, "v": i * 7 - 60} for i in range(21)]
    _assert_agree(case_from_source(source, {"s0": rows}, capacity=21))


@needs_native
def test_fused_group_stores_an_in_place_sink() -> None:
    # ``s2 = K1(s1, s2); s2 = K2(s2, s2)`` fused into one group listed ``s2``
    # as an eliminated intermediate, so the final result was never stored.
    source = """
struct S { float a; };
shader K1(in S src, out S dst) { dst.a = src.a + 1.0; }
shader K2(in S src, out S dst) { dst.a = src.a * 2.0; }
pipeline P {
    stream<S, 11> s0; stream<S, 11> s1; stream<S, 11> s2;
    bind { s1 = K1(s0, s1); s2 = K1(s1, s2); s2 = K2(s2, s2); }
}
"""
    rows = _rows([float(i) for i in range(11)])
    _assert_agree(case_from_source(source, {"s0": rows}, capacity=11))


@needs_native
def test_fused_group_without_intermediates_lowers_every_route() -> None:
    # A group whose only "intermediate" is the in-place sink lowered just its
    # first route and silently dropped the rest.
    source = """
struct S { float a; };
shader K1(in S src, out S dst) { dst.a = src.a + 1.0; }
shader K2(in S src, out S dst) { dst.a = src.a * 2.0; }
pipeline P {
    stream<S, 9> s0; stream<S, 9> s1;
    bind { s1 = K1(s0, s1); s1 = K2(s1, s1); }
}
"""
    rows = _rows([float(i) for i in range(9)])
    _assert_agree(case_from_source(source, {"s0": rows}, capacity=9))


@needs_native
def test_in_place_stage_in_fused_scalar_tail_reads_its_input() -> None:
    # In a fused group's scalar tail, a group that *starts* with an in-place
    # stage (``s1 = K1(s1, s1)``; the fold is a fusion barrier after K0) read
    # K1's input from the uninitialized slot allocated for its own output.
    source = """
struct S { float a; };
shader K0(in S src, out S dst, accum float acc) { dst.a = src.a + 0.5; acc = acc + src.a; }
shader K1(in S src, out S dst) { dst.a = src.a * 3.0; }
shader K2(in S src, out S dst) { dst.a = src.a - 1.0; }
pipeline P {
    stream<S, 9> s0; stream<S, 9> s1; stream<S, 9> s2;
    accumulator<float> acc;
    bind {
        s1 = K0(s0, s1, acc);
        uniform float total = fold sum(acc);
        s1 = K1(s1, s1);
        s2 = K2(s1, s2);
    }
}
"""
    rows = _rows([float(i) for i in range(9)])
    _assert_agree(case_from_source(source, {"s0": rows}, capacity=9))


@needs_native
def test_min_max_sum_avg_are_usable_as_names() -> None:
    # ``min``/``max``/``sum``/``avg`` were lexer keywords (fold operators), so
    # the documented ``min``/``max`` intrinsics could not be called at all.
    source = """
struct S { float a; float b; };
shader K(in S src, out S dst, accum float sum) {
    float avg = (src.a + src.b) * 0.5;
    dst.a = min(src.a, src.b);
    dst.b = max(avg, 0.0);
    sum = sum + avg;
}
pipeline P {
    stream<S, 8> s0; stream<S, 8> s1; accumulator<float> sum;
    bind { s1 = K(s0, s1, sum); uniform float hi = fold max(sum); }
}
"""
    rows = [{"a": float(i), "b": float(7 - 2 * i)} for i in range(8)]
    _assert_agree(case_from_source(source, {"s0": rows}, capacity=8))


# --- Live row counts and fusing through dropping filters --------------------


@needs_native
@pytest.mark.parametrize("capacity", [3, 8, 21, 64])
def test_stage_after_dropping_filter_sees_only_kept_rows(capacity: int) -> None:
    # Compiled streams had no live row count: the stage after a filter ran over
    # the filter output's full capacity, stale tail included, so its folds saw
    # extra rows. Now the count is published and bounds every later stage. The
    # ``pure`` call keeps this on the per-stage path; the next test fuses it.
    source = f"""
struct S {{ float a; int b; }};
pure bool keep(float a) {{ return a > 3.0; }}
filter F(in S src, out S dst) {{ dst.a = src.a; dst.b = src.b; return keep(src.a); }}
shader K(in S src, out S dst, accum float acc, accum int n) {{
    dst.a = src.a + 1.0; dst.b = src.b * 2; acc = acc + src.a; n = n + 1;
}}
pipeline P {{
    stream<S, {capacity}> s0; stream<S, {capacity}> s1; stream<S, {capacity}> s2;
    accumulator<float> acc; accumulator<int> n;
    bind {{
        s1 = F(s0, s1);
        s2 = K(s1, s2, acc, n);
        uniform float mean = fold avg(acc);
        uniform float lo = fold min(acc);
        uniform int kept = fold sum(n);
    }}
}}
"""
    rows = [{"a": float(i % 7), "b": i} for i in range(capacity)]
    _assert_agree(case_from_source(source, {"s0": rows}, capacity=capacity))


@needs_native
@pytest.mark.parametrize("capacity", [3, 8, 21, 64])
def test_fused_group_compacts_through_dropping_filter(capacity: int) -> None:
    # Normalize -> DropLow -> Score fuses into one vector loop: the filter's
    # keep flag masks the lanes, the sink is compress-stored at a running
    # write index, and masked lanes contribute the fold identity.
    source = f"""
struct S {{ float a; int b; bool c; }};
shader Normalize(in S src, out S dst) {{ dst.a = src.a * 0.5; dst.b = src.b + 1; dst.c = src.b > 3; }}
filter DropLow(in S src, out S dst) {{ dst.a = src.a; dst.b = src.b; dst.c = src.c; return src.a > 1.0; }}
shader Score(in S src, out S dst, accum float total, accum int hi) {{
    dst.a = src.a * 3.0; dst.b = src.b - 2; dst.c = !src.c;
    total = total + src.a; hi = hi + 1;
}}
pipeline P {{
    stream<S, {capacity}> s0; stream<S, {capacity}> s1; stream<S, {capacity}> s2;
    stream<S, {capacity}> s3; accumulator<float> total; accumulator<int> hi;
    bind {{
        s1 = Normalize(s0, s1);
        s2 = DropLow(s1, s2);
        s3 = Score(s2, s3, total, hi);
        uniform float sum_total = fold sum(total);
        uniform int kept = fold sum(hi);
    }}
}}
"""
    rows = [{"a": float(i % 5), "b": i, "c": False} for i in range(capacity)]
    case = case_from_source(source, {"s0": rows}, capacity=capacity)
    _assert_agree(case, widths=(4, 8, 16))


@needs_native
def test_every_row_dropped_folds_to_identity() -> None:
    source = """
struct S { float a; };
filter F(in S src, out S dst) { dst.a = src.a; return src.a > 100.0; }
shader K(in S src, out S dst, accum float acc) { dst.a = src.a; acc = acc + src.a; }
pipeline P {
    stream<S, 9> s0; stream<S, 9> s1; stream<S, 9> s2; accumulator<float> acc;
    bind {
        s1 = F(s0, s1); s2 = K(s1, s2, acc);
        uniform float lo = fold min(acc); uniform float mean = fold avg(acc);
    }
}
"""
    rows = _rows([float(i) for i in range(9)])
    _assert_agree(case_from_source(source, {"s0": rows}, capacity=9))


@needs_native
def test_short_filtered_input_reads_as_zero_past_its_count() -> None:
    # Two filters keep different numbers of rows; a stage reading both runs to
    # the longer count and sees zeros past the shorter one's end, like the
    # simulator's padding.
    source = """
struct S { float a; };
filter Big(in S src, out S dst) { dst.a = src.a; return src.a > 2.0; }
filter Odd(in S src, out S dst) { dst.a = src.a; return src.a > 6.0; }
shader Pair(in S x, in S y, out S dst, accum float acc) { dst.a = x.a + y.a * 10.0; acc = acc + y.a; }
pipeline P {
    stream<S, 12> s0; stream<S, 12> big; stream<S, 12> odd; stream<S, 12> pairs;
    accumulator<float> acc;
    bind {
        big = Big(s0, big); odd = Odd(s0, odd); pairs = Pair(big, odd, pairs, acc);
        uniform float total = fold sum(acc);
    }
}
"""
    rows = _rows([float(i) for i in range(12)])
    _assert_agree(case_from_source(source, {"s0": rows}, capacity=12))


@needs_native
def test_fold_covers_the_longest_accumulator_writer() -> None:
    # An accumulator written before and after a dropping filter holds the
    # longer writer's rows; the fold must not stop at the later, shorter one.
    source = """
struct S { float a; };
shader A(in S src, out S dst, accum float acc) { dst.a = src.a; acc = acc + src.a; }
filter F(in S src, out S dst) { dst.a = src.a; return src.a > 5.0; }
pipeline P {
    stream<S, 10> s0; stream<S, 10> s1; stream<S, 10> s2; stream<S, 10> s3;
    accumulator<float> acc;
    bind {
        s1 = A(s0, s1, acc); s2 = F(s1, s2); s3 = A(s2, s3, acc);
        uniform float total = fold sum(acc);
    }
}
"""
    rows = _rows([float(i) for i in range(10)])
    _assert_agree(case_from_source(source, {"s0": rows}, capacity=10))


# --- Simulator numeric model (no toolchain needed) --------------------------


def test_simulator_models_32_bit_integers() -> None:
    source = """
struct S { int a; };
shader K(in S src, out S dst) { dst.a = (src.a * 65536) * 65536 + (src.a % 3) + (src.a / 2); }
pipeline P { stream<S, 2> s0; stream<S, 2> s1; bind { s1 = K(s0, s1); } }
"""
    result = simulate_pipeline_source(source, stream_inputs={"s0": _rows([-5, 7])})
    # (a << 32) wraps to 0; -5 % 3 == -2 and -5 / 2 == -2 in C.
    assert [row["a"] for row in result["streams"]["s1"]] == [-4, 4]


def test_simulator_models_single_precision_float() -> None:
    source = """
struct S { float a; };
shader K(in S src, out S dst) { dst.a = src.a + 0.1; }
pipeline P { stream<S, 1> s0; stream<S, 1> s1; bind { s1 = K(s0, s1); } }
"""
    result = simulate_pipeline_source(source, stream_inputs={"s0": _rows([0.2])})
    assert result["streams"]["s1"][0]["a"] == f32(f32(0.2) + f32(0.1))
    assert result["streams"]["s1"][0]["a"] != 0.2 + 0.1


def test_simulator_out_row_starts_from_target_not_input() -> None:
    # ``out`` used to start as a copy of the *input* row, so an output of a
    # different struct type carried the input's fields.
    source = """
struct A { float x; };
struct B { float y; };
shader K(in A src, out B dst) { dst.y = src.x; }
pipeline P { stream<A, 1> s0; stream<B, 1> s1; bind { s1 = K(s0, s1); } }
"""
    result = simulate_pipeline_source(source, stream_inputs={"s0": [{"x": 2.0}]})
    assert result["streams"]["s1"] == [{"y": 2.0}]


# --- Known divergences needing a design decision ----------------------------


@needs_native
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Saturation semantics differ: the simulator keeps the *last* `capacity` "
        "rows (FIFO windowing), compiled code writes rows in order and keeps "
        "overwriting the final 'trash can' row, as README section 2 describes."
    ),
)
def test_known_divergence_route_into_smaller_stream() -> None:
    source = """
struct S { float a; };
shader K(in S src, out S dst) { dst.a = src.a * 2.0; }
pipeline P { stream<S, 8> s0; stream<S, 4> s1; bind { s1 = K(s0, s1); } }
"""
    rows = _rows([float(i) for i in range(8)])
    _assert_agree(case_from_source(source, {"s0": rows}, capacity=8), widths=(8,))


@needs_native
@pytest.mark.xfail(
    strict=True,
    reason=(
        "An accumulator written by two kernels is one slot per row in compiled "
        "code (the writers' contributions add up per row) but a flat list of "
        "contributions in the simulator, so avg/min/max disagree (sum agrees)."
    ),
)
def test_known_divergence_accumulator_with_two_writers() -> None:
    source = """
struct S { float a; };
shader A(in S src, out S dst, accum float acc) { dst.a = src.a; acc = acc + src.a; }
pipeline P {
    stream<S, 8> s0; stream<S, 8> s1; stream<S, 8> s2; accumulator<float> acc;
    bind { s1 = A(s0, s1, acc); s2 = A(s1, s2, acc); uniform float mean = fold avg(acc); }
}
"""
    rows = _rows([float(i) for i in range(8)])
    _assert_agree(case_from_source(source, {"s0": rows}, capacity=8), widths=(8,))
