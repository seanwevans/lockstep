"""Tests for the resource limits applied to the simulator's reduction
subprocess (the boundary where generated native code executes)."""

from __future__ import annotations

import sys

import pytest

from lockstep_compiler import simulator


posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX resource limits are unavailable on Windows"
)


@posix_only
def test_resource_limits_returns_callable_on_posix() -> None:
    preexec = simulator._subprocess_resource_limits()
    assert callable(preexec)


@posix_only
def test_clamp_rlimit_lowers_soft_limit() -> None:
    import resource

    # RLIMIT_CORE is safe to mutate in-process: 0 just disables core dumps and
    # never terminates the interpreter (unlike RLIMIT_CPU / RLIMIT_AS).
    original_soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
    try:
        simulator._clamp_rlimit(resource.RLIMIT_CORE, 0)
        new_soft, new_hard = resource.getrlimit(resource.RLIMIT_CORE)
        assert new_soft == 0
        assert new_hard == hard
    finally:
        resource.setrlimit(resource.RLIMIT_CORE, (original_soft, hard))


@posix_only
def test_clamp_rlimit_never_raises_above_hard_limit() -> None:
    import resource

    original_soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
    try:
        # Ask for more than the hard limit; the soft limit must not exceed it.
        requested = (
            hard + 1 if hard != resource.RLIM_INFINITY else _SOME_LARGE_VALUE
        )
        simulator._clamp_rlimit(resource.RLIMIT_CORE, requested)
        new_soft, _ = resource.getrlimit(resource.RLIMIT_CORE)
        if hard != resource.RLIM_INFINITY:
            assert new_soft <= hard
    finally:
        resource.setrlimit(resource.RLIMIT_CORE, (original_soft, hard))


_SOME_LARGE_VALUE = 1 << 40
