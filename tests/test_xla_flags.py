"""XLA_FLAGS sanitization for macOS Metal / current XLA."""

import os

from darkhunter_sed.stellar_data import sanitize_xla_flags_env


def test_sanitize_removes_thunk_runtime_flag():
    os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"
    sanitize_xla_flags_env()
    assert "XLA_FLAGS" not in os.environ


def test_sanitize_strips_flag_but_keeps_others():
    os.environ["XLA_FLAGS"] = "--foo=bar --xla_cpu_use_thunk_runtime=false"
    sanitize_xla_flags_env()
    assert "XLA_FLAGS" not in os.environ


def test_sanitize_noop_when_clean():
    os.environ["XLA_FLAGS"] = "--foo=bar"
    sanitize_xla_flags_env()
    assert os.environ.get("XLA_FLAGS") == "--foo=bar"
