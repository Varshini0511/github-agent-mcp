"""Pytest gate for the golden evaluation set.

These are LIVE evals -- they hit the GitHub and Groq APIs and cost time
and rate-limit budget, so they are skipped by default and never run in
the fast offline `pytest -q`. Enable them (locally or in CI) with:

    RUN_EVALS=1 pytest tests/test_evals.py

The offline test suite (test_github_client.py, test_write_ops.py) stays
the fast inner loop; this is the slower quality gate.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_EVALS") != "1",
    reason="Live evals hit real APIs; set RUN_EVALS=1 to run them.",
)


async def test_golden_set_all_pass():
    from evals.run import main

    exit_code = await main([])  # runs every case; 0 only if all pass
    assert exit_code == 0, "One or more golden eval cases failed (see report above)."
