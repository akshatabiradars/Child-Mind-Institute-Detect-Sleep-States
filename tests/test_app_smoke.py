"""Smoke test: the Streamlit app runs end-to-end with no uncaught exceptions.

Uses Streamlit's AppTest, which executes the whole script server-side (every tab,
not just what a screenshot would show). This catches version-incompatible widget
APIs and tab-level errors that a headless screenshot would miss.

Skipped automatically if the environment cannot load the model (e.g. a
scikit-learn version that can only read the legacy pickle is absent).
"""

import warnings

import pytest

warnings.simplefilter("ignore")

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_app_runs_without_exceptions():
    at = AppTest.from_file("app.py", default_timeout=180).run()
    # The default run loads Sample 1 and renders the Analyze tab plus the Model
    # tab's feature-importance chart.
    assert not at.exception, [str(e.value) for e in at.exception]
    # Sanity: the KPI metric cards rendered.
    assert len(at.metric) >= 4
