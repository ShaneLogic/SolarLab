"""Keep real API workers inside the lifetime of their patched test inputs."""

import pytest


@pytest.fixture(autouse=True)
def _finish_backend_jobs(finish_background_jobs: None) -> None:
    """Reuse the opt-in root fixture for the backend integration subtree."""
