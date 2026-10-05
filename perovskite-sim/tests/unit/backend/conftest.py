"""Backend tests own background-job cleanup; ordinary core tests stay import-light."""

import pytest


@pytest.fixture(autouse=True)
def _finish_backend_jobs(finish_background_jobs: None) -> None:
    """Wait for real jobs before the shared monkeypatch fixture is undone."""
