"""The existing ninth linter rule includes the new persistence adapter edge."""

import pytest

from test_package_boundaries import (
    test_actual_linter_rejects_current_and_future_upward_dependencies as assert_linter_rejects,
)


@pytest.mark.parametrize("indirect", [False, True])
def test_persistence_contract_cannot_import_its_storage_adapter(tmp_path, indirect):
    assert_linter_rejects(tmp_path, "solarlab.contracts", "solarlab.io", indirect)
