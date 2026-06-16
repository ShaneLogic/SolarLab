from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.seeds import SEED_NEGATIVE_RESULTS, seed_negative_results


def test_seeds_cover_known_refuted_approaches():
    approaches = " ".join(n.approach.lower() for n in SEED_NEGATIVE_RESULTS)
    for needle in ["dos-cap", "bbd", "1.40 v_bi", "two-sided", "shared-occupancy"]:
        assert needle in approaches


def test_seed_is_idempotent(tmp_path):
    led = Ledger(root=tmp_path)
    seed_negative_results(led)
    n_first = len(led.negatives)
    seed_negative_results(led)              # second call must not duplicate
    assert len(led.negatives) == n_first
    assert led.is_refuted("DOS-cap projection target") is True
