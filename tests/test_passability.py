from beater.sr_memory import PassabilityStore


def test_passability_converges_with_success():
    store = PassabilityStore(class_weight=0.4, seed=0)
    for _ in range(20):
        store.update("grass", "tile-1", success=True)
    estimate = store.get_estimate("grass", "tile-1")
    assert estimate.blended > 0.8


def test_passability_reacts_to_failures():
    store = PassabilityStore(class_weight=0.5, seed=0)
    for _ in range(5):
        store.update("ledge", "tile-9", success=True)
    for _ in range(10):
        store.update("ledge", "tile-9", success=False)
    estimate = store.get_estimate("ledge", "tile-9")
    assert estimate.blended < 0.5
