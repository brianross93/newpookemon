from beater.policy.nav_planner import NavPlanner, NavPlannerConfig
from beater.sr_memory import PassabilityStore


def test_nav_planner_prefers_high_passability_route():
    store = PassabilityStore(class_weight=0.5, seed=0)
    safe_tiles = [
        "room:class_safe_r0c0",
        "room:class_safe_r0c1",
        "room:class_safe_r1c0",
        "room:class_safe_r1c1",
    ]
    blocked_tile = "room:class_blocked_r0c1"
    # Mark safe tiles as traversable.
    for key in safe_tiles:
        tile_class = key.split(":", 1)[-1]
        for _ in range(5):
            store.update(tile_class, key, success=True)
    # Mark blocked tile as failing.
    tile_class = blocked_tile.split(":", 1)[-1]
    for _ in range(10):
        store.update(tile_class, blocked_tile, success=False)

    planner = NavPlanner(store, NavPlannerConfig(thompson_retries=4))
    grid = [
        ["room:class_safe_r0c0", blocked_tile],
        ["room:class_safe_r1c0", "room:class_safe_r1c1"],
    ]
    path = planner.plan(grid, start=(0, 0), goal=(1, 1))
    assert path, "planner should find a path"
    # Ensure the high-cost blocked tile is not visited in the interior of the path.
    assert (0, 1) not in path[1:-1]
