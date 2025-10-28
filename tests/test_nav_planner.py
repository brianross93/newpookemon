from beater.policy.nav_planner import NavPlanner, NavPlannerConfig


def test_nav_planner_prefers_high_passability_route():
    planner = NavPlanner(NavPlannerConfig())
    passable = [
        [True, False],
        [True, True],
    ]
    path = planner.plan(passable, start=(0, 0), goal=(1, 1))
    assert path == [(0, 0), (1, 0), (1, 1)]
