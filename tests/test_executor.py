import numpy as np

from beater.executor import PlanletExecutor, SpriteMovementDetector
from beater.sr_memory import PassabilityStore
from beater.types import Observation, Planlet, ScriptOp


class DummyEnv:
    def __init__(self):
        self._rgb = np.zeros((144, 160, 4), dtype=np.uint8)
        self._ram = np.zeros((65536,), dtype=np.uint8)
        self._step = 0
        self._sprite_col = 60
        self._render()

    def observe(self) -> Observation:
        return Observation(rgb=self._rgb.copy(), ram=self._ram.copy(), step_idx=self._step)

    def step_script(self, script):
        self._sprite_col = min(150, self._sprite_col + 5)
        self._render()
        self._step += 1
        return self.observe()

    def run_planlet(self, planlet: Planlet) -> Observation:
        return self.step_script(planlet.script)

    def _render(self):
        self._rgb.fill(0)
        self._rgb[70:74, self._sprite_col : self._sprite_col + 4, :] = 255


def test_executor_updates_passability():
    env = DummyEnv()
    store = PassabilityStore()
    detector = SpriteMovementDetector(diff_threshold=5.0, min_pixels=4, roi_ratio=0.8, success_px=0.5)
    executor = PlanletExecutor(env, store, detector)
    steps = [
        {
            "script": [ScriptOp(op="WAIT", frames=1)],
            "tile_id": "room:class_safe",
            "tile_class": "class_safe",
            "target_coord": (0, 0),
        }
    ]
    planlet = Planlet(
        id="nav-test",
        kind="NAVIGATE",
        args={"steps": steps, "path": [(0, 0), (0, 1)]},
        script=[ScriptOp(op="WAIT", frames=1)],
        timeout_steps=1,
    )
    executor.run(planlet)
    estimate = store.get_estimate("class_safe", "room:class_safe")
    assert estimate.blended > 0.5
    assert planlet.args.get("nav_success") is True
