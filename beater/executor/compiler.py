"""Executor that runs planlets and updates passability."""

from __future__ import annotations

from typing import Any, Dict, List

from beater.env import PyBoyEnv
from beater.sr_memory import PassabilityStore
from beater.types import Observation, Planlet

from .sprite_tracker import SpriteMovementDetector


class PlanletExecutor:
    def __init__(
        self,
        env: PyBoyEnv,
        store: PassabilityStore,
        detector: SpriteMovementDetector,
    ):
        self.env = env
        self.store = store
        self.detector = detector

    def run(self, planlet: Planlet) -> Observation:
        if planlet.kind == "NAVIGATE" and "steps" in planlet.args:
            return self._run_nav(planlet)
        return self.env.run_planlet(planlet)

    # ------------------------------------------------------------------ helpers
    def _run_nav(self, planlet: Planlet) -> Observation:
        steps: List[Dict[str, Any]] = planlet.args.get("steps", [])
        obs_prev = self.env.observe()
        results = []
        for step in steps:
            script = step["script"]
            obs_next = self.env.step_script(script)
            success, delta = self.detector.evaluate(obs_prev, obs_next)
            self.store.update(step["tile_class"], step["tile_id"], success)
            results.append(
                {
                    "tile_id": step["tile_id"],
                    "success": success,
                    "delta": delta,
                }
            )
            obs_prev = obs_next
        planlet.args["step_results"] = results
        planlet.args["nav_success"] = bool(results and results[-1]["success"])
        return obs_prev
