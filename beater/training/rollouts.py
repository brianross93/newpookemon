"""Environment-grounded rollout collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from torch.distributions import Categorical

from beater.env import PyBoyEnv
from beater.executor import NavPlanletBuilder, PlanletExecutor, SpriteMovementDetector
from beater.perception import Perception, TileDescriptor
from beater.policy import (
    AffordancePrior,
    Controller,
    ControllerState,
    GoalManager,
    NavPlanner,
    OptionBank,
    PlanHead,
)
from beater.brains import GPTBrain, GoalSuggestion
from beater.sr_memory import GraphOp, PassabilityStore
from beater.training.replay import RolloutBuffer, RolloutStep
import logging
from beater.objectives import ObjectiveEngine
from beater.utils.detectors import detect_menu_open, infer_menu_flags, scene_change_delta
from beater.utils.maps import ascii_tile_map
from beater.types import Observation

RewardFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
GRAPH_OPS: tuple[GraphOp, ...] = tuple(GraphOp)


@dataclass(slots=True)
class EnvActor:
    env: PyBoyEnv
    executor: PlanletExecutor
    controller_state: ControllerState
    last_obs: Observation
    context_id: str
    brain_directive: Optional[GoalSuggestion] = None


class GroundedRolloutCollector:
    def __init__(
        self,
        controller: Controller,
        perception: Perception,
        tile_descriptor: TileDescriptor,
        nav_planner: NavPlanner,
        nav_builder: NavPlanletBuilder,
        plan_head: PlanHead,
        affordance: AffordancePrior,
        option_bank: OptionBank,
        goal_manager: GoalManager,
        brain: Optional[GPTBrain],
        brain_candidate_preview: int,
        pass_store: PassabilityStore,
        reward_fn: RewardFn,
        env_factory: Callable[[], PyBoyEnv],
        movement_detector_factory: Callable[[], SpriteMovementDetector],
        num_actors: int,
        steps_per_actor: int,
        objective: Optional[ObjectiveEngine] = None,
    ):
        self.controller = controller
        self.perception = perception
        self.tile_descriptor = tile_descriptor
        self.nav_planner = nav_planner
        self.nav_builder = nav_builder
        self.plan_head = plan_head
        self.skill_vocab = list(plan_head.config.skill_vocab)
        self.affordance = affordance
        self.option_bank = option_bank
        self.goal_manager = goal_manager
        self.brain = brain
        self.brain_candidates = max(1, brain_candidate_preview)
        self.pass_store = pass_store
        self.reward_fn = reward_fn
        self.env_factory = env_factory
        self.movement_detector_factory = movement_detector_factory
        self.num_actors = max(1, num_actors)
        self.steps_per_actor = max(1, steps_per_actor)
        self.objective = objective
        self._logger = logging.getLogger("beater")

    def collect(self) -> RolloutBuffer:
        buffer = RolloutBuffer()
        actors = [self._init_actor(idx) for idx in range(self.num_actors)]
        try:
            for _ in range(self.steps_per_actor):
                for actor in actors:
                    self._step_actor(actor, buffer)
        finally:
            for actor in actors:
                actor.env.close()
                self.perception.reset(actor.context_id)
        return buffer

    # ------------------------------------------------------------------ helpers
    def _init_actor(self, idx: int) -> EnvActor:
        env = self.env_factory()
        executor = PlanletExecutor(env, self.pass_store, self.movement_detector_factory())
        obs = env.observe()
        state = self.controller.init_state(batch_size=1, device=self._device)
        context_id = f"actor_{idx}"
        self.perception.reset(context_id)
        return EnvActor(
            env=env,
            executor=executor,
            controller_state=state,
            last_obs=obs,
            context_id=context_id,
        )

    def _step_actor(self, actor: EnvActor, buffer: RolloutBuffer) -> None:
        rgb_tensor = torch.from_numpy(actor.last_obs.rgb).permute(2, 0, 1).unsqueeze(0)
        ram_tensor = torch.from_numpy(actor.last_obs.ram).unsqueeze(0)
        with torch.no_grad():
            z, _ = self.perception(rgb_tensor, ram_tensor, context=actor.context_id)
            ctrl_out = self.controller(z, actor.controller_state)
            affordance_logits = self.affordance(z)
            gate_dist = Categorical(logits=ctrl_out.gate_logits)
            skill_logits = ctrl_out.skill_logits + affordance_logits
            skill_dist = Categorical(logits=skill_logits)
            gate_action = gate_dist.sample()
            skill_action = skill_dist.sample()
            log_prob = gate_dist.log_prob(gate_action) + skill_dist.log_prob(skill_action)
            gate_idx = int(gate_action.item())
            gate_op = GRAPH_OPS[gate_idx % len(GRAPH_OPS)]
            tile_out = self.tile_descriptor(rgb_tensor)
            tile_keys = self.tile_descriptor.tile_keys(tile_out.class_ids, tile_out.grid_shape)
            tile_grid = self.tile_descriptor.reshape_tokens(tile_keys[0], tile_out.grid_shape)
            candidates = self.goal_manager.peek_candidates(
                tile_out.grid_shape, self.brain_candidates
            )
            # Build candidate metadata for the brain (passability + fail counts)
            cand_meta = []
            for (r, c) in candidates:
                key = tile_grid[r][c]
                cls = key.split(":", 1)[-1]
                est = self.pass_store.get_estimate(cls, key)
                cand_meta.append({
                    "passability": est.blended,
                    "fail_count": self.goal_manager.get_fail_count((r, c)),
                    "portal_score": 0.0,
                })
            ascii_map = ascii_tile_map(tile_grid, self.goal_manager.player_anchor(tile_out.grid_shape), candidates)
            menu_active = detect_menu_open(actor.last_obs.ram, tile_grid)
            brain_goal = None
            brain_skill = None
            brain_notes = None
            if self.brain and candidates:
                directive = actor.brain_directive
                if directive is None and gate_op == GraphOp.HALT:
                    directive = self.brain.suggest_goal(
                        tile_grid,
                        candidates,
                        {
                            "actor": actor.context_id,
                            "step": actor.last_obs.step_idx,
                            "ascii_map": ascii_map,
                            "candidate_meta": cand_meta,
                        },
                    )
                    actor.brain_directive = directive
                if directive:
                    if not getattr(directive, "objective_spec", None):
                        self._logger.warning("Dropping GPT directive without objective_spec (actor=%s)", actor.context_id)
                        directive = None
                        actor.brain_directive = None
                    else:
                        brain_goal = directive.resolve_goal(candidates)
                        brain_skill = directive.skill
                        brain_notes = directive.reasoning
                        # Optionally update objective spec
                        if self.objective:
                            self.objective.set_spec(directive.objective_spec, actor.last_obs.step_idx)
                            try:
                                self._logger.info("Objective updated (rollouts): %s", self.objective.summary())
                            except Exception:
                                pass
            preferred_kind = brain_skill if brain_skill in self.skill_vocab else self.option_bank.suggest()
            if preferred_kind is None:
                preferred_kind = "NAVIGATE"
            planlet = self.plan_head.decode(
                skill_logits,
                ctrl_out.timeout_steps,
                skill_action=skill_action,
                preferred_kind=preferred_kind,
            )
            goal = None
            nav_path = []
            if planlet.kind == "NAVIGATE":
                player_anchor = self.goal_manager.player_anchor(tile_out.grid_shape)
                goal = self.goal_manager.next_goal(
                    tile_out.grid_shape,
                    preferred=planlet.kind,
                    goal_override=brain_goal,
                )
                nav_path = self.nav_planner.plan(tile_grid, start=player_anchor, goal=goal)
                if len(nav_path) > 1:
                    planlet = self.nav_builder.from_path(nav_path, tile_grid, goal=goal)
                else:
                    planlet.args["goal"] = goal
                    planlet.args["nav_success"] = False
            if brain_notes:
                planlet.args.setdefault("brain_notes", brain_notes)
            prev_obs = actor.last_obs
            next_obs = actor.executor.run(planlet)
            reward = self.reward_fn(gate_action, skill_action)
            # Add objective-shaped reward if configured
            if self.objective is not None:
                moved, delta = actor.executor.detector.evaluate(prev_obs, next_obs)
                # Scene change via tile histograms (before vs after)
                next_rgb_tensor = torch.from_numpy(next_obs.rgb).permute(2, 0, 1).unsqueeze(0)
                tile_out_next = self.tile_descriptor(next_rgb_tensor)
                sc_delta = scene_change_delta(tile_out.class_ids, tile_out_next.class_ids, tile_out.grid_shape)
                menu_prog, name_commit = infer_menu_flags(planlet, sc_delta)
                shaped = torch.zeros(1)
                skip_menu_reward = menu_active and planlet.kind not in ("MENU_SEQUENCE", "INTERACT")
                if not skip_menu_reward:
                    shaped = self.objective.reward(
                        gate_action=gate_action,
                        skill_action=skill_action,
                        nav_success=bool(planlet.args.get("nav_success", False)) if planlet.kind == "NAVIGATE" else False,
                        sprite_delta=float(delta),
                        scene_change=float(sc_delta),
                        menu_progress=float(menu_prog),
                        name_committed=float(name_commit),
                    )
                reward = reward + shaped
        buffer.add(
            RolloutStep(
                latent=z.squeeze(0).detach(),
                hidden=actor.controller_state.hidden.squeeze(0).detach(),
                gate_action=gate_action.squeeze(0).detach(),
                skill_action=skill_action.squeeze(0).detach(),
                old_log_prob=log_prob.squeeze(0).detach(),
                reward=reward.squeeze(0).detach(),
            )
        )
        actor.controller_state = ctrl_out.state
        actor.last_obs = next_obs
        if planlet.kind == "NAVIGATE" and goal is not None:
            nav_success = bool(planlet.args.get("nav_success", False))
            self.goal_manager.feedback(goal, nav_success)
        if self.brain and actor.brain_directive:
            if planlet.kind != "NAVIGATE" or planlet.args.get("nav_success", False):
                actor.brain_directive = None
        self.option_bank.record(planlet.kind)

    @property
    def _device(self) -> torch.device:
        return next(self.controller.parameters()).device
