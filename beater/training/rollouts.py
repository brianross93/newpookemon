"""Environment-grounded rollout collection."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import logging
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
from beater.sr_memory import GraphOp
from beater.training.replay import RolloutBuffer, RolloutStep
from beater.objectives import ObjectiveEngine
from beater.utils.detectors import detect_menu_open, infer_menu_flags, scene_change_delta
from beater.utils.map_metadata import MapMetadataLoader
from beater.utils.maps import ascii_tile_map
from beater.utils.ram_map import (
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    W_TILE_MAP_ADDR,
    boundary_exit_candidates,
    decode_bag_items,
    decode_party,
    passable_mask,
    read_tile_map,
    tokens_from_tile_map,
)
from beater.types import Observation

RewardFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
GRAPH_OPS: tuple[GraphOp, ...] = tuple(GraphOp)
MAP_GROUP_ADDR = 0xD35F
MAP_ID_ADDR = 0xD35E
POKERED_ROOT = Path(__file__).resolve().parent.parent / "__tmp_pokered"
MAP_METADATA = MapMetadataLoader(POKERED_ROOT)


def _map_tuple_from_ram(ram: Optional["numpy.ndarray"]) -> Optional[Tuple[int, int]]:
    if ram is None:
        return None
    if len(ram) <= MAP_GROUP_ADDR:
        return None
    return (int(ram[MAP_GROUP_ADDR]), int(ram[MAP_ID_ADDR]))


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
        executor = PlanletExecutor(env, self.movement_detector_factory())
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
            grid_shape = tile_out.grid_shape
            tile_grid: List[List[str]] = []
            passable_grid: List[List[bool]] = []
            tile_map: Optional[List[List[int]]] = None
            ram_data = actor.last_obs.ram
            ram_ready = ram_data is not None and len(ram_data) >= (
                W_TILE_MAP_ADDR + SCREEN_HEIGHT * SCREEN_WIDTH
            )
            if ram_ready:
                try:
                    tile_map = read_tile_map(ram_data)
                    tile_grid = tokens_from_tile_map(tile_map)
                except Exception as exc:
                    self._logger.debug("Actor %s failed to read tile map: %s", actor.context_id, exc)
                    tile_map = None
            if not tile_grid:
                tile_keys = self.tile_descriptor.tile_keys(tile_out.class_ids, grid_shape)
                tile_grid = self.tile_descriptor.reshape_tokens(tile_keys[0], grid_shape)
            if tile_map is not None:
                map_tuple = _map_tuple_from_ram(ram_data)
                collisions = MAP_METADATA.collision_ids(map_tuple[1]) if map_tuple else set()
                passable_grid = passable_mask(tile_map, collisions)
            else:
                passable_grid = [[True for _ in range(grid_shape[1])] for _ in range(grid_shape[0])]

            player_anchor = self.goal_manager.player_anchor(grid_shape)
            boundary_targets = boundary_exit_candidates(
                passable_grid, player_anchor, limit=max(4, self.brain_candidates * 2)
            )
            candidates = list(self.goal_manager.peek_candidates(grid_shape, self.brain_candidates))
            for coord in boundary_targets:
                if coord not in candidates:
                    candidates.insert(0, coord)

            cand_meta = []
            for (r, c) in candidates:
                passable = 0 <= r < len(passable_grid) and 0 <= c < len(passable_grid[r]) and passable_grid[r][c]
                cand_meta.append(
                    {
                        "passability": 1.0 if passable else 0.0,
                        "fail_count": self.goal_manager.get_fail_count((r, c)),
                        "boundary_hint": (r, c) in boundary_targets,
                    }
                )

            ascii_map = ascii_tile_map(tile_grid, player_anchor, candidates)
            menu_active = detect_menu_open(actor.last_obs.ram, tile_grid)

            brain_goal = None
            brain_skill: Optional[str] = None
            brain_notes: Optional[str] = None
            if self.brain and candidates:
                directive = actor.brain_directive
                if directive is None and gate_op == GraphOp.HALT:
                    inventory_summary = decode_bag_items(actor.last_obs.ram)
                    party_summary = decode_party(actor.last_obs.ram)
                    halt_context = {
                        "actor": actor.context_id,
                        "step": actor.last_obs.step_idx,
                        "ascii_map": ascii_map,
                        "candidate_meta": cand_meta,
                    }
                    if inventory_summary:
                        halt_context["inventory"] = inventory_summary[:10]
                    if party_summary:
                        halt_context["party"] = party_summary
                    directive = self.brain.suggest_goal(
                        tile_grid,
                        candidates,
                        halt_context,
                    )
                    actor.brain_directive = directive
                if directive:
                    if not getattr(directive, "objective_spec", None):
                        self._logger.warning(
                            "Dropping GPT directive without objective_spec (actor=%s)",
                            actor.context_id,
                        )
                        directive = None
                        actor.brain_directive = None
                    else:
                        brain_goal = directive.resolve_goal(candidates)
                        brain_skill = directive.skill
                        brain_notes = directive.reasoning
                        if self.objective:
                            self.objective.set_spec(directive.objective_spec, actor.last_obs.step_idx)
                            try:
                                self._logger.info(
                                    "Objective updated (rollouts): %s", self.objective.summary()
                                )
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

            goal: Optional[Tuple[int, int]] = None
            nav_path: List[Tuple[int, int]] = []
            if planlet.kind == "NAVIGATE":
                goal = self.goal_manager.next_goal(
                    grid_shape,
                    preferred=planlet.kind,
                    goal_override=brain_goal,
                )
                if goal is not None:
                    nav_path = self.nav_planner.plan(passable_grid, start=player_anchor, goal=goal)
                if nav_path and len(nav_path) > 1:
                    planlet = self.nav_builder.from_path(nav_path, tile_grid, goal=goal)
                else:
                    if isinstance(planlet.args, dict):
                        planlet.args["goal"] = goal
                        planlet.args["nav_success"] = False

            if brain_notes and isinstance(planlet.args, dict):
                planlet.args.setdefault("brain_notes", brain_notes)

        prev_obs = actor.last_obs
        map_before = _map_tuple_from_ram(prev_obs.ram)
        next_obs = actor.executor.run(planlet)
        map_after = _map_tuple_from_ram(next_obs.ram)
        portal_triggered = bool(
            map_before is not None and map_after is not None and map_after != map_before
        )
        interact_completed = 0.0
        if isinstance(planlet.args, dict) and planlet.kind == "MENU_SEQUENCE":
            if planlet.args.get("known_entity"):
                interact_completed = 1.0

        reward = self.reward_fn(gate_action, skill_action)
        if self.objective is not None:
            moved, delta = actor.executor.detector.evaluate(prev_obs, next_obs)
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
                    nav_success=bool(planlet.args.get("nav_success", False)) if isinstance(planlet.args, dict) else False,
                    sprite_delta=float(delta),
                    scene_change=float(sc_delta),
                    menu_progress=float(menu_prog),
                    name_committed=float(name_commit),
                    portal_triggered=portal_triggered,
                    interact_completed=interact_completed,
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
            nav_success = bool(planlet.args.get("nav_success", False)) if isinstance(planlet.args, dict) else False
            self.goal_manager.feedback(goal, nav_success)

        if self.brain and actor.brain_directive:
            if planlet.kind != "NAVIGATE" or planlet.args.get("nav_success", False):
                actor.brain_directive = None

        self.option_bank.record(planlet.kind)

    @property
    def _device(self) -> torch.device:
        return next(self.controller.parameters()).device
