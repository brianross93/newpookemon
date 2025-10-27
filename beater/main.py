"""Entry-point for experiments/debug runs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, List, Optional

import torch
import yaml
from dotenv import load_dotenv
from torch.distributions import Categorical
import random
import random

from beater.env import EnvConfig, PyBoyEnv
from beater.executor import NavPlanletBuilder, PlanletExecutor, SpriteMovementDetector
from beater.brains import GPTBrain, GoalSuggestion, AsyncBrain
from beater.objectives import ObjectiveEngine
from beater.perception import (
    Perception,
    PerceptionConfig,
    TileDescriptor,
    TileDescriptorConfig,
)
from beater.policy import (
    AffordancePrior,
    Controller,
    ControllerConfig, ControllerState,
    GoalManager,
    NavPlanner,
    NavPlannerConfig,
    OptionBank,
    PlanHead,
    PlanHeadConfig,
)
from beater.sr_memory import EntityGraph, GraphOp, PassabilityStore
from beater.training import (
    GroundedRolloutCollector,
    PPOConfig,
    PPOTrainer,
    gate_skill_reward,
)
from beater.types import PlanletKind

LOGGER = logging.getLogger("beater")
GRAPH_OPS: tuple[GraphOp, ...] = tuple(GraphOp)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PokAcmon Beater agent")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--rom",
        type=Path,
        default=None,
        help="Override ROM path from config",
    )
    parser.add_argument(
        "--window",
        type=str,
        default=None,
        help="Override PyBoy window type",
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Force SDL2 window (ignores headless defaults)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Sequential interactive steps to run before exiting (>=0)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_env(cfg: dict[str, Any], args: argparse.Namespace) -> PyBoyEnv:
    env_cfg = cfg.get("env", {})
    rom_path = Path(args.rom) if args.rom else Path(env_cfg.get("rom", "Pokemon Blue.gb"))
    window = args.window or env_cfg.get("window", "SDL2")
    if args.visual:
        window = "SDL2"
    env_conf = EnvConfig(
        rom_path=str(rom_path),
        window=window,
        speed=float(env_cfg.get("speed", 1.0)),
        ring_slots=int(env_cfg.get("ring_slots", 8)),
    )
    return PyBoyEnv(env_conf)


def build_perception(cfg: dict[str, Any]) -> Perception:
    per_cfg = cfg.get("perception", {})
    config = PerceptionConfig(
        z_dim=int(per_cfg.get("z_dim", 256)),
        smooth_tau=float(per_cfg.get("smooth_tau", 0.9)),
        rnd_dim=int(per_cfg.get("rnd_dim", 128)),
        use_ram=bool(per_cfg.get("use_ram", True)),
    )
    return Perception(config)


def build_tile_descriptor(cfg: dict[str, Any]) -> TileDescriptor:
    td_cfg = cfg.get("tile_descriptor", {})
    config = TileDescriptorConfig(
        tile_size=int(td_cfg.get("tile_size", 8)),
        embedding_dim=int(td_cfg.get("embedding_dim", 64)),
        codebook_size=int(td_cfg.get("codebook_size", 64)),
    )
    return TileDescriptor(config)


def build_policy(cfg: dict[str, Any], latent_dim: int) -> tuple[Controller, PlanHead, list[PlanletKind]]:
    pol_cfg = cfg.get("policy", {})
    controller_cfg = ControllerConfig(
        latent_dim=latent_dim,
        hidden_dim=int(pol_cfg.get("hidden_dim", 256)),
        num_skills=int(pol_cfg.get("num_skills", 4)),
    )
    controller = Controller(controller_cfg)
    default_vocab: List[PlanletKind] = ["NAVIGATE", "INTERACT", "MENU_SEQUENCE", "WAIT"]
    skill_vocab = pol_cfg.get("skill_vocab", default_vocab)
    plan_head = PlanHead(PlanHeadConfig(skill_vocab=skill_vocab))
    return controller, plan_head, skill_vocab


def build_nav_planner(cfg: dict[str, Any], store: PassabilityStore) -> NavPlanner:
    nav_cfg = cfg.get("nav_planner", {})
    config = NavPlannerConfig(
        thompson_retries=int(nav_cfg.get("thompson_retries", 4)),
        epsilon=float(nav_cfg.get("epsilon", 1e-3)),
    )
    return NavPlanner(store, config)


def build_trainer(cfg: dict[str, Any], controller: Controller, affordance: AffordancePrior) -> tuple[PPOTrainer, dict[str, Any]]:
    train_cfg = cfg.get("training", {})
    config = PPOConfig(
        lr=float(train_cfg.get("lr", 3e-4)),
        clip_eps=float(train_cfg.get("clip", 0.1)),
        entropy_coef=float(train_cfg.get("entropy_coef", 0.01)),
    )
    trainer = PPOTrainer(controller, affordance, config)
    return trainer, train_cfg


def build_brain(cfg: dict[str, Any]) -> tuple[Optional[GPTBrain], dict[str, Any]]:
    brain_cfg = cfg.get("brain", {})
    if not brain_cfg.get("enabled", False):
        return None, brain_cfg
    brain = GPTBrain(
        model=brain_cfg.get("model", "gpt-5-mini"),
        max_output_tokens=int(brain_cfg.get("max_output_tokens", 256)),
        base_url=brain_cfg.get("base_url", "https://api.openai.com/v1"),
        enabled=True,
        enable_web_search=bool(brain_cfg.get("enable_web_search", False)),
    )
    if not brain.enabled:
        return None, brain_cfg
    return brain, brain_cfg


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    load_dotenv()
    cfg = load_config(args.config)
    env = build_env(cfg, args)
    perception = build_perception(cfg)
    tile_descriptor = build_tile_descriptor(cfg)
    brain, brain_cfg = build_brain(cfg)
    async_brain: Optional[AsyncBrain] = AsyncBrain(brain) if brain else None
    attach_screenshot = bool((brain_cfg or {}).get("attach_screenshot", True))
    max_directive_age = int((brain_cfg or {}).get("max_directive_age_steps", 300) or 300)
    min_cand_overlap = int((brain_cfg or {}).get("min_candidate_overlap", 1) or 1)
    objective = ObjectiveEngine()
    brain_candidate_preview = int(brain_cfg.get("candidate_preview", 4) or 4)
    brain_for_rollouts = brain if brain and brain_cfg.get("use_for_rollouts", False) else None
    controller, plan_head, skill_vocab = build_policy(cfg, perception.config.z_dim)
    affordance = AffordancePrior(perception.config.z_dim, len(skill_vocab))
    option_bank = OptionBank()
    goal_manager_cli = GoalManager()
    graph = EntityGraph()
    pass_store = PassabilityStore()
    nav_planner = build_nav_planner(cfg, pass_store)
    nav_builder = NavPlanletBuilder()
    movement_detector = SpriteMovementDetector()
    executor = PlanletExecutor(env, pass_store, movement_detector)
    trainer, train_cfg = build_trainer(cfg, controller, affordance)
    LOGGER.info("Environment initialized with ROM=%s", env.config.rom_path)
    controller_state: Optional[ControllerState] = None
    interactive_context = "interactive"
    step_target = args.max_steps if args.max_steps > 0 else 1
    start_step = env.step_idx
    brain_directive_cli: Optional[GoalSuggestion] = None
    last_rollout_stats: dict[str, Any] = {}

    obs = env.observe()
    # Default objective until the first LLM-provided spec arrives
    try:
        default_spec = {
            "phase": "boot",
            "reward_weights": {
                "scene_change": 1.0,
                "sprite_delta": 0.05,
                "nav_success": 0.5,
                "menu_progress": 0.25,
                "name_committed": 0.75,
            },
            "timeouts": {"ttl_steps": 1500},
            "skill_bias": "menu",
        }
        objective.set_spec(default_spec, obs.step_idx)
        LOGGER.info("Objective initialized (default): %s", objective.summary())
    except Exception:
        pass
    while env.step_idx - start_step < step_target:
        rgb_tensor = torch.from_numpy(obs.rgb).permute(2, 0, 1).unsqueeze(0)
        ram_tensor = torch.from_numpy(obs.ram).unsqueeze(0) if obs.ram is not None else None
        if controller_state is None:
            controller_state = controller.init_state(batch_size=1, device=rgb_tensor.device)
        with torch.no_grad():
            z, rnd = perception(rgb_tensor, ram_tensor, context=interactive_context)
            graph.add_entity("frame", z.squeeze(0), {"step_idx": obs.step_idx})
            assoc = graph.assoc(z.squeeze(0), top_k=1)
            tile_out = tile_descriptor(rgb_tensor)
            tile_keys = tile_descriptor.tile_keys(tile_out.class_ids, tile_out.grid_shape)
            tile_grid = tile_descriptor.reshape_tokens(tile_keys[0], tile_out.grid_shape)
            anchor = goal_manager_cli.player_anchor(tile_out.grid_shape)
            anchor_key = tile_grid[anchor[0]][anchor[1]]
            _, tile_class = anchor_key.split(":", 1)
            nav_est = pass_store.update(tile_class, anchor_key, success=True)
            candidates = goal_manager_cli.peek_candidates(
                tile_out.grid_shape, brain_candidate_preview
            )
            ctrl_out = controller(z, controller_state)
            affordance_logits = affordance(z)
            gate_dist = Categorical(logits=ctrl_out.gate_logits)
            skill_logits = ctrl_out.skill_logits + affordance_logits
            skill_dist = Categorical(logits=skill_logits)
            gate_action = gate_dist.sample()
            skill_action = skill_dist.sample()
            gate_idx = int(gate_action.item())
            gate_op = GRAPH_OPS[gate_idx % len(GRAPH_OPS)]
            should_query_brain = (
                async_brain is not None
                and brain_directive_cli is None
                and candidates
                and gate_op == GraphOp.HALT
            )
            if should_query_brain:
                img_bytes = None
                if attach_screenshot:
                    try:
                        import io
                        from PIL import Image

                        # Convert observation RGB(A) to PNG bytes
                        arr = obs.rgb
                        mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
                        im = Image.fromarray(arr, mode)
                        buf = io.BytesIO()
                        im.save(buf, format="PNG")
                        img_bytes = buf.getvalue()
                    except Exception:
                        img_bytes = None
                # Build candidate metadata for the prompt
                cand_meta = []
                for (r, c) in candidates:
                    key = tile_grid[r][c]
                    cls = key.split(":", 1)[-1]
                    est = pass_store.get_estimate(cls, key)
                    cand_meta.append({
                        "passability": est.blended,
                        "fail_count": goal_manager_cli.get_fail_count((r, c)),
                    })

                async_brain.request(
                    tile_grid,
                    candidates,
                    {
                        "step": obs.step_idx,
                        "assoc_matches": len(assoc),
                        "passability": nav_est.blended,
                        "candidate_meta": cand_meta,
                    },
                    step_idx=obs.step_idx,
                    image_bytes=img_bytes,
                )
            brain_goal_override = None
            brain_skill = None
            brain_notes = None
            if brain_directive_cli:
                brain_goal_override = brain_directive_cli.resolve_goal(candidates)
                brain_skill = brain_directive_cli.skill
                brain_notes = brain_directive_cli.reasoning

            preferred_kind = brain_skill if brain_skill in skill_vocab else option_bank.suggest()
            # Bias skills with objective suggestion (menu vs overworld)
            bias = objective.skill_bias()
            if bias == "menu" and preferred_kind == "WAIT":
                preferred_kind = "MENU_SEQUENCE"
            elif bias == "overworld" and preferred_kind == "WAIT":
                preferred_kind = "NAVIGATE"
            if preferred_kind is None:
                preferred_kind = "NAVIGATE"

            planlet = plan_head.decode(
                skill_logits,
                ctrl_out.timeout_steps,
                skill_action=skill_action,
                preferred_kind=preferred_kind,
            )
            # If the brain provided explicit MENU_SEQUENCE ops, override the script.
            if (
                planlet.kind == "MENU_SEQUENCE"
                and brain_directive_cli is not None
                and getattr(brain_directive_cli, "ops", None)
            ):
                from beater.types import ScriptOp  # local import to avoid cycles

                menu_ops = list(brain_directive_cli.ops or [])
                script = []
                for btn in menu_ops:
                    script.append(ScriptOp(op="PRESS", button=btn, frames=2))
                    script.append(ScriptOp(op="WAIT", frames=1))
                    script.append(ScriptOp(op="RELEASE", button=btn, frames=0))
                if script:
                    planlet.args["ops"] = menu_ops
                    planlet.script = script
                    planlet.timeout_steps = len(script)
            goal_coord = None
            nav_path: List[tuple[int, int]] = []
            if planlet.kind == "NAVIGATE":
                goal_coord = goal_manager_cli.next_goal(
                    tile_out.grid_shape,
                    preferred=planlet.kind,
                    goal_override=brain_goal_override,
                )
                nav_path = nav_planner.plan(tile_grid, start=anchor, goal=goal_coord)
                if len(nav_path) > 1:
                    planlet = nav_builder.from_path(nav_path, tile_grid, goal=goal_coord)
                else:
                    planlet.args["goal"] = goal_coord
                    planlet.args["nav_success"] = False
            if brain_notes:
                planlet.args.setdefault("brain_notes", brain_notes)
        obs = executor.run(planlet)
        if planlet.kind == "NAVIGATE" and goal_coord is not None:
            goal_manager_cli.feedback(goal_coord, bool(planlet.args.get("nav_success", False)))
        option_bank.record(planlet.kind)
        controller_state = ctrl_out.state
        # Non-blocking: poll for any async brain directive and stage it for next iteration.
        if async_brain is not None and brain_directive_cli is None:
            polled = async_brain.poll()
            if polled is not None:
                directive, req_step, cand_snap = polled
                # Staleness guard: drop if too old or candidate overlap too small
                age = obs.step_idx - int(req_step)
                overlap = 0
                try:
                    cur_set = set(candidates)
                    snap_set = set(cand_snap)
                    overlap = len(cur_set & snap_set)
                except Exception:
                    overlap = 0
                if age <= max_directive_age and overlap >= min_cand_overlap:
                    brain_directive_cli = directive
                    if getattr(directive, "objective_spec", None):
                        objective.set_spec(directive.objective_spec, obs.step_idx)
                        LOGGER.info("Objective updated: %s", objective.summary())
        if brain_directive_cli:
            if planlet.kind != "NAVIGATE" or planlet.args.get("nav_success", False):
                brain_directive_cli = None
        last_rollout_stats = {
            "obs": obs,
            "assoc": assoc,
            "nav_est": nav_est,
            "planlet": planlet,
            "nav_path_len": len(nav_path),
            "z_shape": tuple(z.shape),
            "rnd_shape": tuple(rnd.shape),
        }
        # Periodic objective summary for visibility
        try:
            if obs.step_idx % 500 == 0:
                LOGGER.info("Objective summary: %s", objective.summary())
        except Exception:
            pass
    perception.reset(context=interactive_context)
    training_goal_manager = GoalManager()
    env_factory = lambda: PyBoyEnv(env.config)
    collector = GroundedRolloutCollector(
        controller=controller,
        perception=perception,
        tile_descriptor=tile_descriptor,
        nav_planner=nav_planner,
        nav_builder=nav_builder,
        plan_head=plan_head,
        affordance=affordance,
        option_bank=option_bank,
        goal_manager=training_goal_manager,
        brain=brain_for_rollouts,
        brain_candidate_preview=brain_candidate_preview,
        pass_store=pass_store,
        reward_fn=gate_skill_reward,
        env_factory=env_factory,
        movement_detector_factory=lambda: SpriteMovementDetector(),
        num_actors=int(train_cfg.get("num_actors", 1)),
        steps_per_actor=int(train_cfg.get("steps_per_actor", 8)),
        objective=objective,
    )
    total_updates = int(train_cfg.get("total_updates", 1))
    train_stats = {}
    for update_idx in range(total_updates):
        buffer = collector.collect()
        train_stats = trainer.update(buffer)
        LOGGER.info("PPO update %s stats %s", update_idx, train_stats)
    summary_obs = last_rollout_stats.get("obs", obs)
    LOGGER.info(
        (
            "Initial observation rgb=%s ram=%s step=%s latent=%s rnd=%s assoc=%s "
            "passability=%s planlet=%s nav_path_len=%s ppo_stats=%s"
        ),
        summary_obs.rgb.shape if summary_obs else obs.rgb.shape,
        summary_obs.ram.shape if summary_obs else obs.ram.shape,
        summary_obs.step_idx if summary_obs else obs.step_idx,
        last_rollout_stats.get("z_shape", ()),
        last_rollout_stats.get("rnd_shape", ()),
        last_rollout_stats.get("assoc"),
        last_rollout_stats.get("nav_est"),
        last_rollout_stats.get("planlet"),
        last_rollout_stats.get("nav_path_len"),
        train_stats,
    )
    if async_brain is not None:
        async_brain.shutdown()
    env.close()


if __name__ == "__main__":  # pragma: no cover
    main()

