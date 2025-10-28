"""Entry-point for experiments/debug runs."""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
import yaml
from dotenv import load_dotenv
from torch.distributions import Categorical
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
from beater.types import Observation, Planlet, PlanletKind, ScriptOp
from beater.utils.detectors import detect_menu_open, scene_change_delta
from beater.utils.maps import ascii_tile_map

LOGGER = logging.getLogger("beater")
GRAPH_OPS: tuple[GraphOp, ...] = tuple(GraphOp)


def _button_script(button: str, press_frames: int = 2, wait_frames: int = 1) -> List[ScriptOp]:
    """Build a simple press/wait/release script for a single button."""

    return [
        ScriptOp(op="PRESS", button=button, frames=press_frames),
        ScriptOp(op="WAIT", frames=wait_frames),
        ScriptOp(op="RELEASE", button=button, frames=0),
    ]


BOOT_SEQUENCE: List[ScriptOp] = []
BOOT_SEQUENCE.append(ScriptOp(op="WAIT", frames=90))
for _ in range(5):
    BOOT_SEQUENCE.extend(_button_script("START", press_frames=8, wait_frames=4))
    BOOT_SEQUENCE.append(ScriptOp(op="WAIT", frames=120))
BOOT_SEQUENCE.append(ScriptOp(op="WAIT", frames=180))

NAMING_GRID = (
    "A B C D E F G H I\n"
    "J K L M N O P Q R\n"
    "S T U V W X Y Z -\n"
    "   END (move to END then press A)"
)

NAMING_LETTER_ROWS: tuple[str, ...] = ("ABCDEFGHI", "JKLMNOPQR", "STUVWXYZ-")
NAMING_LETTER_COORDS: Dict[str, Tuple[int, int]] = {
    letter: (row_idx, col_idx)
    for row_idx, row_letters in enumerate(NAMING_LETTER_ROWS)
    for col_idx, letter in enumerate(row_letters)
}

DOWNSTAIRS_EXIT_CANDIDATES: List[Tuple[int, int]] = [
    (9, 0),
    (0, 10),
]

DOWNSTAIRS_EGRESS_SCRIPT: List[ScriptOp] = []
for _ in range(5):
    DOWNSTAIRS_EGRESS_SCRIPT.extend(_button_script("DOWN", press_frames=2, wait_frames=1))



def _valid_naming_cursor(cursor: Tuple[int, int]) -> bool:
    row, col = cursor
    return 0 <= row <= 2 and 0 <= col <= 8


def _move_cursor_script(
    current: Tuple[int, int], target: Tuple[int, int]
) -> tuple[List[ScriptOp], Tuple[int, int]]:
    """Return script ops to move between two naming-grid coordinates."""

    script: List[ScriptOp] = []
    cur_row, cur_col = current
    tgt_row, tgt_col = target

    # Vertical movement first to avoid overshooting horizontal bars.
    while cur_row < tgt_row:
        script.extend(_button_script("DOWN", press_frames=2, wait_frames=1))
        cur_row += 1
    while cur_row > tgt_row:
        script.extend(_button_script("UP", press_frames=2, wait_frames=1))
        cur_row -= 1

    # Horizontal adjustments.
    while cur_col < tgt_col:
        script.extend(_button_script("RIGHT", press_frames=2, wait_frames=1))
        cur_col += 1
    while cur_col > tgt_col:
        script.extend(_button_script("LEFT", press_frames=2, wait_frames=1))
        cur_col -= 1

    return script, (cur_row, cur_col)


def _is_naming_screen(ram: Optional["np.ndarray"]) -> bool:
    """Heuristic check for the player naming UI by scanning the text buffer."""

    if ram is None or len(ram) < 0xD180:
        return False
    window = ram[0xD158 : 0xD158 + 12]
    pattern = [0x8D, 0x88, 0x8D, 0x93]  # "NAME" in the text encoding.
    return list(window[:4]) == pattern


def _directive_has_naming_ops(directive: GoalSuggestion) -> bool:
    if directive.skill != "MENU_SEQUENCE":
        return False
    ops = getattr(directive, "ops", None)
    return isinstance(ops, list) and len(ops) > 0


def _dismiss_title_screen(
    env: PyBoyEnv,
    tile_descriptor: TileDescriptor,
    obs: Observation,
    max_cycles: int = 3,
) -> tuple[Observation, torch.Tensor, Tuple[int, int]]:
    """Run a deterministic script to skip the title/menu stack before the main loop."""

    rgb_tensor = torch.from_numpy(obs.rgb).permute(2, 0, 1).unsqueeze(0)
    tile_out = tile_descriptor(rgb_tensor)
    base_ids = tile_out.class_ids.clone()
    grid_shape = tile_out.grid_shape
    last_delta = 0.0
    last_obs = obs
    for attempt in range(max_cycles):
        obs = env.step_script(BOOT_SEQUENCE)
        rgb_tensor = torch.from_numpy(obs.rgb).permute(2, 0, 1).unsqueeze(0)
        tile_out = tile_descriptor(rgb_tensor)
        tile_keys = tile_descriptor.tile_keys(tile_out.class_ids, tile_out.grid_shape)
        tile_grid = tile_descriptor.reshape_tokens(tile_keys[0], tile_out.grid_shape)
        if detect_menu_open(obs.ram, tile_grid):
            LOGGER.info(
                "Boot sequence reached interactive menu (attempt=%s step=%s)",
                attempt + 1,
                env.step_idx,
            )
            return obs, tile_out.class_ids.clone(), tile_out.grid_shape
        last_delta = scene_change_delta(base_ids, tile_out.class_ids, tile_out.grid_shape)
        last_obs = obs
    LOGGER.warning(
        "Boot sequence exhausted without detecting menu; continuing (delta=%.3f step=%s)",
        last_delta,
        env.step_idx,
    )
    return last_obs, tile_out.class_ids.clone(), tile_out.grid_shape


def _reset_portal_score(
    tile_grid: List[List[str]], portal_scores: Dict[str, float], coord: Tuple[int, int]
) -> None:
    """Reset the portal accumulator for a concrete tile coordinate."""

    r, c = coord
    if r < 0 or c < 0:
        return
    if r >= len(tile_grid) or c >= len(tile_grid[r]):
        return
    key = tile_grid[r][c]
    if key in portal_scores:
        LOGGER.info("Portal score reset for coord=%s key=%s", coord, key)
    portal_scores[key] = 0.0


def _boost_portal_scores(
    tile_grid: List[List[str]], portal_scores: Dict[str, float], coords: List[Tuple[int, int]], bonus: float = 2.0
) -> None:
    """Give target tiles an immediate portal bonus so goal scoring prefers exits."""

    for (r, c) in coords:
        if r < 0 or c < 0:
            continue
        if r >= len(tile_grid) or c >= len(tile_grid[r]):
            continue
        key = tile_grid[r][c]
        portal_scores[key] = portal_scores.get(key, 0.0) + bonus


def _should_random_walk(
    tile_grid: List[List[str]],
    anchor: Tuple[int, int],
    pass_store: PassabilityStore,
    threshold: float = 0.15,
) -> bool:
    """Heuristic: return True when surrounding tiles are largely unknown/blocked."""

    r, c = anchor
    known = 0
    open_tiles = 0
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if nr < 0 or nc < 0 or nr >= len(tile_grid) or nc >= len(tile_grid[nr]):
            continue
        key = tile_grid[nr][nc]
        cls = key.split(":", 1)[-1]
        est = pass_store.get_estimate(cls, key)
        # Treat as known if instance mean has diverged from neutral prior.
        if abs(est.instance_mean - 0.5) > 0.05:
            known += 1
            if est.blended > threshold:
                open_tiles += 1
    # Random walk when we barely know surroundings or all known tiles are blocked.
    return known <= 1 or open_tiles == 0


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
    seed = int(cfg.get("seed", 0))
    if seed:
        LOGGER.info("Seeding RNGs with %s", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env = build_env(cfg, args)
    perception = build_perception(cfg)
    tile_descriptor = build_tile_descriptor(cfg)
    # Screenshot cache for HALT attachments
    last_img_step = -10**9
    last_img_bytes: Optional[bytes] = None
    last_tile_sig: Optional[bytes] = None
    brain, brain_cfg = build_brain(cfg)
    async_brain: Optional[AsyncBrain] = AsyncBrain(brain) if brain else None
    attach_screenshot = bool((brain_cfg or {}).get("attach_screenshot", True))
    max_directive_age = int((brain_cfg or {}).get("max_directive_age_steps", 2000))
    min_cand_overlap = int((brain_cfg or {}).get("min_candidate_overlap", 0))
    halt_cooldown_steps = int((brain_cfg or {}).get("halt_cooldown_steps", 120))
    objective = ObjectiveEngine()
    brain_candidate_preview = int(brain_cfg.get("candidate_preview", 4))
    brain_for_rollouts = brain if brain and brain_cfg.get("use_for_rollouts", False) else None
    controller, plan_head, skill_vocab = build_policy(cfg, perception.config.z_dim)
    affordance = AffordancePrior(perception.config.z_dim, len(skill_vocab))
    option_bank = OptionBank()
    goal_manager_cli = GoalManager()
    graph = EntityGraph()
    pass_store = PassabilityStore()
    portal_scores: Dict[str, float] = {}
    initial_class_ids: Optional[torch.Tensor] = None
    initial_grid_shape: Optional[Tuple[int, int]] = None
    current_phase_tag = "boot"
    menu_cooldown = 0
    naming_cooldown = 0
    naming_cooldown_steps = int((brain_cfg or {}).get("naming_cooldown_steps", 600))
    committed_goal: Optional[Tuple[int, int]] = None
    commit_steps_remaining = 0
    halt_cooldown = 0
    phase_halt_pending: Optional[str] = None
    downstairs_exit_plan: Optional[Planlet] = None
    downstairs_exit_goal: Optional[Tuple[int, int]] = None
    downstairs_script_done = False
    nav_planner = build_nav_planner(cfg, pass_store)
    nav_builder = NavPlanletBuilder()
    movement_detector = SpriteMovementDetector()
    executor = PlanletExecutor(env, pass_store, movement_detector)
    trainer, train_cfg = build_trainer(cfg, controller, affordance)
    LOGGER.info("Environment initialized with ROM=%s", env.config.rom_path)
    controller_state: Optional[ControllerState] = None
    interactive_context = "interactive"
    step_target = args.max_steps  # 0 => run indefinitely
    start_step = env.step_idx
    brain_directive_cli: Optional[GoalSuggestion] = None
    last_rollout_stats: dict[str, Any] = {}

    obs = env.observe()
    obs, initial_class_ids, initial_grid_shape = _dismiss_title_screen(env, tile_descriptor, obs)
    # Default objective until the first LLM-provided spec arrives
    default_spec = {
        "phase": "boot",
        "reward_weights": {
            "scene_change": 1.0,
            "sprite_delta": 0.05,
            "nav_success": 0.5,
            "menu_progress": 0.25,
            "name_committed": 0.75,
        },
        "timeouts": {"ttl_steps": 500},
        "skill_bias": "menu",
    }
    try:
        objective.set_spec(default_spec, obs.step_idx)
        LOGGER.info("Objective initialized (default): %s", objective.summary())
    except Exception as exc:
        LOGGER.exception("Failed to apply default objective spec: %s", exc)
    while step_target == 0 or (env.step_idx - start_step < step_target):
        phase_state = current_phase_tag
        halt_cooldown = max(halt_cooldown - 1, 0)
        menu_cooldown = max(menu_cooldown - 1, 0)
        naming_cooldown = max(naming_cooldown - 1, 0)
        commit_steps_remaining = max(commit_steps_remaining - 1, 0)
        menu_open = menu_cooldown > 0
        rgb_tensor = torch.from_numpy(obs.rgb).permute(2, 0, 1).unsqueeze(0)
        ram_tensor = torch.from_numpy(obs.ram).unsqueeze(0) if obs.ram is not None else None
        if controller_state is None:
            controller_state = controller.init_state(batch_size=1, device=rgb_tensor.device)
        naming_cursor: Optional[Tuple[int, int]] = None
        naming_current_name: str = ""
        with torch.inference_mode():
            z, rnd = perception(rgb_tensor, ram_tensor, context=interactive_context)
            frame_id = graph.add_entity("frame", z.squeeze(0), {"step_idx": obs.step_idx})
            assoc = graph.assoc(z.squeeze(0), top_k=1)
            tile_out = tile_descriptor(rgb_tensor)
            tile_keys = tile_descriptor.tile_keys(tile_out.class_ids, tile_out.grid_shape)
            tile_grid = tile_descriptor.reshape_tokens(tile_keys[0], tile_out.grid_shape)
            anchor = goal_manager_cli.player_anchor(tile_out.grid_shape)
            anchor_key = tile_grid[anchor[0]][anchor[1]]
            _, tile_class = anchor_key.split(":", 1)
            nav_est = pass_store.get_estimate(tile_class, anchor_key)
            candidates = goal_manager_cli.peek_candidates(
                tile_out.grid_shape, brain_candidate_preview
            )
            unseen_for_prompt = goal_manager_cli.unseen_candidates(
                tile_out.grid_shape, tile_grid, pass_store, brain_candidate_preview
            )
            if unseen_for_prompt:
                interleaved: List[Tuple[int, int]] = []
                for coord in unseen_for_prompt + candidates:
                    if coord not in interleaved:
                        interleaved.append(coord)
                candidates = interleaved[:brain_candidate_preview]
            ascii_map = ascii_tile_map(tile_grid, anchor, candidates)
            if portal_scores:
                for key in list(portal_scores.keys()):
                    portal_scores[key] *= 0.995
                    if portal_scores[key] < 1e-3:
                        portal_scores.pop(key)
            phase_delta = 0.0
            if initial_class_ids is not None and initial_grid_shape is not None:
                phase_delta = scene_change_delta(initial_class_ids, tile_out.class_ids, tile_out.grid_shape)
            new_phase = current_phase_tag
            if current_phase_tag == "boot" and phase_delta > 0.35:
                new_phase = "downstairs"
            if new_phase != current_phase_tag:
                current_phase_tag = new_phase
                portal_scores.clear()
                committed_goal = None
                commit_steps_remaining = 0
                if new_phase == "downstairs":
                    try:
                        _reset_portal_score(tile_grid, portal_scores, (7, 10))
                        _boost_portal_scores(tile_grid, portal_scores, DOWNSTAIRS_EXIT_CANDIDATES, bonus=3.0)
                        phase_halt_pending = "downstairs"
                        downstairs_spec = {
                            "phase": "stairs_down",
                            "reward_weights": {
                                "scene_change": 0.5,
                                "sprite_delta": 0.1,
                                "nav_success": 1.2,
                            },
                            "timeouts": {"ttl_steps": 800},
                            "skill_bias": "overworld",
                        }
                        objective.set_spec(downstairs_spec, obs.step_idx)
                        LOGGER.info(
                            "Objective updated (phase->downstairs door focus): %s",
                            objective.summary(),
                        )
                        if not downstairs_script_done:
                            try:
                                obs = env.step_script(DOWNSTAIRS_EGRESS_SCRIPT)
                                downstairs_script_done = True
                                LOGGER.info("Ran downstairs egress script to push toward door")
                                tensor_after = torch.from_numpy(obs.rgb).permute(2, 0, 1).unsqueeze(0)
                                tile_out_after = tile_descriptor(tensor_after)
                                tile_keys_after = tile_descriptor.tile_keys(
                                    tile_out_after.class_ids, tile_out_after.grid_shape
                                )
                                tile_grid_after = tile_descriptor.reshape_tokens(
                                    tile_keys_after[0], tile_out_after.grid_shape
                                )
                                anchor_after = goal_manager_cli.player_anchor(tile_out_after.grid_shape)
                                best_path: Optional[List[Tuple[int, int]]] = None
                                best_goal: Optional[Tuple[int, int]] = None
                                for goal in DOWNSTAIRS_EXIT_CANDIDATES:
                                    path = nav_planner.plan(tile_grid_after, start=anchor_after, goal=goal)
                                    if path and len(path) > 1:
                                        if best_path is None or len(path) < len(best_path):
                                            best_path = path
                                            best_goal = goal
                                if best_path and best_goal:
                                    downstairs_exit_plan = nav_builder.from_path(
                                        best_path, tile_grid_after, goal=best_goal
                                    )
                                    downstairs_exit_goal = best_goal
                                    LOGGER.info(
                                        "Primed downstairs exit plan goal=%s path_len=%s",
                                        best_goal,
                                        len(best_path),
                                    )
                                initial_class_ids = tile_out_after.class_ids.clone()
                                continue
                            except Exception as exc:
                                downstairs_script_done = True
                                LOGGER.exception("Downstairs egress script failed: %s", exc)
                    except Exception as exc:
                        LOGGER.exception("Failed to set downstairs objective: %s", exc)
            naming_screen = _is_naming_screen(obs.ram)
            if naming_screen:
                menu_cooldown = max(menu_cooldown, 120)
                halt_cooldown = 0
                if current_phase_tag != "naming":
                    current_phase_tag = "naming"
                    naming_cooldown = naming_cooldown_steps
                    try:
                        naming_spec = {
                            "phase": "naming",
                            "reward_weights": {
                                "scene_change": 0.2,
                                "menu_progress": 0.6,
                                "name_committed": 1.0,
                            },
                            "timeouts": {"ttl_steps": 600},
                            "skill_bias": "menu",
                        }
                        objective.set_spec(naming_spec, obs.step_idx)
                        LOGGER.info("Objective updated (phase->naming): %s", objective.summary())
                    except Exception as exc:
                        LOGGER.exception("Failed to apply naming objective: %s", exc)
            elif current_phase_tag == "naming":
                current_phase_tag = "boot"
                naming_cooldown = 0
                try:
                    objective.set_spec(default_spec, obs.step_idx)
                    LOGGER.info("Objective reset post-naming: %s", objective.summary())
                except Exception as exc:
                    LOGGER.exception("Failed to reset naming objective: %s", exc)
            menu_detected = detect_menu_open(obs.ram, tile_grid)
            if menu_detected:
                menu_cooldown = max(menu_cooldown, 60)
            menu_open = menu_open or menu_detected
            phase_state = current_phase_tag
            naming_active = phase_state == "naming"
            ctrl_out = controller(z, controller_state)
            affordance_logits = affordance(z)
            gate_dist = Categorical(logits=ctrl_out.gate_logits)
            skill_logits = ctrl_out.skill_logits + affordance_logits
            skill_dist = Categorical(logits=skill_logits)
            gate_action = gate_dist.sample()
            skill_action = skill_dist.sample()
            gate_idx = int(gate_action.item())
            gate_op = GRAPH_OPS[gate_idx % len(GRAPH_OPS)]
            phase_force_halt = phase_halt_pending == phase_state
            force_brain_halt = ((naming_active and brain_directive_cli is None) or phase_force_halt)
            try:
                if gate_op == GraphOp.ASSOC:
                    assoc_portals = graph.assoc(z.squeeze(0), top_k=3, filter_kind="portal")
                    for portal_id, score in assoc_portals:
                        node = graph.nodes.get(portal_id)
                        coord = node.attrs.get("coord") if node else None
                        if coord is None:
                            continue
                        pr, pc = coord
                        if pr < 0 or pc < 0 or pr >= len(tile_grid) or pc >= len(tile_grid[pr]):
                            continue
                        key = tile_grid[pr][pc]
                        portal_scores[key] = portal_scores.get(key, 0.0) + float(score)
                    if assoc_portals:
                        LOGGER.debug("ASSOC boosted portal scores via %s", assoc_portals)
                elif gate_op == GraphOp.FOLLOW:
                    target_key = None
                    if downstairs_exit_goal:
                        r, c = downstairs_exit_goal
                        if 0 <= r < len(tile_grid) and 0 <= c < len(tile_grid[r]):
                            target_key = tile_grid[r][c]
                    if target_key is None and portal_scores:
                        target_key = max(portal_scores.items(), key=lambda kv: kv[1])[0]
                    if target_key:
                        portal_scores[target_key] = portal_scores.get(target_key, 0.0) + 2.0
                        LOGGER.debug("FOLLOW emphasized portal key %s", target_key)
                elif gate_op == GraphOp.WRITE:
                    graph.add_entity(
                        "anchor",
                        z.squeeze(0),
                        {"coord": anchor, "phase": phase_state, "step": obs.step_idx},
                    )
            except Exception as exc:
                LOGGER.exception("Graph operation handling failed: %s", exc)
            should_query_brain = (
                async_brain is not None
                and brain_directive_cli is None
                and not async_brain.has_pending()
                and candidates
                and (gate_op == GraphOp.HALT or force_brain_halt)
                and halt_cooldown == 0
            )
            if should_query_brain:
                if naming_active and naming_cooldown > 0:
                    continue
                img_bytes = None
                need_screenshot = attach_screenshot or naming_active
                if need_screenshot:
                    try:
                        import io
                        from PIL import Image

                        tile_sig = tile_out.class_ids.detach().cpu().numpy().tobytes()
                        if (obs.step_idx - last_img_step) >= 30 or tile_sig != last_tile_sig:
                            arr = obs.rgb
                            mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
                            target_width = max(1, arr.shape[1] // 2)
                            target_height = max(1, arr.shape[0] // 2)
                            im = Image.fromarray(arr, mode).resize((target_width, target_height))
                            buf = io.BytesIO()
                            im.save(buf, format="PNG")
                            last_img_bytes = buf.getvalue()
                            last_img_step = obs.step_idx
                            last_tile_sig = tile_sig
                        img_bytes = last_img_bytes
                        LOGGER.debug(
                            "HALT screenshot prepared step=%s naming=%s",
                            obs.step_idx,
                            naming_active,
                        )
                    except Exception as exc:
                        LOGGER.exception("Failed to encode screenshot for brain request: %s", exc)
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
                        "portal_score": portal_scores.get(key, 0.0),
                    })

                context_payload = {
                    "step": obs.step_idx,
                    "assoc_matches": len(assoc),
                    "passability": nav_est.blended,
                    "ascii_map": ascii_map,
                    "phase": phase_state,
                    "menu_open": menu_open,
                    "candidate_meta": cand_meta,
                }
                if unseen_for_prompt:
                    context_payload["unseen_candidates"] = unseen_for_prompt[:brain_candidate_preview]
                if naming_screen:
                    context_payload["naming_hint"] = True
                    context_payload["naming_grid"] = NAMING_GRID
                    # Cursor positions (pixel coords; map to grid)
                    try:
                        cursor_y = int(obs.ram[0xCC24] // 16)
                        cursor_x = int(obs.ram[0xCC25] // 16)
                        naming_cursor = (cursor_y, cursor_x)
                        context_payload["cursor_position"] = {"row": cursor_y, "col": cursor_x}
                        # Decode current name buffer
                        def decode_poke_text(bytes_arr):
                            mapping = {
                                0x80: 'A', 0x81: 'B', 0x82: 'C', 0x83: 'D', 0x84: 'E', 0x85: 'F', 0x86: 'G', 0x87: 'H', 0x88: 'I',
                                0x89: 'J', 0x8A: 'K', 0x8B: 'L', 0x8C: 'M', 0x8D: 'N', 0x8E: 'O', 0x8F: 'P', 0x90: 'Q', 0x91: 'R',
                                0x92: 'S', 0x93: 'T', 0x94: 'U', 0x95: 'V', 0x96: 'W', 0x97: 'X', 0x98: 'Y', 0x99: 'Z',
                                0x9A: '-',
                                0x50: '',
                            }
                            name = ''
                            for b in bytes_arr:
                                if b == 0x50:
                                    break
                                name += mapping.get(b, '?')
                            return name
                        name_buffer = obs.ram[0xD158:0xD158 + 11]
                        current_name = decode_poke_text(name_buffer)
                        naming_current_name = current_name
                        context_payload["current_name"] = current_name if current_name else "empty"
                        context_payload["preset_options"] = ["RED", "ASH", "JACK"]
                    except Exception as exc:
                        LOGGER.exception("Failed to gather naming context: %s", exc)

                async_brain.request(
                    tile_grid,
                    candidates,
                    context_payload,
                    step_idx=obs.step_idx,
                    image_bytes=img_bytes,
                )
                LOGGER.info(
                    "HALT queued (gate) step=%s phase=%s candidates=%s",
                    obs.step_idx,
                    phase_state,
                    len(candidates),
                )
                halt_cooldown = halt_cooldown_steps
                if phase_force_halt:
                    phase_halt_pending = None
                if naming_active:
                    naming_cooldown = naming_cooldown_steps
                    brain_goal_override = None
                    brain_skill = None
                    brain_notes = None
            if brain_directive_cli:
                brain_goal_override = brain_directive_cli.resolve_goal(candidates)
                brain_skill = brain_directive_cli.skill
                brain_notes = brain_directive_cli.reasoning

            bias = objective.skill_bias()
            preferred_kind = brain_skill if brain_skill in skill_vocab else option_bank.suggest()
            menu_bias_active = bias == "menu"
            if naming_active:
                preferred_kind = "MENU_SEQUENCE" if brain_directive_cli else "WAIT"
            else:
                if bias == "menu" and preferred_kind == "WAIT":
                    preferred_kind = "MENU_SEQUENCE"
                elif bias == "overworld" and preferred_kind == "WAIT":
                    preferred_kind = "NAVIGATE"
                if menu_open and preferred_kind not in ("MENU_SEQUENCE", "INTERACT"):
                    preferred_kind = "MENU_SEQUENCE"
            if preferred_kind is None:
                preferred_kind = "NAVIGATE"
            # If the brain proposed a concrete skill, force that skill index so
            # the LLM can directly drive plan selection (interactive only).
            if brain_skill in skill_vocab:
                try:
                    skill_idx_override = torch.tensor([
                        int(skill_vocab.index(brain_skill))
                    ], device=skill_action.device)
                    skill_action = skill_idx_override
                except Exception as exc:
                    LOGGER.exception("Failed to override skill index from brain directive: %s", exc)

            planlet = plan_head.decode(
                skill_logits,
                ctrl_out.timeout_steps,
                skill_action=skill_action,
                preferred_kind=preferred_kind,
            )
            if phase_state == "downstairs" and downstairs_exit_plan is not None:
                planlet = downstairs_exit_plan
                downstairs_exit_plan = None
                downstairs_exit_goal = None
                LOGGER.info("Executing primed downstairs exit plan")
            # If the brain provided explicit MENU_SEQUENCE ops, override the script.
            if (
                planlet.kind == "MENU_SEQUENCE"
                and brain_directive_cli is not None
                and getattr(brain_directive_cli, "ops", None)
            ):
                menu_ops = list(brain_directive_cli.ops or [])
                script: List[ScriptOp] = []
                total_frames = 0
                for btn in menu_ops:
                    if btn.startswith("WAIT"):
                        try:
                            frames = int(btn.split("_", 1)[1])
                        except (IndexError, ValueError):
                            frames = 10
                        script.append(ScriptOp(op="WAIT", frames=max(frames, 1)))
                        total_frames += max(frames, 1)
                        continue
                    if btn == "NOOP":
                        script.append(ScriptOp(op="WAIT", frames=1))
                        total_frames += 1
                        continue
                    press_frames = 3 if naming_active else 2
                    settle_frames = 4 if naming_active else 1
                    script.append(ScriptOp(op="PRESS", button=btn, frames=press_frames))
                    total_frames += press_frames
                    script.append(ScriptOp(op="WAIT", frames=settle_frames))
                    total_frames += settle_frames
                    script.append(ScriptOp(op="RELEASE", button=btn, frames=0))
                if script:
                    planlet.args["ops"] = menu_ops
                    planlet.script = script
                    planlet.timeout_steps = max(planlet.timeout_steps, total_frames + 30)
                    LOGGER.info(
                        "Applied brain MENU_SEQUENCE ops step=%s ops=%s",
                        obs.step_idx,
                        menu_ops,
                    )
            menu_bias_active = bias == "menu"
            brain_has_ops = (
                brain_directive_cli is not None
                and getattr(brain_directive_cli, "ops", None)
                and planlet.kind == "MENU_SEQUENCE"
            )
            if menu_open and not naming_active:
                menu_cooldown = max(menu_cooldown, 30)
                allowed_kind = planlet.kind in ("MENU_SEQUENCE", "INTERACT")
                auto_close = False
                if not allowed_kind and not menu_bias_active:
                    auto_close = True
                elif allowed_kind and not menu_bias_active and not brain_has_ops:
                    auto_close = True
                if auto_close:
                    script = _button_script("B")
                    planlet = Planlet(
                        id=str(uuid.uuid4()),
                        kind="MENU_SEQUENCE",
                        args={"auto_menu": True},
                        script=script,
                        timeout_steps=len(script),
            )
            goal_coord = None
            nav_path: List[tuple[int, int]] = []
            if planlet.kind == "NAVIGATE":
                # Passability-weighted goal selection
                def _score_goal(g: tuple[int, int]) -> float:
                    path = nav_planner.plan(tile_grid, start=anchor, goal=g)
                    if not path or len(path) < 2:
                        return -1e9
                    # Score by mean blended passability along first few steps
                    horizon = min(6, len(path))
                    ps: List[float] = []
                    for (r, c) in path[1:horizon]:
                        key = tile_grid[r][c]
                        cls = key.split(":", 1)[-1]
                        est = pass_store.get_estimate(cls, key)
                        ps.append(float(est.blended))
                    avg_pass = float(sum(ps) / max(1, len(ps)))
                    goal_key = tile_grid[g[0]][g[1]]
                    portal_bonus = portal_scores.get(goal_key, 0.0) * 0.5
                    dist = len(path)
                    return avg_pass + portal_bonus - 0.01 * dist

                sel_goal = None
                if commit_steps_remaining > 0 and committed_goal in candidates:
                    sel_goal = committed_goal
                if sel_goal is None and brain_goal_override is not None:
                    sel_goal = brain_goal_override
                if sel_goal is None:
                    try:
                        scored = [(g, _score_goal(g)) for g in candidates]
                        scored.sort(key=lambda kv: kv[1], reverse=True)
                        sel_goal = scored[0][0] if scored else None
                        if obs.step_idx % 500 == 0 and scored:
                            LOGGER.info(
                                "Goal scored chosen=%s score=%.3f best_of=%s",
                                sel_goal,
                                scored[0][1],
                                len(scored),
                            )
                    except Exception as exc:
                        LOGGER.exception("Failed to score navigation candidates: %s", exc)
                        sel_goal = None
                if sel_goal is None:
                    sel_goal = goal_manager_cli.next_goal(
                    tile_out.grid_shape,
                    preferred=planlet.kind,
                    goal_override=brain_goal_override,
                )
                goal_coord = sel_goal
                nav_path = nav_planner.plan(tile_grid, start=anchor, goal=goal_coord)
                if nav_path and len(nav_path) > 1:
                    # Truncate horizon to adapt quickly
                    horizon_nodes = nav_path[: min(4, len(nav_path))]
                    planlet = nav_builder.from_path(horizon_nodes, tile_grid, goal=goal_coord)
                    committed_goal = goal_coord
                    commit_steps_remaining = max(commit_steps_remaining, 6)
                else:
                    if _should_random_walk(tile_grid, anchor, pass_store) and not menu_open:
                        steps = random.randint(3, 5)
                        dirs = random.choices(["UP", "DOWN", "LEFT", "RIGHT"], k=steps)
                        script: List[ScriptOp] = []
                        for btn in dirs:
                            script.extend(_button_script(btn, press_frames=2, wait_frames=1))
                        planlet = Planlet(
                            id=str(uuid.uuid4()),
                            kind="MENU_SEQUENCE",
                            args={"random_walk": True},
                            script=script,
                            timeout_steps=len(script),
                        )
                        goal_coord = None
                        nav_path = []
                        LOGGER.debug("Sparse neighbourhood detected; executing random walk %s", dirs)
                    else:
                        committed_goal = None
                    planlet.args["goal"] = goal_coord
                    planlet.args["nav_success"] = False
            if brain_notes:
                planlet.args.setdefault("brain_notes", brain_notes)
        obs = executor.run(planlet)
        next_rgb_tensor = torch.from_numpy(obs.rgb).permute(2, 0, 1).unsqueeze(0)
        next_tile_out = tile_descriptor(next_rgb_tensor)
        next_tile_keys = tile_descriptor.tile_keys(next_tile_out.class_ids, next_tile_out.grid_shape)
        if planlet.kind == "NAVIGATE":
            step_results = planlet.args.get("step_results", []) if isinstance(planlet.args, dict) else []
            if isinstance(step_results, list):
                for step_result in step_results:
                    tile_key = step_result.get("tile_id")
                    if not isinstance(tile_key, str):
                        rc = step_result.get("rc")
                        if isinstance(rc, (list, tuple)) and len(rc) == 2:
                            rr, cc = rc
                            if (
                                isinstance(rr, (int, float))
                                and isinstance(cc, (int, float))
                            ):
                                rr_i = int(rr)
                                cc_i = int(cc)
                                if (
                                    0 <= rr_i < next_tile_out.grid_shape[0]
                                    and 0 <= cc_i < next_tile_out.grid_shape[1]
                                    and next_tile_keys
                                    and next_tile_keys[0]
                                    and rr_i < len(next_tile_keys[0])
                                    and cc_i < len(next_tile_keys[0][rr_i])
                                ):
                                    tile_key = next_tile_keys[0][rr_i][cc_i]
                    if isinstance(tile_key, str):
                        cls = tile_key.split(":", 1)[-1]
                        pass_store.update(cls, tile_key, success=bool(step_result.get("success", False)))
        sc_delta = scene_change_delta(tile_out.class_ids, next_tile_out.class_ids, next_tile_out.grid_shape)
        if (
            planlet.kind == "NAVIGATE"
            and goal_coord is not None
            and sc_delta > 0.15
            and next_tile_keys
            and next_tile_keys[0]
        ):
            gr, gc = goal_coord
            if 0 <= gr < next_tile_out.grid_shape[0] and 0 <= gc < next_tile_out.grid_shape[1]:
                row_keys = next_tile_keys[0]
                if gr < len(row_keys) and gc < len(row_keys[gr]):
                    goal_key = row_keys[gr][gc]
                    portal_scores[goal_key] = portal_scores.get(goal_key, 0.0) + sc_delta
                    if sc_delta > 0.6:
                        portal_scores[goal_key] *= 0.3
                    LOGGER.info(
                        "Portal discovered! coord=%s delta=%.3f cumulative=%.3f",
                        goal_coord,
                        sc_delta,
                        portal_scores[goal_key],
                    )
                    try:
                        portal_node_id = graph.add_entity(
                            "portal",
                            z.squeeze(0),
                            {"coord": goal_coord, "phase": phase_state, "step": obs.step_idx},
                        )
                        graph.add_relation(frame_id, portal_node_id, "seen")
                    except Exception as exc:
                        LOGGER.exception("Failed to record portal entity: %s", exc)
        if planlet.kind == "NAVIGATE" and goal_coord is not None:
            goal_manager_cli.feedback(goal_coord, bool(planlet.args.get("nav_success", False)))
        option_bank.record(planlet.kind)
        controller_state = ctrl_out.state
        # If stuck, log it but avoid resubmitting HALT requests while one is pending.
        try:
            if planlet.kind == "NAVIGATE":
                sr = planlet.args.get("step_results", []) if isinstance(planlet.args, dict) else []
                if isinstance(sr, list) and sr:
                    fails = sum(1 for s in sr if not bool(s.get("success", False)))
                    mean_delta = sum(float(s.get("delta", 0.0)) for s in sr) / max(1, len(sr))
                    if fails / max(1, len(sr)) >= 0.7 and mean_delta < 0.2:
                        LOGGER.debug(
                            "Stuck detection suppressed (fails=%s, mean_delta=%.3f, step=%s)",
                            fails,
                            mean_delta,
                            obs.step_idx,
                        )
        except Exception as exc:
            LOGGER.exception("Error while handling stuck-plan HALT logic: %s", exc)
        # Non-blocking: poll for any async brain directive and stage it for next iteration.
        if async_brain is not None and brain_directive_cli is None:
            polled = async_brain.poll()
            if polled is not None:
                directive, req_step, cand_snap = polled
                if not getattr(directive, "objective_spec", None):
                    LOGGER.warning("Dropping GPT directive (no objective_spec) step=%s", obs.step_idx)
                    directive = None
                    cand_snap = []
                if directive is not None and naming_active and not _directive_has_naming_ops(directive):
                    LOGGER.warning("Dropping GPT directive without naming ops step=%s", obs.step_idx)
                    directive = None
                    cand_snap = []
                if directive is None:
                    continue
                # Staleness guard: drop if too old or candidate overlap too small
                age = obs.step_idx - int(req_step)
                overlap = 0
                try:
                    cur_set = set(candidates)
                    snap_set = set(cand_snap)
                    overlap = len(cur_set & snap_set)
                except Exception as exc:
                    LOGGER.exception("Failed to compute candidate overlap: %s", exc)
                    overlap = 0
                overlap_ok = overlap >= min_cand_overlap
                if not overlap_ok and overlap == 0:
                    LOGGER.warning("HALT directive overlap=0 step=%s", obs.step_idx)
                    overlap_ok = (min_cand_overlap == 0 and not candidates)
                if age <= max_directive_age and overlap_ok:
                    brain_directive_cli = directive
                    if getattr(directive, "objective_spec", None):
                        objective.set_spec(directive.objective_spec, obs.step_idx)
                        LOGGER.info("Objective updated: %s", objective.summary())
                    try:
                        LOGGER.info(
                            "HALT directive applied step=%s skill=%s goal=%s ops=%s phase=%s",
                            obs.step_idx,
                            brain_directive_cli.skill,
                            brain_directive_cli.goal_index,
                            brain_directive_cli.ops,
                            (brain_directive_cli.objective_spec or {}).get("phase"),
                        )
                    except Exception as exc:
                        LOGGER.exception("Failed logging HALT directive details: %s", exc)
                    halt_cooldown = max(halt_cooldown, halt_cooldown_steps)
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
        except Exception as exc:
            LOGGER.exception("Failed to log periodic objective summary: %s", exc)
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



