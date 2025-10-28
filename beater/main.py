"""Entry-point for experiments/debug runs."""

from __future__ import annotations

import argparse
import json
import logging
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from torch.distributions import Categorical

from beater.env import EnvConfig, PyBoyEnv
from beater.executor import NavPlanletBuilder, PlanletExecutor, SpriteMovementDetector
from beater.brains import AsyncBrain, GPTBrain, GoalSuggestion
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
    ControllerConfig,
    ControllerState,
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
from beater.utils.map_metadata import MapInteractable, MapMetadataLoader, MapPortal

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


POKERED_ROOT = Path(__file__).resolve().parent.parent / "__tmp_pokered"
MAP_METADATA = MapMetadataLoader(POKERED_ROOT)
PLAYER_Y_ADDR = 0xD361
PLAYER_X_ADDR = 0xD362
DEFAULT_PORTAL_BONUS = 6.0
DEFAULT_INTERACT_BONUS = 3.0
SCRIPT_ROUTINE_MAP: Dict[str, str] = {
    "OpenRedsPC": "pc_withdraw_potion",
    "PrintRedSNESText": "snes_check",
}


def _player_map_coord(ram: Optional["np.ndarray"]) -> Optional[Tuple[int, int]]:
    if ram is None or len(ram) <= max(PLAYER_X_ADDR, PLAYER_Y_ADDR):
        return None
    y = int(ram[PLAYER_Y_ADDR])
    x = int(ram[PLAYER_X_ADDR])
    return (x, y)


def _world_to_screen(
    player_xy: Tuple[int, int],
    target_xy: Tuple[int, int],
    anchor: Tuple[int, int],
    grid_shape: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    px, py = player_xy
    tx, ty = target_xy
    row = anchor[0] + (ty - py)
    col = anchor[1] + (tx - px)
    if 0 <= row < grid_shape[0] and 0 <= col < grid_shape[1]:
        return (row, col)
    return None


def _approach_from_facing(x: int, y: int, facing: Optional[str]) -> Tuple[int, int]:
    if facing == "SPRITE_FACING_UP":
        return (x, y + 1)
    if facing == "SPRITE_FACING_DOWN":
        return (x, y - 1)
    if facing == "SPRITE_FACING_LEFT":
        return (x + 1, y)
    if facing == "SPRITE_FACING_RIGHT":
        return (x - 1, y)
    return (x, y + 1)


def _interactable_script_id(interactable: MapInteractable) -> Optional[str]:
    if interactable.routine and interactable.routine in SCRIPT_ROUTINE_MAP:
        return SCRIPT_ROUTINE_MAP[interactable.routine]
    return None





@dataclass(slots=True)
class NamingState:
    active: bool
    kind: str
    subscreen: str
    buffer: str
    buf_len: int


def _decode_poke_text(bytes_arr: "np.ndarray") -> str:
    mapping = {
        0x80: "A",
        0x81: "B",
        0x82: "C",
        0x83: "D",
        0x84: "E",
        0x85: "F",
        0x86: "G",
        0x87: "H",
        0x88: "I",
        0x89: "J",
        0x8A: "K",
        0x8B: "L",
        0x8C: "M",
        0x8D: "N",
        0x8E: "O",
        0x8F: "P",
        0x90: "Q",
        0x91: "R",
        0x92: "S",
        0x93: "T",
        0x94: "U",
        0x95: "V",
        0x96: "W",
        0x97: "X",
        0x98: "Y",
        0x99: "Z",
        0x9A: "-",
        0x50: "",
    }
    out: List[str] = []
    for value in bytes_arr:
        val = int(value)
        if val == 0x50:
            break
        out.append(mapping.get(val, "?"))
    return "".join(out)


def _naming_state(ram: Optional["np.ndarray"]) -> NamingState:
    if ram is None or len(ram) < 0xD800:
        return NamingState(False, "none", "none", "", 0)
    kind_code = int(ram[0xD07D])
    if kind_code == 0:
        kind = "player"
    elif kind_code == 1:
        kind = "rival"
    elif kind_code >= 2:
        kind = "nickname"
    else:
        kind = "none"
    subscreen_code = int(ram[0xCC25])
    if subscreen_code == 0x11:
        subscreen = "grid"
    elif subscreen_code == 0x01:
        subscreen = "presets"
    else:
        subscreen = "none"
    buf = _decode_poke_text(ram[0xCF4B : 0xCF4B + 11])
    active = kind != "none" and subscreen != "none"
    return NamingState(active, kind, subscreen, buf, len(buf))


def _confirm_burst_A(count: int = 3, gap: int = 28) -> List[str]:
    ops: List[str] = []
    for _ in range(count):
        ops.extend(["A", f"WAIT_{gap}"])
    return ops


def _naming_ops_for_state(ns: NamingState) -> List[str]:
    ops: List[str] = []
    if ns.subscreen == "grid":
        ops.extend(["B", "WAIT_12"])
    ops.extend(["DOWN", "WAIT_10", "A", "WAIT_32"])
    ops.extend(_confirm_burst_A(count=2, gap=30))
    return ops


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


def _directive_has_naming_ops(directive: GoalSuggestion) -> bool:
    if directive.skill != "MENU_SEQUENCE":
        return False
    ops = getattr(directive, "ops", None)
    return isinstance(ops, list) and len(ops) > 0


def _map_tuple(ram: Optional["np.ndarray"]) -> Optional[Tuple[int, int]]:
    if ram is None or len(ram) <= 0xD35F:
        return None
    map_group = int(ram[0xD35F])
    map_id = int(ram[0xD35E])
    return (map_group, map_id)


def _coord_in_bounds(coord: Tuple[int, int], grid_shape: Tuple[int, int]) -> bool:
    r, c = coord
    return 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]


def _known_entity_script(script_id: Optional[str]) -> List[ScriptOp]:
    script: List[ScriptOp] = []
    if script_id == "pc_withdraw_potion":
        # Open PC -> Withdraw -> Choose Potion -> Exit menus
        script.extend(_button_script("A", press_frames=3, wait_frames=4))
        script.append(ScriptOp(op="WAIT", frames=45))
        script.extend(_button_script("A", press_frames=3, wait_frames=4))
        script.append(ScriptOp(op="WAIT", frames=45))
        script.extend(_button_script("A", press_frames=3, wait_frames=4))
        script.append(ScriptOp(op="WAIT", frames=50))
        script.extend(_button_script("B", press_frames=3, wait_frames=4))
        script.append(ScriptOp(op="WAIT", frames=35))
        script.extend(_button_script("B", press_frames=3, wait_frames=4))
        script.append(ScriptOp(op="WAIT", frames=30))
    elif script_id == "snes_check":
        script.extend(_button_script("A", press_frames=3, wait_frames=4))
        script.append(ScriptOp(op="WAIT", frames=40))
        script.extend(_button_script("B", press_frames=3, wait_frames=4))
        script.append(ScriptOp(op="WAIT", frames=20))
    else:
        # Default simple interaction tap
        script.extend(_button_script("A", press_frames=3, wait_frames=4))
        script.append(ScriptOp(op="WAIT", frames=20))
    return script


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
    known_entity_state: Dict[Tuple[int, int], Dict[str, set[str]]] = {}
    current_phase_tag = "boot"
    menu_cooldown = 0
    naming_cooldown = 0
    naming_cooldown_steps = int((brain_cfg or {}).get("naming_cooldown_steps", 600))
    naming_retry_guard_steps = int((brain_cfg or {}).get("naming_retry_guard_steps", 240))
    naming_fallback_limit = int((brain_cfg or {}).get("naming_fallback_limit", 6))
    novelty_halt_cooldown_steps = int((brain_cfg or {}).get("novelty_halt_cooldown_steps", 600))
    last_naming_script_step = -10**9
    naming_fallback_attempts = 0
    last_novelty_halt_step = -10**9
    seen_maps: set[Tuple[int, int]] = set()
    pending_novelty_hint: Optional[Dict[str, Any]] = None
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
            "portal_triggered": 0.6,
            "interact_complete": 1.0,
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
        map_tuple_before = _map_tuple(obs.ram)
        if map_tuple_before is not None:
            seen_maps.add(map_tuple_before)
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
            candidates = list(candidates)
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
            active_known_entities: List[Dict[str, Any]] = []
            forced_planlet: Optional[Planlet] = None
            forced_planlet_meta: Optional[Dict[str, Any]] = None
            player_map_xy = _player_map_coord(obs.ram)
            current_map_label: Optional[str] = None
            interact_bonus_map: Dict[Tuple[int, int], float] = {}
            if map_tuple_before is not None:
                map_state = known_entity_state.setdefault(
                    map_tuple_before,
                    {"interact_done": set(), "portal_done": set()},
                )
            else:
                map_state = {"interact_done": set(), "portal_done": set()}
            map_state.setdefault("interact_done", set())
            map_state.setdefault("portal_done", set())
            normalized_portals: set[Tuple[int, int]] = set()
            for coord in map_state.get("portal_done", set()):
                if isinstance(coord, (list, tuple)) and len(coord) == 2:
                    normalized_portals.add((int(coord[0]), int(coord[1])))
            map_state["portal_done"] = normalized_portals
            interact_done: set[str] = map_state.get("interact_done", set())
            portals_done: set[Tuple[int, int]] = normalized_portals

            def _insert_candidate(coord: Tuple[int, int]) -> None:
                nonlocal candidates
                if coord not in candidates:
                    candidates = [coord] + [c for c in candidates if c != coord]

            if map_tuple_before is not None and player_map_xy is not None:
                map_id = map_tuple_before[1]
                map_label = MAP_METADATA.map_label(map_id)
                current_map_label = map_label
                for portal in MAP_METADATA.get_portals(map_id):
                    portal_screen = _world_to_screen(
                        player_map_xy, (portal.x, portal.y), anchor, tile_out.grid_shape
                    )
                    if portal_screen is None:
                        continue
                    portal_coord = (int(portal_screen[0]), int(portal_screen[1]))
                    _insert_candidate(portal_screen)
                    pr, pc = portal_screen
                    already_used = portal_coord in portals_done
                    portal_bonus = DEFAULT_PORTAL_BONUS if not already_used else DEFAULT_PORTAL_BONUS * 0.25
                    try:
                        key = tile_grid[pr][pc]
                        portal_scores[key] = portal_scores.get(key, 0.0) + portal_bonus
                    except (IndexError, ValueError):
                        LOGGER.debug(
                            "Portal coord out of bounds for map %s coord=%s screen=%s",
                            map_tuple_before,
                            (portal.x, portal.y),
                            portal_screen,
                        )
                        continue
                    active_known_entities.append(
                        {
                            "type": "portal",
                            "label": portal.label or f"portal@{portal.x},{portal.y}",
                            "coord": [pr, pc],
                            "map_coord": [portal.y, portal.x],
                            "dest": portal.dest_map,
                            "bonus": portal_bonus,
                            "completed": already_used,
                            "prefer": not already_used,
                            "map_label": map_label,
                        }
                    )
                for interact in MAP_METADATA.get_interactables(map_id):
                    label = interact.label or f"interactable@{interact.x},{interact.y}"
                    script_id = _interactable_script_id(interact)
                    approach_world = _approach_from_facing(interact.x, interact.y, interact.facing)
                    approach_screen = _world_to_screen(
                        player_map_xy, approach_world, anchor, tile_out.grid_shape
                    )
                    coord_screen = _world_to_screen(
                        player_map_xy, (interact.x, interact.y), anchor, tile_out.grid_shape
                    )
                    if approach_screen is not None and label not in interact_done:
                        _insert_candidate(approach_screen)
                        interact_bonus_map[tuple(approach_screen)] = DEFAULT_INTERACT_BONUS
                    entity_entry: Dict[str, Any] = {
                        "type": interact.kind,
                        "label": label,
                        "map_coord": [interact.y, interact.x],
                        "facing": interact.facing,
                        "routine": interact.routine,
                        "text": interact.text_id,
                        "completed": label in interact_done,
                        "prefer": label not in interact_done,
                        "map_label": map_label,
                        "bonus": DEFAULT_INTERACT_BONUS,
                    }
                    if script_id is not None:
                        entity_entry["script"] = script_id
                    if coord_screen is not None:
                        entity_entry["coord"] = [coord_screen[0], coord_screen[1]]
                    if approach_screen is not None:
                        entity_entry["approach"] = [approach_screen[0], approach_screen[1]]
                    if "coord" not in entity_entry and "approach" not in entity_entry:
                        continue
                    active_known_entities.append(entity_entry)
                    if (
                        forced_planlet is None
                        and label not in interact_done
                        and script_id is not None
                        and approach_screen is not None
                        and anchor == approach_screen
                    ):
                        script_ops = _known_entity_script(script_id)
                        timeout = max(len(script_ops) * 2 + 60, 90)
                        forced_planlet = Planlet(
                            id=str(uuid.uuid4()),
                            kind="MENU_SEQUENCE",
                            args={
                                "known_entity": label,
                                "approach": tuple(approach_screen),
                                "target": tuple(coord_screen) if coord_screen is not None else None,
                                "script_id": script_id,
                            },
                            script=script_ops,
                            timeout_steps=timeout,
                        )
                        forced_planlet_meta = {
                            "label": label,
                            "approach": tuple(approach_screen),
                            "map_coord": (interact.x, interact.y),
                        }
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
                                "portal_triggered": 1.0,
                                "interact_complete": 0.5,
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
            ns = _naming_state(obs.ram)
            naming_current_name = ns.buffer
            naming_screen = ns.active
            if naming_screen:
                menu_cooldown = max(menu_cooldown, 120)
                halt_cooldown = 0
                phase_label = f"naming:{ns.kind}:{ns.subscreen}"
                if current_phase_tag != phase_label:
                    current_phase_tag = phase_label
                    naming_cooldown = naming_cooldown_steps
                    naming_fallback_attempts = 0
                    try:
                        naming_spec = {
                            "phase": phase_label,
                            "reward_weights": {
                                "scene_change": 0.2,
                                "menu_progress": 0.8,
                                "name_committed": 1.0,
                                "portal_triggered": 0.0,
                                "interact_complete": 0.0,
                            },
                            "timeouts": {"ttl_steps": 600},
                            "skill_bias": "menu",
                        }
                        objective.set_spec(naming_spec, obs.step_idx)
                        LOGGER.info(
                            "Objective updated (phase->%s kind=%s subscreen=%s): %s",
                            phase_label,
                            ns.kind,
                            ns.subscreen,
                            objective.summary(),
                        )
                    except Exception as exc:
                        LOGGER.exception("Failed to apply naming objective: %s", exc)
            elif current_phase_tag.startswith("naming:"):
                current_phase_tag = "boot"
                naming_cooldown = 0
                naming_fallback_attempts = 0
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
            naming_active = ns.active
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
            brain_goal_override: Optional[Tuple[int, int]] = None
            brain_skill: Optional[str] = None
            brain_notes: Optional[str] = None
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
                and (
                    not naming_active
                    or (
                        naming_cooldown == 0
                        and (obs.step_idx - last_naming_script_step) >= naming_retry_guard_steps
                    )
                )
            )
            if should_query_brain:
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
                prefer_coords: set[Tuple[int, int]] = set()
                for ent in active_known_entities:
                    if not ent.get("prefer"):
                        continue
                    if "approach" in ent:
                        try:
                            prefer_coords.add(tuple(int(v) for v in ent["approach"]))
                        except Exception:
                            continue
                    elif "coord" in ent:
                        try:
                            prefer_coords.add(tuple(int(v) for v in ent["coord"]))
                        except Exception:
                            continue
                cand_meta = []
                for (r, c) in candidates:
                    key = tile_grid[r][c]
                    cls = key.split(":", 1)[-1]
                    est = pass_store.get_estimate(cls, key)
                    cand_meta.append({
                        "passability": est.blended,
                        "fail_count": goal_manager_cli.get_fail_count((r, c)),
                        "portal_score": portal_scores.get(key, 0.0),
                        "interact_bonus": interact_bonus_map.get((r, c), 0.0),
                        "known_prefer": (r, c) in prefer_coords,
                    })

                context_payload = {
                    "step": obs.step_idx,
                    "assoc_matches": len(assoc),
                    "passability": nav_est.blended,
                    "ascii_map": ascii_map,
                    "phase": phase_state,
                    "menu_open": menu_open,
                    "candidate_meta": cand_meta,
                    "web_search_policy": "Use web_search only when uncertain or when a new scene/state change is detected.",
                    "known_entity_hint": "Prioritize known_entities with prefer=true and mark them completed once resolved.",
                }
                if map_tuple_before is not None:
                    map_info = {"group": map_tuple_before[0], "id": map_tuple_before[1]}
                    if current_map_label:
                        map_info["label"] = current_map_label
                    context_payload["map"] = map_info
                if pending_novelty_hint:
                    context_payload["novelty"] = pending_novelty_hint
                if active_known_entities:
                    context_payload["known_entities"] = active_known_entities
                if unseen_for_prompt:
                    context_payload["unseen_candidates"] = unseen_for_prompt[:brain_candidate_preview]
                context_payload["naming_state"] = {
                    "active": naming_screen,
                    "kind": ns.kind,
                    "subscreen": ns.subscreen,
                    "buffer": ns.buffer,
                    "buffer_len": ns.buf_len,
                }
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
                        naming_current_name = ns.buffer
                        context_payload["current_name"] = naming_current_name or "empty"
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
                if pending_novelty_hint:
                    last_novelty_halt_step = obs.step_idx
                    pending_novelty_hint = None
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
                since_script = obs.step_idx - last_naming_script_step
                if since_script < naming_retry_guard_steps:
                    preferred_kind = "WAIT"
                else:
                    preferred_kind = "MENU_SEQUENCE"
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

            if forced_planlet is not None:
                planlet = forced_planlet
            else:
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
                        last_naming_script_step = obs.step_idx
                        LOGGER.info(
                            "Applied brain MENU_SEQUENCE ops step=%s ops=%s",
                            obs.step_idx,
                            menu_ops,
                        )
                elif planlet.kind == "MENU_SEQUENCE" and naming_active:
                    since_script = obs.step_idx - last_naming_script_step
                    if since_script < naming_retry_guard_steps:
                        remaining = max(10, naming_retry_guard_steps - since_script)
                        planlet = Planlet(
                            id=str(uuid.uuid4()),
                            kind="WAIT",
                            args={"naming_guard": True, "remaining": remaining},
                            script=[
                                ScriptOp(op="WAIT", frames=remaining)
                            ],
                            timeout_steps=remaining,
                        )
                        LOGGER.debug(
                            "Naming guard active (remaining=%s); skipping fallback ops at step=%s",
                            remaining,
                            obs.step_idx,
                        )
                    else:
                        if naming_fallback_attempts >= naming_fallback_limit:
                            wait_frames = 90
                            planlet = Planlet(
                                id=str(uuid.uuid4()),
                                kind="WAIT",
                                args={"naming_guard": True, "cooldown": True},
                                script=[ScriptOp(op="WAIT", frames=wait_frames)],
                                timeout_steps=wait_frames,
                            )
                            LOGGER.debug(
                                "Naming fallback limit reached; pausing for GPT (step=%s attempts=%s)",
                                obs.step_idx,
                                naming_fallback_attempts,
                            )
                        else:
                            menu_ops = _naming_ops_for_state(ns)
                            script = []
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
                                press_frames = 3
                                settle_frames = 4
                                script.append(ScriptOp(op="PRESS", button=btn, frames=press_frames))
                                total_frames += press_frames
                                script.append(ScriptOp(op="WAIT", frames=settle_frames))
                                total_frames += settle_frames
                                script.append(ScriptOp(op="RELEASE", button=btn, frames=0))
                            if script:
                                planlet.args["ops"] = menu_ops
                                planlet.script = script
                                planlet.timeout_steps = max(planlet.timeout_steps, total_frames + 30)
                                last_naming_script_step = obs.step_idx
                                naming_fallback_attempts += 1
                                LOGGER.info(
                                    "Applied fallback naming ops step=%s subscreen=%s ops=%s (attempt=%s)",
                                    obs.step_idx,
                                    ns.subscreen,
                                    menu_ops,
                                    naming_fallback_attempts,
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
                    interact_bonus = interact_bonus_map.get((g[0], g[1]), 0.0)
                    dist = len(path)
                    return avg_pass + portal_bonus + interact_bonus - 0.01 * dist

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
        prev_ns = ns
        obs = executor.run(planlet)
        if forced_planlet_meta and map_tuple_before is not None:
            approach_coord = tuple(forced_planlet_meta.get("approach", ()))
            label = forced_planlet_meta.get("label")
            state_entry = known_entity_state.setdefault(
                map_tuple_before,
                {"interact_done": set(), "portal_done": set()},
            )
            state_entry.setdefault("interact_done", set()).add(label)
            if len(approach_coord) == 2:
                try:
                    goal_manager_cli.feedback(approach_coord, True)
                except Exception as exc:
                    LOGGER.debug("Failed to provide feedback for known entity %s: %s", label, exc)
        next_ns = _naming_state(obs.ram)
        map_tuple_after = _map_tuple(obs.ram)
        if prev_ns.active and not next_ns.active:
            brain_directive_cli = None
            halt_cooldown = max(halt_cooldown, halt_cooldown_steps)
            last_naming_script_step = obs.step_idx
            LOGGER.info("Naming complete step=%s buffer='%s'", obs.step_idx, prev_ns.buffer)
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
        if (
            map_tuple_before is not None
            and map_tuple_after is not None
            and map_tuple_after != map_tuple_before
        ):
            portal_scores.clear()
            new_map = map_tuple_after not in seen_maps
            seen_maps.add(map_tuple_after)
            if (
                async_brain is not None
                and phase_halt_pending is None
                and pending_novelty_hint is None
                and (new_map or phase_delta > 0.45)
                and (obs.step_idx - last_novelty_halt_step) >= novelty_halt_cooldown_steps
            ):
                pending_novelty_hint = {
                    "reason": "new_map" if new_map else "scene_shift",
                    "map": {"group": map_tuple_after[0], "id": map_tuple_after[1]},
                    "from": {"group": map_tuple_before[0], "id": map_tuple_before[1]},
                }
                phase_halt_pending = current_phase_tag
            if map_tuple_before is not None and goal_coord is not None:
                state_entry = known_entity_state.setdefault(
                    map_tuple_before,
                    {"interact_done": set(), "portal_done": set()},
                )
                state_entry.setdefault("portal_done", set()).add(
                    (int(goal_coord[0]), int(goal_coord[1]))
                )
            if (
                planlet.kind == "NAVIGATE"
                and goal_coord is not None
                and tile_grid
                and 0 <= goal_coord[0] < len(tile_grid)
                and 0 <= goal_coord[1] < len(tile_grid[goal_coord[0]])
            ):
                goal_key = tile_grid[goal_coord[0]][goal_coord[1]]
                portal_scores[goal_key] = 1.0
                LOGGER.info(
                    "Portal discovered! map %s -> %s coord=%s score=%.3f",
                    map_tuple_before,
                    map_tuple_after,
                    goal_coord,
                    portal_scores[goal_key],
                )
                try:
                    portal_node_id = graph.add_entity(
                        "portal",
                        z.squeeze(0),
                        {"coord": goal_coord, "phase": phase_state, "step": obs.step_idx, "map": map_tuple_after},
                    )
                    graph.add_relation(frame_id, portal_node_id, "seen")
                except Exception as exc:
                    LOGGER.exception("Failed to record portal entity: %s", exc)
            else:
                LOGGER.info(
                    "Map changed %s -> %s without a NAVIGATE goal (plan=%s)",
                    map_tuple_before,
                    map_tuple_after,
                    planlet.kind,
                )
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
                if directive is not None and next_ns.active:
                    has_ops = _directive_has_naming_ops(directive)
                    if directive.skill != "MENU_SEQUENCE" or not has_ops:
                        LOGGER.info(
                            "Dropping GPT directive while naming (skill=%s, has_ops=%s)",
                            directive.skill,
                            has_ops,
                        )
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




