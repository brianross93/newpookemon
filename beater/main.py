"""Entry-point for experiments/debug runs."""

from __future__ import annotations

import argparse
import logging
import random
import uuid
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
from beater.sr_memory import EntityGraph, GraphOp
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
from beater.utils.ram_view import NamingState, RamView, decode_ram_view
from beater.utils.ram_map import (
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    W_TILE_MAP_ADDR,
    boundary_exit_candidates,
    passable_mask,
    read_tile_map,
    tokens_from_tile_map,
)

LOGGER = logging.getLogger("beater")
GRAPH_OPS: tuple[GraphOp, ...] = tuple(GraphOp)
_PASSABLE_LOGGED: set[Tuple[int, int]] = set()


def _passable_window(
    grid: Sequence[Sequence[bool]],
    center: Tuple[int, int],
    radius: int = 2,
) -> str:
    rows: List[str] = []
    for r in range(center[0] - radius, center[0] + radius + 1):
        cols: List[str] = []
        for c in range(center[1] - radius, center[1] + radius + 1):
            if r < 0 or c < 0 or r >= len(grid) or c >= len(grid[r]):
                cols.append(" ")
            elif (r, c) == center:
                cols.append("@")
            else:
                cols.append("." if grid[r][c] else "#")
        rows.append("".join(cols))
    return "\n".join(rows)


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
PASSABLE_OVERRIDES: Dict[int, set[int]] = {
    # Bedroom upstairs doorway tiles mis-labelled as blocked.
    0x26: {0x7F, 0x21, 0x39, 0x64},
    0x18: {0x7F, 0x21, 0x39, 0x64},
    0x12: {0x7F, 0x21, 0x39, 0x64},
    0x00: {0x39, 0x64},
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
        ram_view = decode_ram_view(obs.ram if obs.ram is not None else ())
        input_phase = ram_view.input_state_label
        if input_phase not in ("title_screen", "oak_intro", "disabled"):
            LOGGER.info(
                "Boot sequence reached interactive menu (attempt=%s step=%s input_state=%s)",
                attempt + 1,
                env.step_idx,
                input_phase,
            )
            return obs, tile_out.class_ids.clone(), tile_out.grid_shape
        if input_phase == "disabled":
            continue
        if detect_menu_open(obs.ram, tile_grid):
            LOGGER.info(
                "Boot sequence detected menu via heuristic (attempt=%s step=%s input_state=%s)",
                attempt + 1,
                env.step_idx,
                input_phase,
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
    passable_grid: List[List[bool]],
    anchor: Tuple[int, int],
) -> bool:
    """Heuristic: return True when the agent is boxed in by impassable tiles."""

    if not passable_grid:
        return False
    rows = len(passable_grid)
    cols = len(passable_grid[0])
    r, c = anchor
    known = 0
    open_tiles = 0
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
            continue
        known += 1
        if passable_grid[nr][nc]:
            open_tiles += 1
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


def build_nav_planner(cfg: dict[str, Any]) -> NavPlanner:
    nav_cfg = cfg.get("nav_planner", {})
    config = NavPlannerConfig(
        max_expansions=int(nav_cfg.get("max_expansions", 1000)),
    )
    return NavPlanner(config)


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
    portal_scores: Dict[str, float] = {}
    initial_class_ids: Optional[torch.Tensor] = None
    initial_grid_shape: Optional[Tuple[int, int]] = None
    known_entity_state: Dict[Tuple[int, int], Dict[str, set[str]]] = {}
    current_phase_tag = "boot"
    menu_cooldown = 0
    naming_cooldown = 0
    naming_cooldown_steps = int((brain_cfg or {}).get("naming_cooldown_steps", 600))
    naming_retry_guard_steps = int((brain_cfg or {}).get("naming_retry_guard_steps", 240))
    novelty_halt_cooldown_steps = int((brain_cfg or {}).get("novelty_halt_cooldown_steps", 600))
    last_naming_script_step = -10**9
    last_novelty_halt_step = -10**9
    seen_maps: set[Tuple[int, int]] = set()
    pending_novelty_hint: Optional[Dict[str, Any]] = None
    committed_goal: Optional[Tuple[int, int]] = None
    commit_steps_remaining = 0
    halt_cooldown = 0
    phase_halt_pending: Optional[str] = None
    nav_planner = build_nav_planner(cfg)
    nav_builder = NavPlanletBuilder()
    movement_detector = SpriteMovementDetector()
    executor = PlanletExecutor(env, movement_detector)
    trainer, train_cfg = build_trainer(cfg, controller, affordance)
    LOGGER.info("Environment initialized with ROM=%s", env.config.rom_path)
    controller_state: Optional[ControllerState] = None
    interactive_context = "interactive"
    step_target = args.max_steps  # 0 => run indefinitely
    start_step = env.step_idx
    brain_directive_cli: Optional[GoalSuggestion] = None
    last_rollout_stats: dict[str, Any] = {}
    naming_debug_samples = 0

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
        "skill_bias": "overworld",
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
        ram_view: RamView = decode_ram_view(obs.ram if obs.ram is not None else ())
        if controller_state is None:
            controller_state = controller.init_state(batch_size=1, device=rgb_tensor.device)
        naming_cursor: Optional[Tuple[int, int]] = None
        naming_current_name: str = ""
        with torch.inference_mode():
            z, rnd = perception(rgb_tensor, ram_tensor, context=interactive_context)
            frame_id = graph.add_entity("frame", z.squeeze(0), {"step_idx": obs.step_idx})
            assoc = graph.assoc(z.squeeze(0), top_k=1)
            tile_out = tile_descriptor(rgb_tensor)
            grid_shape = tile_out.grid_shape
            tile_grid: List[List[str]] = []
            passable_grid: List[List[bool]] = []
            tile_map: Optional[List[List[int]]] = None
            ram_data = obs.ram
            ram_ready = (
                ram_data is not None
                and len(ram_data) >= (W_TILE_MAP_ADDR + SCREEN_HEIGHT * SCREEN_WIDTH)
            )
            if ram_ready:
                try:
                    tile_map = read_tile_map(ram_data)
                    tile_grid = tokens_from_tile_map(tile_map)
                except Exception as exc:
                    LOGGER.debug("Failed to read tile map at step=%s: %s", obs.step_idx, exc)
                    tile_map = None
                    tile_grid = []
            if not tile_grid:
                tile_keys = tile_descriptor.tile_keys(tile_out.class_ids, grid_shape)
                tile_grid = tile_descriptor.reshape_tokens(tile_keys[0], grid_shape)
            if tile_map is not None:
                map_id = map_tuple_before[1] if map_tuple_before is not None else None
                collisions = MAP_METADATA.collision_ids(map_id) if map_id is not None else set()
                if map_id is not None and map_id in PASSABLE_OVERRIDES:
                    collisions = set(collisions) - PASSABLE_OVERRIDES[map_id]
                    LOGGER.debug(
                        "Applied passable overrides map=%s tiles=%s",
                        map_id,
                        PASSABLE_OVERRIDES[map_id],
                    )
                if map_id is not None and collisions:
                    LOGGER.debug(
                        "Collision snapshot map=%s contains_0x64=%s size=%s",
                        map_id,
                        (0x64 in collisions),
                        len(collisions),
                    )
                passable_grid = passable_mask(tile_map, collisions)
            else:
                passable_grid = [[True for _ in range(grid_shape[1])] for _ in range(grid_shape[0])]
            anchor = goal_manager_cli.player_anchor(grid_shape)
            anchor_passable = (
                0 <= anchor[0] < len(passable_grid)
                and 0 <= anchor[1] < len(passable_grid[anchor[0]])
                and passable_grid[anchor[0]][anchor[1]]
            )
            if (
                (map_tuple_before and map_tuple_before not in _PASSABLE_LOGGED)
                or obs.step_idx % 500 == 0
                or not anchor_passable
            ):
                tile_preview = ""
                if tile_map is not None:
                    r, c = anchor
                    if 0 <= r - 1 < len(tile_map) and 0 <= c < len(tile_map[r - 1]):
                        tile_up = tile_map[r - 1][c]
                        tile_preview = f" tile_up=0x{tile_up:02X}"
                up_blocked = False
                tile_preview = ""
                if tile_map is not None:
                    r, c = anchor
                    if 0 <= r - 1 < len(tile_map) and 0 <= c < len(tile_map[r - 1]):
                        tile_up = tile_map[r - 1][c]
                        up_blocked = tile_up in collisions if 'collisions' in locals() else False
                        tile_preview = f" tile_up=0x{tile_up:02X} blocked={up_blocked}"
                    if 0 <= r - 2 < len(tile_map) and 0 <= c < len(tile_map[r - 2]):
                        tile_far = tile_map[r - 2][c]
                        tile_preview += f" tile_far=0x{tile_far:02X} far_blocked={(tile_far in collisions) if 'collisions' in locals() else False}"
                else:
                    tile_preview = ""
                LOGGER.debug(
                    "Passable window step=%s map=%s anchor=%s%s\n%s",
                    obs.step_idx,
                    map_tuple_before,
                    anchor,
                    tile_preview,
                    _passable_window(passable_grid, anchor),
                )
                LOGGER.debug(
                    "Nav candidates step=%s map=%s collisions_sample=%s",
                    obs.step_idx,
                    map_tuple_before,
                    sorted(list(collisions))[:5] if 'collisions' in locals() else [],
                )
                if map_tuple_before:
                    _PASSABLE_LOGGED.add(map_tuple_before)
            candidates = list(goal_manager_cli.peek_candidates(grid_shape, brain_candidate_preview))
            boundary_targets = boundary_exit_candidates(
                passable_grid, anchor, limit=max(4, brain_candidate_preview * 2)
            )
            for coord in boundary_targets:
                if coord not in candidates:
                    candidates.insert(0, coord)
            unseen_for_prompt = goal_manager_cli.unseen_candidates(
                grid_shape, tile_grid, brain_candidate_preview
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
            player_map_xy = (ram_view.player.x, ram_view.player.y)
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

            if map_tuple_before in ((0, 0), (18, 38), (38, 38), (0, 38)):
                door_coord = (anchor[0] - 2, anchor[1])
                hallway_coord = (anchor[0] - 1, anchor[1])
                for coord in (door_coord, hallway_coord):
                    if _coord_in_bounds(coord, tile_out.grid_shape):
                        _insert_candidate(coord)
                        interact_bonus_map[coord] = interact_bonus_map.get(coord, 0.0) + DEFAULT_PORTAL_BONUS
                        try:
                            key = tile_grid[coord[0]][coord[1]]
                            portal_scores[key] = portal_scores.get(key, 0.0) + DEFAULT_PORTAL_BONUS * 2.0
                        except Exception:
                            continue

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
                    except Exception as exc:
                        LOGGER.exception("Failed to set downstairs objective: %s", exc)
            ns = ram_view.naming
            naming_current_name = ns.buffer
            naming_screen = ns.active
            if naming_screen and naming_debug_samples < 40:
                try:
                    LOGGER.info(
                        "Naming state sample step=%s kind=%s subscreen=%s type=%02X submit=%02X case=%02X buf='%s'",
                        obs.step_idx,
                        ns.kind,
                        ns.subscreen,
                        ns.raw_kind,
                        ns.submit_flag,
                        ns.alphabet_case,
                        ns.buffer or "",
                    )
                except Exception:
                    pass
                naming_debug_samples += 1
            if naming_screen:
                menu_cooldown = max(menu_cooldown, 120)
                halt_cooldown = 0
                phase_label = f"naming:{ns.kind}:{ns.subscreen}"
                if current_phase_tag != phase_label:
                    current_phase_tag = phase_label
                    naming_cooldown = naming_cooldown_steps
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
                try:
                    objective.set_spec(default_spec, obs.step_idx)
                    LOGGER.info("Objective reset post-naming: %s", objective.summary())
                except Exception as exc:
                    LOGGER.exception("Failed to reset naming objective: %s", exc)
                phase_halt_pending = "boot"
            menu_detected = ram_view.menu.open or ("ui:textbox" in ram_view.tags)
            if not menu_detected:
                menu_detected = detect_menu_open(obs.ram, tile_grid)
            if menu_detected:
                menu_cooldown = max(menu_cooldown, 60)
            menu_open = menu_open or menu_detected
            phase_state = current_phase_tag
            naming_active = ns.active
            menu_allowed = menu_open or naming_active or ("ui:menu" in ram_view.tags)
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
                    if portal_scores:
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
                    if r < 0 or c < 0 or r >= len(tile_grid) or c >= len(tile_grid[r]):
                        continue
                    key = tile_grid[r][c]
                    passable_flag = (
                        0 <= r < len(passable_grid)
                        and 0 <= c < len(passable_grid[r])
                        and passable_grid[r][c]
                    )
                    cand_meta.append({
                        "passability": 1.0 if passable_flag else 0.0,
                        "fail_count": goal_manager_cli.get_fail_count((r, c)),
                        "portal_score": portal_scores.get(key, 0.0),
                        "interact_bonus": interact_bonus_map.get((r, c), 0.0),
                        "boundary_hint": (r, c) in boundary_targets,
                        "known_prefer": (r, c) in prefer_coords,
                    })

                context_payload = {
                    "step": obs.step_idx,
                    "assoc_matches": len(assoc),
                    "passability": 1.0 if anchor_passable else 0.0,
                    "ascii_map": ascii_map,
                    "phase": phase_state,
                    "menu_open": menu_open,
                    "input_state": {
                        "value": ram_view.input_state_value,
                        "label": ram_view.input_state_label,
                        "text_state": {
                            "value": ram_view.text_state_value,
                            "label": ram_view.text_state_label,
                            "flags": sorted(ram_view.text_flag_labels),
                        },
                    },
                    "ui_tags": sorted(ram_view.tags),
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
                    "buffer_len": len(ns.buffer),
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
            fallback_nav_kind = (
                "NAVIGATE"
                if "NAVIGATE" in skill_vocab
                else ("WAIT" if "WAIT" in skill_vocab else skill_vocab[0])
            )
            fallback_nav_idx = skill_vocab.index(fallback_nav_kind)
            if not menu_allowed and brain_skill == "MENU_SEQUENCE":
                LOGGER.info(
                    "Suppressing brain MENU_SEQUENCE request (step=%s) because no menu is open",
                    obs.step_idx,
                )
                brain_skill = fallback_nav_kind if fallback_nav_kind in skill_vocab else None
            preferred_kind = brain_skill if brain_skill in skill_vocab else option_bank.suggest()
            menu_bias_active = bias == "menu"
            if naming_active:
                since_script = obs.step_idx - last_naming_script_step
                if since_script < naming_retry_guard_steps:
                    preferred_kind = "WAIT"
                else:
                    preferred_kind = "MENU_SEQUENCE"
            else:
                if menu_allowed and bias == "menu" and preferred_kind == "WAIT":
                    preferred_kind = "MENU_SEQUENCE"
                elif bias == "overworld" and preferred_kind == "WAIT":
                    preferred_kind = fallback_nav_kind
                if menu_allowed and preferred_kind not in ("MENU_SEQUENCE", "INTERACT"):
                    preferred_kind = "MENU_SEQUENCE"
                if not menu_allowed and preferred_kind == "MENU_SEQUENCE":
                    preferred_kind = fallback_nav_kind
                if (
                    not menu_allowed
                    and preferred_kind == "INTERACT"
                    and brain_skill != "INTERACT"
                ):
                    preferred_kind = fallback_nav_kind
            if preferred_kind is None:
                preferred_kind = fallback_nav_kind
            if not menu_allowed and "MENU_SEQUENCE" in skill_vocab:
                menu_idx = skill_vocab.index("MENU_SEQUENCE")
                if int(skill_action.item()) == menu_idx:
                    skill_action = skill_action.new_tensor(fallback_nav_idx)
                    LOGGER.debug(
                        "Redirected MENU_SEQUENCE skill index to %s at step=%s",
                        fallback_nav_kind,
                        obs.step_idx,
                    )
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
                wait_frames = max(30, naming_retry_guard_steps // 4)
                planlet = Planlet(
                    id=str(uuid.uuid4()),
                    kind="WAIT",
                    args={"naming_idle": True},
                    script=[ScriptOp(op="WAIT", frames=wait_frames)],
                    timeout_steps=wait_frames,
                )
                LOGGER.debug(
                    "Naming idle wait issued step=%s frames=%s",
                    obs.step_idx,
                    wait_frames,
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
            # Ensure later code has defaults regardless of auto-close branch.
            goal_coord = None
            nav_path: List[tuple[int, int]] = []
            if planlet.kind == "NAVIGATE":
                # Passability-weighted goal selection
                def _score_goal(g: tuple[int, int]) -> float:
                    path = nav_planner.plan(passable_grid, start=anchor, goal=g)
                    LOGGER.debug(
                        "Score goal step=%s goal=%s path_len=%s anchor=%s",
                        obs.step_idx,
                        g,
                        len(path) if path else 0,
                        anchor,
                    )
                    if not path or len(path) < 2:
                        return -1e9
                    # Score by mean blended passability along first few steps
                    horizon = min(6, len(path))
                    open_steps = 0
                    for (r, c) in path[1:horizon]:
                        if (
                            0 <= r < len(passable_grid)
                            and 0 <= c < len(passable_grid[r])
                            and passable_grid[r][c]
                        ):
                            open_steps += 1
                    avg_pass = open_steps / max(1, horizon - 1)
                    goal_key = ""
                    if 0 <= g[0] < len(tile_grid) and 0 <= g[1] < len(tile_grid[g[0]]):
                        goal_key = tile_grid[g[0]][g[1]]
                    portal_bonus = portal_scores.get(goal_key, 0.0) * 0.5 if goal_key else 0.0
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
                nav_path = (
                    nav_planner.plan(passable_grid, start=anchor, goal=goal_coord)
                    if goal_coord is not None
                    else []
                )
                LOGGER.debug(
                    "Nav plan step=%s goal=%s path_len=%s anchor=%s path_head=%s",
                    obs.step_idx,
                    goal_coord,
                    len(nav_path) if nav_path else 0,
                    anchor,
                    nav_path[:5] if nav_path else [],
                )
                if nav_path and len(nav_path) > 1:
                    # Truncate horizon to adapt quickly
                    horizon_nodes = nav_path[: min(4, len(nav_path))]
                    planlet = nav_builder.from_path(horizon_nodes, tile_grid, goal=goal_coord)
                    LOGGER.debug(
                        "Nav plan script step=%s path=%s script_len=%s first_ops=%s",
                        obs.step_idx,
                        horizon_nodes,
                        len(planlet.script),
                        planlet.script[:4],
                    )
                    committed_goal = goal_coord
                    commit_steps_remaining = max(commit_steps_remaining, 6)
                else:
                    if _should_random_walk(passable_grid, anchor) and not menu_open:
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
        next_ns = decode_ram_view(obs.ram if obs.ram is not None else ()).naming
        map_tuple_after = _map_tuple(obs.ram)
        if prev_ns.active and not next_ns.active:
            brain_directive_cli = None
            halt_cooldown = max(halt_cooldown, halt_cooldown_steps)
            last_naming_script_step = obs.step_idx
            LOGGER.info("Naming complete step=%s buffer='%s'", obs.step_idx, prev_ns.buffer)
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
            "nav_passable": anchor_passable,
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
        last_rollout_stats.get("nav_passable"),
        last_rollout_stats.get("planlet"),
        last_rollout_stats.get("nav_path_len"),
        train_stats,
    )
    if async_brain is not None:
        async_brain.shutdown()
    env.close()


if __name__ == "__main__":  # pragma: no cover
    main()




