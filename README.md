# Pokemon Beater

Discovery-first SR-FBAM agent targeting Pokemon Blue automation using sparse symbolic memory, tile-level discovery, and planlet-based control.

## System Snapshot

- **Perception:** Slot-friendly encoder with per-actor temporal smoothing plus a tile descriptor that clusters 8x8 RGBA patches into unsupervised terrain classes feeding the Bayesian passability store.
- **SR-Memory & Navigation:** Entity graph for ASSOC/FOLLOW/WRITE/HALT, Bayesian passability with class+instance Betas, Thompson-sampling nav planner, `GoalManager` waypoints, passability/portal-weighted goal scoring with short NAV horizons + stuck→HALT recovery, and a `NavPlanletBuilder` that turns sampled paths into executor-ready planlets.
- **Executor:** Sprite-based pose tracker (central ROI centroid) decides whether movement succeeded and updates passability per-tile with rich telemetry.
- **Policy & Training:** GRU controller + AffordancePrior, OptionBank-biased skills, and an environment-grounded PPO loop (multi-actor rollout collector, replay buffer, learner) so gate/skill decisions learn from real PyBoy traces and waypoint feedback.
- **Tooling:** CLI bootstraps PyBoy, executes a nav planlet end-to-end, logs nav path length + PPO stats, and Pytest covers the symbolic graph, passability convergence, nav planner bias, and executor updates.

## Getting Started

1. **Environment**
   ```powershell
   conda create -n srfbam python=3.10 -y
   conda activate srfbam
   pip install -e .
   ```
2. **Emulator Check**
   ```powershell
   python -m beater.main --config configs/default.yaml
   ```
   Logs show perception latents, SR-memory associations, nav path length, executor outcomes, and PPO update stats.
   _Tips:_ add `--visual` to force the SDL2 window and `--max-steps 700` to run a 700-frame session before exiting.
3. **Tests**
   ```powershell
   pytest
   ```

## LLM Brain (GPT-5 Mini)

1. Ensure `.env` contains `OPENAI_API_KEY=sk-...` (already provided) or export it in your shell.
2. Enable the brain in `configs/default.yaml`:
   ```yaml
   brain:
     enabled: true
     model: "gpt-5-mini" # never use max tokens
     candidate_preview: 4
     enable_web_search: true   # allow GPT to call OpenAI web search tool
     use_for_rollouts: false   # set true only if you want GPT guidance during PPO rollouts
   ```
3. Run `python -m beater.main ...` â€” the CLI log will now include GPT suggestions (skill, goal index, reasoning). When `use_for_rollouts` is true, the rollout collector consults GPT for each actor, so expect slower/paid calls.

## LLM Integration Updates

- HALT-gated and async: on `GraphOp.HALT`, a non-blocking request is queued; gameplay continues while waiting.
- Screenshot context: each HALT includes a small PNG of the frame alongside tile-grid context.
- Candidate metadata: each candidate includes `passability` and recent `fail_count` to bias away from walls.
- Unseen frontier hints: HALT payloads now bubble up a shortlist of low-confidence tiles (via the passability store) and interleave them into the candidate list so GPT can bias exploration away from dead walls.
- Mini-map context: a 11A-11 ASCII map (P = player, digits = candidate indices) plus per-candidate `portal_score` inferred from scene changes helps GPT spot exits (stairs/doors).
- Menu ops: if GPT returns `ops` with `skill="MENU_SEQUENCE"`, those button presses are executed exactly.
- Title skipper: a deterministic START/A boot script now runs before the interactive loop so every session begins past the title splash without waiting on GPT.
- RAM-backed menu guard: a combined RAM/tile detector drops the agent into a menu-handling mode (MENU_SEQUENCE/INTERACT only, auto-`B` closes stray menus, shaped reward pauses) so HALTs no longer freeze gameplay visuals.
- Naming detector: the RAM text buffer is monitored for the “First, what is your name?” dialog; when it appears we immediately queue a HALT with the screenshot plus a 9×9 naming grid so GPT can drive the cursor and type names deterministically. If the LLM stalls, the agent simply keeps waiting—no scripted button fallback.
- Objective spec (optional): GPT may include an `objective_spec` JSON block (`phase`, `reward_weights`, `timeouts.ttl_steps`, `skill_bias`) to steer shaped rewards and bias skills (e.g., menu vs overworld).
- Staleness guard: directives are ignored if too old or if the candidate set has drifted.
- Stuck?+'HALT: if a NAVIGATE planlet fails repeatedly with minimal movement, the loop automatically queues another HALT so GPT can reconsider.

### Brain config keys (configs/default.yaml)
```yaml
brain:
  enabled: true
  model: "gpt-5-mini"            # Responses API model name
  candidate_preview: 4            # number of waypoint candidates to surface
  enable_web_search: true         # allow built-in web_search tool
  use_for_rollouts: false         # enable GPT during PPO collection (slower/paid)
  attach_screenshot: true         # attach a PNG of the current frame on HALT
  max_directive_age_steps: 300    # drop stale directives older than this many steps
  min_candidate_overlap: 1        # require at least N common candidates when applying a directive
```

## Architecture Overview
```
PyBoy Env
  -> Perception (encoders + tile descriptor) -> SR-Memory graph & passability
  -> Nav planner (Thompson sampling) -> GoalManager waypoints -> NavPlanletBuilder
  -> PlanletExecutor (sprite tracker + passability updates)
  -> Controller + AffordancePrior -> PPO (grounded multi-actor rollouts)
  -> OptionBank & GoalManager feedback -> Plan-head biasing
```

Key files:

- `beater/perception/encoders.py`, `tile_desc.py`, `losses.py`
- `beater/sr_memory/graph.py`, `passability.py`
- `beater/policy/controller.py`, `plan_head.py`, `nav_planner.py`, `affordance.py`, `options.py`, `waypoints.py`
- `beater/executor/skills.py`, `compiler.py`, `sprite_tracker.py`
- `beater/brains/gpt.py`, `beater/brains/async_brain.py`
- `beater/training/rollouts.py`, `replay.py`, `learner.py`, `rewards.py`
- `beater/objectives.py`, `beater/utils/detectors.py`
- `beater/main.py`

## Remaining Work

1. **Executor Enhancements:** Extend beyond basic nav planlets (battle/menu skills, watchdog timers, failure recovery) and surface richer telemetry.
2. **LLM Tooling:** Build out additional GPT tools (inventory/state inspectors, cached hints) and cost controls now that web search is live.
3. **Long-Horizon PPO:** Scale the rollout manager to true multi-actor concurrency (async collectors, replay prioritization) and weave in SR-memory reuse/intrinsic rewards.
4. **Option Mining:** Promote mined options into reusable macros (skill sequences + args) and integrate them into plan-head decoding rather than simple biasing.
5. **Telemetry & Viz:** Build metrics/viz hooks (reuse ratio, passability calibration, nav success, PPO diagnostics) for training dashboards.

See `PROJECTOVERVIEW.md` for the full spec and next milestones.

## Objective Engine (Shaped Rewards)

- The LLM can optionally provide an `objective_spec` to steer current priorities:
  - Fields: `phase`, `reward_weights` (e.g., `nav_success`, `sprite_delta`, `scene_change`, `menu_progress`, `name_committed`), `timeouts.ttl_steps`, and `skill_bias` (e.g., `menu`/`overworld`).
- The engine converts local signals (nav success, sprite delta, scene-change via tile histogram deltas, simple menu/name heuristics) into a scalar reward using the provided weights.
- A periodic summary is logged every 500 steps for visibility.

## Demo/Debug Tips

- Use `--visual` and set `env.speed: 1.0` in `configs/default.yaml` to make step counts align with visual frames.
- Async HALT keeps frames moving while GPT responds; reduce the Responses timeout in `beater/brains/gpt.py` for snappier demos if needed.
- With `--log-level DEBUG`, you’ll see `POST /v1/responses` and `GPTBrain API call #N` lines to track LLM activity.
- For a fast regression pass, run `python -m beater.main --config configs/default.yaml --max-steps 5000 --log-level INFO`; review the log for `Objective summary` and `HALT directive applied` entries to confirm the async brain and objective swaps are exercising correctly.
