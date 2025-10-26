# PokAcmon Beater

Discovery-first SR-FBAM agent targeting PokÃ©mon Blue automation using sparse symbolic memory, tile-level discovery, and planlet-based control.

## System Snapshot

- **Perception:** Slot-friendly encoder with per-actor temporal smoothing plus a tile descriptor that clusters 8x8 RGBA patches into unsupervised terrain classes feeding the Bayesian passability store.
- **SR-Memory & Navigation:** Entity graph for ASSOC/FOLLOW/WRITE/HALT, Bayesian passability with class+instance Betas, Thompson-sampling nav planner, `GoalManager` waypoints, and a `NavPlanletBuilder` that turns sampled paths into executor-ready planlets.
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
- `beater/brains/gpt.py`
- `beater/training/rollouts.py`, `replay.py`, `learner.py`, `rewards.py`
- `beater/main.py`

## Remaining Work

1. **Executor Enhancements:** Extend beyond basic nav planlets (battle/menu skills, watchdog timers, failure recovery) and surface richer telemetry.
2. **LLM Tooling:** Build out additional GPT tools (inventory/state inspectors, cached hints) and cost controls now that web search is live.
3. **Long-Horizon PPO:** Scale the rollout manager to true multi-actor concurrency (async collectors, replay prioritization) and weave in SR-memory reuse/intrinsic rewards.
4. **Option Mining:** Promote mined options into reusable macros (skill sequences + args) and integrate them into plan-head decoding rather than simple biasing.
5. **Telemetry & Viz:** Build metrics/viz hooks (reuse ratio, passability calibration, nav success, PPO diagnostics) for training dashboards.

See `PROJECTOVERVIEW.md` for the full spec and next milestones.
