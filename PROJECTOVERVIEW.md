Awesome—thanks for the thoughtful feedback and the extra passability note. Below is a **complete, build‑from‑scratch spec** that **integrates** everything: SR‑FBAM entity‑query amortization, discovery‑based passability, slot‑friendly perception, multi‑actor PPO training with intrinsic rewards, and a clean Planlet→Executor boundary. I’ve kept it **non‑deterministic** (no routes, no overlay if‑else, no collision RAM semantics) and **discovery‑first** throughout. Where I reference the SR‑FBAM principles (ASSOC/FOLLOW/WRITE/HALT; reuse ratio → speed), I cite the paper you shared. 

---

# Pokémon Beater — SR‑FBAM Agent, Discovery‑First (Spec v2)

**Goal.** An agent that *learns* entities, skills, and environment rules by interaction alone—and can pass the “beat Pokémon” test—using **external symbolic memory** with **discrete queries** to skip re‑encoding and generalize. No hardcoded routes, no type chart, no overlay rules, no collision RAM semantics. 

**Core idea.** Perception → latent **zₜ**, SR‑Memory graph stores discovered entities/relations, Controller gates among **{ENCODE, ASSOC, FOLLOW, WRITE}**, Plan‑head emits **Planlets** (skills + args), Executor compiles to button pulses. **Passability** is learned as a **Bayesian posterior** over **discovered tile classes** and refined at **tile instances**, enabling probabilistic navigation (Thompson / expected‑cost A*). Reuse of graph queries yields **4–5× fewer encodes** and scalable long‑horizon behavior. 


## Spec v3 — GPT-5 Mini Planning Brain
- **LLM Brain**: Calls OpenAI's `responses` endpoint with `model="gpt-5-mini"`, `input`, and `max_output_tokens` only, matching https://platform.openai.com/docs/models/gpt-5-mini.
- **State serialization**: Tile descriptors + waypoint candidates are summarized as text so GPT-5 Mini can pick a macro skill (NAVIGATE/INTERACT/MENU/WAIT) and a candidate goal index.
- **Control loop**: Suggestions bias Plan-head decoding, override GoalManager targets, and annotate planlets (`brain_notes`). Optional rollout integration lets each actor ask GPT before building nav paths.
- **Tooling**: When enable_web_search is true, GPT-5 Mini receives OpenAI's built-in web_search tool, letting it pull meta strategies (e.g., best boss routes) before recommending skills/goals.
- **Config**: `brain.enabled`, `model`, `max_output_tokens`, `candidate_preview`, and `use_for_rollouts` live in `configs/default.yaml`. `.env` is loaded via `python-dotenv` to read `OPENAI_API_KEY`.
- **Safety**: Requests log-and-fallback when the API key is missing or the HTTP call fails, so the agent still relies on local heuristics.

---
---

## Build Progress (Oct 25, 2025)

- Bootstrapped perception encoders plus tile descriptor + auxiliary losses feeding the Bayesian passability store (`perception/encoders.py`, `perception/tile_desc.py`, `perception/losses.py`).
- Wired controller + plan-head scaffolding (`policy/controller.py`, `policy/plan_head.py`) into the CLI flow so SR-memory latents now produce concrete Planlets.
- Added guardrail tests for symbolic memory pruning and passability convergence (`tests/test_entity_graph.py`, `tests/test_passability.py`).
- Added tile-aware nav planner that consumes discovered classes via Thompson sampling plus corresponding tests (`policy/nav_planner.py`, `tests/test_nav_planner.py`).
- Built PPO warmup harness (rollout collector, replay buffer, learner, rewards) so controller gates/skills are exercised over short rollouts, and exposed the stats via `beater/main.py`.
- Fed nav-planner paths through the executor (NavPlanletBuilder + movement outcome detector) to update Bayesian passability based on actual outcomes and added regression coverage (`executor/skills.py`, `executor/compiler.py`, `tests/test_executor.py`).
- Replaced the synthetic PPO warmup with environment-grounded, multi-actor rollouts tied to the AffordancePrior + OptionBank so controller gates/skills now train on real PyBoy traces (`training/rollouts.py`, `policy/affordance.py`, `policy/options.py`).
- Swapped pixel-diff heuristics for a sprite/pose tracker so passability updates hinge on real avatar motion instead of global frame differences (`executor/sprite_tracker.py`).
- Introduced dynamic waypoint selection via `GoalManager`, wiring nav planner goals and feedback loops through the controller so long paths no longer collapse to corner walks (`policy/waypoints.py`, `beater/main.py`, `training/rollouts.py`).

---

## 0) Repository layout

```
pokemon-beater/
  pyproject.toml
  README.md
  configs/
    default.yaml
  beater/
    types.py
    env/
      pyboy_env.py
      recorders.py
    perception/
      encoders.py
      tile_desc.py
      losses.py
    sr_memory/
      graph.py
      ops.py
      passability.py
    policy/
      controller.py
      plan_head.py
      affordance.py
      options.py
      nav_planner.py
    executor/
      skills.py
      compiler.py
      sprite_tracker.py
    training/
      rollouts.py
      replay.py
      learner.py
      rewards.py
    utils/
      metrics.py
      viz.py
  main.py
```

---

## 1) Data contracts

```python
# beater/types.py
from dataclasses import dataclass
from typing import Literal, Optional, List, Dict, Tuple

Buttons = Literal["UP","DOWN","LEFT","RIGHT","A","B","START","SELECT","NOOP"]

@dataclass
class Observation:
    rgb: "np.ndarray"    # (160,144,3), uint8
    ram: "np.ndarray"    # raw bytes; no semantics assumed
    step_idx: int
    # Optional: last_save_slot (int) for reproducibility

@dataclass
class ScriptOp:
    op: Literal["PRESS","RELEASE","WAIT"]
    button: Optional[Buttons] = None
    frames: int = 0

@dataclass
class Planlet:
    id: str
    kind: Literal["MENU_SEQUENCE","NAVIGATE","INTERACT","WAIT"]
    args: Dict
    script: List[ScriptOp]          # executor-ready; idempotent
    timeout_steps: int
```

**Invariant:** The emulator only ever sees `ScriptOp[]`. The *policy* decides planlets; the *executor* handles timing and watchdogs. (This Planlet boundary mirrors SR‑FBAM’s “dense vs. query” swap while keeping runtime simple. )

---

## 2) Environment wrapper (+ savestate ring, RAM range)

```python
# beater/env/pyboy_env.py
import numpy as np
from pyboy import PyBoy
from beater.types import Observation, ScriptOp, Buttons

class PyBoyEnv:
    def __init__(self, rom_path: str, window="null", speed=1.0, ring_slots=8):
        self.pyboy = PyBoy(rom_path, window_type=window)
        self.speed = speed
        self.ring_slots = ring_slots
        self._ring_idx = 0
        self._init_ring()

    def _init_ring(self):
        self._saves = [None]*self.ring_slots
        self._save_state()

    def _save_state(self):
        self._saves[self._ring_idx] = self.pyboy.save_state()
        self._ring_idx = (self._ring_idx + 1) % self.ring_slots

    def rollback(self, k_back=1):
        idx = (self._ring_idx - 1 - (k_back-1)) % self.ring_slots
        state = self._saves[idx]
        if state is not None:
            self.pyboy.load_state(state)

    def step_script(self, script: list[ScriptOp]) -> Observation:
        for s in script:
            if s.op == "PRESS" and s.button != "NOOP":
                self.pyboy.send_input(getattr(self.pyboy, s.button))
            elif s.op == "RELEASE" and s.button != "NOOP":
                self.pyboy.stop_input(getattr(self.pyboy, s.button))
            # WAIT or inter-press settling
            for _ in range(max(1, s.frames)):
                self.pyboy.tick()
        # periodic auto-save (cheap; tune cadence)
        if self.pyboy.frame_count % 300 == 0:
            self._save_state()
        return self.observe()

    def observe(self) -> Observation:
        screen = self.pyboy.botsupport_manager().screen()
        rgb = np.array(screen.image)  # (160,144,3) uint8
        # Broader RAM snapshot (WRAM+HRAM subset; still semantics-free)
        # NOTE: do not rely on specific addresses; this is bytes-as-features only.
        wram = self.pyboy.get_memory_value(0xC000, 0x2000)  # 8KB
        hram = self.pyboy.get_memory_value(0xFF80, 0x007F)  # 127B
        ram = np.frombuffer(wram + hram, dtype=np.uint8)
        return Observation(rgb=rgb, ram=ram, step_idx=self.pyboy.frame_count)
```

---

## 3) Perception (slot‑friendly + temporal smoothing + RND)

```python
# beater/perception/encoders.py
import torch, torch.nn as nn, torch.nn.functional as F

class VisualEncoder(nn.Module):
    def __init__(self, z_dim=192):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32,64, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(64,96, 3, 2, 1), nn.ReLU(),
        )
        self.proj = nn.Linear(96*20*18, z_dim)
    def forward(self, x):        # x: (B,3,160,144)
        h = self.conv(x).flatten(1)
        return self.proj(h)

class RamEncoder(nn.Module):
    def __init__(self, in_bytes: int, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_bytes, 512), nn.ReLU(),
            nn.Linear(512, z_dim)
        )
    def forward(self, b):        # (B,N)
        return self.net(b.float()/255.0)

class Perception(nn.Module):
    def __init__(self, ram_bytes: int, z_dim=256):
        super().__init__()
        self.vision = VisualEncoder(192)
        self.ram = RamEncoder(ram_bytes, 64)
        self.smooth = 0.9
        self._z_prev = None
    def forward(self, rgb, ram):
        z = torch.cat([self.vision(rgb), self.ram(ram)], dim=-1)  # (B,256)
        if self._z_prev is None: self._z_prev = z.detach()
        # temporal smoothing (stabilize WRITE triggers)
        z = self.smooth*self._z_prev + (1-self.smooth)*z
        self._z_prev = z.detach()
        return z
```

**Optional:** swap `VisualEncoder` for a tiny **slot module** later; keep API unchanged.

---

## 4) SR‑Memory graph (entities + discrete ops)

```python
# beater/sr_memory/graph.py
import torch
from typing import Dict, List, Tuple

class EntityGraph:
    def __init__(self, dim=128):
        self.next_id = 1
        self.emb: Dict[int, torch.Tensor] = {}
        self.typ: Dict[int, str] = {}
        self.edges: Dict[int, Dict[str, set]] = {}
        self.use_count: Dict[int, int] = {}

    def write(self, emb: torch.Tensor, typ: str, edges: List[Tuple[int,str,int]]|None=None):
        nid = self.next_id; self.next_id += 1
        self.emb[nid] = emb.detach().cpu()
        self.typ[nid] = typ
        self.edges.setdefault(nid, {})
        if edges:
            for src, rel, dst in edges:
                self.edges.setdefault(src, {}).setdefault(rel, set()).add(dst)
        self.use_count[nid] = 0
        return nid

    def assoc(self, q: torch.Tensor, typ: str|None=None, k=5) -> List[int]:
        ids = [i for i in self.emb if typ is None or self.typ[i]==typ]
        if not ids: return []
        E = torch.stack([self.emb[i] for i in ids], 0)
        sims = torch.nn.functional.cosine_similarity(q.cpu().unsqueeze(0), E, dim=-1)
        topk = sims.topk(min(k, len(ids))).indices.tolist()
        out = [ids[i] for i in topk]
        for nid in out: self.use_count[nid]+=1
        return out

    def follow(self, node_id: int, rel: str) -> List[int]:
        return list(self.edges.get(node_id, {}).get(rel, []))

    def prune(self, keep_top=5000):
        # drop least-used nodes to bound memory
        if len(self.emb) <= keep_top: return
        by_use = sorted(self.use_count.items(), key=lambda kv: kv[1], reverse=True)[:keep_top]
        keep = set(n for n,_ in by_use)
        self.emb = {n:self.emb[n] for n in keep}
        self.typ  = {n:self.typ[n] for n in keep}
        self.edges= {n:{r:{m for m in ms if m in keep} for r,ms in self.edges.get(n,{}).items()}
                     for n in keep}
        self.use_count = {n:self.use_count[n] for n in keep}
```

**Why discrete ops?** **ASSOC/FOLLOW/WRITE/HALT** provide O(1) lookups and compositional reasoning across persistent structure, enabling higher accuracy and **4–5×** speedups by skipping dense encodes most steps. 

---

## 5) Discovery‑based passability

### 5.1 Tile descriptor (unsupervised; no terrain labels)

```python
# beater/perception/tile_desc.py
import numpy as np
from typing import Tuple

def tile_descriptor(rgb: np.ndarray, tl: Tuple[int,int], tile_sz: int=8) -> dict:
    x, y = tl
    patch = rgb[y:y+tile_sz, x:x+tile_sz, :].astype(np.float32)/255.0
    mean = patch.mean((0,1))
    gx = patch[:,1:,:]-patch[:,:-1,:]; gy = patch[1:,:,:]-patch[:-1,:,:]
    edge = float(np.mean(np.abs(gx))+np.mean(np.abs(gy)))
    gray = (0.299*patch[:,:,0]+0.587*patch[:,:,1]+0.114*patch[:,:,2])
    small = gray.reshape(4,2,4,2).mean((1,3))
    med = np.median(small); bits = (small>med).astype(np.uint8).flatten()
    h=0
    for b in bits: h=(h<<1)|int(b)
    return {"mean":mean.tolist(),"edge":edge,"hash16":int(h)}

def desc_hash64(desc: dict) -> int:
    # simple stable hash from mean color buckets + hash16
    r,g,b = [int(x*7) for x in desc["mean"]]  # 0..7 bucket
    return (desc["hash16"] | (r<<16) | (g<<19) | (b<<22))
```

### 5.2 Bayesian passability (class prior + instance posterior)

```python
# beater/sr_memory/passability.py
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np, math

@dataclass
class Beta:
    a: float = 1.0
    b: float = 1.0
    def mean(self): return self.a/(self.a+self.b)
    def sample(self): return np.random.beta(self.a, self.b)
    def ucb(self, c=2.0, n=1):
        w = math.sqrt(max(1.0, math.log(max(2,n)))/(self.a+self.b))
        return min(1.0, self.mean()+c*w)

class PassabilityStore:
    def __init__(self, a0=1.0, b0=1.0):
        self.cls: Dict[int, Beta] = {}
        self.inst: Dict[Tuple[str,Tuple[int,int],str], Beta] = {}
        self.a0, self.b0 = a0, b0

    def _cls(self, h): return self.cls.setdefault(h, Beta(self.a0, self.b0))
    def _ins(self, m, xy, d): return self.inst.setdefault((m,xy,d), Beta(self.a0, self.b0))

    def predict(self, map_key: str, xy: tuple[int,int], dir_: str, h: int, mode="mean"):
        c,i = self._cls(h), self._ins(map_key, xy, dir_)
        if mode=="thompson": return np.random.beta(c.a+i.a, c.b+i.b)
        if mode=="ucb":      return max(c.ucb(), i.ucb())
        tot=(c.a+c.b)+(i.a+i.b); 
        return ((c.mean()*(c.a+c.b)+(i.mean()*(i.a+i.b)))/tot) if tot else 0.5

    def update(self, map_key: str, xy: tuple[int,int], dir_: str, h: int, success: bool):
        c,i = self._cls(h), self._ins(map_key, xy, dir_)
        if success: c.a+=1; i.a+=1
        else:       c.b+=1; i.b+=1
```

**Interpretation:** The agent *learns* physics from outcomes only. No collision RAM, no terrain tables. Priors generalize across visually similar tiles; instance posteriors record local quirks (e.g., ledges). Planning uses these **posteriors** via memory **ASSOC/FOLLOW** rather than re‑encoding every step—exactly the SR‑FBAM reuse idea. 

---

## 6) Movement outcome detection (vision‑only)

```python
# beater/executor/move_outcome.py
import numpy as np

def displaced(prev_rgb: np.ndarray, next_rgb: np.ndarray, thresh=0.10) -> bool:
    diff = np.mean(np.abs(prev_rgb.astype(np.float32)-next_rgb.astype(np.float32)))/255.0
    return diff > thresh
```

If your perception later estimates ego‑tile indices, you can switch to exact 1‑tile checks; keep this vision‑only first.

---

## 7) Navigation planner (probabilistic)

**Option A — Thompson pathing:** sample walkability per tile and run BFS; repeat K times; pick shortest success.

```python
# beater/policy/nav_planner.py
from collections import deque
import numpy as np

DIRS = [("UP",( -1, 0)), ("DOWN",(1,0)), ("LEFT",(0,-1)), ("RIGHT",(0,1))]

def neighbors(u, H, W):
    i,j = u
    for name,(di,dj) in DIRS:
        v=(i+di,j+dj)
        if 0<=v[0]<H and 0<=v[1]<W:
            yield v, name

def thompson_path(start_xy, goal_xy, grid_hash, map_key, store, K=4):
    H,W = grid_hash.shape
    best=None
    for _ in range(K):
        walk = np.ones((H,W), dtype=bool)
        for i in range(H):
            for j in range(W):
                h=int(grid_hash[i,j])
                # optimistic passability draw
                ps=[store.predict(map_key,(i,j),d,h,mode="thompson") for d,_ in DIRS]
                walk[i,j]=(max(ps)>0.5)
        q=deque([start_xy]); parent={start_xy:None}
        while q:
            u=q.popleft()
            if u==goal_xy: break
            for v,_ in neighbors(u,H,W):
                if not walk[v] or v in parent: continue
                parent[v]=u; q.append(v)
        if goal_xy in parent:
            path=[]; cur=goal_xy
            while cur is not None: path.append(cur); cur=parent[cur]
            path.reverse()
            if best is None or len(path)<len(best): best=path
    return best
```

**Option B — Expected‑cost A*** (edge cost `-log E[pass]`) is a drop‑in alternative.

---

## 8) Controller (recurrent + discrete gate)

```python
# beater/policy/controller.py
import torch, torch.nn as nn, torch.nn.functional as F

OPS = ["ENCODE","ASSOC","FOLLOW","WRITE"]
SKILLS = ["MENU_SEQUENCE","NAVIGATE","INTERACT","WAIT"]

class Controller(nn.Module):
    def __init__(self, z_dim=256, e_dim=64, hid=384):
        super().__init__()
        self.lstm = nn.LSTM(z_dim+e_dim+64, hid, batch_first=True)
        self.op_gate   = nn.Linear(hid, len(OPS))
        self.skill_head= nn.Linear(hid, len(SKILLS))
        self.arg_head  = nn.Linear(hid, 64)
        self.tout_head = nn.Linear(hid, 1)
        self.prev_emb  = nn.Embedding(len(SKILLS), 64)

    def forward(self, z_t, e_t, prev_skill_id, h=None):
        x=torch.cat([z_t, e_t, self.prev_emb(prev_skill_id)], dim=-1).unsqueeze(1)
        o,h = self.lstm(x,h); o=o[:,-1]
        return (self.op_gate(o), self.skill_head(o),
                self.arg_head(o), F.softplus(self.tout_head(o))+5, h)
```

**Training:** use **Gumbel‑Softmax** on `op_gate` with τ annealed from 1.0→0.1; **inference** uses hard argmax (straight‑through). The gate chooses when to **query memory** vs **re‑encode**; rising **reuse ratio** is your progress signal, per SR‑FBAM. 

---

## 9) Plan‑head (emit Planlets)

```python
# beater/policy/plan_head.py
from beater.types import Planlet, ScriptOp

def _script_menu(seq: list[str]) -> list[ScriptOp]:
    ops=[]
    for b in seq:
        ops += [ScriptOp("PRESS", b, 1), ScriptOp("WAIT", frames=12),
                ScriptOp("RELEASE", b, 1), ScriptOp("WAIT", frames=4)]
    return ops

def decode_planlet(skill_logits, arg_vec, timeout) -> Planlet:
    k = int(skill_logits.argmax().item())
    kind = ["MENU_SEQUENCE","NAVIGATE","INTERACT","WAIT"][k]
    if kind=="INTERACT":
        script=[ScriptOp("PRESS","A",1), ScriptOp("WAIT",frames=12)]
        return Planlet("ia",kind,{},script,int(timeout.item()))
    if kind=="WAIT":
        return Planlet("wt",kind,{},[ScriptOp("WAIT",frames=24)],int(timeout.item()))
    if kind=="MENU_SEQUENCE":
        seq = infer_seq_from_arg(arg_vec)    # learned mapping to short pattern
        return Planlet("ms",kind,{"buttons":seq},_script_menu(seq),int(timeout.item()))
    if kind=="NAVIGATE":
        # arg_vec → local goal / frontier choice; path compiled by nav planner
        target = decode_target(arg_vec)      # {"mode":"offset" or "entity_id":...}
        script = compile_nav_script(target)  # uses passability-aware planner
        return Planlet("nav",kind,target,script,int(timeout.item()))
```

---

## 10) Affordance prior (learned “what causes change now”)

```python
# beater/policy/affordance.py
import torch, torch.nn as nn

class Affordance(nn.Module):
    def __init__(self, z_dim=256, n_skills=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim,128), nn.ReLU(), nn.Linear(128,n_skills))
    def forward(self, z): return self.net(z)  # logits to add into skill_head
```

Label “success” if ‖zₜ₊Δ − zₜ‖ exceeds τ or a **WRITE** occurred. Add as log‑prior to reduce thrashing without rules.

---

## 11) Options (unsupervised macro mining)

```python
# beater/policy/options.py
from collections import Counter

def mine_patterns(button_traces, min_support=50, max_len=5):
    counts=Counter()
    for tr in button_traces:
        for L in range(1,max_len+1):
            for i in range(len(tr)-L+1):
                counts[tuple(tr[i:i+L])] += 1
    return [list(p) for p,c in counts.items() if c>=min_support]
```

Expose mined indices through `infer_seq_from_arg(arg_vec)`.

---

## 12) Executor (watchdogs + passability updates)

```python
# beater/executor/skills.py
from beater.types import Planlet, ScriptOp
from beater.perception.tile_desc import tile_descriptor, desc_hash64
from beater.executor.move_outcome import displaced

class Executor:
    def __init__(self, env, pass_store, tile_meta, watchdog_steps=40):
        self.env, self.store, self.meta = env, pass_store, tile_meta
        self.watchdog = watchdog_steps

    def run(self, planlet: Planlet):
        steps=0; prev_obs=self.env.observe()
        for s in planlet.script:
            obs = self.env.step_script([s])
            steps += (s.op!="WAIT")
            if s.button in {"UP","DOWN","LEFT","RIGHT"} and s.op=="PRESS":
                # update learned passability for attempted move
                tl = tile_top_left(next_xy(prev_obs, s.button, self.meta), self.meta)
                desc = tile_descriptor(prev_obs.rgb, tl)
                h = desc_hash64(desc)
                success = displaced(prev_obs.rgb, obs.rgb)
                self.store.update(map_key(prev_obs), ego_xy(prev_obs), s.button, h, success)
            prev_obs=obs
            if steps>self.watchdog: break
        return prev_obs
```

Idempotent: Planlets may be re‑issued after watchdog expiry; learned posteriors prevent loops.

---

## 13) Rewards (intrinsics + minimal extrinsics)

```python
# beater/training/rewards.py
import torch

def rnd_reward(z, rnd_target, rnd_pred):
    with torch.no_grad(): tgt = rnd_target(z)
    pred = rnd_pred(z)
    return (tgt - pred).pow(2).mean(dim=-1)  # curiosity

def passability_info_gain(before_beta, after_beta):
    # KL approx for Beta(a,b) updates; compute per-edge and sum
    import math
    def H(a,b):
        from math import lgamma, digamma
        A=a+b
        return (lgamma(a)+lgamma(b)-lgamma(A) - (a-1)*digamma(a) - (b-1)*digamma(b)
                + (A-2)*digamma(A))
    return max(0.0, H(after_beta.a, after_beta.b) - H(before_beta.a, before_beta.b))
```

* **Curiosity (RND)** on **z** to bootstrap exploration.
* **Information gain** bonus on passability posteriors when attempts teach the model.
* **Progress (self‑discovered):** +1 when a newly **written** node is still present ≥K steps later (irreversibility).
* Optional **credits detector**: a tiny CNN trained online to recognize long text‑scroll + low motion.

---

## 14) Training (PPO + auxiliaries + multi‑actors)

```python
# beater/training/learner.py (sketch)
for update in range(num_updates):
    batch = rollout.collect(actors=cfg.num_actors, steps=cfg.steps_per_actor)

    # advantages/returns
    adv, ret = gae(batch.values, batch.rewards, batch.dones, gamma=cfg.gamma, lam=cfg.lam)

    # controller forward (z, e, prev_skill)
    pi, v, aux = model(batch.obs, batch.prev_skill)

    # PPO losses
    ratio = (pi.log_prob(batch.action) - batch.logp_old).exp()
    loss_pi = -(torch.min(ratio*adv, torch.clamp(ratio,1-cfg.clip,1+cfg.clip)*adv)).mean()
    loss_v  = 0.5*(ret - v).pow(2).mean()
    entropy = pi.entropy().mean()

    # auxiliaries: recon/contrastive for perception, RND, gate regularizer
    loss_aux = aux["recon"] + aux["contrast"] + cfg.gate_reg*aux["gate_encode_frac"]

    loss = loss_pi + cfg.c1*loss_v - cfg.c2*entropy + loss_aux
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
```

**Schedule.** Start short horizons (100 steps, emu speed 4×), anneal gate τ 1.0→0.1, increase horizon as **reuse ratio** > 0.7. (Reuse ratio & speedup are the SR‑FBAM “health” metrics. )

---

## 15) Metrics & diagnostics

* **Entity reuse ratio**: fraction of steps served by **ASSOC/FOLLOW** vs **ENCODE** (expect ↑ over training; correlates with speedup). 
* **Passability calibration**: reliability diagram of p̂ vs empirical success.
* **Planlet success rate**; **watchdog trips**.
* **FPS & episode wall‑time**; **graph size & prune events**.

---

## 16) Config & CLI

```yaml
# configs/default.yaml
env:
  rom: "Pokemon Blue.gb"
  window: "SDL2"
  speed: 1.0
  ring_slots: 8
perception:
  z_dim: 256
training:
  algo: "ppo"
  num_actors: 16
  steps_per_actor: 512
  total_updates: 5000
  lr: 3e-4
  gamma: 0.997
  lam: 0.95
  clip: 0.1
  c1: 0.5
  c2: 0.01
policy:
  gate_temp_start: 1.0
  gate_temp_end: 0.1
  gate_reg: 0.001
executor:
  watchdog_steps: 40
```

**CLI (first spin):**

```
python -m beater.main --config configs/default.yaml
```

Enable --visual for SDL2 output and pass --max-steps N (e.g., 700) to run a fixed-length interactive session before PPO kicks in.
---

## 17) Inference loop (end‑to‑end)

```python
# main.py (core loop sketch)
import torch, numpy as np
from beater.env.pyboy_env import PyBoyEnv
from beater.perception.encoders import Perception
from beater.sr_memory.graph import EntityGraph
from beater.sr_memory.passability import PassabilityStore
from beater.policy.controller import Controller
from beater.policy.plan_head import decode_planlet
from beater.executor.skills import Executor

env = PyBoyEnv("Pokemon Blue.gb", window="SDL2", speed=1.0)
obs = env.observe()
per = Perception(ram_bytes=obs.ram.size)
graph = EntityGraph()
store = PassabilityStore()
ctrl = Controller()
execu = Executor(env, store, tile_meta={})   # tile_meta: screen->tile mapping helpers

h=None; prev_skill=torch.tensor([0])
while True:
    obs = env.observe()
    rgb = torch.from_numpy(obs.rgb).permute(2,0,1).unsqueeze(0).float()/255.
    ram = torch.from_numpy(obs.ram).unsqueeze(0)
    z = per(rgb, ram)

    # (Optional) form small e_t from recent passability priors or last assoc embedding
    e = torch.zeros((1,64))

    logits_op, skill_logits, arg_vec, timeout, h = ctrl(z, e, prev_skill, h)

    # (Gate) pick op and interact with graph/passability as needed
    # (ASSOC/FOLLOW/WRITE logic goes here; omitted for brevity)

    planlet = decode_planlet(skill_logits, arg_vec, timeout)
    obs = execu.run(planlet)
    prev_skill = torch.tensor([int(skill_logits.argmax())])
```

---

## 18) Tests & reproducibility

* **Unit**:

  * `passability_test.py`: Beta updates converge on synthetic blocked/open tiles.
  * `nav_planner_test.py`: Thompson finds a path when one exists; avoids blocked islands.
  * `controller_gate_test.py`: gate entropy decreases with annealing; reuse ratio increases on a toy grid.

* **Integration**:

  * “Title‑to‑overworld” smoke run at 10k steps; assert graph growth > threshold, reuse ratio > 0.3, options mined ≥ 3.

* **Repro**: savestate ring; YAML‑pinned configs; RNG seeds for PyTorch/NumPy.

---

## 19) Risks & mitigations

* **Early thrashing / sparse rewards** → RND + passability info‑gain; curriculum horizons.
* **Graph bloat** → LRU/usage pruning (`graph.prune`).
* **Perception volatility** → temporal smoothing; later swap to slot‑attention.
* **CPU‑bound emulation** → multi‑proc actors; raise emu speed during training.

---

## 20) Why this works (and scales)

The **same mechanism** that produced **4–5× speedups** and large generalization gains in SR‑FBAM—**entity‑query amortization via ASSOC/FOLLOW/WRITE/HALT**—is what makes long‑horizon gameplay feasible here. You learn the world (entities, tiles, menus) once, **query cheaply thereafter**, and only **ENCODE** when needed; the **reuse ratio** becomes the leading indicator your agent is “getting it.” 

---

### What changed from Spec v1

* **Integrated discovery‑based passability** (class+instance Beta posteriors), Thompon/A*.
* **Perception stability** (temporal smoothing) and **broader RAM bytes** (as features only).
* **Savestate ring**, **NOOP**, **affordance prior**, **option mining** cadence.
* **Graph pruning** and explicit **metrics** for reuse/calibration.

---