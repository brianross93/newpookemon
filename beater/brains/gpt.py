"""GPT-5 Mini integration for high-level planning."""

from __future__ import annotations

import json
import logging
import os
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import base64
import re

import requests

LOGGER = logging.getLogger(__name__)

Coord = Tuple[int, int]

def _naming_preset_ops(target_row: int) -> List[str]:
    """
    Build a deterministic button script for rival naming.

    target_row counts from 1 (RED) down the list. We always reset to NEW NAME first.
    """

    ops: List[str] = ["UP", "UP", "UP", "WAIT_6"]
    for _ in range(target_row):
        ops.extend(["DOWN", "WAIT_6"])
    ops.extend(
        [
            "A",
            "WAIT_20",
            "A",
            "WAIT_25",
            "A",
            "WAIT_30",
            "START",
            "WAIT_25",
            "A",
            "WAIT_20",
        ]
    )
    return ops


NAMING_PRESET_OPS: Dict[int, List[str]] = {
    0: _naming_preset_ops(1),  # default NEW NAME -> pick RED
    1: _naming_preset_ops(1),
    2: _naming_preset_ops(2),
    3: _naming_preset_ops(3),
}


@dataclass(slots=True)
class GoalSuggestion:
    """LLM-provided directive for skills/goals and optional menu ops."""

    skill: Optional[str]
    goal_index: Optional[int]
    reasoning: str
    ops: Optional[List[str]] = None
    objective_spec: Optional[Dict[str, object]] = None

    def resolve_goal(self, candidates: Sequence[Coord]) -> Optional[Coord]:
        if self.goal_index is None:
            return None
        if 0 <= self.goal_index < len(candidates):
            return candidates[self.goal_index]
        return None


class GPTBrain:
    """Thin wrapper over the OpenAI Responses API."""

    def __init__(
        self,
        model: str = "gpt-5-mini",
        max_output_tokens: int = 100_000,
        base_url: str = "https://api.openai.com/v1",
        enabled: bool = True,
        enable_web_search: bool = False,
    ):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.enabled = enabled and bool(self.api_key)
        self.enable_web_search = enable_web_search
        self.call_count = 0  # Track number of API calls
        if not self.enabled:
            LOGGER.warning(
                "GPTBrain disabled (missing OPENAI_API_KEY or brain.enabled set to false)."
            )

    def suggest_goal(
        self,
        tile_grid: List[List[str]],
        candidates: List[Coord],
        context: Optional[Dict[str, object]] = None,
        image_bytes: Optional[bytes] = None,
    ) -> Optional[GoalSuggestion]:
        if not self.enabled or not candidates:
            return None
        context = context or {}
        candidate_lines = []
        for idx, (row, col) in enumerate(candidates):
            tile_key = tile_grid[row][col]
            tile_class = tile_key.split(":", 1)[-1]
            candidate_lines.append(
                f"{idx}: row={row}, col={col}, tile_class={tile_class}"
            )
        # enrich with optional candidate metadata if provided
        cand_meta = context.get("candidate_meta") if isinstance(context, dict) else None
        if isinstance(cand_meta, list) and len(cand_meta) == len(candidates):
            new_lines: List[str] = []
            for i, line in enumerate(candidate_lines):
                meta = cand_meta[i]
                if isinstance(meta, dict):
                    p = meta.get("passability")
                    f = meta.get("fail_count")
                    extras: List[str] = []
                    if isinstance(p, (int, float)):
                        try:
                            extras.append(f"p={float(p):.2f}")
                        except Exception:
                            pass
                    if isinstance(f, int):
                        extras.append(f"fails={f}")
                    portal = meta.get("portal_score")
                    if isinstance(portal, (int, float)) and float(portal) > 0:
                        extras.append(f"portal={float(portal):.2f}")
                    if extras:
                        line = f"{line} (" + ", ".join(extras) + ")"
                new_lines.append(line)
            candidate_lines = new_lines
        top_classes = self._top_tile_classes(tile_grid)
        prompt = textwrap.dedent(
            f"""
            You are the high-level reasoning brain for an automated Pokémon Blue agent.
            Decide which macro skill to execute next and which candidate waypoint offers
            the best exploratory value. Always reply with a compact JSON object.

            Episode context:
            - step: {context.get("step")}
            - assoc_matches: {context.get("assoc_matches")}
            - passability_estimate: {context.get("passability")}
            - top_tile_classes: {top_classes[:5]}

            Candidate waypoints:
            {os.linesep.join(candidate_lines)}

            JSON schema:
            {{
              "skill": "NAVIGATE|INTERACT|MENU_SEQUENCE|WAIT",
              "goal_index": <integer index into the candidate list, or -1 if no navigation>,
              "reasoning": "<brief justification>"
            }}
            """
        ).strip()
        ascii_map = context.get("ascii_map") if isinstance(context, dict) else None
        if isinstance(ascii_map, str) and ascii_map.strip():
            prompt += f"\nMini-map (P=player, digits=candidate indices):\n{ascii_map}\n"
        # Provide the controller map and an optional ops contract to encourage
        # compact button sequences for MENU_SEQUENCE without forcing them.
        prompt_extra = textwrap.dedent(
            """
            If you choose skill="MENU_SEQUENCE", you may include an optional field
            "ops" as an array of controller buttons to press next, chosen from
            ["A","B","START","SELECT","UP","DOWN","LEFT","RIGHT"]. Keep
            sequences short (<= 12 ops). Example:
            {"skill":"MENU_SEQUENCE","goal_index":-1,"ops":["START","A","A"],"reasoning":"..."}

            Controller Map (Game Boy):
            - UP/DOWN/LEFT/RIGHT: Move selection or cursor.
            - A: Confirm/advance dialog/select letter.
            - B: Cancel/backspace (on naming screen).
            - START: Confirm at title; menu accept/confirm name in some screens.
            - SELECT: Rarely used; ignore unless needed.

            Example naming grid (schematic):
            [ A B C D E F G H I ]
            [ J K L M N O P Q R ]
            [ S T U V W X Y Z - ]
            [   END    ]
            Use D-Pad to move to a letter and press A; use START or move to END to confirm.
            Use the screenshot and mini-map to locate exits (stairs, doors) and plan precise moves.

            Always include an "objective_spec" to steer phase and reward priorities.
            Example:
            {"objective_spec": {"phase":"naming","reward_weights":{"scene_change":1.0,"sprite_delta":0.1},
             "timeouts":{"ttl_steps":500}, "skill_bias":"menu"}}

            web_search_policy: Use web_search only when uncertain about the next
            step or immediately after a significant scene change is detected.
            Limit to <=2 searches and summarize findings briefly in "reasoning".
            """
        ).strip()
        if context.get("naming_hint"):
            grid = context.get("naming_grid") or ""
            naming_block = textwrap.dedent(
                """

                Naming screen instructions:
                - Use MENU_SEQUENCE ops like ["DOWN","A","RIGHT","A"] to move the cursor and pick letters.
                - Use B to delete the previous letter.
                - Move to END then press A (or press START) to submit the name.
                - You may also select a preset option (e.g., RED/ASH/JACK) by moving to it and pressing A.
                - When on the naming screen you MUST reply with skill="MENU_SEQUENCE" and provide an "ops" array describing the exact button presses to either enter a name or choose a preset. Do not choose NAVIGATE, WAIT, or INTERACT in this state.
                """
            ).strip()
            prompt_extra += f"\n{naming_block}\n{grid}\n"
        full_prompt = prompt + "\n\n" + prompt_extra
        # Reinforce requirements: always include objective_spec; web_search allowed.
        full_prompt += "\n\nREQUIREMENT: Always include an objective_spec with phase, reward_weights, timeouts.ttl_steps, and skill_bias."
        if self.enable_web_search:
            full_prompt += "\nYou may call the web_search tool up to 2 times to inform your choice, but your final output must be a single JSON object only."
        LOGGER.debug(
            "HALT request context step=%s phase=%s menu=%s candidates=%s",
            context.get("step"),
            context.get("phase"),
            context.get("menu_open"),
            len(candidates),
        )
        if image_bytes is not None:
            b64 = base64.b64encode(image_bytes).decode("ascii")
            data_url = f"data:image/png;base64,{b64}"
            input_blocks = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": full_prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ]
            payload = {
                "model": self.model,
                "input": input_blocks,
                "max_output_tokens": self.max_output_tokens,
            }
        else:
            payload = {
                "model": self.model,
                "input": full_prompt,
                "max_output_tokens": self.max_output_tokens,
            }
        payload["text"] = {"format": {"type": "json_object"}, "verbosity": "low"}
        payload["reasoning"] = {"effort": "low"}
        if self.enable_web_search:
            payload["tools"] = [{"type": "web_search"}]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            self.call_count += 1
            LOGGER.info(
                "GPTBrain API call #%d payload_size=%s bytes ctx_step=%s",
                self.call_count,
                len(json.dumps(payload)),
                (context or {}).get("step"),
            )
            response = self.session.post(
                f"{self.base_url}/responses",
                headers=headers,
                json=payload,
                timeout=30,
            )
            LOGGER.debug(
                "GPTBrain API call #%d status=%s",
                self.call_count,
                response.status_code,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = ""
            if hasattr(exc, "response") and exc.response is not None:
                detail = exc.response.text
            LOGGER.warning(
                "GPTBrain request failed (call #%s): %s %s payload=%s",
                self.call_count,
                exc,
                detail,
                json.dumps(payload),
            )
            return None
        payload_json = response.json()
        text = self._extract_text(payload_json)
        if not text:
            LOGGER.warning(
                "GPTBrain empty output step=%s payload=%s",
                (context or {}).get("step"),
                payload_json,
            )
            return None
        preset_idx: Optional[int] = None
        data = self._parse_json(text)
        if data is None:
            return None
        if context.get("naming_hint"):
            if data.get("skill") != "MENU_SEQUENCE":
                forced_idx = self._coerce_index(data.get("goal_index"))
                if forced_idx not in NAMING_PRESET_OPS:
                    forced_idx = 1
                ops_override = NAMING_PRESET_OPS[forced_idx]
                original_reason = data.get("reasoning") or ""
                data["skill"] = "MENU_SEQUENCE"
                data["goal_index"] = -1
                data["ops"] = ops_override
                data["reasoning"] = (original_reason + " Auto-converted to preset MENU_SEQUENCE.").strip()
                preset_idx = forced_idx
                LOGGER.info(
                    "Naming auto-conversion applied step=%s preset_idx=%s ops=%s raw_skill=%s",
                    (context or {}).get("step"),
                    forced_idx,
                    ops_override,
                    data.get("skill"),
                )
            else:
                preset_idx = self._coerce_index(data.get("goal_index"))
                if preset_idx not in NAMING_PRESET_OPS:
                    preset_idx = 1
        skill = data.get("skill")
        goal_index = self._coerce_index(data.get("goal_index"))
        reasoning = data.get("reasoning") or data.get("notes") or ""
        ops = self._coerce_ops(data.get("ops"))
        if context.get("naming_hint") and not ops:
            idx = preset_idx if preset_idx in NAMING_PRESET_OPS else (goal_index if goal_index in NAMING_PRESET_OPS else 1)
            ops = NAMING_PRESET_OPS.get(idx, ["DOWN", "A"])
        objective_spec = data.get("objective_spec") if isinstance(data.get("objective_spec"), dict) else None
        LOGGER.info(
            "HALT response step=%s skill=%s goal=%s has_spec=%s",
            context.get("step"),
            data.get("skill"),
            data.get("goal_index"),
            bool(objective_spec),
        )
        LOGGER.debug("HALT response json=%s", data)
        return GoalSuggestion(
            skill=skill if isinstance(skill, str) else None,
            goal_index=goal_index,
            reasoning=reasoning.strip(),
            ops=ops,
            objective_spec=objective_spec,
        )

    # ------------------------------------------------------------------ helpers
    def _top_tile_classes(self, tile_grid: List[List[str]]) -> List[Tuple[str, int]]:
        counts: Dict[str, int] = {}
        for row in tile_grid:
            for key in row:
                tile_class = key.split(":", 1)[-1]
                counts[tile_class] = counts.get(tile_class, 0) + 1
        return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    def _extract_text(self, payload: Dict[str, object]) -> Optional[str]:
        output = payload.get("output")
        if isinstance(output, list):
            texts: List[str] = []
            for block in output:
                content = block.get("content", []) if isinstance(block, dict) else []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") in {"output_text", "text"}:
                            text_val = item.get("text")
                            if isinstance(text_val, str):
                                texts.append(text_val)
            if texts:
                return "\n".join(texts).strip()
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list) and content:
                text_val = content[0].get("text")
                if isinstance(text_val, str):
                    return text_val
        return None

    def _parse_json(self, raw: str) -> Optional[Dict[str, object]]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            if "```" in raw:
                raw = raw.split("```", 1)[0]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            LOGGER.warning("GPTBrain returned non-JSON content: %s", raw)
            return None

    @staticmethod
    def _coerce_index(value: object) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("-") and value[1:].isdigit():
                return int(value)
            if value.isdigit():
                return int(value)
        return None

    @staticmethod
    def _coerce_ops(value: object) -> Optional[List[str]]:
        if value is None:
            return None
        if not isinstance(value, list):
            return None
        allowed_buttons = {"UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT", "NOOP"}
        ops: List[str] = []
        for v in value:
            if isinstance(v, str):
                vv = v.strip().upper()
                if vv in allowed_buttons:
                    ops.append(vv)
                    continue
                if vv.startswith("WAIT"):
                    if vv == "WAIT":
                        ops.append("WAIT_10")
                        continue
                    match = re.fullmatch(r"WAIT[_\-](\d+)", vv)
                    if match:
                        ops.append(f"WAIT_{match.group(1)}")
                        continue
        return ops or None
