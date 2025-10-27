"""GPT-5 Mini integration for high-level planning."""

from __future__ import annotations

import json
import logging
import os
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import base64

import requests

LOGGER = logging.getLogger(__name__)

Coord = Tuple[int, int]


@dataclass(slots=True)
class GoalSuggestion:
    """LLM-provided directive for skills/goals and optional menu ops."""

    skill: Optional[str]
    goal_index: Optional[int]
    reasoning: str
    ops: Optional[List[str]] = None

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
        max_output_tokens: int = 256,
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
            """
        ).strip()
        full_prompt = prompt + "\n\n" + prompt_extra
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
        if self.enable_web_search:
            payload["tools"] = [{"type": "web_search"}]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            self.call_count += 1
            LOGGER.info("GPTBrain API call #%d", self.call_count)
            response = self.session.post(
                f"{self.base_url}/responses",
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = ""
            if hasattr(exc, "response") and exc.response is not None:
                detail = exc.response.text
            LOGGER.warning("GPTBrain request failed: %s %s", exc, detail)
            return None
        text = self._extract_text(response.json())
        if not text:
            return None
        data = self._parse_json(text)
        if data is None:
            return None
        skill = data.get("skill")
        goal_index = self._coerce_index(data.get("goal_index"))
        reasoning = data.get("reasoning") or data.get("notes") or ""
        ops = self._coerce_ops(data.get("ops"))
        return GoalSuggestion(
            skill=skill if isinstance(skill, str) else None,
            goal_index=goal_index,
            reasoning=reasoning.strip(),
            ops=ops,
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
        allowed = {"UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"}
        ops: List[str] = []
        for v in value:
            if isinstance(v, str):
                vv = v.strip().upper()
                if vv in allowed:
                    ops.append(vv)
        return ops or None
