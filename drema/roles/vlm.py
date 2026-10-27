#!/usr/bin/env python
"""
Vision-language backends used by the environment-side role providers.

The VLM always runs in a separate process (an OpenAI-compatible server such as vLLM, Ollama or a hosted API):
the environment only sends HTTP requests, so the simulator Python (3.8 for RLBench / PyRep) needs no VLM
dependency. The scene is shown with set-of-marks: each candidate object is labelled with its numeric id on the
image and listed by id and name, so every answer refers to an id the server already tracks.
"""

import base64
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SceneObject:
    object_id: int
    name: str


@dataclass
class GroundedCommand:
    """A human message resolved to one object and one permission."""
    object_id: Optional[int]
    permission: str                 # touch / avoid / fragile / clear
    confidence: float = 0.0
    ambiguous: bool = False
    candidates: List[int] = field(default_factory=list)
    explanation: str = ""


@dataclass
class SafetyFlag:
    """An autonomous restriction proposed by the VLM (never a relaxation)."""
    object_id: int
    permission: str                 # avoid / fragile
    reason: str = ""


def _objects_block(objects: List[SceneObject]) -> str:
    return "\n".join(f"- {o.object_id}: {o.name}" for o in objects)


def _parse_json(text: str) -> Any:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start > 0:
        text = text[start:]
    return json.loads(text)


class VLMBackend:

    def ground_command(self, message: str, objects: List[SceneObject], image: Optional[np.ndarray],
                       instruction: str = "") -> GroundedCommand:
        raise NotImplementedError

    def safety_flags(self, objects: List[SceneObject], image: Optional[np.ndarray],
                     instruction: str = "") -> List[SafetyFlag]:
        raise NotImplementedError


GROUND_PROMPT = """You map a human operator's instruction about a robot manipulation scene to exactly one object.
The image shows the scene; each object is labelled with its numeric id. Objects:
{objects}

Task the robot is doing: {instruction}
Operator message: "{message}"

Decide which object the operator refers to and what the robot is allowed to do with it:
- "touch": the robot may touch, push or grasp it;
- "avoid": the robot must not touch it;
- "fragile": the robot must stay farther away than usual;
- "clear": remove any previous instruction about it.
If more than one object fits the description, set "ambiguous" to true and list them in "candidates".
Answer with JSON only:
{{"object_id": <id or null>, "permission": "touch|avoid|fragile|clear", "confidence": <0..1>,
 "ambiguous": <true|false>, "candidates": [<ids>], "explanation": "<short>"}}"""

SAFETY_PROMPT = """You are the safety supervisor of a robot arm working on a table.
The image shows the scene; each object is labelled with its numeric id. Objects:
{objects}

Task the robot is doing: {instruction}

List only the objects the robot must NOT touch (e.g. a person's hand, glassware, electronics, anything not part of
the task that could break or hurt someone) or must keep a larger distance from. You can only restrict: never
grant permission to touch anything. Answer with JSON only:
{{"objects": [{{"id": <id>, "permission": "avoid|fragile", "reason": "<short>"}}]}}"""


class OpenAICompatibleVLM(VLMBackend):
    """Chat-completions endpoint with image input (vLLM, Ollama /v1, OpenAI and compatible servers)."""

    def __init__(self, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY", timeout_s: float = 60.0,
                 temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.timeout_s = float(timeout_s)
        self.temperature = float(temperature)

    @staticmethod
    def _encode_png(image: np.ndarray) -> Optional[str]:
        try:
            import cv2
            ok, buf = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR))
            return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
        except Exception:
            return None

    def _chat(self, prompt: str, image: Optional[np.ndarray]) -> str:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        png = self._encode_png(image) if image is not None else None
        if png is not None:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}})
        body = json.dumps({
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": content}],
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]

    def ground_command(self, message, objects, image, instruction=""):
        raw = self._chat(GROUND_PROMPT.format(objects=_objects_block(objects), instruction=instruction or "unknown",
                                              message=message), image)
        data = _parse_json(raw)
        oid = data.get("object_id")
        return GroundedCommand(
            object_id=int(oid) if oid is not None else None,
            permission=str(data.get("permission", "")).lower(),
            confidence=float(data.get("confidence", 0.0)),
            ambiguous=bool(data.get("ambiguous", False)),
            candidates=[int(c) for c in data.get("candidates", []) or []],
            explanation=str(data.get("explanation", ""))
        )

    def safety_flags(self, objects, image, instruction=""):
        raw = self._chat(SAFETY_PROMPT.format(objects=_objects_block(objects), instruction=instruction or "unknown"),
                         image)
        data = _parse_json(raw)
        items = data.get("objects", []) if isinstance(data, dict) else data
        return [SafetyFlag(int(it["id"]), str(it.get("permission", "avoid")).lower(), str(it.get("reason", "")))
                for it in items if "id" in it]


class KeywordVLM(VLMBackend):
    """
    Offline stand-in without a model: grounds by word overlap with the object names and reads the permission
    from keywords. Meant for tests and for running the pipeline without a VLM server.
    """

    TOUCH_WORDS = ("can touch", "may touch", "grab", "grasp", "pick", "push", "allowed")
    AVOID_WORDS = ("don't touch", "do not touch", "avoid", "never touch", "stay away")
    FRAGILE_WORDS = ("fragile", "delicate", "careful")
    CLEAR_WORDS = ("reset", "clear", "forget")

    def __init__(self, fragile_name_keywords: Optional[List[str]] = None):
        self.fragile_name_keywords = [k.lower() for k in (fragile_name_keywords or [])]

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 1 and not t.isdigit()]

    def ground_command(self, message, objects, image, instruction=""):
        msg = message.lower()
        if any(w in msg for w in self.CLEAR_WORDS):
            permission = "clear"
        elif any(w in msg for w in self.AVOID_WORDS):
            permission = "avoid"
        elif any(w in msg for w in self.FRAGILE_WORDS):
            permission = "fragile"
        elif any(w in msg for w in self.TOUCH_WORDS):
            permission = "touch"
        else:
            permission = ""

        explicit = [int(m) for m in re.findall(r"\b(\d+)\b", msg)]
        ids = {o.object_id for o in objects}
        if explicit and explicit[0] in ids:
            return GroundedCommand(explicit[0], permission, 1.0, False, [explicit[0]], "explicit id")

        msg_tokens = set(self._tokens(msg))
        scores = {o.object_id: len(msg_tokens & set(self._tokens(o.name))) for o in objects}
        best = max(scores.values(), default=0)
        candidates = [oid for oid, s in scores.items() if s == best and best > 0]
        if not candidates:
            return GroundedCommand(None, permission, 0.0, False, [], "no object name matches the message")
        ambiguous = len(candidates) > 1
        return GroundedCommand(candidates[0] if not ambiguous else None, permission,
                               0.9 if not ambiguous else 0.4, ambiguous, candidates, "name match")

    def safety_flags(self, objects, image, instruction=""):
        return [SafetyFlag(o.object_id, "fragile", "name keyword") for o in objects
                if any(k in o.name.lower() for k in self.fragile_name_keywords)]


def build_vlm_backend(cfg: Dict[str, Any]) -> VLMBackend:
    kind = str(cfg.get("backend", "keyword")).lower()
    if kind in ("openai", "openai_compatible", "vllm", "ollama"):
        return OpenAICompatibleVLM(
            base_url=str(cfg.get("base_url", "http://localhost:8000/v1")),
            model=str(cfg.get("model", "")),
            api_key_env=str(cfg.get("api_key_env", "OPENAI_API_KEY")),
            timeout_s=float(cfg.get("timeout_s", 60.0)),
            temperature=float(cfg.get("temperature", 0.0))
        )
    if kind == "keyword":
        return KeywordVLM(fragile_name_keywords=cfg.get("fragile_name_keywords", []))
    raise ValueError(f"Unknown VLM backend '{kind}' (expected 'openai_compatible' or 'keyword')")
