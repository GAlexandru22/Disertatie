"""
LLaVA-7B vision-language model for context-aware PPE compliance reasoning.
Powered by Ollama — a local model server that handles quantization and VRAM
management automatically, with no bitsandbytes or transformers required.

What is Ollama?
---------------
Ollama (https://ollama.com) is a local model runner that downloads and serves
open-source LLMs/VLMs using the GGUF quantization format.  It starts as a
background service on port 11434 and exposes a simple REST API.  We just send
it an HTTP POST with the frame (base64-encoded) and a text prompt, and it
returns the model's response as JSON.  No GPU memory management code needed —
Ollama handles it all internally.

Setup (one-time):
  1. Install Ollama: https://ollama.com/download  (Windows installer)
  2. Pull the model: ollama pull llava:7b          (~4 GB download)
  3. Ollama runs automatically in the system tray after install.

Model: llava:7b
  - LLaVA 1.5 7B, Q4_0 quantization (~4 GB VRAM or auto-offloaded to RAM)
  - Vision-language model: understands both images and text questions
  - Returns free-text answers, which we parse as structured JSON

Threading model:
  Loading is a quick health-check (< 1 s) against the Ollama server.
  Queries (~2-5 s) run in a background thread so the video is never blocked.
"""

from __future__ import annotations

import base64
import io
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Ollama server address — change if you run Ollama on a different machine
OLLAMA_BASE_URL = "http://localhost:11434"

# The model to use.  "llava:7b" is the 7B LLaVA 1.5 model at Q4_0 quant.
# Alternatives: "llava:13b" (more accurate, needs ~8 GB VRAM)
OLLAMA_MODEL = "llava:7b"

# Maximum seconds to wait for a single VLM response before timing out
QUERY_TIMEOUT_SECONDS = 120


# ---------------------------------------------------------------------------
# Data structures exposed to the rest of the pipeline
# ---------------------------------------------------------------------------

@dataclass
class VLMResult:
    """Parsed output from a single LLaVA query.

    The visualizer reads these fields to draw the result overlay.
    """

    compliant: bool
    """True if LLaVA judges all workers properly equipped for the hazards present."""

    violations: list[str] = field(default_factory=list)
    """Natural-language violation descriptions, one per issue found.

    Example: ["Person 1 has no hardhat while standing near the bulldozer."]
    """

    summary: str = ""
    """One-sentence VLM reasoning summary shown in the overlay panel."""


@dataclass
class VLMState:
    """A thread-safe snapshot of the VLM analyzer's current status.

    The visualizer reads this every frame to decide what to draw.
    """

    status: Literal["idle", "loading", "analyzing", "done"] = "idle"
    """
    idle      — analyzer not yet constructed.
    loading   — checking that Ollama is running and the model is available.
    analyzing — a query is currently running; results not yet available.
    done      — last query finished; result holds the VLMResult.
    """

    result: Optional[VLMResult] = None
    """The result of the most recent completed query, or None if no query has run."""

    timestamp: float = 0.0
    """time.time() value when the result was recorded (for "X seconds ago" display)."""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class VLMAnalyzer:
    """Manages Ollama-backed LLaVA queries for context-aware PPE reasoning.

    Typical lifecycle:
        1. Constructed in detect.py main() when --context is enabled.
        2. Health-check thread verifies Ollama is running; state = "done" when OK.
        3. User presses 'V' (or auto-trigger fires on first person detected).
        4. query_async() fires; state = "analyzing".
        5. Ollama responds; state = "done", state.result populated.
        6. Visualizer reads state every frame and draws the result overlay.
    """

    def __init__(self) -> None:
        self._state = VLMState(status="loading")
        self._lock = threading.Lock()
        # Prevents two overlapping queries from running simultaneously
        self._query_lock = threading.Lock()

        # Health-check runs in a background thread so video starts immediately
        threading.Thread(
            target=self._check_ollama,
            daemon=True,
            name="vlm-health-check",
        ).start()

    # ------------------------------------------------------------------
    # Public read-only API
    # ------------------------------------------------------------------

    @property
    def state(self) -> VLMState:
        """Thread-safe snapshot of the current VLM state."""
        with self._lock:
            return VLMState(
                status=self._state.status,
                result=self._state.result,
                timestamp=self._state.timestamp,
            )

    @property
    def ready(self) -> bool:
        """True once the health check has finished (model confirmed available)."""
        with self._lock:
            return self._state.status not in ("idle", "loading")

    # ------------------------------------------------------------------
    # Public action API
    # ------------------------------------------------------------------

    def query_async(
        self,
        frame: np.ndarray,
        proximity_summary: str,
        ppe_summary: str,
    ) -> None:
        """Fire a non-blocking VLM query in a background thread.

        If the model is not ready or a query is already running, the request
        is silently dropped — the user can press V again.

        Args:
            frame:             The current video frame (BGR, H×W×3).
            proximity_summary: Hazard/person proximity string from ContextDetector,
                               e.g. "1 person near bulldozer".
            ppe_summary:       One-line PPE status from context_rules.build_ppe_summary(),
                               e.g. "Person 1: hardhat=NO, mask=YES, vest=NO".
        """
        if not self.ready:
            print("[vlm] Not ready yet — try again shortly.")
            return

        if not self._query_lock.acquire(blocking=False):
            print("[vlm] A query is already running — press V again when it finishes.")
            return

        with self._lock:
            self._state.status = "analyzing"

        threading.Thread(
            target=self._run_query,
            args=(frame.copy(), proximity_summary, ppe_summary),
            daemon=True,
            name="vlm-query",
        ).start()

    # ------------------------------------------------------------------
    # Background: Ollama health check
    # ------------------------------------------------------------------

    def _check_ollama(self) -> None:
        """Verify that Ollama is running and the llava:7b model is available.

        Steps:
          1. GET http://localhost:11434 — confirms the Ollama server is up.
          2. GET /api/tags — lists available models, checks for llava:7b.
          3. If healthy: mark state as "done" (ready for queries).
          4. If not running or model missing: populate state.result with a
             helpful error message so the overlay shows what to fix.
        """
        try:
            import requests  # standard library alternative if requests unavailable

            # Step 1: ping the server
            try:
                ping = requests.get(f"{OLLAMA_BASE_URL}", timeout=5)
                ping.raise_for_status()
            except Exception:
                raise RuntimeError(
                    "Ollama server is not running.\n"
                    "Start it by opening the Ollama app or running: ollama serve"
                )

            # Step 2: check the model is pulled
            tags_resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
            tags_resp.raise_for_status()
            available_models = [m["name"] for m in tags_resp.json().get("models", [])]

            # Accept both "llava:7b" and "llava:7b-..." variants
            model_present = any(
                OLLAMA_MODEL.split(":")[0] in name for name in available_models
            )
            if not model_present:
                raise RuntimeError(
                    f"Model '{OLLAMA_MODEL}' is not pulled.\n"
                    f"Run: ollama pull {OLLAMA_MODEL}\n"
                    f"Available models: {', '.join(available_models) or 'none'}"
                )

            print(f"[vlm] Ollama ready — using model '{OLLAMA_MODEL}'.")
            with self._lock:
                self._state.status = "done"
                self._state.result = None  # no query run yet

        except Exception as exc:
            print(f"[vlm] Ollama check failed: {exc}")
            with self._lock:
                self._state.status = "done"
                self._state.result = VLMResult(
                    compliant=False,
                    violations=[str(exc)],
                    summary="VLM unavailable — see console for setup instructions.",
                )

    # ------------------------------------------------------------------
    # Background: query execution
    # ------------------------------------------------------------------

    def _run_query(
        self,
        frame: np.ndarray,
        proximity_summary: str,
        ppe_summary: str,
    ) -> None:
        """Send the frame + prompt to Ollama and store the result.  Runs in a thread.

        Steps:
          1. Encode the BGR frame as a JPEG and convert to base64.
          2. Build the structured safety-inspection prompt.
          3. POST to Ollama's /api/chat endpoint with the image + prompt.
          4. Parse the JSON response into a VLMResult.
          5. Update state so the visualizer picks it up on the next frame.
        """
        result: VLMResult
        try:
            import requests

            # Step 1: encode the frame as base64 JPEG.
            # JPEG is compact — reduces the payload size vs PNG without
            # meaningfully affecting the model's visual understanding.
            image_b64 = _encode_frame_b64(frame)

            # Step 2: build the prompt
            prompt = _build_prompt(proximity_summary, ppe_summary)

            # Step 3: call the Ollama chat endpoint.
            # stream=False means we wait for the full response before returning.
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_b64],
                    }
                ],
                "stream": False,
            }

            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=QUERY_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()

            raw_response = resp.json()["message"]["content"].strip()
            print(f"[vlm] Raw response: {raw_response[:300]}")

            # Step 4: parse structured JSON from LLaVA's response
            result = _parse_response(raw_response)

        except Exception as exc:
            print(f"[vlm] Query error: {exc}")
            result = VLMResult(
                compliant=False,
                violations=[f"Query failed: {exc}"],
                summary="Analysis failed — check console for details.",
            )
        finally:
            self._query_lock.release()

        with self._lock:
            self._state.status = "done"
            self._state.result = result
            self._state.timestamp = time.time()


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, no class state)
# ---------------------------------------------------------------------------

def _encode_frame_b64(frame: np.ndarray) -> str:
    """Convert a BGR OpenCV frame to a base64-encoded JPEG string.

    Ollama's API requires images as base64 strings in the `images` field of
    the message.  We use JPEG at quality 85 — high enough for visual detail
    without unnecessary payload size.
    """
    from PIL import Image  # type: ignore

    rgb = frame[:, :, ::-1]        # BGR → RGB (Pillow expects RGB)
    pil_image = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_prompt(proximity_summary: str, ppe_summary: str) -> str:
    """Construct the structured safety-inspection prompt.

    We provide the YOLOv11m PPE sensor readings as additional context so
    LLaVA focuses on reasoning about the environment rather than re-detecting
    low-level PPE items from scratch.  The JSON output instruction is critical
    for reliable parsing.
    """
    return (
        "You are a construction site safety inspector reviewing a live camera feed.\n\n"
        "Scene information detected by sensors:\n"
        f"  Machinery / hazards: {proximity_summary}\n"
        f"  PPE sensor readings: {ppe_summary}\n\n"
        "Task: Look at the image and assess whether all visible workers are wearing "
        "the required personal protective equipment (PPE) for the hazards present. "
        "Required PPE near heavy machinery (bulldozer, excavator, crane, etc.): "
        "hardhat and safety vest at minimum.\n\n"
        "Respond ONLY with a single valid JSON object — no extra text before or after:\n"
        '{"compliant": true_or_false, '
        '"violations": ["describe each violation briefly"], '
        '"summary": "one sentence conclusion"}'
    )


def _parse_response(raw: str) -> VLMResult:
    """Parse LLaVA's text output into a VLMResult.

    Strategy:
      1. Extract a balanced JSON object using bracket counting (handles nested lists).
      2. If JSON parsing fails, fall back to keyword-based heuristics.
    """
    json_str = _extract_balanced_json(raw)
    if json_str:
        try:
            data = json.loads(json_str)
            violations = data.get("violations", [])
            if isinstance(violations, str):
                violations = [violations] if violations else []
            return VLMResult(
                compliant=bool(data.get("compliant", False)),
                violations=[str(v) for v in violations],
                summary=str(data.get("summary", raw[:200])),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Heuristic fallback
    lower = raw.lower()
    positive = {"compliant", "all workers", "properly equipped", "wearing"}
    negative = {"missing", "not wearing", "no hardhat", "violation", "non-compliant"}
    is_compliant = (
        any(w in lower for w in positive) and not any(w in lower for w in negative)
    )
    return VLMResult(
        compliant=is_compliant,
        violations=[] if is_compliant else ["See summary — JSON parsing failed."],
        summary=raw[:300] if raw else "No response from model.",
    )


def _extract_balanced_json(text: str) -> Optional[str]:
    """Extract the first balanced JSON object using bracket counting."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
