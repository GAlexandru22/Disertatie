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
        # Incremented on each reset(); in-flight query threads compare their
        # captured generation against this and discard results if it changed.
        self._reset_generation: int = 0

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

    def reset(self) -> None:
        """Hard-reset VLM state: clear the last result and return to idle-ready.

        Wipes the stored result so the overlay disappears.  If a query is
        currently running it completes in the background and its result is
        discarded (the _reset_generation counter makes the stale write a no-op).
        """
        with self._lock:
            self._state.result = None
            # Bump generation so any in-flight query thread knows its result
            # is stale and should not be written back.
            self._reset_generation += 1
            gen = self._reset_generation
        print(f"[vlm] Hard reset (gen={gen}) — overlay cleared, stale query results will be discarded.")

    def query_async(
        self,
        frame: np.ndarray,
        proximity_summary: str,
        custom_context: str = "",
        person_statuses: Optional[list] = None,
    ) -> None:
        """Fire a non-blocking VLM query in a background thread.

        If the model is not ready or a query is already running, the request
        is silently dropped — the user can press V again.

        LLaVA receives both the hazard context and the YOLO per-worker detection
        data (what each worker IS wearing), then determines requirements and
        produces the final compliance verdict in one step.

        Args:
            frame:             The current video frame or multi-frame composite (BGR).
            proximity_summary: Hazard/person proximity string from ContextDetector.
            custom_context:    Optional free-text note typed by the operator.
            person_statuses:   List of dicts with per-worker YOLO detection results,
                               e.g. [{"worker_num": 1, "hardhat": True, ...}, ...]
        """
        if not self.ready:
            print("[vlm] Not ready yet — try again shortly.")
            return

        if not self._query_lock.acquire(blocking=False):
            print("[vlm] A query is already running — press V again when it finishes.")
            return

        with self._lock:
            self._state.status = "analyzing"
            generation = self._reset_generation   # snapshot before thread starts

        threading.Thread(
            target=self._run_query,
            args=(frame.copy(), proximity_summary, custom_context, person_statuses, generation),
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
        custom_context: str = "",
        person_statuses: Optional[list] = None,
        generation: int = 0,
    ) -> None:
        """Send image + YOLO detections to LLaVA; it determines requirements AND compliance.

        Steps:
          1. Short-circuit if nothing to reason about (no machinery, no operator note).
          2. Build a prompt that includes both hazard context and per-worker YOLO data.
          3. LLaVA determines what PPE is required and compares against what's worn.
          4. Parse the final verdict (compliant, violations, summary) directly.
          5. Discard result if reset() was called while the query was running.
        """
        result: VLMResult
        try:
            import requests

            print(f"[debug:vlm] person_statuses received: {person_statuses}")
            print(f"[debug:vlm] custom_context: '{custom_context}'")

            no_machinery = not proximity_summary or proximity_summary == "no heavy machinery detected"

            if no_machinery and not custom_context.strip():
                print("[debug:vlm] Short-circuit: no hazards and no operator note — skipping VLM call")
                result = VLMResult(
                    compliant=True,
                    violations=[],
                    summary="No hazards detected near workers.",
                )
            else:
                image_b64 = _encode_frame_b64(frame)
                prompt = _build_prompt(proximity_summary, custom_context, person_statuses or [])
                print(f"[debug:vlm] Prompt sent to LLaVA:\n{prompt}\n---")

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

                result = _parse_vlm_verdict(raw_response)
                if result is None:
                    print("[vlm] JSON parse failed — returning compliant with no violations")
                    result = VLMResult(
                        compliant=True,
                        violations=[],
                        summary="Could not parse VLM response.",
                    )
                else:
                    print(f"[debug:vlm] Parsed verdict: compliant={result.compliant} violations={result.violations}")

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
            if self._reset_generation != generation:
                print(f"[vlm] Query result discarded (reset occurred during analysis).")
                return
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


def _build_prompt(
    proximity_summary: str,
    custom_context: str = "",
    person_statuses: Optional[list] = None,
) -> str:
    """Guide LLaVA through two explicit reasoning steps.

    Step 1: identify hazards from the image → determine required PPE.
    Step 2: compare required PPE against YOLO sensor readings → produce verdict.
    """
    context_lines: list[str] = []
    if proximity_summary and proximity_summary != "no heavy machinery detected":
        context_lines.append(f"Proximity sensor: {proximity_summary}")
    if custom_context.strip():
        context_lines.append(f"Operator note: {custom_context}")
    context_section = (
        "\n".join(context_lines) if context_lines
        else "No additional sensor data."
    )

    if person_statuses:
        worker_lines: list[str] = []
        for ps in person_statuses:
            worn = [item for item in ("hardhat", "mask", "vest") if ps.get(item, False)]
            not_worn = [item for item in ("hardhat", "mask", "vest") if not ps.get(item, False)]
            nearby = f", near: {', '.join(ps['nearby_hazards'])}" if ps.get("nearby_hazards") else ""
            worker_lines.append(
                f"  Worker {ps['worker_num']}: "
                f"WEARING [{', '.join(worn) if worn else 'none'}] "
                f"NOT WEARING [{', '.join(not_worn) if not_worn else 'none'}]"
                f"{nearby}"
            )
        workers_section = "\n".join(worker_lines)
    else:
        workers_section = "  No workers detected."

    return (
        "You are a construction site safety inspector. Follow these two steps.\n\n"
        "--- STEP 1: ASSESS THE SCENE ---\n"
        "Look at the image. What hazards are present?\n"
        "Then decide: for each hazard, which PPE item does it require?\n"
        "  hardhat -> cranes, overhead work, falling objects, debris, excavators\n"
        "  vest    -> moving vehicles, forklifts, bulldozers, heavy machinery\n"
        "  mask    -> ONLY if you see dense dust clouds / smoke in the image,\n"
        "             OR the operator note explicitly mentions dust/chemicals/fumes\n\n"
        "Additional context from sensors:\n"
        f"{context_section}\n\n"
        "--- STEP 2: CHECK EACH WORKER ---\n"
        "Using the required PPE you determined in Step 1, check the YOLO sensor\n"
        "readings below. Trust the sensor for what each worker is wearing.\n\n"
        "YOLO sensor readings:\n"
        f"{workers_section}\n\n"
        "For each worker: if they are NOT WEARING a required item -> violation.\n"
        "If no hazards require PPE, all workers are compliant.\n\n"
        "Respond ONLY with valid JSON — no other text:\n"
        '{"hazards_seen": ["..."], "required_ppe": ["..."], '
        '"compliant": true, "violations": ["Worker N missing: item (reason)"], '
        '"summary": "one sentence"}'
    )


def _parse_vlm_verdict(raw: str) -> Optional[VLMResult]:
    """Parse LLaVA's two-step compliance verdict from JSON.

    Expects: {"hazards_seen": [...], "required_ppe": [...],
              "compliant": bool, "violations": [...], "summary": "..."}
    Returns None if the response cannot be parsed.
    """
    json_str = _extract_balanced_json(raw)
    if not json_str:
        return None
    try:
        data = json.loads(json_str)
        violations = [str(v) for v in data.get("violations", [])]
        compliant = bool(data.get("compliant", len(violations) == 0))
        if violations:
            compliant = False
        summary = str(data.get("summary", ""))
        hazards = data.get("hazards_seen", [])
        required = data.get("required_ppe", [])
        if hazards or required:
            print(f"[debug:vlm] Hazards seen: {hazards} -> Required PPE: {required}")
        return VLMResult(compliant=compliant, violations=violations, summary=summary)
    except (json.JSONDecodeError, TypeError):
        return None


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
