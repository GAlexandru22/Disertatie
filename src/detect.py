"""
PPE compliance detection — live webcam and video file modes.

Usage:
  # Live webcam (PPE only, existing behaviour)
  python src/detect.py --weights runs/train/ppe_yolo11/weights/best.pt --source 0

  # Video file (display only)
  python src/detect.py --weights best.pt --source site.mp4

  # Video file (save annotated output)
  python src/detect.py --weights best.pt --source site.mp4 --save --output result.mp4

  # Headless (server / CI)
  python src/detect.py --weights best.pt --source site.mp4 --save --no-display

  # Context-aware mode: YOLO-World + LLaVA VLM
  python src/detect.py --weights best.pt --source 0 --context

Keyboard shortcuts (display window):
  Q      — quit
  V / v  — run a fresh LLaVA VLM safety analysis on the current frame
            (only available when --context is enabled and model has loaded)
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# Allow running from repo root (python src/detect.py) or from src/
sys.path.insert(0, str(Path(__file__).parent))

from compliance import ComplianceChecker
from visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPE compliance detection")
    p.add_argument("--weights", required=True,
                   help="Path to trained YOLO weights (best.pt)")
    p.add_argument("--source",  required=True,
                   help="'0' (or any digit) for webcam, or path to a video file")
    p.add_argument("--conf",    type=float, default=0.35,
                   help="Detection confidence threshold")
    p.add_argument("--iou",     type=float, default=0.45,
                   help="NMS IoU threshold")
    p.add_argument("--device",  default="cpu",
                   help="Inference device: 'cpu', '0' for GPU, 'mps' for Apple Silicon")
    p.add_argument("--save",    action="store_true",
                   help="Save annotated output video")
    p.add_argument("--output",  default="output.mp4",
                   help="Output video path (used with --save)")
    p.add_argument("--no-display", dest="no_display", action="store_true",
                   help="Suppress OpenCV display window (headless mode)")
    p.add_argument("--half",    action="store_true",
                   help="Use FP16 inference (GPU only)")
    p.add_argument(
        "--context", action="store_true",
        help=(
            "Enable context-aware mode: adds YOLO-World machinery detection "
            "and LLaVA-1.5-7B VLM reasoning.  Requires a GPU with ~4 GB VRAM "
            "for the VLM.  Press V in the video window to trigger a VLM query."
        ),
    )
    return p.parse_args()


def open_source(source: str) -> tuple[cv2.VideoCapture, dict]:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        label = "WEBCAM"
        total_frames = -1
    else:
        path = Path(source)
        if not path.exists():
            sys.exit(f"[detect] Video file not found: {path.resolve()}")
        cap = cv2.VideoCapture(str(path))
        label = path.name
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        sys.exit(f"[detect] Could not open source: {source}")

    meta = {
        "label": label,
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": total_frames,
    }
    return cap, meta


def build_writer(meta: dict, output_path: str) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path, fourcc, meta["fps"],
        (meta["width"], meta["height"]),
    )
    if not writer.isOpened():
        sys.exit(f"[detect] Could not open video writer for: {output_path}")
    return writer


def _build_composite_frame(frames: list) -> np.ndarray:
    """Arrange up to 6 sampled frames into a 2-column grid for multi-frame VLM analysis.

    Frames are picked at evenly-spaced intervals from the history so the grid
    covers several seconds rather than just the trigger moment.  A 6-frame
    history at _FRAME_SAMPLE_RATE=20 spans ~4 seconds at 30fps.

    Grid layout:  2 cols × N rows  (N = ceil(frames/2))
      2 frames → 1 row  (2×1)
      4 frames → 2 rows (2×2)
      6 frames → 3 rows (2×3)
    """
    n = min(len(frames), 6)
    if n == 0:
        return np.zeros((360, 640, 3), dtype=np.uint8)
    if n == 1:
        return frames[0].copy()

    # Pick evenly-spaced frames across the full history
    indices = [int(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    selected = [frames[i] for i in indices]

    h, w = selected[0].shape[:2]
    cell_w, cell_h = w // 2, h // 2
    resized = [cv2.resize(f, (cell_w, cell_h)) for f in selected]

    # Pad to even count so every row is full
    if len(resized) % 2 != 0:
        resized.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

    rows = [np.hstack(resized[i:i+2]) for i in range(0, len(resized), 2)]
    return np.vstack(rows)


def _ask_custom_context() -> str:
    """Show a small tkinter dialog so the operator can type a free-text context note.

    Blocks the video loop for as long as the user is typing — that is intentional;
    the video pauses, the user types, the video resumes after OK / Cancel.
    Returns an empty string if the user cancels or types nothing.
    """
    try:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()           # hide the empty root window
        root.attributes("-topmost", True)
        text = simpledialog.askstring(
            title="VLM Analysis — operator context",
            prompt=(
                "Type any additional context for the VLM\n"
                "(e.g. 'this worker is handling live cables').\n"
                "Leave blank to use auto-detected context only."
            ),
            parent=root,
        )
        root.destroy()
        return text.strip() if text else ""
    except Exception as exc:
        print(f"[detect] Dialog unavailable ({exc}); using auto-context only.")
        return ""



_SMOOTHING_WINDOW = 10         # frames to average for stable PPE detection at VLM query time
_VLM_RETRIGGER_INTERVAL = 15.0  # seconds between automatic re-analyses while a context is active
_FRAME_HISTORY_SIZE = 6        # number of sampled frames kept for multi-frame VLM composite
_FRAME_SAMPLE_RATE = 20        # append one frame every N frames (~4 seconds coverage at 30fps)


def _smooth_person_statuses(report, history: deque, max_frames: int | None = None) -> list[dict]:
    """Build per-worker PPE status using a majority vote over recent frames.

    A 75-85% fluctuating mask detection will vote YES across the window and
    be treated as detected, avoiding false violations on single-frame dips.
    Persons are matched across frames by their detection index (person_idx).

    max_frames: if set, only consider the most recent N frames.  Use a small
    value (e.g. 5) for manual triggers so that stale pre-state-change history
    (e.g. "mask was off for 8 frames before the user put it on") does not
    outvote the current state.  Use None (full window) for auto-triggers where
    the scene has been stable for many frames.
    """
    frames = list(history)
    if max_frames is not None:
        frames = frames[-max_frames:]
    print(f"[debug:smooth] Smoothing over {len(frames)} frames")
    smoothed = []
    for rec in report.persons:
        idx = rec.person_idx
        hardhat_yes = sum(1 for r in frames for p in r.persons if p.person_idx == idx and p.hardhat_compliant)
        mask_yes    = sum(1 for r in frames for p in r.persons if p.person_idx == idx and p.mask_compliant)
        vest_yes    = sum(1 for r in frames for p in r.persons if p.person_idx == idx and p.vest_compliant)
        total       = sum(1 for r in frames for p in r.persons if p.person_idx == idx)
        n = max(total, 1)
        print(
            f"[debug:smooth]   P{idx+1}: "
            f"hardhat {hardhat_yes}/{n}={hardhat_yes/n:.0%} -> {hardhat_yes/n > 0.5} | "
            f"mask {mask_yes}/{n}={mask_yes/n:.0%} -> {mask_yes/n > 0.5} | "
            f"vest {vest_yes}/{n}={vest_yes/n:.0%} -> {vest_yes/n > 0.5}"
        )
        smoothed.append({
            "worker_num": idx + 1,
            "hardhat": hardhat_yes / n > 0.5,
            "mask":    mask_yes    / n > 0.5,
            "vest":    vest_yes    / n > 0.5,
            "nearby_hazards": rec.nearby_hazards,
        })
    return smoothed


def _trigger_vlm_query(
    vlm_analyzer, frame, report, hazard_dets, context_detector,
    custom_context: str = "",
    compliance_history: deque | None = None,
    frame_history: list | None = None,
) -> None:
    """Helper: build prompt context strings and fire a VLM query.

    Pulled out as a named function to avoid duplicating the same 4 lines for
    the deferred key-press trigger and the auto-trigger on first person detected.
    """
    proximity_summary = context_detector.get_proximity_summary(
        hazard_dets,
        [rec.detection.xyxy for rec in report.persons],
    )

    # Use rolling-window majority vote when history is available so that a
    # transiently-missed detection (e.g. mask at 75 % confidence) does not
    # produce a false violation in the snapshot sent to the VLM.
    if compliance_history:
        person_statuses = _smooth_person_statuses(report, compliance_history)
    else:
        person_statuses = [
            {
                "worker_num": rec.person_idx + 1,
                "hardhat": rec.hardhat_compliant,
                "mask":    rec.mask_compliant,
                "vest":    rec.vest_compliant,
                "nearby_hazards": rec.nearby_hazards,
            }
            for rec in report.persons
        ]

    # Build a multi-frame composite so LLaVA sees temporal context rather than
    # a single snapshot where a worker may have briefly turned away or been occluded.
    if frame_history and len(frame_history) >= 2:
        analysis_frame = _build_composite_frame(list(frame_history))
        print(f"[detect] VLM composite built from {min(len(frame_history), 4)} frames")
    else:
        analysis_frame = frame

    if custom_context:
        print(f"[detect] VLM query triggered — operator note: '{custom_context}'")
    else:
        print(f"[detect] VLM query triggered — context: {proximity_summary}")
    vlm_analyzer.query_async(analysis_frame, proximity_summary, custom_context, person_statuses)


def run_loop(
    model,
    cap: cv2.VideoCapture,
    checker: ComplianceChecker,
    viz: Visualizer,
    args: argparse.Namespace,
    meta: dict,
    writer: cv2.VideoWriter | None,
    context_detector=None,   # ContextDetector | None
    vlm_analyzer=None,       # VLMAnalyzer    | None
) -> None:
    """Main per-frame inference loop.

    Each iteration:
      1. Read a frame from the capture source.
      2. Run YOLOv11m PPE detection.
      3. (context mode) Run YOLO-World machinery detection.
      4. Run ComplianceChecker to produce per-person compliance flags and
         populate nearby_hazards if hazard detections are available.
      5. Handle keyboard input (Q to quit, V to trigger VLM query).
      6. (context mode) Auto-trigger VLM query on the first frame that has
         at least one person and the model has finished loading.
      7. Draw the annotated frame and optionally save it.
    """
    source_label = meta["label"]
    total = meta["total_frames"]
    frame_idx = 0
    prev_time = time.perf_counter()

    # Auto-trigger flag: fire one initial VLM query as soon as the model is
    # ready and a person is visible.  After that, only manual 'V' key fires.
    vlm_auto_triggered = False

    # Rolling window of recent compliance reports — used to smooth out single-
    # frame detection dips when building person_statuses for VLM queries.
    compliance_history: deque = deque(maxlen=_SMOOTHING_WINDOW)

    # Sampled frames for the multi-frame VLM composite.  Updated every
    # _FRAME_SAMPLE_RATE frames so the 4 stored frames span ~1 second.
    frame_history: list = []

    # When the user presses V, the tkinter dialog BLOCKS the frame loop while
    # they type.  Triggering the VLM immediately after dialog-close would use a
    # compliance snapshot frozen before the scene was ready.  Instead we:
    #   1. Clear the history the moment the dialog closes (flush stale data).
    #   2. Arm pending_vlm_context with the typed note.
    #   3. Wait for _FRESH_FRAMES_NEEDED fresh post-dialog frames to accumulate.
    #   4. Only then fire the VLM with a snapshot that truly reflects what the
    #      camera sees right now.
    _FRESH_FRAMES_NEEDED = 8
    pending_vlm_context: str | None = None

    # The context string currently in effect for continuous re-analysis.
    # None  = no active context; VLM will not auto-repeat.
    # str   = VLM re-fires every _VLM_RETRIGGER_INTERVAL seconds with this context.
    # Set by V key, cleared by R key.
    active_context: str | None = None

    # Wall-clock time of the last VLM query fire (used to throttle re-triggers).
    last_vlm_trigger_time: float = 0.0

    # Window title updates when context mode is on to advertise the V shortcut.
    window_title = (
        "PPE Compliance Detection — Q: quit | V: VLM analysis | R: reset"
        if context_detector else
        "PPE Compliance Detection — press Q to quit"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ------------------------------------------------------------------
        # Step 1: YOLOv11m PPE inference (runs every frame — fast)
        # ------------------------------------------------------------------
        results = model.predict(
            frame,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            half=args.half,
            verbose=False,
        )

        # ------------------------------------------------------------------
        # Step 2: YOLO-World context detection (every frame if --context)
        # YOLO-World is fast (~10-20 ms) because class embeddings were
        # pre-computed at startup via set_classes().
        # ------------------------------------------------------------------
        hazard_dets = context_detector.detect(frame) if context_detector else []

        # ------------------------------------------------------------------
        # Step 3: Compliance analysis
        # Pass hazard_dets as-is when context mode is active — even an empty
        # list signals "context mode on, no hazards detected" which means no
        # PPE is currently required.  Passing None means context mode is off
        # entirely (fall back to requiring all three items).
        # ------------------------------------------------------------------
        boxes = results[0].boxes if results else None
        report = checker.analyze(
            boxes,
            model.names,
            frame.shape[:2],
            hazard_detections=hazard_dets if context_detector is not None else None,
        )
        compliance_history.append(report)

        # Sample frames for the multi-frame VLM composite
        if frame_idx % _FRAME_SAMPLE_RATE == 0:
            frame_history.append(frame.copy())
            if len(frame_history) > _FRAME_HISTORY_SIZE:
                frame_history.pop(0)

        # ------------------------------------------------------------------
        # Debug: live per-frame detection state (every 15 frames when person present)
        # ------------------------------------------------------------------
        if frame_idx % 15 == 0 and report.persons:
            for rec in report.persons:
                mask_raw   = "worn"    if rec.mask      else ("no_mask" if rec.no_mask      else "undetected")
                hat_raw    = "worn"    if rec.hardhat   else ("no_hat"  if rec.no_hardhat   else "undetected")
                vest_raw   = "worn"    if rec.vest      else ("no_vest" if rec.no_vest      else "undetected")
                print(
                    f"[debug:yolo] frame={frame_idx} P{rec.person_idx+1}: "
                    f"hardhat={hat_raw}({rec.hardhat_compliant}) "
                    f"mask={mask_raw}({rec.mask_compliant}) "
                    f"vest={vest_raw}({rec.vest_compliant}) "
                    f"fully={rec.fully_compliant}"
                )

        # Countdown while waiting for fresh frames after V was pressed
        if pending_vlm_context is not None and vlm_analyzer is not None:
            remaining = _FRESH_FRAMES_NEEDED - len(compliance_history)
            if remaining > 0 and frame_idx % 5 == 0:
                print(f"[debug:pending] Waiting for {remaining} more fresh frame(s) before VLM fires...")

        # ------------------------------------------------------------------
        # Deferred VLM trigger (manual 'V' press path)
        # Fire once we have _FRESH_FRAMES_NEEDED post-dialog frames so the
        # compliance snapshot reflects what the camera sees right now, not
        # what was on screen before/during the dialog.
        # ------------------------------------------------------------------
        if (
            pending_vlm_context is not None
            and vlm_analyzer is not None
            and len(compliance_history) >= _FRESH_FRAMES_NEEDED
        ):
            print(f"[debug:pending] {_FRESH_FRAMES_NEEDED} fresh frames collected — firing VLM now")
            _trigger_vlm_query(
                vlm_analyzer, frame, report, hazard_dets, context_detector,
                pending_vlm_context, compliance_history, frame_history,
            )
            last_vlm_trigger_time = time.time()
            pending_vlm_context = None

        # ------------------------------------------------------------------
        # Periodic re-trigger while an active context is set.
        # Fires every _VLM_RETRIGGER_INTERVAL seconds so the VLM tracks
        # compliance continuously throughout the video.
        # Skipped when: a query is already running, the deferred first-fire
        # hasn't happened yet, no persons are in frame, or VLM is still loading.
        # ------------------------------------------------------------------
        if (
            active_context is not None
            and pending_vlm_context is None
            and vlm_analyzer is not None
            and vlm_analyzer.state.status not in ("loading", "analyzing")
            and time.time() - last_vlm_trigger_time >= _VLM_RETRIGGER_INTERVAL
            and report.persons
        ):
            print(f"[detect] Re-triggering VLM with active context: '{active_context}'")
            _trigger_vlm_query(
                vlm_analyzer, frame, report, hazard_dets, context_detector,
                active_context, compliance_history, frame_history,
            )
            last_vlm_trigger_time = time.time()

        # ------------------------------------------------------------------
        # FPS measurement
        # ------------------------------------------------------------------
        now = time.perf_counter()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # ------------------------------------------------------------------
        # Step 4: Draw annotated frame
        # vlm_state is None when context mode is off; the visualizer handles
        # that case by simply not drawing any VLM-related overlay.
        # ------------------------------------------------------------------
        vlm_state = vlm_analyzer.state if vlm_analyzer else None
        annotated = viz.draw(
            frame, report, fps, source_label,
            vlm_state=vlm_state,
            hazard_detections=hazard_dets,
        )

        if writer is not None:
            writer.write(annotated)

        # ------------------------------------------------------------------
        # Step 5: Display + keyboard handling
        # ------------------------------------------------------------------
        if not args.no_display:
            display = cv2.resize(annotated, (1280, 720), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(window_title, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            # 'V' or 'v' — open operator-context dialog, then arm the deferred
            # VLM trigger.  We do NOT fire the query immediately here because the
            # dialog blocks the frame loop, so compliance_history is frozen at
            # whatever state it was in before the dialog opened (potentially
            # "mask was off").  Instead: flush the stale history now and let
            # _FRESH_FRAMES_NEEDED fresh frames accumulate before firing.
            if key in (ord("v"), ord("V")) and vlm_analyzer is not None:
                custom_context = _ask_custom_context()
                compliance_history.clear()   # discard all frames captured before/during dialog
                pending_vlm_context = custom_context
                active_context = custom_context  # arm continuous re-analysis with this context
                last_vlm_trigger_time = 0.0      # reset timer so re-trigger doesn't fire immediately after first
                print(f"[detect] Context set: '{custom_context}' — re-analysing every {_VLM_RETRIGGER_INTERVAL:.0f}s")

            # 'R' or 'r' — hard reset: clear the VLM overlay, active context,
            # compliance history, any pending deferred trigger, and re-arm the
            # auto-trigger.
            if key in (ord("r"), ord("R")) and vlm_analyzer is not None:
                vlm_analyzer.reset()
                compliance_history.clear()
                pending_vlm_context = None
                active_context = None            # stop continuous re-analysis
                vlm_auto_triggered = False
                print("[detect] Hard reset — context cleared, history flushed, auto-trigger re-armed.")

        # ------------------------------------------------------------------
        # Step 6: Auto-trigger — fire once when the model is ready and a
        # person is first detected.  This gives an initial safety assessment
        # without requiring the user to press V manually on startup.
        # ------------------------------------------------------------------
        if (
            vlm_analyzer is not None
            and not vlm_auto_triggered
            and vlm_analyzer.ready
            and report.persons
        ):
            _trigger_vlm_query(vlm_analyzer, frame, report, hazard_dets, context_detector,
                               compliance_history=compliance_history,
                               frame_history=frame_history)
            last_vlm_trigger_time = time.time()
            active_context = ""   # enable continuous re-analysis from startup
            vlm_auto_triggered = True

        # ------------------------------------------------------------------
        # Progress reporting (file sources only)
        # ------------------------------------------------------------------
        frame_idx += 1
        if total > 0 and frame_idx % 30 == 0:
            pct = frame_idx / total * 100
            print(f"\r[detect] {frame_idx}/{total} frames ({pct:.1f}%)  FPS={fps:.1f}", end="", flush=True)

    if total > 0:
        print()  # newline after progress line


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Load the PPE model (YOLOv11m fine-tuned on construction PPE dataset)
    # ------------------------------------------------------------------
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is not installed. Run: pip install -r requirements.txt")

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"[detect] Weights not found: {weights.resolve()}")

    print(f"[detect] Loading PPE model: {weights}")
    model = YOLO(str(weights))

    cap, meta = open_source(args.source)
    print(f"[detect] Source : {meta['label']}  ({meta['width']}x{meta['height']} @ {meta['fps']:.1f} fps)")
    if meta["total_frames"] > 0:
        print(f"[detect] Frames : {meta['total_frames']}")

    checker = ComplianceChecker()
    viz = Visualizer(model.names, context_mode=args.context)

    writer = None
    if args.save:
        writer = build_writer(meta, args.output)
        print(f"[detect] Saving annotated output to: {args.output}")

    # ------------------------------------------------------------------
    # Context mode: load YOLO-World + start LLaVA in background thread
    # ------------------------------------------------------------------
    context_detector = None
    vlm_analyzer = None

    if args.context:
        print("[detect] Context mode enabled — loading YOLO-World and LLaVA VLM.")
        print("[detect] LLaVA loads in the background; video starts immediately.")

        from context_detector import ContextDetector
        from vlm_analyzer import VLMAnalyzer

        # ContextDetector loads synchronously (YOLO-World is small, ~100 MB).
        context_detector = ContextDetector(device=args.device)

        # VLMAnalyzer spawns a background thread that downloads/loads LLaVA.
        # The video loop starts right away; the VLM becomes available once loaded.
        vlm_analyzer = VLMAnalyzer()

    # ------------------------------------------------------------------
    # Main inference loop
    # ------------------------------------------------------------------
    try:
        run_loop(
            model, cap, checker, viz, args, meta, writer,
            context_detector=context_detector,
            vlm_analyzer=vlm_analyzer,
        )
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[detect] Output saved: {args.output}")
        cv2.destroyAllWindows()

    print("[detect] Done.")


if __name__ == "__main__":
    main()
