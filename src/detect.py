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


def _trigger_vlm_query(vlm_analyzer, frame, report, hazard_dets, context_detector) -> None:
    """Helper: build prompt context strings and fire a VLM query.

    Pulled out as a named function to avoid duplicating the same 4 lines for
    the key-press trigger and the auto-trigger on first person detected.

    Args:
        vlm_analyzer:     VLMAnalyzer instance.
        frame:            Current BGR video frame.
        report:           ComplianceReport for this frame (contains per-person PPE flags).
        hazard_dets:      List[HazardDetection] from ContextDetector.detect().
        context_detector: ContextDetector instance (for get_proximity_summary).
    """
    # Import here (not at module top) so that context_rules is only loaded when
    # --context is active; it also avoids importing context modules in no-context mode.
    from context_rules import build_ppe_summary

    # Build the two natural-language strings that form the VLM prompt context.
    proximity_summary = context_detector.get_proximity_summary(
        hazard_dets,
        [rec.detection.xyxy for rec in report.persons],
    )
    ppe_summary = build_ppe_summary(report)

    print(f"[detect] VLM query triggered — context: {proximity_summary}")
    vlm_analyzer.query_async(frame, proximity_summary, ppe_summary)


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

    # Window title updates when context mode is on to advertise the V shortcut.
    window_title = (
        "PPE Compliance Detection — Q: quit | V: VLM analysis"
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
        # Passing hazard_dets populates PersonRecord.nearby_hazards so the
        # visualizer can draw proximity zones and build_ppe_summary can include
        # hazard context in the LLaVA prompt.
        # ------------------------------------------------------------------
        boxes = results[0].boxes if results else None
        report = checker.analyze(
            boxes,
            model.names,
            frame.shape[:2],
            hazard_detections=hazard_dets if hazard_dets else None,
        )

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
            cv2.imshow(window_title, annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            # 'V' or 'v' — trigger a fresh VLM analysis on the current frame.
            # The query runs in a background thread so the video is not paused.
            if key in (ord("v"), ord("V")) and vlm_analyzer is not None:
                _trigger_vlm_query(vlm_analyzer, frame, report, hazard_dets, context_detector)

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
            _trigger_vlm_query(vlm_analyzer, frame, report, hazard_dets, context_detector)
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
        context_detector = ContextDetector()

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
