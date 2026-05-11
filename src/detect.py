"""
PPE compliance detection — live webcam and video file modes.

Usage:
  # Live webcam
  python src/detect.py --weights runs/train/ppe_yolo11/weights/best.pt --source 0

  # Video file (display only)
  python src/detect.py --weights best.pt --source Date_pentru_testare/site.mp4

  # Video file (save annotated output)
  python src/detect.py --weights best.pt --source site.mp4 --save --output result.mp4

  # Headless (server / CI)
  python src/detect.py --weights best.pt --source site.mp4 --save --no-display

Press 'q' to quit the display window.
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


def run_loop(
    model,
    cap: cv2.VideoCapture,
    checker: ComplianceChecker,
    viz: Visualizer,
    args: argparse.Namespace,
    meta: dict,
    writer: cv2.VideoWriter | None,
) -> None:
    source_label = meta["label"]
    total = meta["total_frames"]
    frame_idx = 0
    prev_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model.predict(
            frame,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            half=args.half,
            verbose=False,
        )

        # Compliance analysis
        boxes = results[0].boxes if results else None
        report = checker.analyze(boxes, model.names, frame.shape[:2])

        # FPS
        now = time.perf_counter()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # Draw
        annotated = viz.draw(frame, report, fps, source_label)

        if writer is not None:
            writer.write(annotated)

        if not args.no_display:
            cv2.imshow("PPE Compliance Detection — press Q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if total > 0 and frame_idx % 30 == 0:
            pct = frame_idx / total * 100
            print(f"\r[detect] {frame_idx}/{total} frames ({pct:.1f}%)  FPS={fps:.1f}", end="", flush=True)

    if total > 0:
        print()  # newline after progress line


def main() -> None:
    args = parse_args()

    # Load model
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is not installed. Run: pip install -r requirements.txt")

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"[detect] Weights not found: {weights.resolve()}")

    print(f"[detect] Loading weights: {weights}")
    model = YOLO(str(weights))

    cap, meta = open_source(args.source)
    print(f"[detect] Source : {meta['label']}  ({meta['width']}x{meta['height']} @ {meta['fps']:.1f} fps)")
    if meta["total_frames"] > 0:
        print(f"[detect] Frames : {meta['total_frames']}")

    checker = ComplianceChecker()
    viz = Visualizer(model.names)

    writer = None
    if args.save:
        writer = build_writer(meta, args.output)
        print(f"[detect] Saving annotated output to: {args.output}")

    try:
        run_loop(model, cap, checker, viz, args, meta, writer)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[detect] Output saved: {args.output}")
        cv2.destroyAllWindows()

    print("[detect] Done.")


if __name__ == "__main__":
    main()
