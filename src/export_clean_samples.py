"""
Export clean sample images from the training set for dissertation figures.

Selects images with a low number of annotation boxes (minimal visual clutter),
draws the bounding boxes with class labels, and copies them to an output folder.

Usage:
  python src/export_clean_samples.py
  python src/export_clean_samples.py --min-boxes 2 --max-boxes 5 --count 20 --out samples/clean
  python src/export_clean_samples.py --split valid    # use validation set instead
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


# Class index → display name (matches construction_ppe.yaml)
CLASS_NAMES = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "Machinery",
    9: "Vehicle",
}

# Per-class BGR colours for annotation boxes
CLASS_COLORS = {
    0: (0,   200, 0),    # Hardhat         — green
    1: (0,   200, 0),    # Mask             — green
    2: (0,   0,   220),  # NO-Hardhat       — red
    3: (0,   0,   220),  # NO-Mask          — red
    4: (0,   0,   220),  # NO-Safety Vest   — red
    5: (200, 200, 200),  # Person           — light grey
    6: (0,   165, 255),  # Safety Cone      — orange
    7: (0,   200, 0),    # Safety Vest      — green
    8: (128, 0,   128),  # Machinery        — purple
    9: (128, 0,   128),  # Vehicle          — purple
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export clean annotated sample images")
    p.add_argument("--data",       default="css-data",     help="Dataset root (contains train/valid/test)")
    p.add_argument("--split",      default="train",        choices=["train", "valid", "test"])
    p.add_argument("--min-boxes",  type=int, default=2,    help="Minimum boxes per image (skip empty images)")
    p.add_argument("--max-boxes",  type=int, default=6,    help="Maximum boxes per image (skip cluttered images)")
    p.add_argument("--count",      type=int, default=15,   help="How many images to export")
    p.add_argument("--out",        default="samples/clean", help="Output directory")
    p.add_argument("--seed",       type=int, default=42,   help="Random seed for reproducibility")
    p.add_argument("--no-labels",  action="store_true",    help="Draw boxes only, no text labels")
    p.add_argument("--thickness",  type=int, default=2,    help="Box line thickness in pixels")
    return p.parse_args()


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                  img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Convert normalised YOLO xywh to absolute pixel xyxy."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def imread_unicode(path: Path) -> np.ndarray | None:
    """cv2.imread replacement that handles Unicode paths on Windows."""
    buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def draw_annotations(img: np.ndarray, label_path: Path, thickness: int, no_labels: bool) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, w, h)
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if not no_labels:
            label = CLASS_NAMES.get(cls_id, str(cls_id))
            font_scale = max(0.4, min(w, h) / 1200)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            bg_y1 = max(0, y1 - th - baseline - 2)
            cv2.rectangle(out, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - baseline - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # Resolve dataset root relative to this script or cwd
    script_dir = Path(__file__).parent
    data_root = (script_dir.parent / args.data).resolve()
    if not data_root.exists():
        data_root = Path(args.data).resolve()
    if not data_root.exists():
        raise SystemExit(f"Dataset root not found: {args.data}\nTried: {data_root}")

    img_dir = data_root / args.split / "images"
    lbl_dir = data_root / args.split / "labels"
    if not img_dir.exists():
        raise SystemExit(f"Image directory not found: {img_dir}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect candidates in the requested box-count range
    candidates: list[tuple[Path, Path, int]] = []
    for lbl_path in lbl_dir.glob("*.txt"):
        lines = [l for l in lbl_path.read_text().splitlines() if l.strip()]
        n = len(lines)
        if args.min_boxes <= n <= args.max_boxes:
            # Find the corresponding image (jpg or png)
            for ext in (".jpg", ".jpeg", ".png"):
                img_path = img_dir / (lbl_path.stem + ext)
                if img_path.exists():
                    candidates.append((img_path, lbl_path, n))
                    break

    if not candidates:
        print(f"No images found with {args.min_boxes}–{args.max_boxes} boxes in {args.split} split.")
        return

    print(f"Found {len(candidates)} candidate images ({args.min_boxes}–{args.max_boxes} boxes).")
    random.shuffle(candidates)
    selected = candidates[: args.count]

    for i, (img_path, lbl_path, n_boxes) in enumerate(selected, 1):
        img = imread_unicode(img_path)
        if img is None:
            print(f"  [skip] Could not read {img_path.name}")
            continue

        annotated = draw_annotations(img, lbl_path, args.thickness, args.no_labels)

        out_name = f"{i:02d}_{n_boxes}boxes_{img_path.stem}.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  [{i:02d}] {n_boxes} boxes -> {out_name}")

    print(f"\nDone - {len(selected)} images saved to: {str(out_dir.resolve()).encode('ascii', errors='replace').decode()}")


if __name__ == "__main__":
    main()
