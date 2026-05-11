"""
Training entry point for the PPE detection model.

Usage (on the CUDA training machine after git clone):
  # Smoke test — verify dataset paths and GPU setup (3 epochs)
  python src/train.py --epochs 3 --batch 8

  # Full training run
  python src/train.py --epochs 100 --batch 16 --device 0

  # Resume an interrupted run
  python src/train.py --resume

  # Export best weights to ONNX after training
  python src/train.py --epochs 100 --export

After training, commit or copy:
  runs/train/ppe_yolo11/weights/best.pt
back to the inference machine.
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv11 for PPE detection")
    p.add_argument("--model",    default="yolo11m.pt",
                   help="Ultralytics model to fine-tune (auto-downloads pretrained weights)")
    p.add_argument("--data",     default="data/construction_ppe.yaml",
                   help="Path to dataset YAML (relative to repo root)")
    p.add_argument("--epochs",   type=int, default=100)
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--batch",    type=int, default=16,
                   help="Batch size per GPU; reduce to 8 if OOM")
    p.add_argument("--device",   default="0",
                   help="'0' for first GPU, '0,1' for multi-GPU, 'cpu' for CPU")
    p.add_argument("--project",  default="runs/train")
    p.add_argument("--name",     default="ppe_yolo11")
    p.add_argument("--resume",   action="store_true",
                   help="Resume training from the last checkpoint")
    p.add_argument("--export",   action="store_true",
                   help="Export best.pt to ONNX after training finishes")
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "ultralytics is not installed.\n"
            "Run: pip install -r requirements.txt"
        )

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"Dataset YAML not found: {data_path.resolve()}")

    print(f"[train] Model   : {args.model}")
    print(f"[train] Data    : {data_path.resolve()}")
    print(f"[train] Epochs  : {args.epochs}")
    print(f"[train] Batch   : {args.batch}")
    print(f"[train] Device  : {args.device}")
    print(f"[train] Project : {args.project}/{args.name}")

    model = YOLO(args.model)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        # Augmentation — mirrors the Roboflow preprocessing applied to the dataset
        fliplr=0.5,
        flipud=0.0,
        degrees=12.0,
        shear=2.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,     # helps with underrepresented classes (Mask, Vehicle)
        mixup=0.1,
        # Loss weights
        cls=0.5,
        # Training control
        patience=30,
        save=True,
        save_period=10,
        plots=True,
        val=True,
        verbose=True,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n[train] Done. Best weights saved to: {best_weights.resolve()}")

    # Print key metrics summary
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"):
            if key in metrics:
                print(f"  {key}: {metrics[key]:.4f}")

    if args.export:
        _export_onnx(best_weights)


def _export_onnx(weights_path: Path) -> None:
    from ultralytics import YOLO
    if not weights_path.exists():
        print(f"[export] Weights not found at {weights_path}, skipping ONNX export.")
        return
    print(f"\n[export] Exporting {weights_path} to ONNX …")
    model = YOLO(str(weights_path))
    model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    print("[export] ONNX export complete.")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
