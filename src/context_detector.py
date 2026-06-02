"""
YOLO-World open-vocabulary machinery detector.

What is YOLO-World?
-------------------
YOLO-World (Tencent AI Lab, CVPR 2024) is an open-vocabulary real-time
object detector.  Unlike a standard YOLO model that can only detect the fixed
set of classes it was trained on, YOLO-World accepts *text prompts* as class
definitions at inference time.  You describe what you want to find in plain
language, and the model uses a pre-trained vision-language backbone (CLIP-like)
to locate those objects in the image — all without additional training.

How we use it here:
  1. We pass a list of construction-site machinery names as text prompts.
  2. YOLO-World finds those machines in every video frame.
  3. We calculate a "danger zone" around each machine and check whether any
     person (from the YOLOv11m detector) is inside that zone.
  4. The proximity result enriches the LLaVA VLM prompt in vlm_analyzer.py.

Model choice — yolov8s-worldv2:
  The "s" (small) variant runs fast enough alongside YOLOv11m at video frame
  rates.  The "v2" suffix means it supports ONNX/TensorRT export and has
  slightly better accuracy than the original V1 release.
  Weights download automatically from Ultralytics on first run (~100 MB).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from context_rules import PROXIMITY_MULTIPLIER

# ---------------------------------------------------------------------------
# Hazard class names — these are the text prompts given to YOLO-World.
# Add or remove entries here without any retraining.
# ---------------------------------------------------------------------------

HAZARD_CLASSES: list[str] = [
    "bulldozer",
    "excavator",
    "crane",
    "forklift",
    "dump truck",
    "concrete mixer",
    "road roller",
    "heavy machinery",
    "construction vehicle",
]


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class HazardDetection:
    """A single piece of detected machinery returned by YOLO-World."""

    class_name: str
    xyxy: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    confidence: float

    @property
    def cx(self) -> float:
        """Horizontal center of the bounding box."""
        return (self.xyxy[0] + self.xyxy[2]) / 2

    @property
    def cy(self) -> float:
        """Vertical center of the bounding box."""
        return (self.xyxy[1] + self.xyxy[3]) / 2

    @property
    def zone_xyxy(self) -> tuple[float, float, float, float]:
        """Expanded danger-zone bounding box (PROXIMITY_MULTIPLIER × size).

        The zone is centered on the machine and extends beyond its edges,
        so workers standing near (but not inside) the machine are caught.
        """
        x1, y1, x2, y2 = self.xyxy
        w = (x2 - x1) * PROXIMITY_MULTIPLIER
        h = (y2 - y1) * PROXIMITY_MULTIPLIER
        return (
            self.cx - w / 2,
            self.cy - h / 2,
            self.cx + w / 2,
            self.cy + h / 2,
        )


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class ContextDetector:
    """Wraps YOLO-World to provide open-vocabulary machinery detection.

    Usage:
        detector = ContextDetector()     # downloads yolov8s-worldv2.pt on first run
        hazards  = detector.detect(frame)
        summary  = detector.get_proximity_summary(hazards, person_boxes)
    """

    # The small YOLO-World V2 model — best balance of speed and accuracy for
    # per-frame use while YOLOv11m is also running in the same loop.
    _WEIGHTS = "yolov8s-worldv2.pt"

    def __init__(self, conf_threshold: float = 0.20) -> None:
        """Load YOLO-World and register the text prompt classes.

        Args:
            conf_threshold: Minimum detection confidence to report.
                            Set lower than the default (0.25) because
                            construction machinery is often partially occluded.
        """
        # Deferred import: ultralytics is large and only needed in context mode.
        from ultralytics import YOLOWorld

        print(f"[context_detector] Loading {self._WEIGHTS}…")
        self._model = YOLOWorld(self._WEIGHTS)

        # This is the defining YOLO-World API call.
        # set_classes() converts the text list into language embeddings that
        # are baked into the model's neck as re-parameterized weights.
        # After this call, the model behaves like a conventional detector but
        # only for the classes we specified.
        self._model.set_classes(HAZARD_CLASSES)

        self._conf = conf_threshold
        print(
            f"[context_detector] Ready — watching {len(HAZARD_CLASSES)} hazard classes: "
            f"{', '.join(HAZARD_CLASSES)}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[HazardDetection]:
        """Run YOLO-World on a BGR frame and return machinery detections.

        This is called every frame inside detect.py's run loop.  It is fast
        (< 20 ms on a mid-range GPU) because YOLO-World's text embeddings were
        already computed at init time via set_classes().

        Args:
            frame: BGR image (H, W, 3) from cv2.VideoCapture.

        Returns:
            List of HazardDetection objects, one per detected machine.
            Empty list if no machinery is visible.
        """
        results = self._model.predict(frame, conf=self._conf, verbose=False)
        detections: list[HazardDetection] = []

        if not results or results[0].boxes is None:
            return detections

        # results[0].names is a dict like {0: "bulldozer", 1: "excavator", ...}
        # built from HAZARD_CLASSES in the order we passed them to set_classes().
        names: dict[int, str] = results[0].names

        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            class_name = names.get(cls_id, HAZARD_CLASSES[cls_id] if cls_id < len(HAZARD_CLASSES) else "unknown")
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(HazardDetection(
                class_name=class_name,
                xyxy=(x1, y1, x2, y2),
                confidence=float(box.conf.item()),
            ))

        return detections

    def get_proximity_summary(
        self,
        hazard_detections: list[HazardDetection],
        person_boxes: list[tuple[float, float, float, float]],
    ) -> str:
        """Build a natural-language proximity description for the LLaVA prompt.

        Example outputs:
            "1 person near bulldozer; 2 persons near crane"
            "machinery present (forklift) but no workers in proximity zone"
            "no heavy machinery detected"

        Args:
            hazard_detections: Output of detect().
            person_boxes:      List of (x1,y1,x2,y2) tuples, one per person
                               detected by YOLOv11m.

        Returns:
            A one-line string suitable for embedding in a LLaVA text prompt.
        """
        if not hazard_detections:
            return "no heavy machinery detected"

        summaries: list[str] = []
        for hazard in hazard_detections:
            nearby_count = sum(
                1 for pb in person_boxes if self._person_in_zone(pb, hazard)
            )
            if nearby_count > 0:
                noun = "person" if nearby_count == 1 else "persons"
                summaries.append(f"{nearby_count} {noun} near {hazard.class_name}")

        if not summaries:
            # Machinery is in the frame but no worker is inside the danger zone
            names_seen = ", ".join(sorted({h.class_name for h in hazard_detections}))
            return f"machinery present ({names_seen}) but no workers in proximity zone"

        return "; ".join(summaries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _person_in_zone(
        person_xyxy: tuple[float, float, float, float],
        hazard: HazardDetection,
    ) -> bool:
        """Return True if the person's center falls inside the hazard's danger zone.

        We test the person's *center point* (not the full box) against the
        expanded hazard zone.  Center-point testing avoids false positives when
        a person's bounding box clips the edge of a machine that is far away.
        """
        zx1, zy1, zx2, zy2 = hazard.zone_xyxy
        px1, py1, px2, py2 = person_xyxy
        # Person center
        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2
        return zx1 <= pcx <= zx2 and zy1 <= pcy <= zy2
