"""
Person–PPE association and compliance checking.

Architecture:
  1. Parse raw YOLO detections into typed Detection objects.
  2. Associate each PPE detection with the nearest Person via containment ratio
     (intersection_area / ppe_area) — more robust than IoU for small PPE vs
     large person boxes.
  3. Apply spatial position validation (helmet near head zone, vest near torso)
     as a secondary sanity filter.
  4. Derive compliance flags per person and per frame.
  5. (Optional) If hazard detections from YOLO-World are provided, populate
     each PersonRecord.nearby_hazards with the names of machines whose danger
     zone overlaps the person's center point.  This data is consumed by
     context_rules.build_ppe_summary() and the visualizer's proximity overlay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    class_id: int
    class_name: str
    conf: float
    xyxy: tuple[float, float, float, float]  # x1, y1, x2, y2 in absolute pixels
    cx: float
    cy: float
    width: float
    height: float


@dataclass
class PersonRecord:
    detection: Detection
    person_idx: int
    hardhat: Optional[Detection] = None     # class 0 — worn
    mask: Optional[Detection] = None        # class 1 — worn
    vest: Optional[Detection] = None        # class 7 — worn
    no_hardhat: Optional[Detection] = None  # class 2 — not worn
    no_mask: Optional[Detection] = None     # class 3 — not worn
    no_vest: Optional[Detection] = None     # class 4 — not worn
    hardhat_compliant: bool = False
    mask_compliant: bool = False
    vest_compliant: bool = False
    hardhat_position_valid: bool = False
    mask_position_valid: bool = False
    vest_position_valid: bool = False
    fully_compliant: bool = False
    # Names of YOLO-World-detected machines whose danger zone overlaps this person.
    # Populated by ComplianceChecker.analyze() when hazard_detections are provided.
    # Used by context_rules.build_ppe_summary() and the visualizer proximity overlay.
    nearby_hazards: list[str] = field(default_factory=list)


@dataclass
class ComplianceReport:
    persons: list[PersonRecord] = field(default_factory=list)
    unassociated_ppe: list[Detection] = field(default_factory=list)
    frame_compliant: bool = False
    violation_count: int = 0
    compliant_count: int = 0


# ---------------------------------------------------------------------------
# Compliance checker
# ---------------------------------------------------------------------------

class ComplianceChecker:
    WORN_PPE = {0: "hardhat", 1: "mask", 7: "vest"}
    NOTWORN_PPE = {2: "hardhat", 3: "mask", 4: "vest"}
    PERSON_CLASS = 5
    IGNORED_CLASSES = {6, 8, 9}  # cone, machinery, vehicle

    HEAD_ZONE_FRACTION = 0.30   # top 30 % of person box = head
    MASK_ZONE_FRACTION = 0.35   # top 35 % = face / mask region
    TORSO_TOP_FRACTION = 0.15   # torso starts at 15 %
    TORSO_BOT_FRACTION = 0.75   # torso ends at 75 %
    MIN_OVERLAP = 0.30          # minimum containment ratio to assign PPE to person

    def analyze(
        self,
        boxes,                                    # ultralytics Boxes tensor
        class_names: dict[int, str],
        frame_shape: tuple[int, int],             # (H, W)
        hazard_detections: Optional[list] = None, # list[HazardDetection] from context_detector
    ) -> ComplianceReport:
        """Run PPE compliance analysis on a single frame.

        Args:
            boxes:             Ultralytics Boxes object from model.predict().
            class_names:       Dict mapping class_id → class name (model.names).
            frame_shape:       (height, width) of the frame in pixels.
            hazard_detections: Optional list of HazardDetection objects from
                               ContextDetector.detect().  When provided, each
                               PersonRecord.nearby_hazards is populated with the
                               names of machines whose danger zone contains the
                               person's center.  Pass None to skip (no-context mode).

        Returns:
            ComplianceReport with per-person compliance flags and summary counts.
        """
        detections = self._parse(boxes, class_names)
        persons_det = [d for d in detections if d.class_id == self.PERSON_CLASS]
        ppe_items = [
            d for d in detections
            if d.class_id not in self.IGNORED_CLASSES
            and d.class_id != self.PERSON_CLASS
        ]

        if not persons_det:
            return ComplianceReport(unassociated_ppe=ppe_items)

        records, unassociated = self._associate(persons_det, ppe_items)
        for rec in records:
            self._compute_compliance(rec)

        # Hazard proximity annotation — only runs when context mode is active.
        # We keep this logic here so the rest of the pipeline only needs to
        # read PersonRecord.nearby_hazards without knowing where it came from.
        if hazard_detections:
            self._populate_nearby_hazards(records, hazard_detections)

        compliant = sum(1 for r in records if r.fully_compliant)
        violations = len(records) - compliant

        return ComplianceReport(
            persons=records,
            unassociated_ppe=unassociated,
            frame_compliant=violations == 0 and len(records) > 0,
            violation_count=violations,
            compliant_count=compliant,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # The proximity zone multiplier must match context_rules.PROXIMITY_MULTIPLIER.
    # Defined here as a class constant so compliance.py stays self-contained
    # (no import from context_rules which would introduce a dependency cycle
    # via the TYPE_CHECKING guard in that module).
    _HAZARD_PROXIMITY_MULTIPLIER: float = 1.5

    @staticmethod
    def _populate_nearby_hazards(
        records: list[PersonRecord],
        hazard_detections: list,   # list[HazardDetection] — typed as list to avoid import
    ) -> None:
        """Fill PersonRecord.nearby_hazards for each person in proximity to a machine.

        For each (person, hazard) pair we expand the hazard's bounding box by
        _HAZARD_PROXIMITY_MULTIPLIER around its center, then check whether the
        person's center point falls inside that expanded zone.

        This is intentionally a center-point test (not a full-box overlap) to
        avoid false positives when a person's bounding box clips the far edge
        of a machine that is several metres away.

        Args:
            records:           PersonRecord list built by _associate().
            hazard_detections: List of HazardDetection objects from ContextDetector.
                               Accessed via duck typing (.xyxy, .class_name, .cx, .cy).
        """
        mult = ComplianceChecker._HAZARD_PROXIMITY_MULTIPLIER

        for hazard in hazard_detections:
            hx1, hy1, hx2, hy2 = hazard.xyxy
            hw = (hx2 - hx1) * mult
            hh = (hy2 - hy1) * mult
            hcx = (hx1 + hx2) / 2
            hcy = (hy1 + hy2) / 2
            zone = (hcx - hw / 2, hcy - hh / 2, hcx + hw / 2, hcy + hh / 2)

            for rec in records:
                pcx = rec.detection.cx
                pcy = rec.detection.cy
                if zone[0] <= pcx <= zone[2] and zone[1] <= pcy <= zone[3]:
                    # Avoid duplicate entries if the same machine fires twice
                    if hazard.class_name not in rec.nearby_hazards:
                        rec.nearby_hazards.append(hazard.class_name)

    @staticmethod
    def _parse(boxes, class_names: dict[int, str]) -> list[Detection]:
        detections: list[Detection] = []
        if boxes is None or len(boxes) == 0:
            return detections
        for box in boxes:
            cls_id = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(
                class_id=cls_id,
                class_name=class_names.get(cls_id, str(cls_id)),
                conf=float(box.conf.item()),
                xyxy=(x1, y1, x2, y2),
                cx=(x1 + x2) / 2,
                cy=(y1 + y2) / 2,
                width=x2 - x1,
                height=y2 - y1,
            ))
        return detections

    def _associate(
        self,
        persons: list[Detection],
        ppe_items: list[Detection],
    ) -> tuple[list[PersonRecord], list[Detection]]:
        records = [PersonRecord(detection=p, person_idx=i) for i, p in enumerate(persons)]
        unassociated: list[Detection] = []

        for ppe in ppe_items:
            best_rec: Optional[PersonRecord] = None
            best_ratio = self.MIN_OVERLAP

            for rec in records:
                ratio = self._containment_ratio(ppe, rec.detection)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_rec = rec

            if best_rec is None:
                unassociated.append(ppe)
                continue

            role_map = {**{k: k for k in self.WORN_PPE}, **{k: k for k in self.NOTWORN_PPE}}
            if ppe.class_id in self.WORN_PPE:
                role = self.WORN_PPE[ppe.class_id]
                self._assign_worn(best_rec, role, ppe)
            elif ppe.class_id in self.NOTWORN_PPE:
                role = self.NOTWORN_PPE[ppe.class_id]
                self._assign_notworn(best_rec, role, ppe)

        return records, unassociated

    @staticmethod
    def _containment_ratio(ppe: Detection, person: Detection) -> float:
        px1, py1, px2, py2 = person.xyxy
        ex1, ey1, ex2, ey2 = ppe.xyxy
        ix1 = max(px1, ex1)
        iy1 = max(py1, ey1)
        ix2 = min(px2, ex2)
        iy2 = min(py2, ey2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        intersection = (ix2 - ix1) * (iy2 - iy1)
        ppe_area = ppe.width * ppe.height
        if ppe_area <= 0:
            return 0.0
        return intersection / ppe_area

    @staticmethod
    def _assign_worn(rec: PersonRecord, role: str, det: Detection) -> None:
        if role == "hardhat":
            if rec.hardhat is None or det.conf > rec.hardhat.conf:
                rec.hardhat = det
        elif role == "mask":
            if rec.mask is None or det.conf > rec.mask.conf:
                rec.mask = det
        elif role == "vest":
            if rec.vest is None or det.conf > rec.vest.conf:
                rec.vest = det

    @staticmethod
    def _assign_notworn(rec: PersonRecord, role: str, det: Detection) -> None:
        if role == "hardhat":
            if rec.no_hardhat is None or det.conf > rec.no_hardhat.conf:
                rec.no_hardhat = det
        elif role == "mask":
            if rec.no_mask is None or det.conf > rec.no_mask.conf:
                rec.no_mask = det
        elif role == "vest":
            if rec.no_vest is None or det.conf > rec.no_vest.conf:
                rec.no_vest = det

    def _validate_position(
        self,
        ppe: Detection,
        person: Detection,
        role: str,
    ) -> bool:
        p_top = person.xyxy[1]
        p_h = person.height
        if p_h <= 0:
            return True  # cannot validate, give benefit of the doubt
        relative_y = (ppe.cy - p_top) / p_h
        if role == "hardhat":
            return relative_y <= self.HEAD_ZONE_FRACTION
        if role == "mask":
            return relative_y <= self.MASK_ZONE_FRACTION
        if role == "vest":
            return self.TORSO_TOP_FRACTION <= relative_y <= self.TORSO_BOT_FRACTION
        return True

    def _compute_compliance(self, rec: PersonRecord) -> None:
        person = rec.detection

        # Hardhat
        if rec.no_hardhat is not None and rec.hardhat is None:
            rec.hardhat_compliant = False
        elif rec.hardhat is not None and rec.no_hardhat is None:
            valid = self._validate_position(rec.hardhat, person, "hardhat")
            rec.hardhat_position_valid = valid
            rec.hardhat_compliant = valid
        elif rec.hardhat is not None and rec.no_hardhat is not None:
            # model ambiguity — trust higher confidence
            if rec.hardhat.conf >= rec.no_hardhat.conf:
                valid = self._validate_position(rec.hardhat, person, "hardhat")
                rec.hardhat_position_valid = valid
                rec.hardhat_compliant = valid
            else:
                rec.hardhat_compliant = False
        else:
            # neither detected — conservative: non-compliant
            rec.hardhat_compliant = False

        # Mask
        if rec.no_mask is not None and rec.mask is None:
            rec.mask_compliant = False
        elif rec.mask is not None and rec.no_mask is None:
            valid = self._validate_position(rec.mask, person, "mask")
            rec.mask_position_valid = valid
            rec.mask_compliant = valid
        elif rec.mask is not None and rec.no_mask is not None:
            if rec.mask.conf >= rec.no_mask.conf:
                valid = self._validate_position(rec.mask, person, "mask")
                rec.mask_position_valid = valid
                rec.mask_compliant = valid
            else:
                rec.mask_compliant = False
        else:
            rec.mask_compliant = False

        # Safety Vest
        if rec.no_vest is not None and rec.vest is None:
            rec.vest_compliant = False
        elif rec.vest is not None and rec.no_vest is None:
            valid = self._validate_position(rec.vest, person, "vest")
            rec.vest_position_valid = valid
            rec.vest_compliant = valid
        elif rec.vest is not None and rec.no_vest is not None:
            if rec.vest.conf >= rec.no_vest.conf:
                valid = self._validate_position(rec.vest, person, "vest")
                rec.vest_position_valid = valid
                rec.vest_compliant = valid
            else:
                rec.vest_compliant = False
        else:
            rec.vest_compliant = False

        rec.fully_compliant = (
            rec.hardhat_compliant and rec.mask_compliant and rec.vest_compliant
        )
