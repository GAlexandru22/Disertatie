"""
OpenCV-based drawing utilities for PPE compliance visualisation.

Draw order (back to front):
  1. Person bounding boxes (colored by compliance)
  2. PPE bounding boxes with association lines to their person
  3. Compliance badges [H][M][V] above each person
  4. HUD overlay (source, FPS, counts, frame status)
"""

from __future__ import annotations

import cv2
import numpy as np

from compliance import ComplianceReport, Detection, PersonRecord


# BGR color palette
_GREEN  = (0,   200,   0)
_RED    = (0,     0, 220)
_ORANGE = (0,   165, 255)
_GRAY   = (128, 128, 128)
_WHITE  = (255, 255, 255)
_BLACK  = (0,     0,   0)
_DARK   = (30,   30,  30)


class Visualizer:
    def __init__(self, class_names: dict[int, str]) -> None:
        self.class_names = class_names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        report: ComplianceReport,
        fps: float,
        source_label: str,
    ) -> np.ndarray:
        out = frame.copy()
        self._draw_person_boxes(out, report.persons)
        self._draw_ppe_boxes(out, report.persons, report.unassociated_ppe)
        self._draw_compliance_badges(out, report.persons)
        self._draw_hud(out, report, fps, source_label)
        return out

    # ------------------------------------------------------------------
    # Layer 1 — person boxes
    # ------------------------------------------------------------------

    def _draw_person_boxes(
        self, frame: np.ndarray, persons: list[PersonRecord]
    ) -> None:
        for rec in persons:
            d = rec.detection
            if rec.hardhat_compliant or rec.mask_compliant or rec.vest_compliant:
                color = _GREEN if rec.fully_compliant else _ORANGE
            else:
                color = _RED

            x1, y1, x2, y2 = map(int, d.xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"P{rec.person_idx + 1}"
            self._put_label(frame, label, x1, y1, color, font_scale=0.55, thickness=1)

    # ------------------------------------------------------------------
    # Layer 2 — PPE boxes + association lines
    # ------------------------------------------------------------------

    def _draw_ppe_boxes(
        self,
        frame: np.ndarray,
        persons: list[PersonRecord],
        unassociated: list[Detection],
    ) -> None:
        # Build map: person_idx -> person center for drawing association lines
        centers: dict[int, tuple[int, int]] = {}
        for rec in persons:
            d = rec.detection
            centers[rec.person_idx] = (int(d.cx), int(d.cy))

        for rec in persons:
            pc = centers[rec.person_idx]
            self._draw_single_ppe(frame, rec.hardhat,    _GREEN, pc, "Hardhat")
            self._draw_single_ppe(frame, rec.mask,       _GREEN, pc, "Mask")
            self._draw_single_ppe(frame, rec.vest,       _GREEN, pc, "Vest")
            self._draw_single_ppe(frame, rec.no_hardhat, _RED,   pc, "NO-Hardhat")
            self._draw_single_ppe(frame, rec.no_mask,    _RED,   pc, "NO-Mask")
            self._draw_single_ppe(frame, rec.no_vest,    _RED,   pc, "NO-Vest")

        for det in unassociated:
            x1, y1, x2, y2 = map(int, det.xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), _GRAY, 1)
            lbl = f"{det.class_name} {det.conf:.2f}"
            self._put_label(frame, lbl, x1, y1, _GRAY, font_scale=0.4, thickness=1)

    @staticmethod
    def _draw_single_ppe(
        frame: np.ndarray,
        det: Detection | None,
        color: tuple[int, int, int],
        person_center: tuple[int, int],
        label_prefix: str,
    ) -> None:
        if det is None:
            return
        x1, y1, x2, y2 = map(int, det.xyxy)
        cx, cy = int(det.cx), int(det.cy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # thin association line from PPE center to person center
        cv2.line(frame, (cx, cy), person_center, color, 1, cv2.LINE_AA)
        lbl = f"{label_prefix} {det.conf:.2f}"
        Visualizer._put_label(frame, lbl, x1, y1, color, font_scale=0.4, thickness=1)

    # ------------------------------------------------------------------
    # Layer 3 — compliance badges [H][M][V]
    # ------------------------------------------------------------------

    def _draw_compliance_badges(
        self, frame: np.ndarray, persons: list[PersonRecord]
    ) -> None:
        badge_w, badge_h = 22, 20
        gap = 3

        for rec in persons:
            x1 = int(rec.detection.xyxy[0])
            y1 = int(rec.detection.xyxy[1])

            badges = [
                ("H", rec.hardhat_compliant, rec.hardhat is not None or rec.no_hardhat is not None),
                ("M", rec.mask_compliant,    rec.mask    is not None or rec.no_mask    is not None),
                ("V", rec.vest_compliant,    rec.vest    is not None or rec.no_vest    is not None),
            ]

            total_w = len(badges) * badge_w + (len(badges) - 1) * gap
            bx = x1
            by = max(0, y1 - badge_h - 4)

            for letter, compliant, has_detection in badges:
                if has_detection:
                    bg = _GREEN if compliant else _RED
                else:
                    bg = _GRAY
                cv2.rectangle(frame, (bx, by), (bx + badge_w, by + badge_h), bg, -1)
                cv2.putText(
                    frame, letter,
                    (bx + 5, by + badge_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA,
                )
                bx += badge_w + gap

    # ------------------------------------------------------------------
    # Layer 4 — HUD overlay
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_hud(
        frame: np.ndarray,
        report: ComplianceReport,
        fps: float,
        source_label: str,
    ) -> None:
        h, w = frame.shape[:2]
        panel_w, panel_h = 230, 145
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), _DARK, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        lines = [
            (f"Source : {source_label}", _WHITE),
            (f"FPS    : {fps:>5.1f}",   _WHITE),
            (f"Persons: {len(report.persons)}", _WHITE),
            (f"Safe   : {report.compliant_count}", _GREEN),
            (f"Violat.: {report.violation_count}", _RED if report.violation_count else _WHITE),
        ]
        for i, (text, color) in enumerate(lines):
            cv2.putText(
                frame, text,
                (8, 20 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA,
            )

        # Large status banner bottom-right
        status_text = "SAFE" if report.frame_compliant else "VIOLATION"
        status_color = _GREEN if report.frame_compliant else _RED
        font_scale = 1.4
        thickness = 3
        (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        tx = w - tw - 12
        ty = h - 12
        cv2.putText(
            frame, status_text,
            (tx, ty),
            cv2.FONT_HERSHEY_DUPLEX, font_scale, status_color, thickness, cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _put_label(
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
        font_scale: float = 0.5,
        thickness: int = 1,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        label_y = y - 4 if y - th - 4 >= 0 else y + th + 4
        cv2.rectangle(frame, (x, label_y - th - 2), (x + tw + 2, label_y + 2), _DARK, -1)
        cv2.putText(frame, text, (x + 1, label_y), font, font_scale, color, thickness, cv2.LINE_AA)
