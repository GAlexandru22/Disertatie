"""
OpenCV-based drawing utilities for PPE compliance visualisation.

Draw order (back to front):
  1. Hazard proximity zones (dashed yellow rectangles, context mode only)
  2. Person bounding boxes (colored by compliance)
  3. PPE bounding boxes with association lines to their person
  4. Compliance badges [H][M][V] above each person
  5. HUD overlay (source, FPS, counts, frame status, V-key hint)
  6. VLM result overlay panel (context mode only — right side of frame)
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from compliance import ComplianceReport, Detection, PersonRecord


# BGR color palette
_GREEN  = (0,   200,   0)
_RED    = (0,     0, 220)
_ORANGE = (0,   165, 255)
_YELLOW = (0,   200, 255)
_GRAY   = (128, 128, 128)
_WHITE  = (255, 255, 255)
_BLACK  = (0,     0,   0)
_DARK   = (30,   30,  30)


class Visualizer:
    def __init__(self, class_names: dict[int, str], context_mode: bool = False) -> None:
        self.class_names = class_names
        # context_mode controls whether the 'V' key hint and VLM overlay are drawn.
        self._context_mode = context_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        report: ComplianceReport,
        fps: float,
        source_label: str,
        vlm_state=None,            # VLMState | None — from vlm_analyzer.py
        hazard_detections=None,    # list[HazardDetection] | None — from context_detector.py
    ) -> np.ndarray:
        """Compose all visual layers onto a copy of frame and return it.

        New optional parameters (both default to None for backward compat):
          vlm_state:         Current VLMState snapshot; draws the result panel.
          hazard_detections: List of HazardDetection objects; draws danger zones.
        """
        out = frame.copy()

        # Layer 1: Hazard proximity zones (drawn first so person boxes appear on top)
        if hazard_detections:
            self._draw_hazard_zones(out, hazard_detections)

        # Layers 2-4: existing PPE compliance drawing
        self._draw_person_boxes(out, report.persons)
        self._draw_ppe_boxes(out, report.persons, report.unassociated_ppe)
        self._draw_compliance_badges(out, report.persons)

        # Layer 5: HUD (always on top of detections, behind VLM panel)
        self._draw_hud(out, report, fps, source_label)

        # Layer 6: VLM result panel (drawn last — highest z-order)
        if vlm_state is not None:
            self._draw_vlm_overlay(out, vlm_state)

        return out

    # ------------------------------------------------------------------
    # Layer 1 — hazard proximity zones (NEW, context mode only)
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_hazard_zones(
        frame: np.ndarray,
        hazard_detections: list,   # list[HazardDetection]
    ) -> None:
        """Draw dashed yellow rectangles for each detected hazard's danger zone.

        The danger zone is the expanded bounding box (zone_xyxy property on
        HazardDetection) that represents the area where PPE is required.
        A dashed outline makes it visually distinct from solid PPE/person boxes.
        """
        for hazard in hazard_detections:
            # Clamp zone coordinates to frame boundaries
            h, w = frame.shape[:2]
            zx1, zy1, zx2, zy2 = hazard.zone_xyxy
            zx1, zy1 = max(0, int(zx1)), max(0, int(zy1))
            zx2, zy2 = min(w - 1, int(zx2)), min(h - 1, int(zy2))

            _draw_dashed_rect(frame, (zx1, zy1), (zx2, zy2), _YELLOW, dash_len=10, gap_len=6)

            # Label the machine type at the top-left of the zone
            label = f"{hazard.class_name} ({hazard.confidence:.2f})"
            Visualizer._put_label(frame, label, zx1, zy1, _YELLOW, font_scale=0.45, thickness=1)

    # ------------------------------------------------------------------
    # Layer 2 — person boxes
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
    # Layer 3 — PPE boxes + association lines
    # ------------------------------------------------------------------

    def _draw_ppe_boxes(
        self,
        frame: np.ndarray,
        persons: list[PersonRecord],
        unassociated: list[Detection],
    ) -> None:
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
        cv2.line(frame, (cx, cy), person_center, color, 1, cv2.LINE_AA)
        lbl = f"{label_prefix} {det.conf:.2f}"
        Visualizer._put_label(frame, lbl, x1, y1, color, font_scale=0.4, thickness=1)

    # ------------------------------------------------------------------
    # Layer 4 — compliance badges [H][M][V]
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
    # Layer 5 — HUD overlay
    # ------------------------------------------------------------------

    def _draw_hud(
        self,
        frame: np.ndarray,
        report: ComplianceReport,
        fps: float,
        source_label: str,
    ) -> None:
        h, w = frame.shape[:2]

        # Extra row in the HUD panel when context mode is on (for the V-key hint)
        panel_h = 167 if self._context_mode else 145
        panel_w = 230
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

        # When context mode is active, remind the user about the VLM shortcut
        if self._context_mode:
            lines.append(("[V] Run VLM Analysis", _YELLOW))

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
    # Layer 6 — VLM result overlay panel (NEW, context mode only)
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_vlm_overlay(frame: np.ndarray, vlm_state) -> None:  # vlm_state: VLMState
        """Draw the LLaVA analysis result as a semi-transparent panel.

        Visual states:
          loading   — small gray badge top-right corner ("VLM: Loading...")
          analyzing — small yellow badge top-right corner ("VLM: Analyzing...")
          done      — full result panel on the right side of the frame showing:
                      * Header: VLM CONTEXT ANALYSIS
                      * Green/red compliance indicator
                      * Violation list (one line each)
                      * Reasoning summary
                      * "Analyzed Xs ago" timestamp
        """
        h, w = frame.shape[:2]
        status = vlm_state.status

        if status in ("loading", "idle"):
            # Small status badge in the top-right corner
            badge_text = "VLM: Loading..."
            _draw_status_badge(frame, badge_text, w, _GRAY)
            return

        if status == "analyzing":
            # Pulsing yellow badge while the query is running
            # Use time.time() to alternate the trailing dots for a simple animation
            dots = "." * (1 + int(time.time() * 2) % 3)
            badge_text = f"VLM: Analyzing{dots}"
            _draw_status_badge(frame, badge_text, w, _YELLOW)
            return

        # status == "done" — draw the full result panel
        result = vlm_state.result
        if result is None:
            return

        # ---------------------------------------------------------------
        # Measure content to size the panel correctly before drawing it
        # ---------------------------------------------------------------
        FONT       = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SM    = 0.45
        FONT_MED   = 0.52
        LINE_H     = 20     # pixels per text line
        PADDING    = 10     # horizontal padding inside the panel
        MAX_W      = min(380, w // 2)  # panel never wider than half the frame

        # Build the text lines we will draw
        elapsed = time.time() - vlm_state.timestamp
        header_lines = ["VLM CONTEXT ANALYSIS"]
        status_line  = "All workers compliant" if result.compliant else "VIOLATIONS DETECTED"
        violation_lines = [_truncate(v, MAX_W - 2 * PADDING, FONT, FONT_SM) for v in result.violations]
        summary_lines   = _wrap_text(result.summary, MAX_W - 2 * PADDING, FONT, FONT_SM)
        time_line    = f"Analyzed {elapsed:.0f}s ago"

        total_lines = (
            len(header_lines) + 1   # header + spacing
            + 1                     # compliance status
            + len(violation_lines)
            + len(summary_lines)
            + 1                     # timestamp
        )
        panel_h = total_lines * LINE_H + PADDING * 3
        panel_h = max(panel_h, 100)

        # Panel occupies the right side of the frame
        px1 = w - MAX_W
        py1 = 0
        px2 = w
        py2 = min(panel_h, h)

        # ---------------------------------------------------------------
        # Draw semi-transparent background
        # ---------------------------------------------------------------
        panel_color = _DARK
        overlay = frame.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), panel_color, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Coloured left accent bar: green if compliant, red/orange if not
        accent_color = _GREEN if result.compliant else _ORANGE
        cv2.rectangle(frame, (px1, py1), (px1 + 4, py2), accent_color, -1)

        # ---------------------------------------------------------------
        # Draw text lines inside the panel
        # ---------------------------------------------------------------
        cx = px1 + PADDING + 6   # x start (after accent bar)
        cy = py1 + PADDING + LINE_H

        def _put(text: str, color: tuple, scale: float = FONT_MED) -> None:
            nonlocal cy
            cv2.putText(frame, text, (cx, cy), FONT, scale, color, 1, cv2.LINE_AA)
            cy += LINE_H

        _put("VLM CONTEXT ANALYSIS", _WHITE, scale=0.52)
        cy += 4  # extra spacing after header

        compliance_color = _GREEN if result.compliant else _RED
        _put(status_line, compliance_color, scale=0.52)

        if violation_lines:
            cy += 4
            for vline in violation_lines:
                _put(f"  {vline}", _ORANGE, scale=FONT_SM)

        if summary_lines:
            cy += 4
            for sline in summary_lines:
                _put(sline, _GRAY, scale=FONT_SM)

        cy += 4
        _put(time_line, _GRAY, scale=0.42)

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


# ---------------------------------------------------------------------------
# Module-level drawing helpers (not bound to the Visualizer class)
# ---------------------------------------------------------------------------

def _draw_dashed_rect(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    dash_len: int = 10,
    gap_len: int = 6,
    thickness: int = 2,
) -> None:
    """Draw a rectangle using dashed lines.

    OpenCV has no built-in dashed-line primitive, so we approximate it by
    drawing a series of short solid segments separated by gaps.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    step = dash_len + gap_len

    # Top and bottom edges (horizontal dashes)
    for x in range(x1, x2, step):
        ex = min(x + dash_len, x2)
        cv2.line(frame, (x, y1), (ex, y1), color, thickness)
        cv2.line(frame, (x, y2), (ex, y2), color, thickness)

    # Left and right edges (vertical dashes)
    for y in range(y1, y2, step):
        ey = min(y + dash_len, y2)
        cv2.line(frame, (x1, y), (x1, ey), color, thickness)
        cv2.line(frame, (x2, y), (x2, ey), color, thickness)


def _draw_status_badge(
    frame: np.ndarray,
    text: str,
    frame_width: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a small pill-shaped status badge in the top-right corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.45, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 5
    bx2 = frame_width - 6
    bx1 = bx2 - tw - pad * 2
    by1, by2 = 6, 6 + th + pad * 2
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), _DARK, -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1)
    cv2.putText(frame, text, (bx1 + pad, by2 - pad), font, scale, color, thickness, cv2.LINE_AA)


def _truncate(text: str, max_px: int, font, scale: float) -> str:
    """Truncate text to fit within max_px width, appending '…' if cut."""
    (tw, _), _ = cv2.getTextSize(text, font, scale, 1)
    if tw <= max_px:
        return text
    while text and tw > max_px:
        text = text[:-1]
        (tw, _), _ = cv2.getTextSize(text + "…", font, scale, 1)
    return text + "…"


def _wrap_text(text: str, max_px: int, font, scale: float) -> list[str]:
    """Wrap text into lines that each fit within max_px pixels wide."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        (tw, _), _ = cv2.getTextSize(test, font, scale, 1)
        if tw <= max_px:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines if lines else [text[:60]]
