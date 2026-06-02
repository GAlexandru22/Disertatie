"""
PPE requirement rules per detected hazard type, and helpers for building
human-readable strings that feed into the LLaVA VLM prompt.

This module is intentionally kept free of heavy dependencies — it only holds
data and pure-Python logic so it can be imported cheaply anywhere in the
pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# TYPE_CHECKING guard prevents a circular import at runtime: compliance.py
# imports nothing from here, but we need ComplianceReport only for the
# type annotation in build_ppe_summary().
if TYPE_CHECKING:
    from compliance import ComplianceReport


# ---------------------------------------------------------------------------
# Hazard → required PPE mapping
# ---------------------------------------------------------------------------

# Keys must match the class names passed to YOLO-World in context_detector.py.
# Values are frozensets of PPE role strings: "hardhat", "mask", "vest".
# These rules inform the LLaVA prompt so the VLM knows what to look for.
CONTEXT_PPE_RULES: dict[str, frozenset[str]] = {
    "bulldozer":            frozenset({"hardhat", "vest"}),
    "excavator":            frozenset({"hardhat", "vest"}),
    "crane":                frozenset({"hardhat"}),
    "forklift":             frozenset({"vest"}),
    "dump truck":           frozenset({"hardhat", "vest"}),
    "concrete mixer":       frozenset({"hardhat", "vest"}),
    "road roller":          frozenset({"hardhat", "vest"}),
    "heavy machinery":      frozenset({"hardhat", "vest"}),
    "construction vehicle": frozenset({"hardhat", "vest"}),
}

# Multiplier applied to a hazard bounding box to define its "danger zone".
# A value of 1.5 means the zone extends 25 % beyond each edge of the machine.
# This catches workers who are near (but not touching) the equipment.
PROXIMITY_MULTIPLIER: float = 1.5


# ---------------------------------------------------------------------------
# Prompt-building helpers
# ---------------------------------------------------------------------------

def build_ppe_summary(report: ComplianceReport) -> str:
    """Convert a ComplianceReport into a one-line PPE status string.

    This string is inserted into the LLaVA prompt so the VLM receives the
    output of our fast YOLOv11m detector as additional context, reducing
    the burden on the VLM for low-level PPE detection.

    Example output:
        "Person 1: hardhat=NO, mask=YES, vest=NO | Person 2: hardhat=YES, mask=YES, vest=YES"

    If no persons were detected, returns a sentinel string so the VLM still
    receives a meaningful (empty-scene) prompt.
    """
    if not report.persons:
        return "No workers detected in frame."

    parts: list[str] = []
    for rec in report.persons:
        h = "YES" if rec.hardhat_compliant else "NO"
        m = "YES" if rec.mask_compliant    else "NO"
        v = "YES" if rec.vest_compliant    else "NO"
        nearby = f" [near: {', '.join(rec.nearby_hazards)}]" if rec.nearby_hazards else ""
        parts.append(f"Person {rec.person_idx + 1}: hardhat={h}, mask={m}, vest={v}{nearby}")

    return " | ".join(parts)


def required_ppe_for_hazards(hazard_names: list[str]) -> frozenset[str]:
    """Return the union of all PPE required by a list of hazard class names.

    Used when building a concise summary of what the VLM should check for.

    Example:
        required_ppe_for_hazards(["bulldozer", "crane"])
        → frozenset({"hardhat", "vest"})   # union of both rule sets
    """
    required: set[str] = set()
    for name in hazard_names:
        required |= CONTEXT_PPE_RULES.get(name, frozenset())
    return frozenset(required)
