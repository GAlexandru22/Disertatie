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
    """Convert a ComplianceReport into a natural-language PPE status string.

    Uses plain English ("has hardhat, missing vest") rather than key=value
    pairs so the sentence integrates naturally into the LLaVA prompt and is
    easier for the model to reason about.

    Example output:
        Worker 1 (near crane): has mask | missing hardhat, vest
        Worker 2: fully equipped (hardhat, mask, vest)
    """
    if not report.persons:
        return "No workers detected in frame."

    parts: list[str] = []
    for rec in report.persons:
        worn    = [item for item, ok in (("hardhat", rec.hardhat_compliant),
                                         ("mask",    rec.mask_compliant),
                                         ("vest",    rec.vest_compliant)) if ok]
        missing = [item for item, ok in (("hardhat", rec.hardhat_compliant),
                                         ("mask",    rec.mask_compliant),
                                         ("vest",    rec.vest_compliant)) if not ok]
        nearby = f" (near: {', '.join(rec.nearby_hazards)})" if rec.nearby_hazards else ""
        label  = f"Worker {rec.person_idx + 1}{nearby}"

        if not missing:
            parts.append(f"{label}: fully equipped ({', '.join(worn)})")
        elif not worn:
            parts.append(f"{label}: no PPE detected")
        else:
            parts.append(f"{label}: has {', '.join(worn)} | missing {', '.join(missing)}")

    return "\n  ".join(parts)


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
