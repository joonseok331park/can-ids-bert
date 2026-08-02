"""Canonical class definitions for the public four-class prototype."""

from __future__ import annotations

from typing import Final


CLASS_NAMES: Final[tuple[str, ...]] = (
    "Benign",
    "DoS",
    "Fuzzy",
    "Malfunction",
)
CLASS_LABELS: Final[tuple[int, ...]] = tuple(range(len(CLASS_NAMES)))
CLASS_TO_LABEL: Final[dict[str, int]] = {
    name: label for label, name in enumerate(CLASS_NAMES)
}
NUM_CLASSES: Final[int] = len(CLASS_NAMES)
