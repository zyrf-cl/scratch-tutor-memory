"""Predefined Scratch concepts the memory module tracks mastery for.

Closed enum keeps queries / aggregations clean — no free-form concept strings
sneaking in via the API.
"""

from enum import Enum


class ScratchConcept(str, Enum):
    LOOP = "loop"
    CONDITIONAL = "conditional"
    EVENT = "event"
    VARIABLE = "variable"
    BROADCAST = "broadcast"
    CLONE = "clone"
    SPRITE = "sprite"
    SOUND = "sound"
    MOTION = "motion"
    SENSING = "sensing"
    OPERATOR = "operator"
    LIST = "list"


ALL_CONCEPTS: list[str] = [c.value for c in ScratchConcept]


def is_valid_concept(concept_id: str) -> bool:
    return concept_id in ALL_CONCEPTS


__all__ = ["ScratchConcept", "ALL_CONCEPTS", "is_valid_concept"]
