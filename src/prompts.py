"""Prompt helpers for the Knowledge Causality 3D task."""

from typing import Any, Dict, List, Optional


# Template strings — use .format(tower_floors=N) to fill in placeholders.
PROMPTS = {
    "iron": [
        (
            "There is a heavy iron ball at the top of a slope. "
            "The tower ahead has {tower_floors} wooden blocks. "
            "The ball rolls down and hits the tower. "
            "Generate the collision result."
        ),
    ],
    "plastic": [
        (
            "There is a lightweight plastic ball at the top of a slope. "
            "The tower ahead has {tower_floors} wooden blocks. "
            "The ball rolls down and hits the tower. "
            "Generate the collision result."
        ),
    ],
}


def get_prompt(
    ball_type: str = "iron",
    tower_floors: int = 5,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a contextual prompt for the causality task.

    Args:
        ball_type: "iron" or "plastic"
        tower_floors: Number of tower floors
        metadata: Optional extra metadata (reserved for future use)

    Returns:
        A natural-language prompt string.
    """
    key = ball_type if ball_type in PROMPTS else "iron"
    template = PROMPTS[key][0]
    return template.format(tower_floors=tower_floors)


def get_all_prompts(ball_type: str = "iron", tower_floors: int = 5) -> List[str]:
    """Return all prompts for a ball type, with placeholders filled."""
    templates = PROMPTS.get(ball_type, PROMPTS["iron"])
    return [t.format(tower_floors=tower_floors) for t in templates]
