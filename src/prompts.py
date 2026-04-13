"""Prompt helpers for the Knowledge Causality 3D task."""

from typing import Any, Dict, List, Optional


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
    ball_desc = "heavy iron" if ball_type == "iron" else "lightweight plastic"

    return (
        f"There is a {ball_desc} ball at the top of a slope. "
        f"The tower ahead has {tower_floors} wooden blocks. "
        "The ball rolls down and hits the tower. "
        "Generate the collision result."
    )


def get_all_prompts(ball_type: str = "iron") -> List[str]:
    """Return all static fallback prompts for a ball type."""
    return PROMPTS.get(ball_type, PROMPTS["iron"])
