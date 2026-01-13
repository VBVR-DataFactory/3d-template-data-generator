"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           YOUR TASK PROMPTS                                   ║
║                                                                               ║
║  CUSTOMIZE THIS FILE to define prompts/instructions for your task.            ║
║  Prompts are selected based on task type and returned to the model.           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import random


# ══════════════════════════════════════════════════════════════════════════════
#  DEFINE YOUR PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

PROMPTS = {
    "default": [
        "Animate the chess pieces to show white delivering checkmate in one move. The winning piece should move smoothly to its destination square, capturing if necessary, resulting in the opponent's king being in checkmate.",
        "Show white making the winning move that checkmates black. The piece should move clearly from its starting position to deliver mate, with smooth animation.",
        "Demonstrate white's checkmate in one. Move the attacking piece to its final square, showing the decisive blow that ends the game.",
    ],
    
    "back_rank": [
        "Show the rook or queen delivering a back-rank checkmate. The attacking piece should slide horizontally along the back rank to trap the enemy king.",
        "Animate a classic back-rank mate. The attacking piece moves along the eighth rank to checkmate the trapped king behind its own pawns.",
    ],
    
    "queen_mate": [
        "Show the queen delivering checkmate. The queen should move decisively to its final square, supported by the king, to trap the opponent's king.",
        "Animate the queen administering checkmate. She should glide to her destination, working with the friendly king to corner the enemy monarch.",
    ],
    
    "rook_mate": [
        "Show the rook delivering checkmate. The rook should move cleanly along its file or rank to trap the enemy king.",
        "Animate the rook administering mate. It should slide smoothly to its destination square, cutting off the king's escape.",
    ],
}


def get_prompt(task_type: str = "default") -> str:
    """
    Select a random prompt for the given task type.
    
    Args:
        task_type: Type of task (key in PROMPTS dict)
        
    Returns:
        Random prompt string from the specified type
    """
    prompts = PROMPTS.get(task_type, PROMPTS["default"])
    return random.choice(prompts)


def get_all_prompts(task_type: str = "default") -> list[str]:
    """Get all prompts for a given task type."""
    return PROMPTS.get(task_type, PROMPTS["default"])
