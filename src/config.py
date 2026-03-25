from core.base_blender_generator import GenerationConfig
from pydantic import Field


class TaskConfig(GenerationConfig):
    """
    Task-specific configuration for Knowledge Causality.
    """
    domain:         str             = Field(default="knowledge_causality")
    image_size:     tuple[int, int] = Field(default=(800, 500))
    video_frames:   int             = Field(default=60)   # 2 seconds @ 30 fps
    render_samples: int             = Field(default=50)
