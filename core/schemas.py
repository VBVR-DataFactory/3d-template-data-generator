"""Pydantic schemas for VBVR-3D task data."""

from typing import Optional, Any, Dict
from pydantic import BaseModel


class TaskPair(BaseModel):
    """
    One complete training sample.

    Fields:
      task_id             — unique identifier, e.g. "knowledge_causality_00000042"
      domain              — task domain, e.g. "knowledge_causality"
      prompt              — the natural-language question given to the video model
      first_image         — path to the rendered first-frame PNG
      final_image         — (optional) path to a goal-state PNG
      ground_truth_video  — (optional) path to the physics-rendered MP4
      metadata            — dict of randomised parameters for this sample
    """
    task_id:            str
    domain:             str
    prompt:             str
    first_image:        Any                   # str path or PIL Image
    final_image:        Optional[Any] = None
    first_video:        Optional[str] = None  # path to first segment .mp4
    last_video:         Optional[str] = None  # path to last segment .mp4
    ground_truth_video: Optional[str] = None  # path to full .mp4
    metadata:           Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True
