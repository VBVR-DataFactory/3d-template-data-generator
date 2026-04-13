"""
Core framework for VBVR-3D Blender generators.

DO NOT MODIFY — this is shared framework code.
Customise files in src/ to define your task.
"""

from .base_blender_generator import BaseBlenderGenerator, GenerationConfig
from .schemas      import TaskPair
from .output_writer import OutputWriter
from .metadata_builder import build_metadata, verify_metadata

__all__ = [
    "BaseBlenderGenerator",
    "GenerationConfig",
    "TaskPair",
    "OutputWriter",
    "build_metadata",
    "verify_metadata",
]
