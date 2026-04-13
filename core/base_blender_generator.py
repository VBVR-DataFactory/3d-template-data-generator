"""
Base generator class for VBVR-3D Blender-powered generators.

This is the CORE FRAMEWORK — do not modify this file.
Implement your task logic by subclassing BaseBlenderGenerator in src/generator.py.
"""

import os
import subprocess
import shutil
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from .schemas import TaskPair


# ── Configuration ──────────────────────────────────────────────────────────────

class GenerationConfig(BaseModel):
    """All generation hyperparameters. Subclass in src/config.py."""
    num_samples:    int
    domain:         str
    difficulty:     Optional[str] = None
    output_dir:     Path = Path("data/questions")
    image_size:     tuple[int, int] = (800, 500)
    video_fps:      int  = 30
    video_frames:   int  = 60          # 60 frames @ 30fps = 2 seconds
    render_samples: int  = 50
    random_seed:    Optional[int] = None
    no_video:       bool = False        # skip video rendering (fast preview mode)
    dry_run:        bool = False        # print params only, do not render


# ── Base Generator ──────────────────────────────────────────────────────────────

class BaseBlenderGenerator(ABC):
    """
    Abstract base class for 3D Blender task generators.

    Subclass this and implement generate_task_pair() in src/generator.py.
    The framework handles dataset loops, file output, video assembly,
    timing, skip-if-exists, and dry-run.

    GPU note:
      Mac (Apple Silicon): device='CPU' — Metal stalls without a display.
      Linux / RunPod:      device='GPU', compute_device_type='CUDA' or 'OPTIX'.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config

        if config.random_seed is not None:
            import random
            random.seed(config.random_seed)

        import bpy
        self.bpy = bpy
        self._setup_blender_env()

    def _configure_cycles_device(self, scene):
        """
        macOS headless: CPU only (Metal without display is unreliable).
        Linux / Windows cloud GPUs: prefer OPTIX, then CUDA, else CPU.
        """
        if sys.platform == "darwin":
            scene.cycles.device = "CPU"
            return
        try:
            prefs = self.bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "OPTIX"
            if hasattr(prefs, "refresh_devices"):
                prefs.refresh_devices()
            elif hasattr(prefs, "get_devices"):
                prefs.get_devices()
            for d in prefs.devices:
                d.use = d.type != "CPU"
            if not any(d.use for d in prefs.devices):
                scene.cycles.device = "CPU"
                return
            scene.cycles.device = "GPU"
        except Exception as exc:
            print(f"⚠️  Cycles GPU init failed ({exc}); using CPU.")
            scene.cycles.device = "CPU"

    def _setup_blender_env(self):
        """
        Configure default Blender / Cycles rendering settings.
        Override in your subclass if you need different quality settings.
        """
        scene = self.bpy.context.scene
        scene.render.engine             = 'CYCLES'
        self._configure_cycles_device(scene)
        scene.cycles.samples            = self.config.render_samples
        scene.render.resolution_x       = self.config.image_size[0]
        scene.render.resolution_y       = self.config.image_size[1]
        scene.render.resolution_percentage = 100
        scene.render.fps                = self.config.video_fps
        scene.frame_start               = 1
        scene.frame_end                 = self.config.video_frames

    def clear_scene(self):
        """Reset Blender to an empty scene before building the next sample."""
        self.bpy.ops.wm.read_factory_settings(use_empty=True)
        self._setup_blender_env()

    def render_first_frame(self, output_path: str) -> str:
        """Render a single still image at frame 1."""
        self.bpy.context.scene.frame_set(1)
        self.bpy.context.scene.render.image_settings.file_format = 'PNG'
        self.bpy.context.scene.render.filepath = output_path
        self.bpy.ops.render.render(write_still=True)
        return output_path

    def render_video_segment(
        self,
        output_mp4: str,
        frame_start: int,
        frame_end: int,
        *,
        bake_physics: bool = False,
    ) -> Optional[str]:
        """
        Render a subset of frames as MP4 (PNG sequence → ffmpeg).

        Use this to produce first_video.mp4 or last_video.mp4 segments.
        Call *after* baking physics (or pass bake_physics=True on the first
        call) so that the simulation cache is available.

        Args:
            output_mp4:   Destination path for the video file.
            frame_start:  First frame to render (inclusive, 1-based).
            frame_end:    Last frame to render (inclusive, 1-based).
            bake_physics: If True, bake rigid-body physics before rendering.

        Returns:
            Path to MP4, or None if --no-video or ffmpeg not found.
        """
        if self.config.no_video:
            return None
        if shutil.which("ffmpeg") is None:
            print("⚠️  ffmpeg not found — skipping video segment.")
            return None

        if bake_physics:
            self.bpy.ops.ptcache.bake_all()

        frame_dir = tempfile.mkdtemp(prefix="vbvr3d_seg_")
        try:
            scene = self.bpy.context.scene
            scene.render.image_settings.file_format = 'PNG'
            for f in range(frame_start, frame_end + 1):
                scene.frame_set(f)
                scene.render.filepath = os.path.join(frame_dir, f"{f:04d}.png")
                self.bpy.ops.render.render(write_still=True)

            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(self.config.video_fps),
                "-start_number", str(frame_start),
                "-i",         os.path.join(frame_dir, "%04d.png"),
                "-c:v",       "libx264",
                "-pix_fmt",   "yuv420p",
                "-crf",       "20",
                output_mp4,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ ffmpeg error: {result.stderr[:300]}")
                return None
            return output_mp4
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

    def render_video(self, output_mp4: str, *, bake_physics: bool = True) -> Optional[str]:
        """
        Render a physics-baked animation as MP4 (PNG sequence → ffmpeg).
        Returns path to MP4, or None if --no-video or ffmpeg not found.
        """
        if self.config.no_video:
            return None
        if shutil.which("ffmpeg") is None:
            print("⚠️  ffmpeg not found — skipping video.")
            return None

        if bake_physics:
            self.bpy.ops.ptcache.bake_all()

        frame_dir = tempfile.mkdtemp(prefix="vbvr3d_frames_")
        try:
            scene = self.bpy.context.scene
            scene.render.image_settings.file_format = 'PNG'
            for f in range(1, self.config.video_frames + 1):
                scene.frame_set(f)
                scene.render.filepath = os.path.join(frame_dir, f"{f:04d}.png")
                self.bpy.ops.render.render(write_still=True)

            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(self.config.video_fps),
                "-i",         os.path.join(frame_dir, "%04d.png"),
                "-c:v",       "libx264",
                "-pix_fmt",   "yuv420p",
                "-crf",       "20",
                output_mp4,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ ffmpeg error: {result.stderr[:300]}")
                return None
            return output_mp4
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

    # ── Metadata & deduplication ────────────────────────────────────────────────

    def _task_signature(self, task_data: dict) -> tuple:
        """
        Generic _task_signature method for generators without custom implementation.
        Creates a signature from task_data by quantizing floats and serializing values.
        """
        def q(v: float, step: int = 5) -> int:
            return int(round(v / step) * step)

        def serialize_value(v):
            if isinstance(v, (int, str, bool, type(None))):
                return v
            if isinstance(v, float):
                return q(v, 5)
            if isinstance(v, tuple):
                return tuple(serialize_value(item) for item in v)
            if isinstance(v, list):
                return tuple(sorted(serialize_value(item) for item in v))
            if isinstance(v, dict):
                return tuple((dk, serialize_value(dv)) for dk, dv in sorted(v.items()))
            return str(v)

        skip_keys = {'temp_path', 'temp_dir', 'temp_file', 'video_temp_path',
                     'image_temp_path', '_cache', '_internal', '_temp', 'seed', 'random_seed'}

        items = []
        for key, value in sorted(task_data.items()):
            if any(skip in key.lower() for skip in skip_keys):
                continue
            items.append((key, serialize_value(value)))

        return tuple(items)

    def _build_metadata(self, task_id: str, task_data: dict) -> dict:
        """
        Build standardized metadata for a task.

        Args:
            task_id: Task ID
            task_data: Parameters dict from _generate_task_data()

        Returns:
            Standardized metadata dict
        """
        from .metadata_builder import build_metadata

        return build_metadata(
            task_id=task_id,
            generator_name=self.config.domain,
            parameters=task_data,
            seed=self.config.random_seed,
        )

    # ── Dataset loop ───────────────────────────────────────────────────────────

    @abstractmethod
    def generate_task_pair(self, task_id: str) -> TaskPair:
        """
        Generate a single task sample.  **Implement this in src/generator.py.**

        Contract:
          1. Call self.clear_scene()
          2. Build 3D scene with randomised parameters
          3. Call self.render_first_frame(path)
          4. Optionally call self.render_video(path)
          5. Return a TaskPair
        """
        pass

    def _task_dir(self, task_id: str) -> Path:
        return (
            Path(self.config.output_dir)
            / f"{self.config.domain}_task"
            / task_id
        )

    def _already_done(self, task_id: str) -> bool:
        """Return True if this sample was already generated (skip-if-exists)."""
        return (self._task_dir(task_id) / "first_frame.png").exists()

    def generate_dataset(self) -> List[TaskPair]:
        """
        Run the full generation loop.
        Called by examples/generate.py.

        Features:
          - Skip already-generated samples (checkpoint/resume)
          - Per-sample timing
          - Dry-run mode (print params, no render)
          - --no-video fast preview mode
          - Total elapsed time summary

        Returns:
          List of TaskPair objects. The caller is responsible for writing
          them to disk via OutputWriter.
        """
        pairs  = []
        skipped = 0
        t_total_start = time.time()

        mode_flags = []
        if self.config.no_video: mode_flags.append("no-video")
        if self.config.dry_run:  mode_flags.append("dry-run")
        mode_str = f"  [{', '.join(mode_flags)}]" if mode_flags else ""
        print(f"\n{'─'*55}")
        print(f"  VBVR-3D Generator — {self.config.domain}")
        print(f"  Samples: {self.config.num_samples}{mode_str}")
        print(f"  Output : {self.config.output_dir}")
        if self.config.random_seed is not None:
            print(f"  Seed   : {self.config.random_seed}")
        print(f"{'─'*55}\n")

        for i in range(self.config.num_samples):
            task_id = f"{self.config.domain}_{i:08d}"

            # ── Skip if already done ──────────────────────────────────────
            if self._already_done(task_id):
                print(f"  ⏩ Skipping {task_id}  (already exists)")
                skipped += 1
                continue

            # ── Dry run ───────────────────────────────────────────────────
            if self.config.dry_run:
                print(f"  [dry-run] Would generate: {task_id}")
                continue

            # ── Generate ──────────────────────────────────────────────────
            print(f"  ⏳ Generating {task_id} ({i+1}/{self.config.num_samples})…")
            t_start = time.time()

            pair = self.generate_task_pair(task_id)
            pairs.append(pair)

            elapsed = time.time() - t_start
            video_info = "  🎬 +video" if pair.ground_truth_video else ""
            print(f"  ✅ {task_id}  ({elapsed:.1f}s){video_info}")

        # ── Summary ───────────────────────────────────────────────────────
        total = time.time() - t_total_start
        generated = len(pairs)
        print(f"\n{'─'*55}")
        print(f"  Done.  Generated: {generated}  |  Skipped: {skipped}  |  Total: {total:.1f}s")
        if generated > 0:
            print(f"  Avg per sample: {total/generated:.1f}s")
        print(f"{'─'*55}\n")

        return pairs
