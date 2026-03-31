# 3d-template-data-generator

A **Blender-powered** synthetic data generator template for 3D video reasoning tasks.
Built for [VBVR (Very Big Video Reasoning)](https://github.com/Video-Reason).

Every generator produces deterministic, parametrically varied training samples
for I2V (Image-to-Video) foundation models.

---

## Output Format

```
data/questions/{domain}_task/{task_id}/
├── first_frame.png        # Initial state (REQUIRED) — first frame of first_video
├── final_frame.png        # Optional goal/after-state PNG — last frame of last_video
├── first_video.mp4        # Opening segment video (optional)
├── last_video.mp4         # Closing segment video (optional)
├── ground_truth.mp4       # Full video, beginning to end (optional)
├── prompt.txt             # Natural-language task question
└── metadata.json          # All randomised parameters for this sample
```

---

## Quick Start

```bash
# 1) Install dependencies into Blender's embedded Python
BLENDER_PY="/Applications/Blender.app/Contents/Resources/5.1/python/bin/python3.13"
"$BLENDER_PY" -m pip install -r requirements.txt

# 2) Generate 10 samples (headless, no UI)
/Applications/Blender.app/Contents/MacOS/Blender -b \
    -P examples/generate_blender.py -- --num-samples 10
```

---

## Prerequisites
- `ffmpeg` must be installed and available on `PATH` (used to assemble `ground_truth.mp4`).
- The project relies on Blender's Python runtime (`bpy`), so all Python deps must be installed into Blender's embedded Python.
- On macOS headless runs, the core generator forces Cycles to `CPU` for stability.

---

## Repository Structure

```
3d-template-data-generator/
│
├── core/                          # ✅ Framework — DO NOT MODIFY
│   ├── base_blender_generator.py  # BaseBlenderGenerator + render helpers
│   ├── schemas.py                 # TaskPair dataclass
│   └── output_writer.py           # Saves output to standardised folders
│
├── src/                           # ✏️  YOUR TASK — customise these 3 files
│   ├── generator.py               # Implement generate_task_pair()
│   ├── config.py                  # Task hyperparameters (image size, fps, …)
│   └── __init__.py
│
├── examples/
│   └── generate_blender.py        # Entry point — run this with Blender
│
└── data/questions/                # Generated output appears here
```

---

## How to Write a New 3D Task Generator

You only need to touch **3 files** in `src/`:

### 1. `src/config.py` — Define your hyperparameters

```python
from core.base_blender_generator import GenerationConfig
from pydantic import Field

class TaskConfig(GenerationConfig):
    domain: str         = Field(default="my_new_task")
    image_size: tuple[int, int] = Field(default=(800, 500))
    video_frames: int   = Field(default=60)   # 2 sec @ 30 fps
    render_samples: int = Field(default=50)

    # Add task-specific params here:
    # max_objects: int = Field(default=5)
```

### 2. `src/generator.py` — Build the scene

```python
from core.base_blender_generator import BaseBlenderGenerator
from core.schemas import TaskPair

class MyTaskGenerator(BaseBlenderGenerator):

    def generate_task_pair(self, task_id: str) -> TaskPair:
        self.clear_scene()          # Always start fresh

        # ── Randomise parameters ──────────────────────────────────────────
        n_objects = random.randint(2, 5)  # placeholder: use your task-specific ranges

        # ── Build the 3D scene with bpy ───────────────────────────────────
        self._sky_world()           # (optional) add sky lighting
        # ... add meshes, materials, camera, lights ...

        # ── Render ───────────────────────────────────────────────────────
        output_dir = os.path.join(str(self.config.output_dir),
                                  f"{self.config.domain}_task", task_id)
        os.makedirs(output_dir, exist_ok=True)

        first_frame = os.path.join(output_dir, "first_frame.png")
        video       = os.path.join(output_dir, "ground_truth.mp4")

        self.render_first_frame(first_frame)      # renders frame 1
        self.render_video(video, bake_physics=True)  # renders full animation

        # ── Return TaskPair ───────────────────────────────────────────────
        return TaskPair(
            task_id=task_id,
            domain=self.config.domain,
            prompt="<your task question here>",
            first_image=first_frame,
            ground_truth_video=video if os.path.exists(video) else None,
            metadata={"n_objects": n_objects, ...}
        )
```

### 3. `src/__init__.py` — Export your new class

```python
from .config    import TaskConfig
from .generator import MyTaskGenerator
```

### 4. Update `examples/generate_blender.py`

Replace `CausalityGenerator` with `MyTaskGenerator`.

---

## BaseBlenderGenerator API Reference

| Method | Description |
|:---|:---|
| `self.clear_scene()` | Resets Blender to empty scene |
| `self.render_first_frame(path)` | Renders frame 1 as PNG |
| `self.render_video(path, bake_physics=True)` | Bakes physics + renders full MP4 |
| `self.render_video_segment(path, frame_start, frame_end)` | Renders a frame range as MP4 (for first/last video) |
| `self.bpy` | Direct access to the `bpy` Blender API |
| `self.config` | Your `TaskConfig` instance |

---

## Performance Guide

| Machine | Per-frame time (50 samples) | 60-frame video | Notes |
|:---|:---|:---|:---|
| Mac M2 (CPU) | ~1–2 s | ~90 s | Good for dev/debug |
| RunPod RTX 4090 | ~0.1–0.2 s | ~8–12 s | Production batch |

## RunPod / GPU Notes

On macOS headless runs, the core generator forces Cycles to `CPU` for stability.
On Linux/RunPod, it tries `OPTIX` first, then `CUDA`, falling back to `CPU`.
If you need to force a specific device/compute backend, edit `_configure_cycles_device()` in `core/base_blender_generator.py`.

---

## Five Core Tasks (VBVR-3D)

| # | Domain | Task | Key 3D Property Tested |
|:--|:---|:---|:---|
| 1 | Knowledge | Material-Momentum Causality | Mass / material → collision outcome |
| 2 | Spatiality | Egocentric Navigation | 3D spatial layout → 1st-person path |
| 3 | Transformation | Object Permanence & Occlusion | Tracking hidden objects in 3D |
| 4 | Perception | Mirror / Refraction Reasoning | Inverse optical physics |
| 5 | Abstraction | 3D Topological Analogy | Topological knot reasoning |
