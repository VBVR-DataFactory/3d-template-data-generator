"""Output writer — saves TaskPairs to the standard VBVR-3D folder layout."""

import json
import shutil
from pathlib import Path
from typing import List

from .schemas import TaskPair


class OutputWriter:
    """
    Writes each TaskPair to:

        {output_dir}/{domain}_task/{task_id}/
            first_frame.png
            final_frame.png        (if provided)
            first_video.mp4        (if provided)
            last_video.mp4         (if provided)
            ground_truth.mp4       (if provided)
            prompt.txt
            metadata.json          (if provided)
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_task_pair(self, task_pair: TaskPair) -> Path:
        task_dir = self.output_dir / f"{task_pair.domain}_task" / task_pair.task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # ── Images ──────────────────────────────────────────────────────────
        self._copy_or_save_image(task_pair.first_image, task_dir / "first_frame.png")
        if task_pair.final_image:
            self._copy_or_save_image(task_pair.final_image, task_dir / "final_frame.png")

        # ── Prompt ──────────────────────────────────────────────────────────
        (task_dir / "prompt.txt").write_text(task_pair.prompt)

        # ── Videos ──────────────────────────────────────────────────────────
        if task_pair.first_video:
            src = Path(task_pair.first_video)
            if src.exists():
                dst = task_dir / f"first_video{src.suffix}"
                if src.resolve() != dst.resolve():
                    shutil.copy(src, dst)

        if task_pair.last_video:
            src = Path(task_pair.last_video)
            if src.exists():
                dst = task_dir / f"last_video{src.suffix}"
                if src.resolve() != dst.resolve():
                    shutil.copy(src, dst)

        if task_pair.ground_truth_video:
            src = Path(task_pair.ground_truth_video)
            if src.exists():
                dst = task_dir / f"ground_truth{src.suffix}"
                if src.resolve() != dst.resolve():
                    shutil.copy(src, dst)

        # ── Metadata ────────────────────────────────────────────────────────
        if task_pair.metadata is not None:
            (task_dir / "metadata.json").write_text(
                json.dumps(task_pair.metadata, ensure_ascii=False, indent=2)
            )

        return task_dir

    def write_dataset(self, task_pairs: List[TaskPair]) -> Path:
        for pair in task_pairs:
            self.write_task_pair(pair)
        return self.output_dir

    @staticmethod
    def _copy_or_save_image(image: any, dest: Path):
        if isinstance(image, (str, Path)):
            src = Path(image)
            if src.resolve() != dest.resolve() and src.exists():
                shutil.copy(src, dest)
        elif image is not None:
            # PIL Image fallback
            image.convert("RGB").save(dest)
