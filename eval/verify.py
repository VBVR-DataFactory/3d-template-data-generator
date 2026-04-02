#!/usr/bin/env python3
"""
Automated evaluation script for verifying model-generated videos against ground truth.

This script is self-contained — it does not depend on the core/ or src/ modules.
All evaluation logic is included here. Customize the TaskEvaluator class for your task.

Usage:
    # Single video evaluation
    python eval/verify.py --video generated.mp4 --gt-dir data/questions/knowledge_causality_task/knowledge_causality_00000000/

    # Batch evaluation
    python eval/verify.py --videos-dir model_outputs/ --gt-dir data/questions/

    # Save results to JSON
    python eval/verify.py --video generated.mp4 --gt-dir data/questions/knowledge_causality_task/knowledge_causality_00000000/ --output results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np


# ============================================================================
#  Evaluation Utilities (self-contained, no external dependencies)
# ============================================================================

def load_video_frames(video_path: str, max_frames: int = 100) -> List[np.ndarray]:
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames and max_frames < total_frames:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
    else:
        indices = list(range(total_frames))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def load_image(path: str) -> Optional[np.ndarray]:
    """Load an image file, return None if not found."""
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    return img


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index between two images."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1 ** 2
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2 ** 2
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim_map.mean())


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    mse = float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255 ** 2 / mse))


def compute_frame_difference(f1: np.ndarray, f2: np.ndarray) -> float:
    """Mean absolute difference between two frames, normalised to [0, 1]."""
    if f1.shape != f2.shape:
        f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]))
    return float(np.mean(np.abs(f1.astype(np.float64) - f2.astype(np.float64))) / 255.0)


def normalize_frame_size(frame: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Resize frame to match target dimensions."""
    if frame.shape == target.shape:
        return frame
    return cv2.resize(frame, (target.shape[1], target.shape[0]))


def compute_optical_flow_magnitude(f1: np.ndarray, f2: np.ndarray) -> float:
    """Compute mean optical flow magnitude between two frames."""
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY) if len(f1.shape) == 3 else f1
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY) if len(f2.shape) == 3 else f2
    if g1.shape != g2.shape:
        g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]))
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.mean(mag))


# ============================================================================
#  Standard Evaluation Dimensions (matching VBVR-EvalKit)
# ============================================================================

STANDARD_WEIGHTS = {
    "first_frame_consistency": 0.15,
    "final_frame_accuracy": 0.35,
    "temporal_smoothness": 0.15,
    "visual_quality": 0.10,
    "task_specific": 0.25,
}


def evaluate_first_frame(first_frame: np.ndarray, gt_first: np.ndarray) -> float:
    """Score first frame consistency (SSIM-based)."""
    if first_frame.shape != gt_first.shape:
        gt_first = normalize_frame_size(gt_first, first_frame)
    ssim = compute_ssim(first_frame, gt_first)
    if ssim >= 0.95:
        return 1.0
    elif ssim >= 0.85:
        return 0.8 + (ssim - 0.85) / 0.10 * 0.2
    elif ssim >= 0.70:
        return 0.5 + (ssim - 0.70) / 0.15 * 0.3
    else:
        return ssim / 0.70 * 0.5


def evaluate_final_frame(final_frame: np.ndarray, gt_final: np.ndarray) -> float:
    """Score final frame accuracy (SSIM + PSNR)."""
    if final_frame.shape != gt_final.shape:
        gt_final = normalize_frame_size(gt_final, final_frame)
    ssim = compute_ssim(final_frame, gt_final)
    psnr = compute_psnr(final_frame, gt_final)
    psnr_score = max(0.0, min(1.0, (psnr - 20) / 20)) if psnr != float("inf") else 1.0
    return 0.7 * ssim + 0.3 * psnr_score


def evaluate_temporal_smoothness(frames: List[np.ndarray]) -> float:
    """Score temporal smoothness (frame-to-frame consistency)."""
    if len(frames) < 2:
        return 1.0
    diffs = [compute_frame_difference(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
    std_diff = float(np.std(diffs))
    max_diff = float(np.max(diffs))
    variance_score = 1.0 - min(1.0, std_diff / 0.1)
    jump_score = 1.0 - min(1.0, max_diff / 0.3)
    return 0.6 * variance_score + 0.4 * jump_score


def evaluate_visual_quality(frames: List[np.ndarray]) -> float:
    """Score visual quality (sharpness + noise)."""
    if not frames:
        return 0.0
    scores = []
    for frame in frames[:: max(1, len(frames) // 10)]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(1.0, laplacian_var / 1000)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_est = np.mean(np.abs(gray.astype(float) - blurred.astype(float)))
        noise_score = 1.0 - min(1.0, noise_est / 30)
        scores.append(0.6 * sharpness + 0.4 * noise_score)
    return float(np.mean(scores))


# ============================================================================
#  Task-Specific Evaluator — CUSTOMIZE THIS
# ============================================================================

class TaskEvaluator:
    """
    Task-specific evaluation logic for 3D physics tasks.

    CUSTOMIZE THIS CLASS for your task. The included example evaluates a
    causality task (heavy ball knocks tower vs. lightweight ball bouncing off).
    Replace with logic relevant to your 3D task.

    The method evaluate_task_specific() should return a float between 0 and 1.
    """

    # Sub-criteria for the causality task example.
    TASK_WEIGHTS = {
        "collision_outcome": 0.40,
        "physics_plausibility": 0.35,
        "object_integrity": 0.25,
    }

    def evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Evaluate task-specific criteria.

        Args:
            video_frames: Frames from the model-generated video.
            gt_frames: Frames from the ground truth video (may be empty).
            gt_first_frame: Ground truth first frame image.
            gt_final_frame: Ground truth final frame image (may be None for 3D tasks).
            metadata: Task metadata (e.g., ball type, tower floors).

        Returns:
            Score between 0.0 and 1.0.
        """
        if not video_frames:
            return 0.0

        scores = {}

        # ── Collision outcome (40%) ───────────────────────────────────────
        # Compare motion magnitude in GT vs generated video.
        # Heavy ball should cause large displacement; plastic ball minimal.
        scores["collision_outcome"] = self._score_collision_outcome(
            video_frames, gt_frames, metadata
        )

        # ── Physics plausibility (35%) ────────────────────────────────────
        # Check that motion follows a plausible trajectory (optical flow).
        scores["physics_plausibility"] = self._score_physics(video_frames)

        # ── Object integrity (25%) ────────────────────────────────────────
        # Objects should remain recognisable throughout the video.
        scores["object_integrity"] = self._score_object_integrity(video_frames)

        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)

    # ── Helper methods (replace these for your task) ──────────────────────

    def _score_collision_outcome(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        metadata: Optional[Dict[str, Any]],
    ) -> float:
        """Compare final-state displacement between GT and generated video."""
        if not gt_frames or len(gt_frames) < 2 or len(video_frames) < 2:
            return 0.5  # Not enough data to judge

        # Measure total motion in second half of video (post-collision)
        mid = len(video_frames) // 2
        gen_motion = sum(
            compute_optical_flow_magnitude(video_frames[i], video_frames[i + 1])
            for i in range(mid, len(video_frames) - 1)
        ) / max(1, len(video_frames) - mid - 1)

        gt_mid = len(gt_frames) // 2
        gt_motion = sum(
            compute_optical_flow_magnitude(gt_frames[i], gt_frames[i + 1])
            for i in range(gt_mid, len(gt_frames) - 1)
        ) / max(1, len(gt_frames) - gt_mid - 1)

        if gt_motion == 0:
            return 1.0 if gen_motion < 0.5 else 0.5

        ratio = gen_motion / gt_motion
        # Score based on how close the motion magnitudes are
        return max(0.0, 1.0 - abs(1.0 - ratio))

    def _score_physics(self, frames: List[np.ndarray]) -> float:
        """Check for physically plausible motion (smooth optical flow)."""
        if len(frames) < 3:
            return 0.5

        magnitudes = []
        step = max(1, len(frames) // 20)
        for i in range(0, len(frames) - 1, step):
            mag = compute_optical_flow_magnitude(frames[i], frames[i + 1])
            magnitudes.append(mag)

        if not magnitudes:
            return 0.5

        # Penalise sudden jumps in motion (non-physical teleportation)
        diffs = [abs(magnitudes[i + 1] - magnitudes[i]) for i in range(len(magnitudes) - 1)]
        max_jump = max(diffs) if diffs else 0
        mean_motion = float(np.mean(magnitudes))

        jump_score = 1.0 - min(1.0, max_jump / max(mean_motion * 3, 1.0))
        # Some motion should exist (static video = bad)
        motion_score = min(1.0, mean_motion / 2.0) if mean_motion > 0.1 else 0.3

        return 0.5 * jump_score + 0.5 * motion_score

    def _score_object_integrity(self, frames: List[np.ndarray]) -> float:
        """Check that objects remain recognisable (no melting/disappearing)."""
        if len(frames) < 2:
            return 0.5

        # Sample pairs of nearby frames — they should be similar
        step = max(1, len(frames) // 10)
        similarities = []
        for i in range(0, len(frames) - step, step):
            ssim = compute_ssim(frames[i], frames[min(i + step, len(frames) - 1)])
            similarities.append(ssim)

        if not similarities:
            return 0.5

        # Frames shouldn't change too drastically between nearby samples
        min_sim = min(similarities)
        mean_sim = float(np.mean(similarities))
        return 0.5 * min(1.0, min_sim / 0.5) + 0.5 * min(1.0, mean_sim / 0.7)


# ============================================================================
#  Main Evaluation Runner
# ============================================================================

def evaluate_single(
    video_path: str,
    gt_dir: str,
    task_evaluator: Optional[TaskEvaluator] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single generated video against ground truth.

    Args:
        video_path: Path to the model-generated video.
        gt_dir: Path to the GT sample folder (contains first_frame.png, etc.).
        task_evaluator: Optional custom evaluator instance.

    Returns:
        Dict with overall score and per-dimension breakdown.
    """
    if task_evaluator is None:
        task_evaluator = TaskEvaluator()

    # Load generated video
    video_frames = load_video_frames(video_path)
    if not video_frames:
        return {"score": 0.0, "error": "Could not load video frames", "dimensions": {}}

    # Load ground truth
    gt_first = load_image(os.path.join(gt_dir, "first_frame.png"))
    gt_final = load_image(os.path.join(gt_dir, "final_frame.png"))
    gt_video_path = os.path.join(gt_dir, "ground_truth.mp4")
    gt_frames = load_video_frames(gt_video_path) if os.path.exists(gt_video_path) else []

    # Load metadata if available
    metadata_path = os.path.join(gt_dir, "metadata.json")
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

    # Normalise frame sizes to GT
    target = gt_first if gt_first is not None else gt_final
    if target is not None and video_frames[0].shape != target.shape:
        video_frames = [normalize_frame_size(f, target) for f in video_frames]

    # Score each dimension
    dimensions: Dict[str, float] = {}

    if gt_first is not None:
        dimensions["first_frame_consistency"] = evaluate_first_frame(video_frames[0], gt_first)
    else:
        dimensions["first_frame_consistency"] = 0.5

    if gt_final is not None:
        dimensions["final_frame_accuracy"] = evaluate_final_frame(video_frames[-1], gt_final)
    else:
        # For 3D tasks that may not have a static final frame, fall back to
        # comparing the last frame of GT video vs generated video.
        if gt_frames:
            dimensions["final_frame_accuracy"] = evaluate_final_frame(
                video_frames[-1], gt_frames[-1]
            )
        else:
            dimensions["final_frame_accuracy"] = 0.0

    dimensions["temporal_smoothness"] = evaluate_temporal_smoothness(video_frames)
    dimensions["visual_quality"] = evaluate_visual_quality(video_frames)

    task_score = task_evaluator.evaluate_task_specific(
        video_frames, gt_frames, gt_first, gt_final, metadata
    )
    dimensions["task_specific"] = max(0.0, min(1.0, task_score))

    # Weighted overall
    overall = sum(
        dimensions[k] * STANDARD_WEIGHTS[k] for k in STANDARD_WEIGHTS if k in dimensions
    )
    overall = max(0.0, min(1.0, overall))

    return {
        "video_path": video_path,
        "gt_dir": gt_dir,
        "score": overall,
        "dimensions": dimensions,
    }


def find_video_gt_pairs(videos_dir: str, gt_dir: str) -> List[Tuple[str, str]]:
    """
    Discover (video, gt_folder) pairs for batch evaluation.

    Supports two layouts:
      1. Flat: videos_dir/{task_id}.mp4 matched to gt_dir/{domain}_task/{task_id}/
      2. Nested: videos_dir/{domain}_task/{task_id}/generated.mp4

    Returns list of (video_path, gt_sample_dir) tuples.
    """
    pairs = []

    # Walk gt_dir to find all sample folders (those containing first_frame.png)
    gt_samples: Dict[str, str] = {}
    for root, _dirs, files in os.walk(gt_dir):
        if "first_frame.png" in files:
            sample_id = os.path.basename(root)
            gt_samples[sample_id] = root

    # Strategy 1: flat layout — {task_id}.mp4
    for fname in sorted(os.listdir(videos_dir)):
        if fname.endswith(".mp4"):
            task_id = fname.replace(".mp4", "")
            if task_id in gt_samples:
                pairs.append((os.path.join(videos_dir, fname), gt_samples[task_id]))

    if pairs:
        return pairs

    # Strategy 2: nested layout
    for root, _dirs, files in os.walk(videos_dir):
        for fname in files:
            if fname.endswith(".mp4"):
                video_path = os.path.join(root, fname)
                parent = os.path.basename(root)
                if parent in gt_samples:
                    pairs.append((video_path, gt_samples[parent]))

    return pairs


def print_result(result: Dict[str, Any]) -> None:
    """Pretty-print a single evaluation result."""
    gt_name = os.path.basename(result.get("gt_dir", ""))
    print(f"\nTask: {gt_name}")

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return

    for dim, weight in STANDARD_WEIGHTS.items():
        score = result["dimensions"].get(dim, 0.0)
        print(f"  {dim:<30s}  {score:.4f}  (weight: {weight:.2f})")

    print(f"  {'─' * 50}")
    print(f"  {'Overall':<30s}  {result['score']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model-generated videos against ground truth."
    )

    # Single video mode
    parser.add_argument("--video", type=str, help="Path to a single generated video")
    parser.add_argument("--gt-dir", type=str, help="Path to the GT sample folder")

    # Batch mode
    parser.add_argument("--videos-dir", type=str, help="Directory of generated videos")

    # Output
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")

    args = parser.parse_args()

    evaluator = TaskEvaluator()
    results = []

    if args.video and args.gt_dir:
        # Single video evaluation
        result = evaluate_single(args.video, args.gt_dir, evaluator)
        results.append(result)
        print_result(result)

    elif args.videos_dir and args.gt_dir:
        # Batch evaluation
        pairs = find_video_gt_pairs(args.videos_dir, args.gt_dir)
        if not pairs:
            print("No video–GT pairs found.")
            sys.exit(1)

        print(f"Found {len(pairs)} video(s) to evaluate.\n")
        all_scores = []
        for video_path, gt_sample_dir in pairs:
            result = evaluate_single(video_path, gt_sample_dir, evaluator)
            results.append(result)
            print_result(result)
            all_scores.append(result["score"])

        print(f"\n{'=' * 55}")
        print(f"  Mean score: {np.mean(all_scores):.4f}  ({len(all_scores)} videos)")
        print(f"{'=' * 55}")

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python eval/verify.py --video out.mp4 --gt-dir data/questions/knowledge_causality_task/knowledge_causality_00000000/")
        print("  python eval/verify.py --videos-dir model_outputs/ --gt-dir data/questions/")
        sys.exit(1)

    # Save results
    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
