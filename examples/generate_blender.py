"""
VBVR-3D Generator entry point.

Usage:
    blender -b -P examples/generate_blender.py -- [options]

Options:
    --num-samples N    Number of samples to generate (default: 3)
    --no-video         Only render first_frame.png, skip video (~10s vs ~5min)
    --seed N           Fix random seed for reproducibility
    --samples N        Override Cycles samples per frame
    --dry-run          Print task IDs and params, do not render anything
    --output PATH      Override output directory

Examples:
    # Quick preview — images only, 5 samples
    blender -b -P examples/generate_blender.py -- --num-samples 5 --no-video

    # Full production run, reproducible
    blender -b -P examples/generate_blender.py -- --num-samples 100 --seed 42

    # See what would be generated
    blender -b -P examples/generate_blender.py -- --num-samples 10 --dry-run
"""

import sys
import os
import argparse

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import bpy
from src.config    import TaskConfig
from src.generator import CausalityGenerator


def parse_args():
    """Parse custom args from after '--' in the Blender command line."""
    argv = sys.argv
    custom = argv[argv.index("--") + 1:] if "--" in argv else []

    parser = argparse.ArgumentParser(
        description="VBVR-3D Generator — Knowledge Causality task"
    )
    parser.add_argument("--num-samples", type=int,  default=3,
                        help="Number of samples to generate")
    parser.add_argument("--no-video",    action="store_true",
                        help="Only render first_frame.png (fast mode)")
    parser.add_argument("--seed",        type=int,  default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--samples",     type=int,  default=None,
                        help="Override Cycles samples per frame")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print tasks without rendering")
    parser.add_argument("--output",      type=str,  default=None,
                        help="Override output directory")
    return parser.parse_args(custom)


def main():
    args = parse_args()

    config = TaskConfig(
        num_samples  = args.num_samples,
        domain       = "knowledge_causality",
        output_dir   = args.output or os.path.join(root_dir, "data", "questions"),
        no_video     = args.no_video,
        dry_run      = args.dry_run,
        random_seed  = args.seed,
        render_samples = args.samples or 50,
    )

    CausalityGenerator(config).generate_dataset()
    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
