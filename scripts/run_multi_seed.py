#!/usr/bin/env python3
"""Run multi-seed experiments for statistical analysis."""

import subprocess
import argparse
import os
from pathlib import Path


def run_multi_seed(
    script: str,
    seeds: list,
    base_args: list,
    output_base: str
):
    """Run experiment with multiple seeds.

    Args:
        script: Training script (e.g., 'train_basic.sh')
        seeds: List of random seeds
        base_args: Base command-line arguments
        output_base: Base output directory
    """
    result_dirs = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running with seed {seed}")
        print(f"{'='*60}\n")

        # Create output directory
        output_dir = Path(output_base) / f"seed_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        env = os.environ.copy()
        env['SEED'] = str(seed)
        env['OUTPUT_DIR'] = str(output_dir)

        # Run training
        cmd = ['bash', script] + base_args
        subprocess.run(cmd, env=env, check=True)

        result_dirs.append(str(output_dir))

    return result_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', default='train_basic.sh',
                        help='Training script to run')
    parser.add_argument('--seeds', default='42,43,44',
                        help='Comma-separated list of random seeds')
    parser.add_argument('--output', default='results/multi_seed',
                        help='Base output directory')
    parser.add_argument('--args', nargs=argparse.REMAINDER, default=[],
                        help='Additional arguments to pass to training script')

    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    result_dirs = run_multi_seed(
        script=args.script,
        seeds=seeds,
        base_args=args.args,
        output_base=args.output
    )

    print(f"\nCompleted {len(result_dirs)} runs:")
    for d in result_dirs:
        print(f"  - {d}")


if __name__ == '__main__':
    main()
