#!/usr/bin/env python3
"""Analyze multi-seed experimental results."""

import argparse
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils.statistical import (
    aggregate_seed_results,
    pairwise_t_tests,
    anova_test,
    generate_comparison_table
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', nargs='+', required=True,
                        help='List of result directories')
    parser.add_argument('--metric', default='top1',
                        help='Metric to analyze')
    parser.add_argument('--output', default='analysis',
                        help='Output directory')

    args = parser.parse_args()

    # Aggregate results
    print("Aggregating results...")
    aggregated = aggregate_seed_results(args.results, args.metric)

    if not aggregated:
        print("Error: No results found")
        return

    print(f"\n{args.metric.upper()} Results:")
    print(f"  Mean: {aggregated['mean']:.3f}")
    print(f"  Std:  {aggregated['std']:.3f}")
    print(f"  95% CI: [{aggregated['ci_lower']:.3f}, {aggregated['ci_upper']:.3f}]")
    print(f"  N seeds: {aggregated['n_seeds']}")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
