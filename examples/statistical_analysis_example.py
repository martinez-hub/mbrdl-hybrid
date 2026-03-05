#!/usr/bin/env python3
"""
Example: Statistical Analysis for Algorithm Comparison

This script demonstrates how to use the statistical testing tools to compare
different training algorithms across multiple random seeds.

Usage:
    python examples/statistical_analysis_example.py
"""

import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils.statistical import (
    aggregate_seed_results,
    welch_test,
    paired_t_test,
    anova_test,
    generate_comparison_table
)


def example_1_welch_test():
    """Example 1: Pairwise Welch's t-test for algorithm comparison."""
    print("="*70)
    print("Example 1: Welch's T-Test (Pairwise Comparisons)")
    print("="*70)

    # Simulated results from 5 random seeds for each algorithm
    # These represent accuracy on CURE-TSR Snow (Severity 5)
    algorithm_results = {
        'erm':  [0.276, 0.274, 0.278, 0.275, 0.277],
        'mda':  [0.562, 0.564, 0.560, 0.563, 0.561],
        'mrt':  [0.585, 0.583, 0.587, 0.584, 0.586],
        'mat':  [0.613, 0.611, 0.615, 0.612, 0.614],
        'mdat': [0.591, 0.589, 0.593, 0.590, 0.592],
        'mrat': [0.604, 0.602, 0.606, 0.603, 0.605]
    }

    print("\nAlgorithm Results (Accuracy):")
    for alg, results in algorithm_results.items():
        mean = sum(results) / len(results)
        print(f"  {alg.upper():6s}: {mean:.3f} ± {(max(results)-min(results))/2:.3f}")

    # Perform pairwise Welch's t-tests
    print("\nPerforming pairwise Welch's t-tests...")
    p_values = welch_test(algorithm_results)

    print("\nP-values (rows vs columns):")
    print(p_values.round(4))

    print("\nInterpretation:")
    print("  p < 0.001: *** (very strong evidence)")
    print("  p < 0.01:  **  (strong evidence)")
    print("  p < 0.05:  *   (moderate evidence)")
    print("  p ≥ 0.05:  n.s. (not significant)")

    # Example comparison
    p_mrat_vs_mat = p_values.loc['mrat', 'mat']
    print(f"\nMRAT vs MAT: p = {p_mrat_vs_mat:.4f}", end=" ")
    if p_mrat_vs_mat < 0.05:
        print("(significantly different)")
    else:
        print("(no significant difference)")


def example_2_paired_test():
    """Example 2: Paired t-test for matched seed comparison."""
    print("\n" + "="*70)
    print("Example 2: Paired T-Test (Same Seeds)")
    print("="*70)

    # Same random seeds used for both algorithms
    seeds = [42, 43, 44, 45, 46]
    mdat_results = [0.591, 0.589, 0.593, 0.590, 0.592]
    mrat_results = [0.604, 0.602, 0.606, 0.603, 0.605]

    print(f"\nResults with seeds {seeds}:")
    print(f"  MDAT: {mdat_results}")
    print(f"  MRAT: {mrat_results}")

    # Paired t-test
    t_stat, p_value = paired_t_test(mdat_results, mrat_results)

    print(f"\nPaired t-test results:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p_value:.4f}")

    if p_value < 0.05:
        mean_diff = sum(mrat_results)/len(mrat_results) - sum(mdat_results)/len(mdat_results)
        print(f"\n✓ MRAT significantly outperforms MDAT (p < 0.05)")
        print(f"  Mean difference: {mean_diff:.3f}")
    else:
        print(f"\n✗ No significant difference (p ≥ 0.05)")


def example_3_anova():
    """Example 3: One-way ANOVA for multiple algorithms."""
    print("\n" + "="*70)
    print("Example 3: One-Way ANOVA (Overall Comparison)")
    print("="*70)

    algorithm_results = {
        'erm':  [0.276, 0.274, 0.278, 0.275, 0.277],
        'mda':  [0.562, 0.564, 0.560, 0.563, 0.561],
        'mrt':  [0.585, 0.583, 0.587, 0.584, 0.586],
        'mat':  [0.613, 0.611, 0.615, 0.612, 0.614],
        'mdat': [0.591, 0.589, 0.593, 0.590, 0.592],
        'mrat': [0.604, 0.602, 0.606, 0.603, 0.605]
    }

    print("\nTesting H₀: All algorithms have equal mean accuracy")

    f_stat, p_value = anova_test(algorithm_results)

    print(f"\nANOVA results:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value:     {p_value:.6f}")

    if p_value < 0.05:
        print(f"\n✓ Reject H₀: Significant differences exist among algorithms")
        print(f"  Recommendation: Proceed with pairwise tests (e.g., Welch's t-test)")
    else:
        print(f"\n✗ Cannot reject H₀: No significant differences detected")


def example_4_aggregation():
    """Example 4: Multi-seed result aggregation with confidence intervals."""
    print("\n" + "="*70)
    print("Example 4: Multi-Seed Aggregation with 95% CI")
    print("="*70)

    # Simulated MRAT results from 5 seeds
    mrat_values = [0.604, 0.602, 0.606, 0.603, 0.605]

    import numpy as np
    from scipy import stats as sp_stats

    mean = np.mean(mrat_values)
    std = np.std(mrat_values, ddof=1)  # Sample std
    n = len(mrat_values)
    sem = sp_stats.sem(mrat_values)
    ci = sem * sp_stats.t.ppf((1 + 0.95) / 2, n - 1)

    print(f"\nMRAT results from {n} random seeds:")
    print(f"  Values:  {mrat_values}")
    print(f"  Mean:    {mean:.4f}")
    print(f"  Std:     {std:.4f}")
    print(f"  SEM:     {sem:.4f}")
    print(f"  95% CI:  [{mean-ci:.4f}, {mean+ci:.4f}]")

    print(f"\nReporting format:")
    print(f"  MRAT: {mean:.3f} ± {ci:.3f} (mean ± 95% CI, n={n})")


def example_5_comparison_table():
    """Example 5: Generate publication-ready comparison table."""
    print("\n" + "="*70)
    print("Example 5: Comparison Table Generation")
    print("="*70)

    # Simulated aggregated results
    import numpy as np

    algorithm_results = {
        'erm': {
            'accuracy': {'mean': 0.276, 'ci_upper': 0.278, 'ci_lower': 0.274},
            'ece': {'mean': 0.321, 'ci_upper': 0.325, 'ci_lower': 0.317}
        },
        'mda': {
            'accuracy': {'mean': 0.562, 'ci_upper': 0.564, 'ci_lower': 0.560},
            'ece': {'mean': 0.201, 'ci_upper': 0.205, 'ci_lower': 0.197}
        },
        'mrt': {
            'accuracy': {'mean': 0.585, 'ci_upper': 0.587, 'ci_lower': 0.583},
            'ece': {'mean': 0.208, 'ci_upper': 0.212, 'ci_lower': 0.204}
        },
        'mat': {
            'accuracy': {'mean': 0.613, 'ci_upper': 0.615, 'ci_lower': 0.611},
            'ece': {'mean': 0.197, 'ci_upper': 0.201, 'ci_lower': 0.193}
        },
        'mdat': {
            'accuracy': {'mean': 0.591, 'ci_upper': 0.593, 'ci_lower': 0.589},
            'ece': {'mean': 0.203, 'ci_upper': 0.207, 'ci_lower': 0.199}
        },
        'mrat': {
            'accuracy': {'mean': 0.604, 'ci_upper': 0.606, 'ci_lower': 0.602},
            'ece': {'mean': 0.200, 'ci_upper': 0.204, 'ci_lower': 0.196}
        }
    }

    df = generate_comparison_table(
        algorithm_results,
        metrics=['accuracy', 'ece']
    )

    print("\nComparison Table:")
    print(df.to_string(index=False))

    print("\nNote: Table formatted as 'mean ± error' where error = (ci_upper - mean)")
    print("      This can be exported to LaTeX using output_path parameter")


def main():
    """Run all examples."""
    print("\n" + "#"*70)
    print("# Statistical Analysis Examples for MBRDL Hybrid Algorithms")
    print("#"*70)

    example_1_welch_test()
    example_2_paired_test()
    example_3_anova()
    example_4_aggregation()
    example_5_comparison_table()

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("""
Statistical Testing Workflow:

1. Run multi-seed experiments (5+ seeds recommended)
2. Use ANOVA to test if any algorithm differs
3. If ANOVA significant (p < 0.05), use pairwise tests:
   - Welch's t-test: Different algorithms, different variances
   - Paired t-test: Same seeds used (higher power)
4. Report results with mean ± 95% CI
5. Apply Bonferroni correction if doing many pairwise tests:
   - Adjusted α = 0.05 / number_of_comparisons

Key Functions:
  - welch_test(results): Pairwise Welch's t-test
  - paired_t_test(alg1, alg2): Paired t-test
  - anova_test(results): One-way ANOVA
  - aggregate_seed_results(dirs, metric): Aggregate with CI
  - generate_comparison_table(results): LaTeX table

See README.md for more details.
""")


if __name__ == '__main__':
    main()
