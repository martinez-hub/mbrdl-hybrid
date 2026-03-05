"""Statistical analysis utilities for multi-seed experiments."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple


def aggregate_seed_results(result_dirs: List[str], metric: str = 'top1') -> Dict:
    """Aggregate results from multiple seed runs.

    Args:
        result_dirs: List of result directories
        metric: Metric to aggregate (e.g., 'top1', 'ece')

    Returns:
        Dictionary with mean, std, ci_lower, ci_upper
    """
    values = []

    for result_dir in result_dirs:
        # Try to load from checkpoint or results file
        result_path = Path(result_dir)

        # Look for results.json or similar
        if (result_path / 'results.json').exists():
            with open(result_path / 'results.json') as f:
                data = json.load(f)
                values.append(data[metric])
        else:
            print(f"Warning: Could not find results in {result_dir}")

    if not values:
        return {}

    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)

    # 95% confidence interval using t-distribution
    n = len(values)
    sem = stats.sem(values)
    ci = sem * stats.t.ppf((1 + 0.95) / 2, n - 1)

    return {
        'mean': float(mean),
        'std': float(std),
        'ci_lower': float(mean - ci),
        'ci_upper': float(mean + ci),
        'n_seeds': n,
        'values': values.tolist(),
    }


def pairwise_t_tests(algorithm_results: Dict[str, List[float]]) -> pd.DataFrame:
    """Perform pairwise t-tests between algorithms.

    Args:
        algorithm_results: Dict mapping algorithm to list of values (one per seed)

    Returns:
        DataFrame with p-values for all pairs
    """
    algorithms = list(algorithm_results.keys())
    n = len(algorithms)

    p_values = np.zeros((n, n))

    for i, alg1 in enumerate(algorithms):
        for j, alg2 in enumerate(algorithms):
            if i == j:
                p_values[i, j] = 1.0
            else:
                _, p = stats.ttest_ind(
                    algorithm_results[alg1],
                    algorithm_results[alg2]
                )
                p_values[i, j] = p

    df = pd.DataFrame(p_values, index=algorithms, columns=algorithms)
    return df


def anova_test(algorithm_results: Dict[str, List[float]]) -> Tuple[float, float]:
    """One-way ANOVA across algorithms.

    Args:
        algorithm_results: Dict mapping algorithm to list of values

    Returns:
        Tuple of (F-statistic, p-value)
    """
    groups = [values for values in algorithm_results.values()]
    f_stat, p_value = stats.f_oneway(*groups)
    return f_stat, p_value


def generate_comparison_table(
    algorithm_results: Dict[str, Dict],
    metrics: List[str] = ['accuracy', 'ece'],
    output_path: str = None
) -> pd.DataFrame:
    """Generate LaTeX comparison table.

    Args:
        algorithm_results: Dict mapping algorithm to aggregated results
        metrics: List of metrics to include
        output_path: Optional path to save LaTeX table

    Returns:
        DataFrame with formatted results
    """
    rows = []

    for alg_name, results in algorithm_results.items():
        row = {'Algorithm': alg_name.upper()}

        for metric in metrics:
            if metric in results:
                mean = results[metric]['mean']
                ci_upper = results[metric]['ci_upper']
                error = ci_upper - mean
                row[metric] = f"{mean:.3f} ± {error:.3f}"

        rows.append(row)

    df = pd.DataFrame(rows)

    if output_path:
        latex = df.to_latex(index=False, escape=False)
        Path(output_path).write_text(latex)

    return df
