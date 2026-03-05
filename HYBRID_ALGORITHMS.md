# Hybrid Training Algorithms

This repository extends the original MBRDL with two novel hybrid algorithms from ["From Snow to Rain: Evaluating Robustness, Calibration, and Complexity of Model-Based Robust Training"](https://arxiv.org/abs/2601.09153).

## Overview

The hybrid algorithms combine different initialization strategies with adversarial refinement to improve robustness while managing computational cost.

## MDAT (MDA → MAT)

**Model-based Data Augmentation → Adversarial Training**

Combines random initialization with gradient-based refinement.

### Algorithm

1. **Initialization (MDA):** Sample δ₀ ~ Uniform(-1, 1)^d
2. **Refinement (MAT):** For t = 0 to T-1:
   - δₜ₊₁ = δₜ + α · ∇_δ L(f(G(x, δₜ)), y)
   - δₜ₊₁ = clip(δₜ₊₁, -1, 1)
3. **Augmentation:** Return [x; G(x, δ_T)], [y; y]

### Complexity

**Θ(T)** - Same as MAT (T gradient steps)

### Hyperparameters

- `T`: Number of gradient steps (default: 10)
- `alpha`: Step size (default: 0.1)
- `delta_dim`: Nuisance space dimension (default: 8)

### Usage

```bash
python -m core.train \
  --dataset cure-tsr \
  --train-data-dir path/to/data \
  --source-of-nat-var snow \
  --model-paths path/to/munit.pt \
  --mdat -k 10
```

### When to Use

- **Mild to moderate corruptions** (severity 1-3)
- **Computational budget** allows for gradient computation
- Want **better robustness than MDA** without full MAT cost

## MRAT (MRT → MAT)

**Model-based Robust Training → Adversarial Training**

Combines worst-of-k initialization with gradient-based refinement.

### Algorithm

1. **Initialization (MRT):**
   - Sample {δ₁, ..., δₖ} ~ Uniform(-1, 1)^d
   - δ₀ = argmax_i L(f(G(x, δᵢ)), y)
2. **Refinement (MAT):** For t = 0 to T-1:
   - δₜ₊₁ = δₜ + α · ∇_δ L(f(G(x, δₜ)), y)
   - δₜ₊₁ = clip(δₜ₊₁, -1, 1)
3. **Augmentation:** Return [x; G(x, δ_T)], [y; y]

### Complexity

**Θ(k + T)** - Combines MRT sampling (k) + MAT refinement (T)

### Hyperparameters

- `k`: Number of random samples for worst-case selection (default: 10)
- `T`: Number of gradient steps (default: 10)
- `alpha`: Step size (default: 0.1)
- `delta_dim`: Nuisance space dimension (default: 8)

### Usage

```bash
python -m core.train \
  --dataset cure-tsr \
  --train-data-dir path/to/data \
  --source-of-nat-var snow \
  --model-paths path/to/munit.pt \
  --mrat -k 10 --T 10
```

### When to Use

- **Severe corruptions** (severity 4-5)
- Want **maximum robustness**
- Computational budget allows for k+T forward passes

## Comparison with Base Algorithms

| Algorithm | Complexity | Best For | Accuracy | Calibration | Cost |
|-----------|------------|----------|----------|-------------|------|
| ERM       | Θ(1)       | Baseline | Low      | Good        | Lowest |
| MDA       | Θ(1)       | Efficiency | Moderate | Good   | Low |
| MRT       | Θ(k)       | Fast robust | Moderate | Good  | Medium |
| MAT       | Θ(T)       | Max robust | High     | Moderate | High |
| **MDAT**  | **Θ(T)**   | **Balanced** | **Moderate-High** | **Good** | **High** |
| **MRAT**  | **Θ(k+T)** | **Best robust** | **Highest** | **Moderate** | **Highest** |
| PGD       | Θ(S)       | Pixel attack | Moderate | Poor | High |

### Expected Performance (CURE-TSR Snow, Severity 5)

From the paper:
- **MAT**: 61.3% accuracy, 19.7% ECE
- **MRAT**: 60.4% accuracy, 20.0% ECE
- **MDAT**: 59.1% accuracy, 20.3% ECE
- **MRT**: 58.5% accuracy, 20.8% ECE
- **MDA**: 56.2% accuracy, 20.1% ECE

## Implementation Notes

- Both hybrids reuse existing infrastructure (G generator, training loop)
- MDAT has same computational cost as MAT but different initialization
- MRAT has higher cost but strongest adversarial examples
- For multi-seed experiments, use `scripts/run_multi_seed.py`

## Evaluation Metrics

### Expected Calibration Error (ECE)

Measures the difference between predicted confidence and actual accuracy:

```
ECE = Σᵢ (nᵢ/N) |acc(Bᵢ) - conf(Bᵢ)|
```

where predictions are binned by confidence level.

### Mean Reciprocal Rank (MRR)

Ranks algorithms by performance:

```
MRR(algorithm) = 1 / rank(algorithm)
```

Best algorithm gets MRR = 1.0, second gets 0.5, etc.

## Multi-Seed Experiments

Run experiments with multiple random seeds for statistical significance:

```bash
python scripts/run_multi_seed.py \
  --script train_basic.sh \
  --seeds 42,43,44 \
  --output results/mrat_snow \
  --args --mrat -k 10 --T 10
```

Analyze results:

```bash
python scripts/analyze_results.py \
  --results results/mrat_snow/seed_* \
  --metric top1 \
  --output analysis/mrat_snow
```

## Citation

```bibtex
@article{martinez2026snow,
  title={From Snow to Rain: Evaluating Robustness, Calibration, and Complexity of Model-Based Robust Training},
  author={Mart{\'\i}nez-Mart{\'\i}nez, Josu{\'e} and Brown, Olivia and Zeno, Giselle and Khorrami, Pooya and Caceres, Rajmonda},
  journal={arXiv preprint arXiv:2601.09153},
  year={2026}
}
```

## Troubleshooting

### CUDA Out of Memory

MRAT has the highest memory footprint (k + T forward passes). Solutions:
- Reduce batch size
- Reduce k or T
- Use MDAT instead (lower memory, still good performance)

### Poor Calibration (High ECE)

- MAT/MRAT tend to have higher ECE than MDA/MRT
- Consider temperature scaling post-hoc
- Use label smoothing during training

### Training Too Slow

- Use MDA for fastest training (Θ(1))
- Use MRT for moderate speed (Θ(k))
- MDAT and MAT have same speed (Θ(T))
- MRAT is slowest (Θ(k+T))
