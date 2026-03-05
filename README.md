# MBRDL-Hybrid: Enhanced Model-Based Robust Deep Learning

**Josué Martínez-Martínez, Olivia Brown, Giselle Zeno, Pooya Khorrami, Rajmonda Caceres**

Extended implementation of [MBRDL](https://github.com/arobey1/mbrdl) with novel hybrid training algorithms, comprehensive statistical analysis, and performance optimizations.

**Paper**: ["From Snow to Rain: Evaluating Robustness, Calibration, and Complexity of Model-Based Robust Training"](https://arxiv.org/abs/2601.09153)
**arXiv**: [2601.09153](https://arxiv.org/abs/2601.09153) | **Year**: 2026

## Key Features

✨ **New Hybrid Algorithms**
- **MDAT (MDA→MAT)**: Random initialization + gradient refinement [Θ(T)]
- **MRAT (MRT→MAT)**: Worst-of-k initialization + gradient refinement [Θ(k+T)]

📊 **Enhanced Evaluation**
- ECE (Expected Calibration Error) for model calibration
- MRR (Mean Reciprocal Rank) for algorithm comparison
- Training time tracking

🔬 **Statistical Analysis**
- Welch's t-test, Paired t-test, ANOVA
- Multi-seed experiment automation
- Confidence interval computation
- Publication-ready comparison tables

⚡ **Performance Optimizations**
- 5-10% faster training algorithms
- 10-15% faster metrics computation
- 40% faster statistical tests
- Reduced memory allocations

🧪 **Comprehensive Testing**
- 15 automated tests (unit + integration)
- Performance benchmarks
- CPU/GPU compatibility

---

## Quick Start

### Installation

```bash
git clone https://github.com/martinez-hub/mbrdl-hybrid.git
cd mbrdl-hybrid
pip install -r requirements.txt
```

### Train with Hybrid Algorithms

```bash
# MDAT: Random initialization + gradient refinement
python -m core.train \
  --dataset cure-tsr \
  --train-data-dir ./datasets/cure_tsr/raw_data \
  --source-of-nat-var snow \
  --model-paths ./models/cure-tsr-snow.pt \
  --architecture resnet18 \
  --mdat -k 10 --T 10

# MRAT: Worst-of-k initialization + gradient refinement
python -m core.train \
  --dataset cure-tsr \
  --train-data-dir ./datasets/cure_tsr/raw_data \
  --source-of-nat-var snow \
  --model-paths ./models/cure-tsr-snow.pt \
  --architecture resnet18 \
  --mrat -k 10 --T 10
```

### Multi-Seed Experiments

```bash
# Run with multiple seeds for statistical analysis
python scripts/run_multi_seed.py \
  --seeds 42,43,44,45,46 \
  --output results/mrat_snow \
  --args --mrat -k 10 --T 10

# Analyze results
python scripts/analyze_results.py \
  --results results/mrat_snow/seed_* \
  --metric top1 \
  --output analysis/mrat_snow
```

---

## Algorithms

| Algorithm | Initialization | Refinement | Complexity | Best For |
|-----------|---------------|------------|------------|----------|
| **ERM** | - | - | Θ(1) | Baseline |
| **MDA** | Random | - | Θ(1) | Fast augmentation |
| **MRT** | Worst-of-k | - | Θ(k) | Robust sampling |
| **MAT** | Zero | Gradient ascent | Θ(T) | Max robustness |
| **MDAT** ⭐ | Random | Gradient ascent | Θ(T) | Balanced |
| **MRAT** ⭐ | Worst-of-k | Gradient ascent | Θ(k+T) | Best robustness |
| **PGD** | Zero | Gradient ascent | Θ(S) | Pixel perturbations |

**Expected Performance** (CURE-TSR Snow, Severity 5):
- **MRAT**: 60.4% accuracy, 20.0% ECE
- **MDAT**: 59.1% accuracy, 20.3% ECE
- **MAT**: 61.3% accuracy, 19.7% ECE

See [HYBRID_ALGORITHMS.md](HYBRID_ALGORITHMS.md) for detailed algorithm descriptions.

---

## Statistical Analysis

### Quick Example

```python
from core.utils.statistical import welch_test, paired_t_test, anova_test

# Compare algorithms with Welch's t-test (recommended)
algorithm_results = {
    'mda': [0.56, 0.57, 0.55],
    'mrat': [0.60, 0.61, 0.59]
}
p_values = welch_test(algorithm_results)

# Paired comparison (same seeds)
t_stat, p_value = paired_t_test(mdat_results, mrat_results)

# ANOVA for multiple algorithms
f_stat, p_value = anova_test(algorithm_results)
```

### Interactive Examples

```bash
# Run comprehensive statistical analysis examples
python examples/statistical_analysis_example.py
```

**Covers**: Welch's t-test, paired t-test, ANOVA, multi-seed aggregation, comparison tables

**See**: [README Statistical Analysis section](https://github.com/martinez-hub/mbrdl-hybrid#statistical-analysis-full) for complete API documentation.

---

## Testing & Validation

```bash
# Unit tests (4 tests)
python tests/test_hybrid_algorithms.py

# Integration tests (11 tests)
python tests/smoke_test.py

# Performance benchmarks
python tests/benchmark_optimizations.py
```

**All 15 tests passing** ✅

---

## Performance

### Training Speed (CPU, batch=32)
- MDA: 6.94 ms/iter (1.04x faster vs baseline)
- MDAT: 283.76 ms/iter (1.05x faster)
- MRAT: 360.39 ms/iter (1.05x faster)

### Metrics Speed
- ECE computation: 17% faster
- Pairwise t-tests: 40% faster

See [OPTIMIZATIONS.md](OPTIMIZATIONS.md) for detailed optimization breakdown.

---

## Documentation

| File | Description |
|------|-------------|
| `README.md` | This file - quick start and API overview |
| `HYBRID_ALGORITHMS.md` | Detailed algorithm descriptions and complexity analysis |
| `OPTIMIZATIONS.md` | Performance optimizations and benchmarks |
| `VERIFICATION.md` | Test verification report (15/15 tests) |
| `IMPLEMENTATION_SUMMARY.md` | Complete implementation details |

---

## Citation

### This Work (MBRDL-Hybrid)

```bibtex
@article{martinez2026snow,
  title={From Snow to Rain: Evaluating Robustness, Calibration, and Complexity of Model-Based Robust Training},
  author={Mart{\'\i}nez-Mart{\'\i}nez, Josu{\'e} and Brown, Olivia and Zeno, Giselle and Khorrami, Pooya and Caceres, Rajmonda},
  journal={arXiv preprint arXiv:2601.09153},
  year={2026}
}
```

### Original MBRDL

```bibtex
@article{robey2020model,
  title={Model-Based Robust Deep Learning},
  author={Robey, Alexander and Hassani, Hamed and Pappas, George J},
  journal={arXiv preprint arXiv:2005.10247},
  year={2020}
}
```

---

## License

This project extends the original [MBRDL repository](https://github.com/arobey1/mbrdl). See LICENSE.md for details.

---

## Acknowledgments

Built upon the excellent [MBRDL framework](https://github.com/arobey1/mbrdl) by Robey et al. This repository adds hybrid training algorithms, statistical analysis tools, and performance optimizations as described in our 2026 paper.

For the original MBRDL documentation, datasets, and pre-trained models, visit the [original repository](https://github.com/arobey1/mbrdl).
