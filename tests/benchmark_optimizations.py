#!/usr/bin/env python3
"""Benchmark script to measure optimization improvements.

Compares original vs optimized training algorithms for:
- Speed (time per iteration)
- Memory usage (peak GPU memory)
- Throughput (samples per second)
"""

import torch
import torch.nn as nn
import time
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import training algorithms (now optimized)
sys.path.insert(0, str(Path(__file__).parent.parent / 'core' / 'training'))
import train_algs as current


class SimpleGenerator(nn.Module):
    """Simple generator for benchmarking."""
    def __init__(self, delta_dim=8):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(delta_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * 32 * 32),
            nn.Tanh()
        )

    def forward(self, x, delta):
        batch_size = x.size(0)
        delta_flat = delta.view(batch_size, -1)
        perturbation = self.transform(delta_flat)
        perturbation = perturbation.view(batch_size, 3, 32, 32)
        return torch.clamp(x + 0.1 * perturbation, 0, 1)


class SimpleClassifier(nn.Module):
    """Simple classifier for benchmarking."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return torch.log_softmax(self.fc(x), dim=1)


class MockArgs:
    """Mock args for testing."""
    def __init__(self, k=10, T=10):
        self.k = k
        self.T = T
        self.delta_dim = 8


def benchmark_algorithm(alg_fn, name, images, target, model, criterion, G, args, num_iterations=100):
    """Benchmark a single algorithm."""
    print(f"\nBenchmarking {name}...")

    # Warmup
    for _ in range(10):
        _ = alg_fn(images.clone(), target.clone(), model, criterion, G, args)

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    start_time = time.time()

    for _ in range(num_iterations):
        _ = alg_fn(images.clone(), target.clone(), model, criterion, G, args)

    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    time_per_iter = total_time / num_iterations
    samples_per_sec = (images.size(0) * num_iterations) / total_time

    # Memory usage
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    else:
        peak_memory = 0

    return {
        'time_per_iter': time_per_iter,
        'samples_per_sec': samples_per_sec,
        'peak_memory_mb': peak_memory
    }


def compare_algorithms(batch_size=32, device='cpu', num_iterations=100):
    """Compare original vs optimized algorithms."""
    print("="*70)
    print(f"Optimization Benchmark ({device.upper()})")
    print("="*70)
    print(f"Batch size: {batch_size}")
    print(f"Iterations: {num_iterations}")

    # Setup
    model = SimpleClassifier().to(device)
    G = SimpleGenerator().to(device)
    criterion = nn.NLLLoss()
    args = MockArgs(k=10, T=10)

    images = torch.randn(batch_size, 3, 32, 32).to(device)
    target = torch.randint(0, 10, (batch_size,)).to(device)

    algorithms = [
        ('MDA', current.mda_train, False),
        ('MRT', current.mrt_train, True),
        ('MAT', current.mat_train, True),
        ('MDAT', current.mdat_train, True),
        ('MRAT', current.mrat_train, True),
    ]

    results = {}

    for alg_name, alg_fn, needs_criterion in algorithms:
        print(f"\n{'─'*70}")
        print(f"Algorithm: {alg_name}")
        print(f"{'─'*70}")

        # Prepare function call
        if needs_criterion:
            fn_call = lambda imgs, tgt, m, c, g, a: alg_fn(imgs, tgt, m, c, g, a)
        else:
            fn_call = lambda imgs, tgt, m, c, g, a: alg_fn(imgs, tgt, m, g, a)

        # Benchmark algorithm
        stats = benchmark_algorithm(
            fn_call, f"{alg_name} (optimized)",
            images, target, model, criterion, G, args, num_iterations
        )

        print(f"\n  Results:")
        print(f"    Time/iter: {stats['time_per_iter']*1000:.2f} ms")
        print(f"    Throughput: {stats['samples_per_sec']:.1f} samples/sec")

        if torch.cuda.is_available():
            print(f"    Peak memory: {stats['peak_memory_mb']:.1f} MB")

        results[alg_name] = stats

    # Summary
    print(f"\n{'='*70}")
    print("Performance Summary (Optimized Implementation)")
    print(f"{'='*70}")

    print("\nPerformance by algorithm:")
    for alg_name, stats in results.items():
        print(f"  {alg_name:6s}: {stats['time_per_iter']*1000:6.2f} ms/iter, "
              f"{stats['samples_per_sec']:6.1f} samples/sec")

    return results


def main():
    """Run benchmarks."""
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "#"*70)
    print("# MBRDL Hybrid Algorithms - Optimization Benchmark")
    print("#"*70)

    # Run benchmark on CPU
    cpu_results = compare_algorithms(batch_size=32, device='cpu', num_iterations=50)

    # Run on GPU if available
    if torch.cuda.is_available():
        print("\n\n")
        gpu_results = compare_algorithms(batch_size=64, device='cuda', num_iterations=100)

    print("\n" + "="*70)
    print("Benchmark Complete")
    print("="*70)
    print("\nOptimizations applied in current implementation:")
    print("  ✓ In-place tensor operations (.add_(), .clamp_())")
    print("  ✓ Removed redundant device transfers")
    print("  ✓ Efficient gradient zeroing")
    print("  ✓ Efficient target repetition (.repeat() vs list comprehension)")
    print("  ✓ Tensor comparisons (avoid .item() overhead)")
    print("\nNote: All algorithms are now optimized.")
    print("      See OPTIMIZATIONS.md for detailed performance comparisons.")


if __name__ == '__main__':
    main()
