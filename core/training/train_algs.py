"""Optimized training algorithms for MBRDL.

Performance improvements:
- Removed redundant .to(device) calls
- Optimized tensor operations (in-place, pre-allocation)
- Improved memory efficiency
- Vectorized operations where possible
- Reduced unnecessary tensor copies
"""

import torch
import torch.nn.functional as F


def mda_train(images, target, model, G, args):
    """Model-based Data Augmentation training algorithm (optimized).

    Optimizations:
    - Use repeat() instead of list comprehension for targets
    - Removed redundant device transfers
    """
    device = images.device
    all_mb_images = [images]

    with torch.no_grad():
        for _ in range(args.k):
            delta = torch.rand(images.size(0), args.delta_dim, 1, 1, device=device)
            mb_images = G(images, delta)
            all_mb_images.append(mb_images)

    images = torch.cat(all_mb_images, dim=0)
    # Optimized: use repeat instead of list comprehension
    target = target.repeat(len(all_mb_images))

    return images, target


def mrt_train(images, target, model, criterion, G, args):
    """Model-based Robust Training training algorithm (optimized).

    Optimizations:
    - Removed redundant device transfers
    - Keep tensor comparisons (faster than .item())
    """
    device = images.device
    max_loss = torch.tensor(0.0, device=device)
    worst_imgs = None

    with torch.no_grad():
        for _ in range(args.k):
            delta = torch.rand(images.size(0), args.delta_dim, 1, 1, device=device)
            mb_images = G(images, delta)
            mb_output = model(mb_images)
            mb_loss = criterion(mb_output, target)

            if mb_loss > max_loss:
                worst_imgs = mb_images
                max_loss = mb_loss

    images = torch.cat([images, worst_imgs], dim=0)
    target = target.repeat(2)

    return images, target


def mat_train(images, target, model, criterion, G, args, alpha=0.1):
    """Model-based Adversarial Training training algorithm (optimized).

    Optimizations:
    - In-place clamp operations
    - Removed redundant operations
    """
    device = images.device
    adv_delta = torch.zeros(images.size(0), args.delta_dim, 1, 1, device=device,  requires_grad=True)

    for _ in range(args.k):
        mb_images = G(images, adv_delta)
        loss = F.nll_loss(model(mb_images), target)
        loss.backward()

        # Update delta with in-place operations
        grad_delta = adv_delta.grad.detach()
        adv_delta.data.add_(alpha * grad_delta).clamp_(-1, 1)
        adv_delta.grad.zero_()

    # Generate final images
    mb_images = G(images, adv_delta.detach())
    images = torch.cat([images, mb_images], dim=0)
    target = target.repeat(2)

    return images, target


def mdat_train(images, target, model, criterion, G, args, alpha=0.1):
    """Model-based Data Augmentation → Adversarial Training (MDAT) - optimized.

    Optimizations:
    - In-place operations for delta updates
    - Removed redundant operations
    """
    device = images.device

    # Initialize with random sample (MDA initialization)
    adv_delta = (torch.rand(images.size(0), args.delta_dim, 1, 1, device=device) * 2 - 1)
    adv_delta.requires_grad_(True)

    # Determine number of gradient steps
    num_steps = args.T if hasattr(args, 'T') and args.T is not None else args.k

    # Adversarial refinement (MAT-style gradient ascent)
    for _ in range(num_steps):
        mb_images = G(images, adv_delta)
        loss = F.nll_loss(model(mb_images), target)
        loss.backward()

        # Update delta with in-place operations
        grad_delta = adv_delta.grad.detach()
        adv_delta.data.add_(alpha * grad_delta).clamp_(-1, 1)
        adv_delta.grad.zero_()

    # Generate final augmented images
    mb_images = G(images, adv_delta.detach())
    images = torch.cat([images, mb_images], dim=0)
    target = target.repeat(2)

    return images, target


def mrat_train(images, target, model, criterion, G, args, alpha=0.1):
    """Model-based Robust Training → Adversarial Training (MRAT) - optimized.

    Optimizations:
    - In-place operations for delta updates
    - Removed redundant operations
    """
    device = images.device

    # Step 1: MRT initialization - find worst-of-k
    max_loss = torch.tensor(0.0, device=device)
    worst_delta = None

    with torch.no_grad():
        for _ in range(args.k):
            delta = torch.rand(images.size(0), args.delta_dim, 1, 1, device=device) * 2 - 1
            mb_images = G(images, delta)
            mb_output = model(mb_images)
            mb_loss = criterion(mb_output, target)

            if mb_loss > max_loss:
                worst_delta = delta
                max_loss = mb_loss

    # Step 2: MAT refinement starting from worst-case
    adv_delta = worst_delta.clone().requires_grad_(True)

    # Use args.T for gradient steps if available
    num_steps = args.T if hasattr(args, 'T') and args.T is not None else args.k

    for _ in range(num_steps):
        mb_images = G(images, adv_delta)
        loss = F.nll_loss(model(mb_images), target)
        loss.backward()

        # Update delta with in-place operations
        grad_delta = adv_delta.grad.detach()
        adv_delta.data.add_(alpha * grad_delta).clamp_(-1, 1)
        adv_delta.grad.zero_()

    # Generate final augmented images
    mb_images = G(images, adv_delta.detach())
    images = torch.cat([images, mb_images], dim=0)
    target = target.repeat(2)

    return images, target


def pgd_train(images, target, model, criterion, num_iter=10, alpha=0.01, epsilon=8/255.):
    """PGD Adversarial training algorithm (optimized).

    Optimizations:
    - In-place operations for delta updates
    """
    delta = torch.zeros_like(images, requires_grad=True)

    for _ in range(num_iter):
        loss = criterion(model(images + delta), target)
        loss.backward()

        # Update delta with in-place operations
        delta.data.add_(alpha * delta.grad.detach()).clamp_(-epsilon, epsilon)
        delta.grad.zero_()

    return images + delta.detach(), target
