import torch
import torch.nn.functional as F

def mda_train(images, target, model, G, args):
    """Model-based Data Augmentation training algorithm.
    
    Params:
        images: Batch of training imagse.
        target: Labels corresponding to training batch.
        model: Classifier instance.
        G: Model of natural variation.
        args: Command line arguments.

    Returns:
        Images and target augmented with model-based data.
    """

    all_mb_images = [images]

    for _ in range(args.k):
        with torch.no_grad():
            delta = torch.rand(images.size(0), args.delta_dim, 1, 1).cuda()
            mb_images = G(images, delta)
            all_mb_images.append(mb_images)

    images = torch.cat(all_mb_images, dim=0).cuda()
    target = torch.cat([target for _ in range(len(all_mb_images))])

    return images, target

def mrt_train(images, target, model, criterion, G, args):
    """Model-based Robust Training training algorithm.
    
    Params:
        images: Batch of training imagse.
        target: Labels corresponding to training batch.
        model: Classifier instance.
        criterion: Loss function.
        G: Model of natural variation.
        args: Command line arguments.

    Returns:
        Images and target augmented with model-based data.
    """

    max_loss, worst_imgs = torch.tensor(0.0).cuda(), None

    for _ in range(args.k):
        with torch.no_grad():
            delta = torch.rand(images.size(0), args.delta_dim, 1, 1).cuda()
            mb_images = G(images, delta)
            mb_output = model(mb_images)
            mb_loss = criterion(mb_output, target)
            if mb_loss > max_loss:
                worst_imgs = mb_images
                max_loss = mb_loss

    images = torch.cat([images, worst_imgs.cuda()], dim=0)
    target = torch.cat([target, target])

    return images, target

def mat_train(images, target, model, criterion, G, args, alpha=0.1):
    """Model-based Adversarial Training training algorithm.
    
    Params:
        images: Batch of training imagse.
        target: Labels corresponding to training batch.
        model: Classifier instance.
        criterion: Loss function.
        G: Model of natural variation.
        args: Command line arguments.
        alpha: Step size for adversarial training.

    Returns:
        Images and target augmented with model-based data.
    """

    adv_delta = torch.zeros(images.size(0), args.delta_dim, 1, 1).cuda()
    adv_delta.requires_grad_(True)
    for _ in range(args.k):
        mb_images = G(images, adv_delta)
        loss = F.nll_loss(model(mb_images), target)
        loss.backward()
        grad_delta = adv_delta.grad.detach() # / torch.norm(adv_delta.grad.detach())
        adv_delta.data = (adv_delta + alpha * grad_delta).clamp(-1, 1)
        adv_delta.grad.zero_()

    adv_delta = adv_delta.detach().requires_grad_(False)
    mb_images = G(images, adv_delta)
    images = torch.cat([images, mb_images.cuda()], dim=0)
    target = torch.cat([target, target])

    return images, target

def mdat_train(images, target, model, criterion, G, args, alpha=0.1):
    """Model-based Data Augmentation → Adversarial Training (MDAT).

    Hybrid algorithm combining random initialization with gradient refinement.

    Params:
        images: Batch of training images [B, C, H, W]
        target: Labels [B]
        model: Classifier instance
        criterion: Loss function
        G: Model of natural variation
        args: Command line arguments with args.k (gradient steps)
        alpha: Step size for adversarial training (default: 0.1)

    Returns:
        Images and target augmented with adversarially refined model-based data.

    Algorithm:
        1. Initialize: δ_0 ~ Uniform(-1, 1)^delta_dim  (MDA-style random init)
        2. Refine for T steps:
           δ_{t+1} = clip(δ_t + α * ∇_δ L(f(G(x, δ_t)), y), -1, 1)
        3. Return: [x; G(x, δ_T)], [y; y]

    Complexity: Θ(T) - same as MAT
    """
    # Initialize with random sample (MDA initialization)
    device = images.device
    adv_delta = torch.rand(images.size(0), args.delta_dim, 1, 1).to(device) * 2 - 1  # Uniform[-1,1]
    adv_delta.requires_grad_(True)

    # Determine number of gradient steps
    num_steps = args.T if hasattr(args, 'T') and args.T is not None else args.k

    # Adversarial refinement (MAT-style gradient ascent)
    for _ in range(num_steps):
        mb_images = G(images, adv_delta)
        loss = F.nll_loss(model(mb_images), target)
        loss.backward()
        grad_delta = adv_delta.grad.detach()
        adv_delta.data = (adv_delta + alpha * grad_delta).clamp(-1, 1)
        adv_delta.grad.zero_()

    # Generate final augmented images
    adv_delta = adv_delta.detach().requires_grad_(False)
    mb_images = G(images, adv_delta)
    images = torch.cat([images, mb_images.to(device)], dim=0)
    target = torch.cat([target, target])

    return images, target


def mrat_train(images, target, model, criterion, G, args, alpha=0.1):
    """Model-based Robust Training → Adversarial Training (MRAT).

    Hybrid algorithm combining worst-of-k initialization with gradient refinement.

    Params:
        images: Batch of training images [B, C, H, W]
        target: Labels [B]
        model: Classifier instance
        criterion: Loss function
        G: Model of natural variation
        args: Command line arguments with args.k (samples) and args.T (gradient steps)
        alpha: Step size for adversarial training (default: 0.1)

    Returns:
        Images and target augmented with adversarially refined model-based data.

    Algorithm:
        1. Initialize: δ_0 = argmax_{δ_i} L(f(G(x, δ_i)), y) over k samples (MRT init)
        2. Refine for T steps:
           δ_{t+1} = clip(δ_t + α * ∇_δ L(f(G(x, δ_t)), y), -1, 1)
        3. Return: [x; G(x, δ_T)], [y; y]

    Complexity: Θ(k + T) - combines MRT and MAT costs
    """
    # Step 1: MRT initialization - find worst-of-k
    device = images.device
    max_loss, worst_delta = torch.tensor(0.0).to(device), None

    for _ in range(args.k):
        with torch.no_grad():
            delta = torch.rand(images.size(0), args.delta_dim, 1, 1).to(device) * 2 - 1
            mb_images = G(images, delta)
            mb_output = model(mb_images)
            mb_loss = criterion(mb_output, target)
            if mb_loss > max_loss:
                worst_delta = delta
                max_loss = mb_loss

    # Step 2: MAT refinement starting from worst-case
    adv_delta = worst_delta.clone().requires_grad_(True)

    # Use args.T for gradient steps if available, otherwise use args.k
    num_steps = args.T if hasattr(args, 'T') and args.T is not None else args.k

    for _ in range(num_steps):
        mb_images = G(images, adv_delta)
        loss = F.nll_loss(model(mb_images), target)
        loss.backward()
        grad_delta = adv_delta.grad.detach()
        adv_delta.data = (adv_delta + alpha * grad_delta).clamp(-1, 1)
        adv_delta.grad.zero_()

    # Generate final augmented images
    adv_delta = adv_delta.detach().requires_grad_(False)
    mb_images = G(images, adv_delta)
    images = torch.cat([images, mb_images.to(device)], dim=0)
    target = torch.cat([target, target])

    return images, target


def pgd_train(images, target, model, criterion, num_iter=10, alpha=0.01, epsilon=8/255.):
    """PGD Adversarial training algorithm.

    Params:
        images: Batch of training imagse.
        target: Labels corresponding to training batch.
        model: Classifier instance.
        criterion: Loss function.
        num_iter: Number of steps of gradient ascent.
        alpha: Step size for gradient ascent.
        epsilon: Maximum (l_infinity) adversarial perturbation size.

    Adversarially perturbed data.
    """

    delta = torch.zeros_like(images, requires_grad=True)
    for t in range(num_iter):
        loss = criterion(model(images + delta), target)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach()).clamp(-epsilon, epsilon)
        delta.grad.zero_()

    return images + delta.detach(), target