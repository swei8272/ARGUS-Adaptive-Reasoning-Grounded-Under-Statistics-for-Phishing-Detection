"""Shared CPU, Apple Metal (MPS), and CUDA runtime policy."""
import random

import numpy as np
import torch


def select_device() -> torch.device:
    """Prefer Apple Silicon MPS, then NVIDIA CUDA, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def device_dtype(device, dtype=torch.float32) -> torch.dtype:
    """MPS computation is explicitly float32; other backends preserve dtype."""
    return torch.float32 if torch.device(device).type == 'mps' else dtype


def seed_everything(seed: int, device=None) -> None:
    """Seed host RNGs and the selected backend; not a bitwise determinism claim.

    Seed the CPU generator directly because torch.manual_seed also dispatches
    accelerator seeds. Never call CUDA APIs as a substitute for MPS seeding.
    """
    device = select_device() if device is None else torch.device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if device.type == 'mps':
        torch.mps.manual_seed(seed)
    elif device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
