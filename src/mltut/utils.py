"""Small, dependency-light utilities used across notebooks."""

import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducible runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Best available accelerator: ``cuda`` > ``mps`` > ``cpu``."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
