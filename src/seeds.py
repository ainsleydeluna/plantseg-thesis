"""Reproducibility seed utility.

Implements the seeding half of Blocker B6 (see docs/IMPLEMENTATION_CONTRACT.md):
seed 42 across python random / numpy / torch / torch.cuda, with cuDNN set to
deterministic and non-benchmark mode.
"""

import os
import random

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = 42) -> int:
    """Seed all RNGs and enforce the contract-B6 determinism protocol.

    Sets cuDNN deterministic mode, torch deterministic algorithms (warn_only=True), and
    CUBLAS_WORKSPACE_CONFIG. Returns the seed used so callers can log it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS determinism must be configured BEFORE the first CUDA op / cuBLAS handle is created;
    # set_seed runs before any device transfer in the training path (contract B6).
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # all CUDA devices

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)  # warn (not error) on non-deterministic ops

    return seed
