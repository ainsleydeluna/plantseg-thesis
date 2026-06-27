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
    """Seed all RNGs and force deterministic cuDNN behaviour.

    Returns the seed used so callers can log it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # all CUDA devices

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed
