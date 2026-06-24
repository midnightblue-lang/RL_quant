import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set deterministic seeds for current stage dependencies."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
