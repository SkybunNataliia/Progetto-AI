import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Per garantire determinismo in CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False