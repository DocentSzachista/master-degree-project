import numpy as np
import torch
import random


def set_workstation(device: str, seed=0):
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    torch.device(device=device)
