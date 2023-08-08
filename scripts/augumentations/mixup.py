import numpy as np
import torch
import torch.nn as nn


def mixup_criterion(
    lambda_: float, image_1: torch.Tensor, image_2: torch.Tensor
) -> torch.Tensor:
    """Mixes two images with each other"""
    return lambda_ * image_1 + (1 - lambda_) * image_2
