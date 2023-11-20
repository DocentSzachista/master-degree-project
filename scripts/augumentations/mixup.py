import numpy as np
import torch
import torch.nn as nn


def mixup_criterion(
    lambda_: float, image_1: np.ndarray, image_2: np.ndarray
) -> np.ndarray:
    """Mixes two images with each other"""
    return (lambda_ * image_1 + (1 - lambda_) * image_2).astype(np.uint8)
