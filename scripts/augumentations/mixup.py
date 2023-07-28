import torch.nn as nn
import numpy as np
import torch 

def mixup_data(x, y, alpha=1.0):
    """Mixes data."""
    if alpha > 0:
        lambda_ = np.random.beta(alpha, alpha)
    else: 
        lambda_ = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lambda_ * x + (1- lambda_) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lambda_


def mixup_criterion(criterion: nn.CrossEntropyLoss, pred, y_a, y_b, lambda_: float):
    return lambda_ * criterion(pred, y_a) + (1 - lambda_) * criterion(pred, y_b)
