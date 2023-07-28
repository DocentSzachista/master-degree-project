import random
import torch


def create_and_shuffle_indexes(matrix_shape: tuple):
    indexes = [
        i * 32 + j for i in range(matrix_shape[0]) for j in range(matrix_shape[1])
    ]
    random.shuffle(indexes)
    return indexes


def apply_noise_to_image(
    shuffled_indexes: list,
    image: torch.Tensor,
    mask: torch.Tensor,
    start: int,
    stop: int,
):
    """Apply part of mask to the image basing on pixels_affected parameter"""
    img_length = 32
    for index in range(start, stop):
        i = shuffled_indexes[index] // img_length
        j = shuffled_indexes[index] % img_length
        image[:, i, j] += mask[:, i, j]
    return image


def generate_mask(shape: tuple):
    torch.manual_seed(0)
    return torch.randn(shape)
