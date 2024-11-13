"""Metrics helper functions."""

from enum import Enum

import numpy as np

import torch as pt
from matplotlib import pyplot as plt


class Result(Enum):
    """Result Declaration for Scoring Functions."""

    SUCCESS = 0
    NEUTRAL = 1
    FAILURE = 2


def is_one_hot_encoded(mask: pt.Tensor) -> bool:
    """
    Check if a given semantic segmentation mask is one-hot encoded.

    Args:
        mask (pt.Tensor): Tensor representing the segmentation mask.
                          Can be of shape (C, H, W) for one-hot encoding
                          or (H, W) for class indices.


    Returns:
        bool: True if the mask is one-hot encoded, False otherwise.
    """
    channelled_image_size = 3
    if mask.dim() == channelled_image_size and mask.size(0) > 1:
        return True

    return False


def one_hot_decode(one_hot_mask: pt.Tensor) -> pt.Tensor:
    """
    Convert a one-hot encoded mask to a class index mask.

    Args:
        one_hot_mask (pt.Tensor): One-hot encoded mask of shape (C, H, W),
                                  where C is the number of classes.


    Returns:
        pt.Tensor: Class index mask of shape (H, W).
    """
    channelled_image_size = 3
    if one_hot_mask.dim() != channelled_image_size:
        raise ValueError("Input mask must be a 3D tensor with shape (C, H, W)")

    class_index_mask = pt.argmax(one_hot_mask, dim=0)
    return class_index_mask


def compare(source1, source2, value):
    """Create comparison images to analize semseg results."""

    def mask(image, value):
        return (image == [value, value, value]).all(axis=2)

    image1 = source1.squeeze(0).cpu().numpy()
    image1 = np.stack((image1,) * 3, axis=-1)
    image2 = source2.cpu().numpy()
    image2 = np.stack((image2,) * 3, axis=-1)

    highlighted1 = image1.copy()
    highlighted1[mask(image1, value)] = (255, 0, 0)
    image1 = highlighted1

    highlighted2 = image2.copy()
    highlighted2[mask(image2, value)] = (255, 0, 0)
    image2 = highlighted2

    image3 = (image1 == image2).astype(int) * 255
    # image3 = np.abs(image1 - image2)

    # Plot the images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image1)
    axes[0].set_title("Pred")
    axes[0].axis("off")

    axes[1].imshow(image2)
    axes[1].set_title("GT")
    axes[1].axis("off")

    axes[2].imshow(image3)
    axes[2].set_title("Comparison")
    axes[2].axis("off")

    plt.savefig(f".ignore/tests/semseg/test_comparison_{value}.jpg")
