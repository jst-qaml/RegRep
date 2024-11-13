"""Loss Functions for Models."""
from repair.semseg.metrics import jaccard_index

import torch
import torch.nn.functional as F  # noqa N812


def jaccard_loss(y_pred: torch.Tensor, y_true: torch.Tensor, smooth=1e-6):
    """
    Compute the Jaccard loss, also known as the Intersection over Union (IoU) loss.

    Args:
        y_pred (torch.Tensor): Predicted tensor of shape (N, C, H, W) or (N, H, W).
        y_true (torch.Tensor): Ground truth tensor of shape (N, C, H, W) or (N, H, W).
        smooth (float): Smoothing value to avoid division by zero.

    Returns:
        torch.Tensor: Computed Jaccard loss.
    """
    # adapt true shape to predict shape
    if (
        y_pred.shape != y_true.shape
        and len(y_pred.shape) == 3  # noqa PLR2004
        and y_pred.shape[0] > 1
    ):
        y_true = F.one_hot(y_true.long(), y_pred.shape[0]).permute(2, 0, 1).float()

    # Compute the Jaccard index
    jaccard_index_ = jaccard_index(y_pred, y_true, one_hot=True, return_tensor=True)

    # Jaccard loss is 1 - Jaccard index
    return 1 - jaccard_index_
