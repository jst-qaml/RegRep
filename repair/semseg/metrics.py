"""Metrics used for the pytorch framework."""

from enum import Enum
from functools import partial
from typing import Any, Callable, List, Tuple

import torch as pt

class Result(Enum):
    """Result Declaration for Scoring Functions."""

    SUCCESS = 0
    NEUTRAL = 1
    FAILURE = 2

# core functions


def evaluate_test_result(
    value: Any, criteria_success: Callable, criteria_failure: Callable
):
    """Evaluate the passed and/or fail condition for a test value."""
    if criteria_success(value):
        test_result = Result.SUCCESS
    elif criteria_failure(value):
        test_result = Result.FAILURE
    else:
        test_result = Result.NEUTRAL

    return test_result


def run_batch_scoring_function(
    predicted_masks_list: List[pt.Tensor],
    ground_truth_masks_list: List[pt.Tensor],
    device: str,
    scoring_function: Callable,
    upper_threshold: float,
    lower_threshold: float,
    **kwargs,
) -> List[Tuple[Result, float]]:
    """
    Compute mean IoU for lists of predicted and ground truth masks.

    Args:
        predicted_masks_list (List[pt.Tensor]): List of predicted masks, each of
            shape (H, W).
        ground_truth_masks_list (List[pt.Tensor]): List of ground truth masks, each of
            shape (H, W).
        device: The device to move tensors to (e.g., 'cuda' or 'cpu').


    Returns:
        List[float]: List of mean IoU scores for each pair of masks.
    """
    score_values = [
        scoring_function(pred_mask.to(device), gt_mask.to(device), **kwargs)
        for pred_mask, gt_mask in zip(predicted_masks_list, ground_truth_masks_list)
    ]

    def success_criteria(x):
        return x >= upper_threshold

    def fail_criteria(x):
        return x < lower_threshold

    test_results = [
        evaluate_test_result(score, success_criteria, fail_criteria)
        for score in score_values
    ]

    return list(zip(test_results, score_values))


# semseg scoring functions

def jaccard_index(y_pred: pt.Tensor, y_true: pt.Tensor, smooth=1e-6, one_hot=False, return_tensor=False):
    """
    Compute the Jaccard loss, also known as the Intersection over Union (IoU) loss.

    Args:
        y_pred (torch.Tensor): Predicted tensor of shape (N, C, H, W) or (N, H, W).
        y_true (torch.Tensor): Ground truth tensor of shape (N, C, H, W) or (N, H, W).
        smooth (float): Smoothing value to avoid division by zero.

    Returns:
        torch.Tensor: Computed Jaccard loss.
    """

    if not one_hot:
        y_true = pt.nn.functional.one_hot(y_true.long(), 175).permute(2, 0, 1).float()
        y_pred = pt.nn.functional.one_hot(y_pred.long(), 175).permute(2, 0, 1).float()
    else:
        y_pred = pt.nn.functional.softmax(y_pred,dim=0)
    
    y_pred = y_pred.contiguous()
    y_true = y_true.contiguous()

    # Calculate intersection and union
    intersection = (y_pred * y_true).sum(dim=(-2, -1))
    sum_ = y_true.sum(dim=(-2, -1)) + y_pred.sum(dim=(-2, -1))
    union = sum_ - intersection
    # Compute the Jaccard index
    jaccard_index = (intersection + smooth) / (union + smooth)

    jaccard_index_mean = jaccard_index.mean()

    return jaccard_index_mean if return_tensor else jaccard_index_mean.item()


def per_class_iou(
    pred_mask: pt.Tensor, gt_mask: pt.Tensor, class_weights: List[float] = None
) -> Tuple[Result, float]:
    """
    mean class IoU for a single pair of predicted and ground truth with class indices.

    This IoU is calculated per class, instead of the whole image

    Args:
        pred_mask (pt.Tensor): Predicted mask of shape (H, W) with class indices.
        gt_mask (pt.Tensor): Ground truth mask of shape (H, W) with class indices.
        class_weights: (List[float]): Optional weights for classes.
            Should be of size equal to num_classes.
            the sum of all weights should be 1.


    Returns:
        float: Mean class IoU score.
    """
    mean_iou = 0.0

    ious = []
    pred_classes = pt.unique(pred_mask).tolist()
    gt_classes = pt.unique(gt_mask).tolist()
    all_classes = set(pred_classes + gt_classes)
    compared_classes = []

    image_pixels = gt_mask.numel()

    for cls in all_classes:
        pred_cls = pred_mask == cls
        gt_cls = gt_mask == cls

        intersection = (pred_cls & gt_cls).float().sum()
        union = (pred_cls | gt_cls).float().sum()

        # # if the union greater than 1% of the entire image
        if union > image_pixels * 0.01:
            compared_classes.append(cls)
            iou = (intersection / union).item()
            ious.append(iou)

    # adjust weights to scale with compared classes
    # weights must continue to sum 1 and be proportional with the original scale
    if class_weights is not None:
        used_weights = [class_weights[cls] for cls in compared_classes]
        adjust_proportion = 1 / sum(used_weights)
        adjusted_weights = [weight * adjust_proportion for weight in used_weights]
        ious = [ious[i] * adjusted_weights[i] for i in range(len(ious))]

    if len(ious) > 0:
        mean_iou = sum(ious) / len(ious)
    else:
        mean_iou = 0

    # assure mean iou is float
    mean_iou = mean_iou if not isinstance(mean_iou, pt.Tensor) else mean_iou.item()

    return mean_iou


def fwiou(preds: pt.Tensor, labels: pt.Tensor):
    """
    Frequency Weighted IoU (FWIoU) for multiclass semantic segmentation.

    Args:
        preds (torch.Tensor): Predicted segmentation masks of shape (H, W)
        labels (torch.Tensor): Ground truth segmentation masks of shape (H, W)

    Returns:
        float: Frequency Weighted IoU
    """
    fwiou = 0
    total_pixels = labels.numel()

    pred_classes = pt.unique(preds).tolist()
    gt_classes = pt.unique(labels).tolist()
    all_classes = set(pred_classes + gt_classes)

    for cls in all_classes:
        pred_class = (preds == cls).int()
        label_class = (labels == cls).int()

        intersection = pt.sum(pred_class & label_class).float()
        union = pt.sum(pred_class | label_class).float()
        class_pixels = pt.sum(label_class).float()

        if union != 0:
            fwiou += (class_pixels / total_pixels) * (intersection / union)

    fwiou = fwiou if not isinstance(fwiou, pt.Tensor) else fwiou.item()

    return fwiou


def pixel_wise_accuracy(preds: pt.Tensor, labels: pt.Tensor):
    """
    Calculate the overall accuracy for multiclass semantic segmentation.

    Args:
        preds (torch.Tensor): Predicted segmentation masks of shape (H, W)
        labels (torch.Tensor): Ground truth segmentation masks of shape (H, W)

    Returns:
        float: Overall accuracy
    """
    correct_pixels = pt.sum(preds == labels).item()
    total_pixels = labels.numel()

    accuracy = correct_pixels / total_pixels
    return accuracy

SEMSEG_METRICS = []

class_based_iou = partial(
    run_batch_scoring_function,
    scoring_function=per_class_iou,
    upper_threshold=0.7,
    lower_threshold=0.2,
)
SEMSEG_METRICS.append(class_based_iou)

pixel_accuracy = partial(
    run_batch_scoring_function,
    scoring_function=pixel_wise_accuracy,
    upper_threshold=0.7,
    lower_threshold=0.2,
)
SEMSEG_METRICS.append(pixel_accuracy)