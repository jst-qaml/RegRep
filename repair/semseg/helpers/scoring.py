"""Helper functions for scoring."""
import threading
from multiprocessing import Manager
from multiprocessing.pool import ThreadPool
from typing import Callable, Dict, List, Tuple

from dataset.test import Result
from repair.semseg.models import EIARepairModel

import torch
import tqdm


def predict_batch(
    b,  # list of tuple of data, gt
    model: EIARepairModel,
    device: str,
    use_postprocessing: bool=True,
) -> torch.Tensor:
    """
    Run model prediction over a batch.

    Moves batch to device
    """
    images = b[0]
    # try to move and stack data
    try:
        images = [image.to(device) for image in images]
        images = torch.stack(images)

    # else keep data as is
    except Exception:
        images = b[0]

    # move gt to device
    # the move is made in place to avoid ruff alerts
    for j, gt in enumerate(b[1]):
        b[1][j] = model.move_gt_to_device(gt, device)

    # Generate predictions using the current model
    predictions = model(images, use_postprocessing=use_postprocessing)

    return predictions


def score_batch(
    b: List,
    model: EIARepairModel,
    device,
    score_foo: Callable,
):
    """
    Score a batch of predictions.

    Args:
        b (List): Batch of data containing images, ground truth bounding boxes,
            and identifiers.
        devices (List[torch.device]): List of devices for computation.
        model_copies (List[torch.nn.Module]): List of model copies for
            evaluation.
        semaphores (List[threading.Semaphore]): List of semaphores for
            synchronization.
        score_foo (callable): Function to score the problem
            This function should return two values,
            One is if the result is deemed a success, a failure or a neutral example
            The second one is the score value, generally a float

    Returns:
        List[Tuple, float]: A list of tuples containing identifiers and corresponding
            scores.
    """
    with torch.no_grad():
        predictions = predict_batch(b, model, device)

        # Score the predictions using the specified scoring function
        scores = list(
            score_foo(predictions, b[1], device),
        )

        return scores


def score_batch_parallel(
    devices,
    b: List,
    model_copies: List[EIARepairModel],
    semaphores: List[threading.Semaphore],
    score_foo,
) -> List[Tuple[int, Result, float]]:
    """
    Score a batch of predictions.

    Args:
        b (List): Batch of data containing images, ground truth,
            and identifiers.
        devices (List[torch.device]): List of devices for computation.
        model_copies (List[torch.nn.Module]): List of model copies for
            evaluation.
        semaphores (List[threading.Semaphore]): List of semaphores for
            synchronization.
        score_foo (callable): Function to score the bounding box predictions
            (default is bb_score_imgs).

    Returns:
        List[Tuple[identifier, Result ,float]: A list of tuples containing identifiers
            and corresponding scores.
    """
    scored = False
    while not scored:
        for i in range(len(devices)):
            # Try to acquire the semaphore to access the current device
            if semaphores[i].acquire(blocking=False):
                scores = score_batch(b, model_copies[i], devices[i], score_foo)

                # Release the semaphore after evaluation
                semaphores[i].release()
                # Mark the batch as scored and exit the loop
                scored = True
                break

    # Return the zipped list of identifiers and corresponding scores

    zipped_scores = []
    for i in range(len(b[2])):
        zipped_scores.append((b[2][i], scores[i][0], scores[i][1]))

    return zipped_scores


def score_tuples_batch(model_copies, devices, problem, repair_data_loader, sort=True):
    """
    Score Data Points in repair_data_loader using the problem scoring function.

    Args:
        model_copies (List[EAIRepairModel]):
            List of Models to use for scoring the Data Points
        devices (List[device]): Model Copy Devices
        problem (EAIRepairProblem): Problem associated to this repair
        repair_data_loader (Dataloader): Dataloader containing Data Points

    Returns:
        List[Tuple[Identifier, Result, Score]]
    """
    # prepare scoring for fl if required
    with Manager() as manager, ThreadPool(len(devices)) as pool:
        # Create semaphores for each device
        semaphores = [manager.Semaphore() for _ in devices]

        # Calculate scores for batches of repair data using ThreadPool
        score_tuples = list(
            tqdm.tqdm(
                pool.imap_unordered(
                    lambda b: score_batch_parallel(
                        devices,
                        b,
                        model_copies,
                        semaphores,
                        problem.default_scoring_function,
                    ),
                    repair_data_loader,
                ),
                desc="Calculating scores to sample repair data",
                total=len(repair_data_loader),
            )
        )
        # Flatten the list of score tuples
        score_tuples = [item for sublist in score_tuples for item in sublist]
        # score_tuples is now a list of (identifier, Result, value)

    if sort:
        # Sort score tuples based on the second element (score)
        score_tuples = sorted(score_tuples, key=lambda x: x[2])
    return score_tuples


def calculate_metrics(
    model: EIARepairModel,
    metrics: Dict[str, Callable],
    data_loader: torch.utils.data.DataLoader,
    device: str,
) -> Dict[str, float]:
    """Calculates the avarage for all metrics of the model over the given dataset."""
    metrics_results = {}

    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="Computing metrics"):
            predictions = predict_batch(batch, model, device)
            ground_truth = batch[1]

            for metric_name, scoring_func in metrics.items():
                scores = list(
                    map(
                        lambda x: x[1],
                        scoring_func(predictions, ground_truth, device),
                    )
                )

                if metric_name not in metrics_results:
                    metrics_results[metric_name] = []
                metrics_results[metric_name] += scores

    for metric, values in metrics_results.items():
        average = sum(values) / len(values)
        metrics_results[metric] = average

    return metrics_results
