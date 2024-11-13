"""Localizations functions for defective weight detection."""
import json
from pathlib import Path
import random
import threading
from copy import deepcopy
from multiprocessing import Manager
from multiprocessing.pool import ThreadPool
from typing import Dict, List, Optional, Tuple, Union

from repair.semseg.helpers.scoring import calculate_metrics, score_tuples_batch
from repair.semseg.models import EIARepairModel
from repair.semseg.problems import (
    EAIRepairProblem,
    ImprovementType,
    SemSegRepairProblem,
)
from repair.semseg.weights_repair import repair_weights

import torch
import tqdm

from .helpers.weight_selection import (
    compute_backward_loss,
    compute_forward_impact_filters,
    compute_forward_impact_weights,
    get_suspicious_objects_relative,
    instrument_model,
)


def fi_bl_batch(  # noqa PLR0912 "too many branches"
    batch: List,
    devices: List,
    model_copies: List[EIARepairModel],
    problem: EAIRepairProblem,
    fwd_traces: List[dict],
    scores: List[float],
    per_filter: bool,
    semaphores: List[threading.Semaphore],
    suspicious_filters: Optional[List] = None,
    ignore_missing_scores=False,
) -> Dict[str, Dict[str, Union[Tuple[float, float, float], List[int]]]]:
    """
    Compute the forward impact (FI) and gradient loss (BL) for a batch of images.

    Args:
        batch (List): Batch of data containing images, ground truth bounding boxes,
            and image identifiers.
        devices (List[torch.device]): List of devices for computation.
        model_copies (List[torch.nn.Module]): List of model copies for evaluation.
        problem (EAIRepairProblem): Problem being solved.
        fwd_traces (List[Dict]): List of dictionaries containing forward traces
            for each device.
        scores (List[float]): List of scores for each image.
        per_filter (bool): Flag indicating whether to compute per-filter FI.
        semaphores (List[threading.Semaphore]): List of semaphores for synchronization.
        suspicious_filters (Optional[List]): List of suspicious filters for BL
            computation (default is None).

    Returns:
        Dict[str, Dict[str, Union[Tuple[float, float, float], List[int]]]]: A dictionary
            containing FI and BL for each module.
            - Keys: Module names.
            - Values: Nested dictionaries with keys 'FI' and 'BL', each containing a tuple
                of FI or BL values and lists of input identifiers.
    """
    # Initialize variables for loop control and result storage
    done = False
    result = dict()

    # Continue until all devices are done processing
    while not done:
        # Iterate over devices for parallel processing
        for i in range(len(devices)):
            # Try to acquire the semaphore to access the current device
            if semaphores[i].acquire(blocking=False):
                # Iterate over images info in the batch
                for image, ground_truth, image_index, image_path in zip(batch[0], batch[1], batch[2], batch[3]):
                    # Clear the forward trace for the current device
                    # The forward tracing clearing is needed to ensure that the forward
                    # trace is appropriately reset before computing the forward impact
                    # (FI) or backward loss (BL) for each image in the batch.
                    fwd_traces[i].clear()

                    # Move image and ground truth bounding boxes to the current device
                    try:
                        image = image.to(devices[i])  # noqa PLW2901
                        images = torch.stack([image], dim=0)
                    except Exception:  # noqa BLE001
                        images = [image]

                    gt = model_copies[i].move_gt_to_device(ground_truth, devices[i])

                    prediction = model_copies[i](images)[0]
                    raw_prediction = model_copies[i](images, use_postprocessing=False)[0]
                    raw_prediction.image_path_ = Path(image_path).stem #

                    # Compute score for the current image if not already computed
                    if image_index not in scores:
                        if ignore_missing_scores:
                            # The Image does not belong to the currently selected Batch
                            continue
                        scores[image_index] = problem.default_scoring_function(
                            [prediction], [gt], devices[i]
                        )[0][1]

                    # Compute Forward Impact (FI)
                    if per_filter:
                        FIs = compute_forward_impact_filters(  # noqa N806
                            model_copies[i],
                            fwd_traces[i],
                            raw_prediction,
                            devices[i],
                            scores[image_index],
                        )
                    else:
                        FIs = compute_forward_impact_weights(  # noqa N806
                            model_copies[i],
                            fwd_traces[i],
                            raw_prediction,
                            devices[i],
                            scores[image_index],
                            suspicious_filters,
                        )

                    # Compute BL
                    BLs = compute_backward_loss(  # noqa N806
                        model_copies[i],
                        problem.default_loss_function,  # loss function
                        raw_prediction,
                        gt,
                        devices[i],
                        scores[image_index],
                        per_filter=per_filter,
                        suspicious_filters=suspicious_filters,
                    )

                    # Update the result dictionary with computed FI or BL values
                    for module_name in FIs:
                        module_fi_bl = result.setdefault(module_name, dict())
                        module_fi_bl["FI"] = tuple(
                            map(
                                sum,
                                zip(module_fi_bl.get("FI", (0, 0, 0)), FIs[module_name]),
                            )
                        )
                        module_fi_bl.setdefault("inputs_with_data_fi", list()).append(
                            image_index
                        )
                    for module_name in BLs:
                        module_fi_bl = result.setdefault(module_name, dict())
                        module_fi_bl["BL"] = tuple(
                            map(
                                sum,
                                zip(module_fi_bl.get("BL", (0, 0, 0)), BLs[module_name]),
                            )
                        )
                        module_fi_bl.setdefault("inputs_with_data_bl", list()).append(
                            image_index
                        )
                done = True
                semaphores[i].release()
                break
    return result


def random_fl(
    model: Union[EIARepairModel, torch.nn.Module],
    max_layer_amount: int,
    max_weight_amount: int = 1,
):
    """Select random weights to fix."""
    weights_to_repair = set()

    # Get a list of named modules in the model
    layers = list(model.named_modules())

    # filter for repairable layers
    layers = [layer for layer in layers if model.is_elegible_layer(layer)]

    # Randomly select layers to modify
    random_layers = []
    for _i in range(min(len(layers), max_layer_amount)):
        layer = random.choice(layers)
        while (
            (not hasattr(layer[1], "weight"))
            or (layer[1].weight is None)
            or (not isinstance(layer[1].weight.size(), torch.Size))
        ):
            layer = random.choice(layers)

        random_layers.append(layer)

    for _i in range(max(max_weight_amount, 1)):
        # Randomly select coordinates within the weight dimensions of the layer
        layer = random.choice(random_layers)
        coords = []
        for dim in layer[1].weight.size():
            coords.append(random.randint(0, dim - 1))
        weights_to_repair.add((layer[0], tuple(coords)))

    return list(weights_to_repair)


def fis_bls_filter(
    model_copies: List[EIARepairModel],
    problem: EAIRepairProblem,
    devices,
    score_tuples,
    repair_data_loader,
    per_filter=True,
    suspicious_filters=None,
    ignore_missing_scores=False,
):
    """Calculate the FI and BL for the selected targets (weights, layers)
    """
    # Set deterministic algorithms for PyTorch
    # QUESTION: Why? What does it do?
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Initialize forward traces for each model copy
    fwd_traces = [dict() for _ in model_copies]

    # Initialize instrumentation hooks
    # Instrumentation: collect forward traces for each model copy
    instrumentation_hooks = []
    for i in range(len(model_copies)):
        # Find modules to instrument (Linear and Conv2d layers)
        modules_to_instrument = list(
            filter(
                lambda x: (
                    model_copies[i].is_elegible_layer(x)
                    and isinstance(x[1], (torch.nn.Linear, torch.nn.Conv2d))
                ),
                model_copies[i].named_modules(),
            )
        )
        modules_to_instrument = [module for _, module in modules_to_instrument]

        # Instrument model and collect forward traces
        instrumentation_hooks.extend(
            instrument_model(model_copies[i], fwd_traces[i], modules_to_instrument)
        )

    # Initialize filters for Forward Impact Score (FIS)
    fis_bls_filters = dict()
    # Convert score tuples to dictionary
    # identifier: score values (FI, BL)
    scores = {identifier: value for identifier, _, value in score_tuples}

    with Manager() as manager, ThreadPool(len(devices)) as pool:
        semaphores = [manager.Semaphore() for _ in devices]

        # Iterate over repair data loader batches and apply Forward Impact (FI) and
        # Backward Loss (BL) filters
        for batch_dict in tqdm.tqdm(
            pool.imap(
                lambda batch: fi_bl_batch(
                    batch,
                    devices,
                    model_copies,
                    problem,
                    fwd_traces,
                    scores,
                    per_filter,
                    semaphores,
                    suspicious_filters=suspicious_filters,
                    ignore_missing_scores=ignore_missing_scores,
                ),
                repair_data_loader,
            ),
            desc="Filters FI and BL for each image in dataset",
            total=len(repair_data_loader),
        ):
            # Iterate over modules in the batch dictionary
            for module_name in batch_dict:
                # Initialize module's FI and BL in the filters dictionary
                module_fi_bl = fis_bls_filters.setdefault(module_name, dict())
                if "FI" in batch_dict[module_name]:
                    # Update FI and inputs_with_data_fi for the module
                    module_fi_bl["FI"] = tuple(
                        map(
                            sum,
                            zip(
                                module_fi_bl.get("FI", (0, 0, 0)),
                                batch_dict[module_name]["FI"],
                            ),
                        )
                    )
                    module_fi_bl.setdefault("inputs_with_data_fi", list()).extend(
                        batch_dict[module_name]["inputs_with_data_fi"]
                    )
                if "BL" in batch_dict[module_name]:
                    # Update BL and inputs_with_data_bl for the module
                    module_fi_bl["BL"] = tuple(
                        map(
                            sum,
                            zip(
                                module_fi_bl.get("BL", (0, 0, 0)),
                                batch_dict[module_name]["BL"],
                            ),
                        )
                    )
                    module_fi_bl.setdefault("inputs_with_data_bl", list()).extend(
                        batch_dict[module_name]["inputs_with_data_bl"]
                    )
    # Calculate suspicious filters based on FI and BL
    # this calculation is made via a pareto front
    return fis_bls_filters


def relative_2step_conv(
    model_copies: List[EIARepairModel],
    problem: EAIRepairProblem,
    devices,
    repair_data_loader,
    mode,
    return_groups=False,
    selection="pareto",
    per_layer=5,
    stop_at_grad_calculation=False,
):
    """
    Uses the best and worst-scoring Data Points to perform relative selection of Weights.

    Args:
        model_copies (List[EAIRepairModel]):
            List of Models to use for scoring the Data Points
        problem (EAIRepairProblem): Problem associated to this repair
        devices (List[device]): Model Copy Devices
        repair_data_loader (Dataloader): Dataloader containing Data Points

    Returns:
        List[String, Tuple, FI, BL] suspicious Weights
    """
    # Score positive Samples, in case we want to have a relational comparison
    score_tuples_pos = score_tuples_batch(
        model_copies, devices, problem, repair_data_loader
    )
    score_tuples_pos = score_tuples_pos[0 : len(score_tuples_pos) // 2]
    # Score negative Samples, in case we want to have a relational comparison
    score_tuples_neg = score_tuples_batch(
        model_copies, devices, problem, repair_data_loader
    )
    score_tuples_neg = score_tuples_neg[-len(score_tuples_neg) // 2 :]
    print(f"Positive Score Tuples: {len(score_tuples_pos)}")
    print(f"Negative Score Tuples: {len(score_tuples_neg)}")

    # Get FI and BL for respective Samples
    fi_bl_pos = fis_bls_filter(
        model_copies,
        problem,
        devices,
        score_tuples_pos,
        repair_data_loader,
        ignore_missing_scores=True,
    )
    if stop_at_grad_calculation:
        return
    fi_bl_neg = fis_bls_filter(
        model_copies,
        problem,
        devices,
        score_tuples_neg,
        repair_data_loader,
        ignore_missing_scores=True,
    )

    # Get Suspicious Objects and Groups for all Layers
    suspicious_filters, grouped_suspicious_objects = get_suspicious_objects_relative(
        fi_bl_pos, fi_bl_neg, mode, selection, per_layer=per_layer
    )

    # Get FI and BL for respective Samples,
    # but now using entire Weights instead of Filters for Conv2D and Linear
    fi_bl_pos = fis_bls_filter(
        model_copies,
        problem,
        devices,
        score_tuples_pos,
        repair_data_loader,
        False,
        suspicious_filters,
        ignore_missing_scores=True,
    )
    fi_bl_neg = fis_bls_filter(
        model_copies,
        problem,
        devices,
        score_tuples_neg,
        repair_data_loader,
        False,
        suspicious_filters,
        ignore_missing_scores=True,
    )


    identified_weights, grouped_suspicious_weights = get_suspicious_objects_relative(
        fi_bl_pos, fi_bl_neg, mode, selection, per_layer=per_layer
    )
    print(grouped_suspicious_weights)

    if return_groups:
        return grouped_suspicious_weights
    else:
        return identified_weights


class Filtering:
    """Wrapper Class for selecting next Weights."""

    def __init__(
        self,
        available_weights: dict,
        selected_weights: List[Tuple],
        budget: int,
        arguments: dict,
    ) -> None:
        """Initialize the Algorithm."""
        self.available_weights = available_weights
        self.selected_weights = selected_weights
        self.discarded_weights = []
        self.budget = budget
        self.arguments = arguments

    def has_budget(self):
        """Return whether the Algorithm has more budget left."""
        raise NotImplementedError

    def get_next(self):
        """Returns the next set of Weights for filtering."""
        raise NotImplementedError

    def evaluate_result(self):
        """Evaluate, whether the current set of weights had a significant impact."""
        raise NotImplementedError

    def finalize(self):
        """Finalize selected Weights."""
        raise NotImplementedError


class GroupFiltering(Filtering):
    """Wrapper Class for selecting next Weights."""

    def __init__(
        self,
        available_weights: dict,
        budget: int,
        arguments: dict,
    ) -> None:
        """Initialize the Algorithm."""
        sorted_keys = sorted(
            available_weights,
            key=lambda x: (sum([w[2] * w[3] for w in available_weights[x]]) / len(x)),
            reverse=True,
        )
        self.sorted_keys = sorted_keys
        self.all_evaluated = False
        super().__init__(available_weights, [], budget, arguments)

    def has_budget(self):
        """Return whether the Algorithm has more budget left."""
        return (
            len(self.selected_weights) < self.budget
            and bool(self.available_weights)
            and not self.all_evaluated
        )

    def get_next(self):
        """Returns the next set of Weights for filtering."""
        chosen_key = None
        # Retrieve the next Layer Candidate according to our Ordering
        for key in self.sorted_keys:
            chosen_key = key
            break

        # If chosen_key is None, we have no new layers left and will stop in has_budget
        if chosen_key is None:
            self.all_evaluated = True
            return None, None, None

        weights = self.available_weights[chosen_key]
        self.sorted_keys.remove(chosen_key)
        return key, [tuple([w[0], w[1]]) for w in weights], dict()

    def evaluate_result(
        self, key, weights, original_metric, final_metric, metric_type, modified_weights
    ):
        """Evaluate, whether the current set of weights had a significant impact."""
        # get how much percentage of the original metric the deviation represents
        deviation = final_metric - original_metric
        deviation_percentage = deviation / original_metric

        # calculate the significance based on the metric scoring type
        significant = (
            deviation_percentage > self.arguments["significance"]
            if metric_type == ImprovementType.higher_is_better
            else deviation_percentage < -self.arguments["significance"]
        )

        if significant:
            self.selected_weights.extend(weights)
        else:
            self.discarded_weights.extend(weights)
        return significant

    def finalize(self):
        """Finalize selected Weights."""
        return self.selected_weights

def filter_weights(
    model_copies: List[EIARepairModel],
    problem: EAIRepairProblem,
    devices,
    repair_data_loader,
    test_data_loader,
    weights,
    generations=3,
    population=25,
    significance=0.0005,
    experiment_name="",
    max_weights=20,
    modified_weights=None,
    filtering_mode="search",
):
    """Filters Layers by ability of Suspicious Weights to improve Score.

    Args:
        model_copies (List[EAIRepairModel]):
            List of Models to use for scoring the Data Points
        problem (EAIRepairProblem): Problem associated to this repair
        devices (List[device]): Model Copy Devices
        repair_data_loader (Dataloader): Dataloader containing Data Points
        test_data_loader (Dataloader): Dataloader containing Data Points
        weights (List[String, Tuple]): Weights to try and repair
        generations (int): Number of Generations for the Genetic Algorithm
        population (int): Number of Populations to test in each Generation
        significance (float): Minimum percentual improvement to consider the repair
            meaningful

    Returns:
        List[String, Tuple] suspicious Weights
    """
    if filtering_mode == "group":
        filtering = GroupFiltering(weights, max_weights, dict(significance=significance))
        bounds = (0, 2)
    else: # arachne weight formatting
        with open(f"./results/{experiment_name}/grouped_weights.json", "w") as handle:
            print_weights = deepcopy(weights)
            print_weights = [
                tuple([name, tuple(map(int, coord)), float(fi), float(bl)])
                for name, coord, fi, bl in print_weights
            ]
            json.dump(dict(weights=print_weights), handle)
        parsed_weights = [[w[0], w[1], w[2] * w[3]] for w in weights]
        parsed_weights = sorted(parsed_weights, key=lambda x: x[2], reverse=True)
        selected = parsed_weights[0:max_weights]
        return [[w[0], w[1]] for w in selected]

    with open(f"./results/{experiment_name}/grouped_weights.json", "w") as handle:
        print_weights = deepcopy(weights)
        for layer in print_weights:
            print_weights[layer] = [
                tuple([name, tuple(map(int, coord)), float(fi), float(bl)])
                for name, coord, fi, bl in print_weights[layer]
            ]
        json.dump(dict(weights=print_weights), handle)

    print(f"Evaluating Suspicious Objects for {len(weights)} Layers")
    with torch.no_grad():
        while filtering.has_budget():
            print(
                f"Selected Weights [{len(filtering.finalize())}]: {filtering.finalize()}"
            )
            module_name, weights_to_repair, others = filtering.get_next()
            if module_name is None or weights_to_repair is None:
                continue
            if filtering_mode == "proximity":
                modified_weights = others["modified_weights"]

            target_model = model_copies[0]

            # avoid GPU memory leaking
            filter_problem = None
            if problem == SemSegRepairProblem:
                filter_problem = SemSegRepairProblem
            torch.cuda.empty_cache()

            if modified_weights is not None:
                target_model = deepcopy(model_copies[0]).to(devices[0])
                named_modules = dict(target_model.named_modules())
                for key in modified_weights:
                    named_modules[key[0]].weight.data[key[1]] = (
                        modified_weights[key] * named_modules[key[0]].weight.data[key[1]]
                    )

            print("...")
            print(f"Repairing Suspicious Objects for {module_name}: {weights_to_repair}")

            # setup for origin model score calculation
            original_metrics = calculate_metrics(
                target_model,
                filter_problem.available_metrics,
                repair_data_loader,
                devices[0],
            )

            original_score = original_metrics[filter_problem.target_metric]

            folder = f"{experiment_name}/filtering"
            # Perform the repair process using the repair_weights function
            # this functions performs a search through the weights to repair
            repaired_model = repair_weights(
                target_model,
                weights_to_repair,
                repair_data_loader,
                filter_problem,
                "PSO",
                devices,
                bounds=bounds,
                save_history=False,
                generations=generations,
                population=population,
                folder=folder,
                name=module_name,
                retain_model=modified_weights is not None,
                modified_weights=modified_weights,
            ).to(devices[0])

            # get repaired model results (to compare with the original)
            repaired_metrics = calculate_metrics(
                repaired_model,
                filter_problem.available_metrics,
                repair_data_loader,
                devices[0],
            )
            repaired_score = repaired_metrics[filter_problem.target_metric]

            repaired_model.cpu()
            del repaired_model
            del filter_problem
            torch.cuda.empty_cache()

            significant = filtering.evaluate_result(
                module_name,
                weights_to_repair,
                original_score,
                repaired_score,
                problem.improvement_type,
                modified_weights,
            )

            print(f"Score for Module {module_name} is significant: {significant}")
            for metric, value in original_metrics.items():
                original = value
                repaired = repaired_metrics[metric]
                change_str = f"Scoring Changes: {metric} {original} => {repaired}"
                print(change_str)
            print("...")

    chosen_weights = filtering.finalize()
    discarded_weights = filtering.discarded_weights

    print(f"Selected {len(chosen_weights)} Weights!")
    print("---------------------------------------------")
    print(chosen_weights)
    print("---------------------------------------------")
    with open(f"./results/{experiment_name}/chosen_weights.json", "w") as handle:
        print_weights = [
            tuple([name, tuple(map(int, coord))]) for name, coord in chosen_weights
        ]
        json.dump(dict(weights=print_weights), handle)
    with open(f"./results/{experiment_name}/discarded_weights.json", "w") as handle:
        print_weights = [
            tuple([name, tuple(map(int, coord))]) for name, coord in discarded_weights
        ]
        json.dump(dict(weights=print_weights), handle)
    return chosen_weights
