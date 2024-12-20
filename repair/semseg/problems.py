"""Define problems to be solved by the repair tools."""

import copy
from enum import Enum
from multiprocessing import Manager
from multiprocessing.pool import ThreadPool
from typing import Any, Callable, List, Tuple, Union
from functools import partial

import numpy

from dataset.test import (
    bb_score_imgs,
)
from repair.semseg.losses import (
    edge_loss,
    jaccard_loss,
)
from repair.semseg.helpers.scoring import (
    score_batch,
)
from repair.semseg.metrics import (
    ball_pixel_accuracy,
    class_based_iou,
    focused_pixel_accuracy,
    frequency_weighted_iou,
    high_level_metrics,
    iou,
    mean_edge_error,
    mean_relative_error,
    mean_weighted_threshold,
    pixel_accuracy,
)
from repair.semseg.models import EIARepairModel

import torch
from pymoo.core.problem import Problem
from torchvision.ops import complete_box_iou_loss
from tqdm import tqdm


class ImprovementType(Enum):
    """Enum used to indicate if a problem score improves."""

    lower_is_better = 0
    higher_is_better = 1


class EAIRepairProblem(Problem):
    """Template class used to define repairable problems.

    Repair Problems have two parts:
    - An uninstanciated class has information/methods general to the problem
        (the static methods)
    - Once the problem get instanciated, the it can be used as a general pymoo problem to
        solve search problem
    """

    # EXAMPLE, this method should be rewritten in the new problem class
    # the scoring function should return a list of tuples where the first value is an
    # enum indicating the type of result and the second value is the effective score
    default_scoring_function = bb_score_imgs
    default_loss_function = complete_box_iou_loss
    available_metrics = {
        "mre": mean_relative_error,
        "met": mean_weighted_threshold,
    }
    improvement_type = ImprovementType.higher_is_better

    def __init__(
        self,
        original_model: EIARepairModel,
        weights_coords: List[Tuple[str, int]],
        data_loader: torch.utils.data.DataLoader,
        devices: List[torch.device],
        bounds: Tuple[int, int],
        retain_model: bool,
        score_foo: Callable,
        improvement_type: ImprovementType,
        original_modules: dict = None,
    ):
        """
        Initialize EAIRepairProblem instance.

        Args:
            original_model (torch.nn.Module): The original PyTorch model.
            weights_coords (List[Tuple[str, int]]): A list of tuples containing the module
                names and the corresponding weight coordinates to be modified.
            data_loader (torch.utils.data.DataLoader): DataLoader containing the dataset.
            devices (List[torch.device]): List of devices to distribute the computation.
            score_foo (callable): A function to score the predictions.
        """
        # Get the number of decision variables required for the Problem class
        n_var = len(weights_coords)

        # Initialize the class as a Problem
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=0, xl=bounds[0], xu=bounds[1])

        # Initialize own attributes
        self.data_loader = data_loader
        self.devices = devices
        self.weights_coords = weights_coords
        self.retain_model = retain_model
        self.score_foo = score_foo

        print(f"Setting up Problem with Config retain Model: [{retain_model}]")
        # Create copies of the original model for each device
        # this means that each GPU used will have it's own copy of the models
        if not self.retain_model:
            # Store the named modules of the original model (saved as dict)
            self.original_named_modules = dict(original_model.named_modules())

            self.current_models = [
                copy.deepcopy(original_model).to(device) for device in devices
            ]

            self.current_named_modules = [
                dict(current_model.named_modules())
                for current_model in self.current_models
            ]
        else:
            assert len(self.devices) == 1
            assert original_modules is not None
            # Store the named modules of the original model (saved as dict)
            self.original_named_modules = original_modules

            self.current_models = [original_model]
            self.current_named_modules = [dict(original_model.named_modules())]

        self.invert_score = (
            True if improvement_type == ImprovementType.higher_is_better else False
        )

        self.pbar: Union[tqdm, None] = None

    # Overriding deepcopy to exclude models so that algo history does not take all GPU
    # memory
    def __deepcopy__(self, memo):
        """
        Overriding method to exclude models so algo history does not take all GPU memory.

        Args:
            memo (dict): A dictionary to keep track of objects already copied to avoid
                infinite recursion.

        Returns:
            BBoxRepairProblem: A deepcopy of the current instance with models excluded.
        """
        # Get the class of the current instance
        cls = self.__class__
        # Create a new instance of the class
        result = cls.__new__(cls)
        # Add the current instance to the memo dictionary to keep track of it
        memo[id(self)] = result

        # Iterate over the attributes of the current instance
        # Check if the attribute is not one of the models that use GPU memory
        for k, v in self.__dict__.items():
            if k == "pbar":
                setattr(result, k, self.pbar)
            elif k not in [
                "data_loader",
                "original_named_modules",
                "current_models",
                "current_named_modules",
            ]:
                # If not a model, perform a deepcopy of the attribute and set it to the
                # new instance
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                # If it's one of the models, set it to None in the new instance to exclude
                # it
                setattr(result, k, None)
        return result

    def evaluate_single_solution(self, x: Any, device_identifier: int) -> float:
        """
        Evaluate a single solution.

        Args:
            x (np.ndarray): Input candidate.
            device_identifier (int): Id to identify the device used for execution

        Returns:
            float: Evaluation score.
        """
        raise NotImplementedError

    def apply_candidate_repair(self, candidate, device_identifier):
        """
        Applies the expected changes from the candidate solution to the model.

        Current repair is: Wi + (1 + Xi),
        Where Wi is the current weight value and Xi is the current candidate output.
        """
        for w_i, w_c in enumerate(self.weights_coords):
            module_name, weight_coord = w_c

            candidate_value = candidate[w_i]

            self.current_named_modules[device_identifier][module_name].weight.data[
                weight_coord
            ] = (
                candidate_value
                * self.original_named_modules[module_name].weight.data[weight_coord]
            )

    def _evaluate_single_solution(self, x, semaphores):
        """
        Evaluate a single solution.

        As the Problem interface requires.

        Args:
            x (np.ndarray): Input candidate.
            semaphores: Semaphore to synchronize processes.

        Returns:
            float: Evaluation score.
        """
        # Initialize variables for evaluation control and score calculation
        evaluated = False

        while not evaluated:
            # Iterate over devices for parallel evaluation
            for i in range(len(self.devices)):
                # Try to acquire the semaphore to access the current device
                if semaphores[i].acquire(blocking=False):
                    # if acquired, modify the weights of the current model based on the
                    # input vector x
                    # x represents a candidate for the param search
                    score = 0.0

                    self.apply_candidate_repair(x, i)

                    # Perform inference with the modified model and compute the score
                    with torch.no_grad():
                        for batch in self.data_loader:
                            scores = score_batch(
                                batch,
                                self.current_models[i],
                                self.devices[i],
                                self.score_foo,
                            )

                            # for now only, ignore the result when evaluating solutions
                            scores = [value for _, value in scores]
                            score += sum(scores)
                    score /= len(self.data_loader.dataset)
                    # Mark evaluation completion and release the semaphore
                    evaluated = True
                    semaphores[i].release()

                    break  # Exit the loop over devices

        # Return the negative of the accumulated score (since the optimizer minimizes)
        torch.cuda.empty_cache()
        if self.pbar is not None:
            self.pbar.update()
        return -score if self.invert_score else score

    def set_pbar(self, pbar: tqdm):
        """Sets a pbar to track progress."""
        self.pbar = pbar

    def hide_pbar(self):
        """Removes the pbar without closing it."""
        self.pbar = None

    def close_pbar(self):
        """Closes and remove pbar."""
        self.pbar.close()
        self.pbar = None

    def _evaluate(self, X, out, *args, **kwargs):  # noqa N803
        """
        Evaluate multiple solutions in a concurrent way.

        Args:
            X (np.ndarray): Input vectors.
            out: Output dictionary.
        """
        # Initialize a Manager to create shared objects for synchronization among threads
        # Create a ThreadPool with the number of devices for parallel evaluation
        with Manager() as manager, ThreadPool(len(self.devices)) as pool:
            # Create semaphores to synchronize access to shared resources among threads
            # this shared resources are the GPU devices available
            semaphores = [manager.Semaphore() for _ in self.devices]

            # Evaluate solutions in parallel using ThreadPool and multiprocessing
            # Use ThreadPool's imap function to apply _evaluate_single_solution to each
            # solution in X
            # Note: imap allows for lazy evaluation, so the results are computed only when
            # needed
            # It returns an iterator that yields the results as they become available
            F = list(  # noqa N806
                pool.imap(lambda x: self._evaluate_single_solution(x, semaphores), X)
            )

            # Store the evaluation results in the output dictionary
            out["F"] = numpy.array(F)


class SemSegRepairProblem(EAIRepairProblem):
    """
    A class to evaluate the semantic segmentation problem for PyTorch DNN repair.

    This class inherits from the Problem class of the pymoo library,
    which is used for defining optimization problems.

    The semantic segmentation repair problem involves modifying weights of a
    PyTorch DNN model to improve its performance.

    Attributes:
        data_loader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        devices (List[torch.device]): List of devices to distribute the computation.
        original_named_modules (dict): Dictionary containing the original model's named
            modules.
        current_models (List[torch.nn.Module]): List of current models, each with an
            associated device.
        current_named_modules (List[dict]): List of dictionaries containing named
            modules of current models.
        weights_coords (List[Tuple[str, int]]): A list of tuples containing the module
            names and the corresponding weight coordinates to be modified.
        score_foo (callable): A function to score the semantic segmentation predictions.
    """

    default_scoring_function = class_based_iou # mIoU
    target_metric = "mIoU"
    default_loss_function = jaccard_loss
    available_metrics = {
        "mIoU": class_based_iou,
        "acc": pixel_accuracy,
    }
    improvement_type = ImprovementType.higher_is_better

    def __init__(
        self,
        original_model: torch.nn.Module,
        weights_coords: List[Tuple[str, int]],
        data_loader: torch.utils.data.DataLoader,
        devices: List[torch.device],
        bounds: Tuple[int, int] = (0, 2),
        retain_model: bool = False,
        score_foo: Callable = class_based_iou,  # noqa B008
        original_modules: dict = None,
    ):
        super().__init__(
            original_model,
            weights_coords,
            data_loader,
            devices,
            bounds,
            retain_model,
            score_foo=SemSegRepairProblem.default_scoring_function,
            improvement_type=SemSegRepairProblem.improvement_type,
            original_modules=original_modules,
        )

class DepthEstRepairProblem(EAIRepairProblem):
    """
    A class to evaluate the Depth Estimation problem for PyTorch DNN repair.

    This class inherits from the Problem class of the pymoo library,
    which is used for defining optimization problems.

    The Depth Estimation repair problem involves modifying weights of a PyTorch DNN model
    to improve its performance.

    Attributes:
        data_loader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        devices (List[torch.device]): List of devices to distribute the computation.
        original_named_modules (dict):
            Dictionary containing the original model's named modules.
        current_models (List[torch.nn.Module]):
            List of current models, each with an associated device.
        current_named_modules (List[dict]):
            List of dictionaries containing named modules of current models.
        weights_coords (List[Tuple[str, int]]):
            A list of tuples containing the module names
            and the corresponding weight coordinates to be modified.
        score_foo (callable): A function to score the Depth Estimation predictions.
    """

    metric_interval = [0, 20]
    lambda1 = 5.0
    default_loss_function = partial(
        edge_loss, metric_interval=metric_interval
    )
    target_metric = "mee"
    available_metrics = {
        "mee": partial(mean_edge_error, metric_interval=metric_interval, lambda1=lambda1),
        "mre": partial(mean_relative_error, metric_interval=metric_interval),
        "mee_full": partial(mean_edge_error, metric_interval=[0,80], use_rel=False),
        "mre_full": partial(mean_relative_error, metric_interval=[0,80]),
    }
    default_scoring_function = available_metrics[target_metric]
    improvement_type = ImprovementType.lower_is_better

    def __init__(
        self,
        original_model: torch.nn.Module,
        weights_coords: List[Tuple[str, int]],
        data_loader: torch.utils.data.DataLoader,
        devices: List[torch.device],
        bounds: Tuple[int, int] = (0, 2),
        retain_model: bool = False,
        score_foo: Callable = available_metrics[target_metric],
        original_modules: dict = None,
    ):
        super().__init__(
            original_model,
            weights_coords,
            data_loader,
            devices,
            bounds,
            retain_model,
            score_foo,
            improvement_type=DepthEstRepairProblem.improvement_type,
            original_modules=original_modules,
        )
        