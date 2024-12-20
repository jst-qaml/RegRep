"""Repairing functions."""
import datetime
import os
import pickle
import time
from copy import deepcopy
from typing import List

import numpy

from repair.semseg.helpers.scoring import score_tuples_batch
from repair.semseg.models import EIARepairModel
from repair.semseg.problems import EAIRepairProblem, ImprovementType

import dill
import torch
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.sampling import Sampling
from pymoo.termination import get_termination
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
from tqdm import tqdm


class MyOutput(Output):
    """visual output table to use while search is running."""

    def __init__(self, data_size):
        super().__init__()
        self.f_min = Column("f_min", width=17)
        self.columns += [self.f_min]
        self.data_size = data_size

    def update(self, algorithm):
        """Update the cli output after a generation."""
        super().update(algorithm)
        self.f_min.set(numpy.min(algorithm.pop.get("F")))


class FloatRandomSampling(Sampling):
    """Provide Random Initial Sampling for Genetic Algorithms."""

    def _do(self, problem, n_samples, **kwargs):
        x = numpy.random.random((n_samples, problem.n_var))

        if problem.has_bounds():
            xl, xu = problem.bounds()
            assert numpy.all(xu >= xl)
            x = xl + (xu - xl) * x

        for i in range(problem.n_var):
            # Ensure the original model is in the initial population
            x[0, i] = 1

        return x


def repair_weights(  # noqa PLR0912
    original_model: EIARepairModel,
    target_weights: List,
    dataloader: torch.utils.data.DataLoader,
    problem_type: EAIRepairProblem,
    search_method: str,
    devices: List[torch.device],
    bounds=(0, 2),
    save_history: bool = False,
    generations=100,
    population=80,
    folder="",
    name="",
    checkpoint="",
    retain_model=False,
    modified_weights=None,
    use_precomputing: bool = True,
) -> EIARepairModel:
    """
    Repairs the weights of the model based on specified target weights.

    Performs the repair by executing a search over the candidate weights.

    Args:
        original_model (torch.nn.Module): The original model to be repaired.
        target_weights (List): List of target weights to be repaired.
        dataloader (DataLoader): DataLoader for the dataset.
        problem_type (EAIRepairProblem): The type of problem being solved
        devices (List[torch.device]): List of devices (e.g., GPUs) to use for computation.
        repair_indexes: (Specify the type)
        save_history (bool, optional): Whether to save repair history. Defaults to False.
        generations (int, optional): Number of generations for the repair process.
            Defaults to 100.

    Returns:
        torch.nn.Module: Repaired model.
    """
    # prepare precomputing to save time if the model supports it
    if use_precomputing:
        original_model.enable_precomputing()

    original_modules = None
    if retain_model:
        model_copy = deepcopy(original_model).to(devices[0])
        if modified_weights is not None:
            named_modules = dict(model_copy.named_modules())
            for key in modified_weights:
                named_modules[key[0]].weight.data[key[1]] = (
                    modified_weights[key] * named_modules[key[0]].weight.data[key[1]]
                )
        original_modules = dict(model_copy.named_modules())

    # Create directory for storing results if it doesn't exist
    os.makedirs(f"./results/{folder}", exist_ok=True)  # noqa PTH103
    logs_prefix = f"./results/{folder}/repair_{name}_{int(time.time())}"

    # Open a CSV file to log repair progress
    with open(logs_prefix + ".csv", "w") as handle:
        handle.write("generation")
        for i in range(len(target_weights)):
            handle.write(",x" + str(i))
        handle.write(",fitness\n")

    # Declare the repair problem
    problem: EAIRepairProblem = problem_type(
        original_model,
        target_weights,
        dataloader,
        devices,
        bounds=bounds,
        retain_model=retain_model,
        original_modules=original_modules,
    )
    # in this instance, the search is performed by a genetic algorithm\
    # Initialize genetic algorithm
    population_size = population
    if search_method == "GA":
        algorithm = GA(
            pop_size=population_size,
            eliminate_duplicates=True,
            sampling=FloatRandomSampling(),
            output=MyOutput(len(dataloader)),
        )
    elif search_method == "PSO":
        algorithm = PSO(
            pop_size=population_size,
            sampling=FloatRandomSampling(),
            output=MyOutput(len(dataloader)),
        )
    generations = generations
    termination = get_termination("n_gen", generations)
    algorithm.setup(
        problem, termination=termination, seed=1, save_history=save_history, verbose=True
    )

    if checkpoint != "":
        with open(checkpoint, "rb") as f:
            algorithm = dill.load(f)
            print("Loaded Checkpoint:", algorithm)

    # Perform genetic algorithm (logging the process)
    with open(logs_prefix + ".csv", "a", buffering=1) as log_handle:
        # open a tqdm progress bar to monitor generational progress
        # account for individuals already evaluated from a previous checkpoint
        total_individuals_evaluated = generations * population_size
        individuals_already_evaluated = (
            (algorithm.n_gen - 1) * population_size if algorithm.n_gen is not None else 0
        )
        individuals_left_to_evaluate = (
            total_individuals_evaluated - individuals_already_evaluated
        )
        pbar = tqdm(total=individuals_left_to_evaluate, ascii=" >=")
        problem.set_pbar(pbar)

        # while algorithm has budget
        while algorithm.has_next():
            # perform the next evaluation
            algorithm.next()
            for xs, fs in zip(algorithm.pop.get("X"), algorithm.pop.get("F")):
                log_handle.write(str(algorithm.n_gen - 1))
                for x in xs:
                    log_handle.write(",")
                    log_handle.write(str(x))
                for f in fs:
                    log_handle.write(",")
                    log_handle.write(str(f))
                log_handle.write("\n")

            # save checkpoints every so often
            if (algorithm.n_gen - 1) % 5 == 0:
                # hide pbar from the dill dumper
                problem.hide_pbar()
                with open(f"{logs_prefix}_checkpoint", "wb") as f:
                    dill.dump(algorithm, f)
                # restore pbar
                problem.set_pbar(pbar)

        problem.close_pbar()
    # get the result
    result = algorithm.result()
    # remove unpickable pbars from the result
    print("Removing tqdm pbars from results")
    for result_data in result.history:
        result_data.problem.pbar = None

    # Copy original and output models
    if retain_model:
        output_model = model_copy
    else:
        output_model = deepcopy(original_model).to(devices[0])
    output_named_modules = dict(output_model.named_modules())

    # Update weights of the output model based on repair results
    # in this instance, the modified will result would be:
    # modified weight = original weight * (1 + GA result)
    if modified_weights is None:
        modified_weights = dict()

    for i, w_c in enumerate(target_weights):
        module_name, weight_coord = w_c
        modifier = result.X[i]
        if (module_name, weight_coord) not in modified_weights:
            modified_weights[module_name, weight_coord] = modifier
        else:
            current_modifier = modified_weights[module_name, weight_coord]
            modified_weights[module_name, weight_coord] = current_modifier * modifier
        print(f"Modifying {module_name} at {weight_coord} using {modifier}")
        output_named_modules[module_name].weight.data[weight_coord] = (
            result.X[i] * output_named_modules[module_name].weight.data[weight_coord]
        )

    # Save repair history if specified
    if save_history:
        with open(logs_prefix + ".zip", "bw") as handle:
            pickle.dump(
                {
                    "weights_to_repair": target_weights,
                    "generations": generations,
                    "history": result.history,
                    "algorithm": search_method,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    if use_precomputing:
        original_model.disable_precomputing()
        output_model.disable_precomputing()

    return output_model

def interval_filtering(  # noqa PLR0912
    original_model: EIARepairModel,
    target_weights: List,
    dataloader: torch.utils.data.DataLoader,
    problem_type: EAIRepairProblem,
    devices: List[torch.device],
    bounds=(0, 2),
    folder="",
    name="",
    modified_weights=None,
):
    """
    Repairs the weights of the model based on specified target weights using Interval.

    Args:
        original_model (torch.nn.Module): The original model to be repaired.
        target_weights (List): List of target weights to be repaired.
        dataloader (DataLoader): DataLoader for the dataset.
        problem_type (EAIRepairProblem): The type of problem being solved
        devices (List[torch.device]): List of devices (e.g., GPUs) to use for computation.
        repair_indexes: (Specify the type)
        save_history (bool, optional): Whether to save repair history. Defaults to False.
        generations (int, optional): Number of generations for the repair process.
            Defaults to 100.

    Returns:
        torch.nn.Module: Repaired model.
    """
    assert (
        len(target_weights) == 1
    ), "Interval_Filtering currently only supports single weights"

    model_copy = deepcopy(original_model).to(devices[0])
    if modified_weights is not None:
        named_modules = dict(model_copy.named_modules())
        for key in modified_weights:
            named_modules[key[0]].weight.data[key[1]] = (
                modified_weights[key] * named_modules[key[0]].weight.data[key[1]]
            )
    named_modules = dict(model_copy.named_modules())
    original_modules = dict(original_model.named_modules())

    # Create directory for storing results if it doesn't exist
    os.makedirs(f"./results/{folder}", exist_ok=True)  # noqa PTH103
    logs_prefix = f"./results/{folder}/repair_{name}_{int(time.time())}"

    # Open a CSV file to log repair progress
    with open(logs_prefix + ".csv", "w") as handle:
        handle.write("generation")
        for i in range(len(target_weights)):
            handle.write(",x" + str(i))
        handle.write(",fitness\n")

    # setup for origin model score calculation
    scores = score_tuples_batch(
        [model_copy],
        [devices[0]],
        problem_type,
        dataloader,
        False,
    )
    scores = [x[2] for x in scores]

    original_score = sum(scores) / len(scores)

    best = [1.0, original_score]
    budget = 10
    i = 0

    lower, middle, upper = bounds[0], 1.0, bounds[1]
    while budget > 0:
        candidates = [
            middle - (abs(middle - lower) / 2),
            middle + (abs(upper - middle) / 2),
        ]
        candidate_scores = [None, None]

        for j, v in enumerate(candidates):
            key = target_weights[0]
            named_modules[key[0]].weight.data[key[1]] = (
                float(v) * original_modules[key[0]].weight.data[key[1]]
            )

            # get repaired model results (to compare with the original)
            repaired_scores = score_tuples_batch(
                [model_copy],
                [devices[0]],
                problem_type,
                dataloader,
                False,
            )
            repaired_scores = [x[2] for x in repaired_scores]

            repaired_score = sum(repaired_scores) / len(repaired_scores)
            candidate_scores[j] = repaired_score

            with open(logs_prefix + ".csv", "a", buffering=1) as log_handle:
                log_handle.write(f"{i}, {v}, {repaired_score}\n")

            ct = datetime.datetime.now()
            print(f"[{ct}] Result for {v} is [{repaired_score}] from [{lower};{upper}]")
            print(f"[{ct}] Current Best is: {best} at {i}")

            i += 1
            budget -= 1

        # Assign new Best and new Interval Bounds
        if problem_type.improvement_type == ImprovementType.lower_is_better:
            if candidate_scores[0] < best[1] or candidate_scores[1] < best[1]:
                # Select the best of the two Candidates
                index = 0 if candidate_scores[0] < candidate_scores[1] else 1
                best = [candidates[index], candidate_scores[index]]
        else:
            # Select the best of the two Candidates
            if candidate_scores[0] > best[1] or candidate_scores[1] > best[1]:
                index = 0 if candidate_scores[0] > candidate_scores[1] else 1
                best = [candidates[index], candidate_scores[index]]

        if candidates[0] == best[0] or candidates[1] == best[0]:
            # One of the new Values is better than the previous
            if candidates[0] == best[0]:
                # Select the strict lower Range [lower;middle]
                lower, middle, upper = lower, candidates[0], middle
            else:
                # Select the strict higher Range [middle;upper]
                lower, middle, upper = middle, candidates[1], upper
        else:
            # Neither is better, just move towards the better part
            if problem_type.improvement_type == ImprovementType.lower_is_better:
                if candidate_scores[0] < candidate_scores[1]:
                    # Select the conservative Lower Range [Lower;candidate[1]]
                    lower, middle, upper = (
                        lower,
                        lower + ((candidates[1] - lower) / 2),
                        candidates[1],
                    )
                else:
                    # Select the conservative Higher Range [candidate[0];Upper]
                    lower, middle, upper = (
                        candidates[0],
                        candidates[0] + ((upper - candidates[0]) / 2),
                        upper,
                    )
            else:
                if candidate_scores[0] > candidate_scores[1]:
                    # Select the conservative Lower Range [Lower;candidate[1]]
                    lower, middle, upper = (
                        lower,
                        lower + ((candidates[1] - lower) / 2),
                        candidates[1],
                    )
                else:
                    # Select the conservative Higher Range [candidate[0];Upper]
                    lower, middle, upper = (
                        candidates[0],
                        candidates[0] + ((upper - candidates[0]) / 2),
                        upper,
                    )

    if modified_weights is None:
        modified_weights = dict()

    for w_c in target_weights:
        module_name, weight_coord = w_c
        modifier = best[0]
        if (module_name, weight_coord) not in modified_weights:
            modified_weights[module_name, weight_coord] = modifier
        else:
            current_modifier = modified_weights[module_name, weight_coord]
            modified_weights[module_name, weight_coord] = current_modifier * modifier
        print(f"Modifying {module_name} at {weight_coord} using {modifier}")
        named_modules[module_name].weight.data[weight_coord] = (
            best[0] * original_modules[module_name].weight.data[weight_coord]
        )

    return model_copy
