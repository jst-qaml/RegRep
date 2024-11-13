"""CLI for pytorch repair."""
import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Literal, Union

from repair.semseg.helpers.scoring import (
    calculate_metrics,
)
from repair.semseg.weight_selection import (
    filter_weights,
    random_fl,
    relative_2step_conv,
)
from repair.semseg.models import (
    EIARepairModel,
    SemsegModel,
)
from repair.semseg.problems import (
    EAIRepairProblem,
    SemSegRepairProblem,
)
from repair.semseg.weights_repair import repair_weights

import torch


def main(  # noqa PLR0912
    model: EIARepairModel,
    problem: EAIRepairProblem,
    fl_method: Literal["arachne", "random", "filtering"],
    search_method: Literal["GA", "PSO"],
    device: Union[str, None],
    repair_data_path: str,
    test_data_path: str,
    batch_size: int,
    generations: int,
    max_population: int,
    xl: int,
    xu: int,
    max_weights: int,
    checkpoint: str,
    use_precomputation: bool,
    mode: str,
    significance: float = 0.001,
    weights_per_layer: int = 10,
    stop_at_fl: bool = False,
):
    """Main repair function."""

    repair_type = f"I{xu-xl}W{max_weights}_{fl_method}{mode}_fl_{search_method}"
    hyperparameters = (
        f"_rep_g{generations}_p{max_population}_{int(time.time())}"
    )
    experiment_name = repair_type + hyperparameters
    Path.mkdir(Path(f"./results/{experiment_name}"), exist_ok=True, parents=True)
    print(f"Executing run for {experiment_name}")

    print("Loading devices")
    if device is None:
        # use all avaialble devices
        devices = []
        # Check if GPU is available
        if torch.cuda.is_available():
            # Create a list of only available CUDA devices
            devices = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
        else:
            # Append CPU device to the list of devices
            devices.append(torch.device("cpu"))
    else:
        # use specified device
        devices = [torch.device(device)]

    print(f"Selected devices: {devices}")

    if len(devices) > 1:
        model_copies = [deepcopy(model).to(device) for device in devices]
    else:
        model_copies = [model.to(devices[0])]

    for m in model_copies:
        m.eval()

    # get model compatible dataset loaders
    print("Generating dataloaders")
    repair_data_loader, test_data_loader = model.get_repair_and_test_dataloaders(
        repair_data_path, test_data_path, batch_size=batch_size
    )
    print(model)

    print(f"repairing with {len(repair_data_loader.dataset)} images")

    print(f"Starting {fl_method} Weight Selection")

    if checkpoint == "":
        if fl_method == "filtering":
            # weight scoring
            suspicious_objects = relative_2step_conv(
                model_copies,
                problem,
                devices,
                repair_data_loader,
                mode="relative",
                return_groups=True,
                selection="top",
                per_layer=weights_per_layer,
                stop_at_grad_calculation=stop_at_fl,
            )
            if stop_at_fl:
                return
            # weight filtering
            weights_to_repair = filter_weights(
                model_copies,
                problem,
                devices,
                repair_data_loader,
                test_data_loader,
                suspicious_objects,
                significance=significance,
                experiment_name=experiment_name,
                max_weights=max_weights,
                filtering_mode=mode,
            )
            population = max_population
        elif fl_method == "arachne":
            # gather weights with the pareto front
            suspicious_objects = relative_2step_conv(
                model_copies,
                problem,
                devices,
                repair_data_loader,
                mode="relative",
                return_groups=False,
                selection="pareto",
                per_layer=weights_per_layer,
                stop_at_grad_calculation=stop_at_fl,
            )
            if stop_at_fl:
                return
            # filtering arachne weights just format the chosen weights
            # as needed for the next steps
            weights_to_repair = filter_weights(
                model_copies,
                problem,
                devices,
                repair_data_loader,
                test_data_loader,
                suspicious_objects,
                experiment_name=experiment_name,
                max_weights=max_weights,
                filtering_mode="arachne",
            )
            population = max_population
        else:  # random
            weights_to_repair = random_fl(model_copies[0], max_layer_amount=12)
    else:
        weights_to_repair = []
        population = max_population

    print("SUSPICIOUS WEIGHTS")
    print(weights_to_repair)

    # get all problem metrics
    # these metrics then are used to compare the old model with the new one
    if checkpoint == "":
        original_metrics = calculate_metrics(
            model_copies[0],
            problem.available_metrics,
            test_data_loader,
            devices[0],
        )
        # Save it to a JSON file
        with open(f"./results/{experiment_name}/original_metrics.json", "w") as file:
            json.dump(original_metrics, file)
    else:
        json_path = Path(checkpoint).parent / "original_metrics.json"
        with open(json_path) as file:
            original_metrics = dict(json.load(file))
        # Save it to a JSON file
        with open(f"./results/{experiment_name}/original_metrics.json", "w") as file:
            json.dump(original_metrics, file)

    for metric, value in original_metrics.items():
        print(f"Original {metric}: {value}")

    # Perform the repair process using the repair_weights function
    # this functions performs a search through the weights to repair
    print("Performing model repair over the suspected weights")
    repaired_model = repair_weights(
        model,
        weights_to_repair,
        repair_data_loader,
        problem,
        search_method,
        devices,
        bounds=(xl, xu),
        save_history=True,
        generations=generations,
        population=population,
        folder=experiment_name,
        name=experiment_name,
        checkpoint=checkpoint,
        use_precomputing=use_precomputation,
    ).to(devices[0])

    # get repaired model results (to compare with the original)
    repaired_metrics = calculate_metrics(
        repaired_model,
        problem.available_metrics,
        test_data_loader,
        devices[0],
    )

    for metric, value in original_metrics.items():
        print(f"Original {metric}: {value}")
        print(f"Repaired {metric}: {repaired_metrics[metric]}")

    # log/report results
    logs_prefix = (
        f"./results/{experiment_name}/repair_{experiment_name}_{int(time.time())}"
    )
    with open(logs_prefix + ".txt", "w") as results_f:
        for metric, value in original_metrics.items():
            results_f.write(f"Original {metric}: {value}")
            results_f.write(f"Repaired {metric}: {repaired_metrics[metric]}")

    with open(logs_prefix + ".txt", "a") as results_f:
        results_f.write("repaired weights")
        results_f.write(str(weights_to_repair))

    # Debugging
    with open(f"{logs_prefix}{experiment_name}ModelWeights.txt", "a") as results_f:
        original_model = model
        test_model = model_copies[0]
        results_f.write(f"Original Model: {original_model.state_dict()} \n")
        results_f.write(f"Original Model: {test_model.state_dict()} \n")
        results_f.write(f"Original Model: {repaired_model.state_dict()} \n")

    print("Saving Model!")
    fpath = f"./results/{experiment_name}/{experiment_name}_ckpt.pth"

    repaired_model.save_inner_model(fpath)

    return experiment_name, original_metrics, repaired_metrics


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description="""Pytorch repair script.""")
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the model being repaired."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to a trained model checkpoint.",
    )
    parser.add_argument(
        "--train_data_path",
        default="",
        type=str,
        required=False,
        help="Path to the training data set.",
    )
    parser.add_argument(
        "--repair_data_path",
        default="",
        type=str,
        required=False,
        help="Path to the repairing data set.",
    )
    parser.add_argument(
        "--test_data_path",
        default="",
        type=str,
        required=False,
        help="Path to the test data set.",
    )
    parser.add_argument(
        "--fl_method",
        type=str,
        required=True,
        help="Fault Localization method used (currently: random, 2step_conv, filtering)",
    )
    parser.add_argument(
        "--search_method",
        type=str,
        required=True,
        default="GA",
        help="Fault Localization method used (currently: random, 2step_conv, filtering)",
    )
    parser.add_argument(
        "--problem",
        type=str,
        required=True,
        help="Problem being solved (currently: semseg, bbox)",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default=None,
        help="Device to use, None means use all available",
    )
    parser.add_argument(
        "--batch",
        type=int,
        required=False,
        default=2,
        help="Batch size",
    )
    parser.add_argument(
        "--generations",
        type=int,
        required=False,
        default=100,
        help="How many generations to evaluate in the GA",
    )
    parser.add_argument(
        "--max_population",
        type=int,
        required=False,
        default=40,
        help="How big of a population to use in each GA generation",
    )
    parser.add_argument(
        "--xl",
        type=int,
        required=False,
        default=0,
        help="Lower Bound for Problem Variables",
    )
    parser.add_argument(
        "--xu",
        type=int,
        required=False,
        default=2,
        help="Upper Bound for Problem Variables",
    )
    parser.add_argument(
        "--max_weights",
        type=int,
        required=False,
        default=30,
        help="Max Number of Weights to Repair",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default="",
        help="Checkpoint for the Repair Process",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        default="",
        help="Mode for executing Filtering",
    )
    parser.add_argument(
        "--significance",
        type=float,
        required=False,
        default=0.001,
        help="Significance to evaluate improvement on the Filtering process",
    )
    parser.add_argument("--use_precompute", dest="use_precompute", action="store_true")
    parser.add_argument("--no_precompute", dest="use_precompute", action="store_false")
    parser.set_defaults(use_precompute=True)
    args = parser.parse_args()

    # define the model CFG
    # used for meta model definitions, such as:
    # model loading
    # dataloaders
    if args.model == "semsegmodel":
        model = SemsegModel(
            args.model_path,
            args.device,
        )
    else:
        raise Exception(f"Invalid model: {args.model}")  # noqa TRY002

    # define the problem
    # a problem include some predefinitions by itself as a class
    # such as how to score batches for that problem or how to handle the output data
    # but also once initiated can be used to effectively solve the repair problem
    if args.problem == "semseg":
        problem = SemSegRepairProblem
    else:
        raise Exception(f"Invalid problem: {args.problem}")  # noqa TRY002

    # define the desired fl method
    fl_method = args.fl_method
    if fl_method not in ["random", "arachne", "filtering"]:
        raise Exception(f"Invalid fl_method: {fl_method}")  # noqa TRY002

    # define the desired search method
    search_method = args.search_method
    if search_method not in ["GA", "PSO"]:
        raise Exception(f"Invalid search_method: {search_method}")  # noqa TRY002

    # check that the device is available
    device = args.device
    available_devices = ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if device is not None and device not in available_devices:
        raise Exception(f"Device not available: {device}")  # noqa TRY002
    
    main(
        model,
        problem,
        fl_method,
        search_method,
        device,
        args.repair_data_path,
        args.test_data_path,
        args.batch,
        args.generations,
        args.max_population,
        args.xl,
        args.xu,
        args.max_weights,
        args.checkpoint,
        args.use_precompute,
        args.mode,
        args.significance,
    )
