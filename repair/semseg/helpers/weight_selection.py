"""Core functions to perform fault localization."""
import logging
import math
import os
from pathlib import Path
import pickle
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

import torch
import tqdm

logger = logging.getLogger(__name__)


def instrument_model(
    model: torch.nn.Module,
    forward_trace_dict: Dict,
    modules_to_instrument: List[torch.nn.Module],
) -> List[torch.utils.hooks.RemovableHandle]:
    """Add forward hooks to targeted layers.

    This hooks assure the gradient requirements.

    For each target module given we register a forward hook that saves the module's
    input and output in forward_trace_dict and turn requires_grad to true for the
    output. Other modules in the model have requires_grad set to false on their output

    :param model: the model to instrument
    :param forward_trace_dict: dict where the hooks will store the execution trace of
        the model
    :param modules_to_instrument: list of modules to instruments
    :return: The list of hooks added to the model
    """

    def get_activation(m_name: str, _forward_trace_dict: dict):
        def hook(m, inp, output):
            output.requires_grad_()
            output.retain_grad()
            if len(inp) == 1:
                _forward_trace_dict[m_name] = (inp[0].detach(), output)
            else:
                logger.warning(f"Unexpected number of inputs, skipping layer {m_name}")

        return hook

    def hook_freeze(m, inp, output):
        if isinstance(output, torch.Tensor) and output.is_leaf:
            output.requires_grad_(False)

    hooks = []
    for name, module in model.named_modules():
        if module in modules_to_instrument:
            hooks.append(
                module.register_forward_hook(get_activation(name, forward_trace_dict))
            )
            if hasattr(module, "weight"):
                module.weight.requires_grad = True
                module.weight.retain_grad()
        else:
            hooks.append(module.register_forward_hook(hook_freeze))
    return hooks

def save_tensors(tensor_dict, filepath):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(tensor_dict, f)
    # print(f"Saved tensor data to {filepath}")

# Function to load the tensor dictionary from a file
def load_tensors(filepath):
    with open(filepath, 'rb') as f:
        tensor_dict = pickle.load(f)
    # print(f"Loaded tensor data from {filepath}")
    return tensor_dict


def _get_output_backward(model, model_output, forward_trace_dict, device):

    model.zero_grad()
    model_output.requires_grad_()
    model_output.retain_grad()
    if model_output.grad is not None:
        model_output.grad.zero_()

    
    precomputed_grads_path = ".ignore/precomputed_grads/"
    os.makedirs(precomputed_grads_path, exist_ok=True)

    model_output_has_name = hasattr(model_output, "image_path_")

    output_name = (model_output.image_path_ if model_output_has_name else "") + ".pkl"

    output_path = precomputed_grads_path + output_name
    
    is_output_precomputed = output_name in os.listdir(precomputed_grads_path)

    model_output.backward(
        torch.ones(model_output.shape, device=device), retain_graph=True
    )

    if not is_output_precomputed:
        save_tensors(model_output.grad, output_path)
    else:
        model_output.grad = load_tensors(output_path).to(device)

def _get_FIs(module, module_name, image_name, func, trace):

    precomputed_grads_path = ".ignore/precomputed_FIs/"
    os.makedirs(precomputed_grads_path, exist_ok=True)

    output_name = f"{image_name}_{module_name}.pkl"

    output_path = precomputed_grads_path + output_name
    
    is_output_precomputed = output_name in os.listdir(precomputed_grads_path)

    if not is_output_precomputed:
        FIs = func(module, trace)
        save_tensors(FIs, output_path)
    else:
        FIs = load_tensors(output_path)
    
    return FIs

def compute_forward_impact_filters(
    model: torch.nn.Module,
    forward_trace_dict: Dict,
    model_output: torch.Tensor,
    device,
    img_score: float,
) -> dict:
    """Computes FI of the instrumented convolutional layers' filters for a given output.

    Computes the forward impact of the instrumented layers' filters by following the
    formula in https://doi.org/10.1145/3563210 for each weight of the previously
    instrumented layers.

    :param model: the model whose weights' forward impact we want to compute
    :param forward_trace_dict: dict where the execution traces of the modules are stored
    :param model_output: The output of the model for which we compute the forward impact
    :param device: device that processes the tensors
    :param FIs_BLs: Dict to save the sums of the FIs per filter, main purpose of this
        function is to modify FIs_BLS
    :param img_score: How well the model detected object in the current image
    :param lock: lock that ensures thread-safe update of FIs_BLs
    :return: The modules that activated and had a FI for this img
    """
    result = dict()

    # model.zero_grad()
    # model_output.requires_grad_()
    # model_output.retain_grad()
    # if model_output.grad is not None:
    #     model_output.grad.zero_()
    # model_output.backward(
    #     torch.ones(model_output.shape, device=device), retain_graph=True
    # )

    _get_output_backward(model, model_output, forward_trace_dict, device)

    named_modules = dict(model.named_modules())

    # For each layer that had an activation
    for module_name, trace in forward_trace_dict.items():
        module = named_modules[module_name]
        if isinstance(module, torch.nn.Conv2d):
            FIs = _get_forward_impact_conv2d_filters(module, trace)
        else:
            continue

        if FIs is None:
            continue

        FIs_squared = np.square(FIs)
        FIs_score = FIs * torch.tensor(img_score).reshape(-1, 1, 1).numpy()

        
        result[module_name] = (FIs, FIs_squared, FIs_score)

    return result


def compute_forward_impact_weights(
    model: torch.nn.Module,
    forward_trace_dict: dict,
    model_output: torch.Tensor,
    device,
    img_score: float,
    suspicious_filters: List = None,
) -> dict:
    """Compute the forward impact of the instrumented layers' weights for a given output.

    Computes the forward impact of the instrumented layers' weights by following the
    formula in https://doi.org/10.1145/3563210 for each weight of the previously
    instrumented layers.

    :param model: the model whose weights' forward impact we want to compute
    :param forward_trace_dict: dict where the execution traces of the modules are stored
    :param model_output: The output of the model for which we compute the forward impact
    :param device: device that processes the tensors
    :param img_score: How well the model detected object in the current image
    :param suspicious_filters: list of convolutional layer filters in which to compute
        the FI. All filters used if None
    :return: The modules that activated and had a FI for this img
    """
    # model.zero_grad()
    # model_output.requires_grad_()
    # model_output.retain_grad()
    # if model_output.grad is not None:
    #     model_output.grad.zero_()
    # model_output.backward(
    #     torch.ones(model_output.shape, device=device), retain_graph=True
    # )

    _get_output_backward(model, model_output, forward_trace_dict, device)


    named_modules = dict(model.named_modules())

    result = dict()
    # For each layer that had an activation
    for module_name, trace in forward_trace_dict.items():
        module = named_modules[module_name]
        if isinstance(module, torch.nn.Linear):
            FIs = _get_forward_impact_linear(module, trace)
        elif isinstance(module, torch.nn.Conv2d):
            FIs = _get_forward_impact_conv2d_weights(
                module, module_name, trace, suspicious_filters
            )
        else:
            continue


        if FIs is None:
            continue

        FIs_squared = np.square(FIs)
        FIs_score = FIs * torch.tensor(img_score).reshape(-1, 1, 1).numpy()
        # if doning classic arachne instead of correlation based scoring,
        # assigning all zeroes instead of the actual squared value and FI_score
        # can speed up experiments
        # FIs_squared = np.zeros((len(img_score), 1,1))
        # FIs_score = np.zeros((len(img_score), 1,1))

        result[module_name] = (FIs, FIs_squared, FIs_score)
    return result


def _get_forward_impact_linear(
    module: torch.nn.Linear, forward_trace: Tuple
) -> np.ndarray:
    """Compute forward impact of all weights for a Linear module.

    Follows formula in https://doi.org/10.1145/3563210 to compute a Linear layer's
    weights' forward impact. For each weight w in each neuron j (w coordinates i,j),
    computes the contribution of w to the neuron's output by multiplying its
    corresponding inputs by w and then normalise by the contribution of all weights
    in the neuron (output - bias).

    :param module: the module under consideration
    :param forward_trace: inputs and outputs of the layers, with the recorded gradients
    :return: An array of the same size as the layer's weights, containing their forward
    impact
    """
    # forward trace has the module_input and module_output for each image in the batch
    module_inputs, module_outputs = forward_trace

    if module_outputs.grad is None:
        return None
    # did not get gradient of the output for this layer so
    # can not compute forward impact.

    module_weights = module.weight.detach()  # shape (out_feat, in_feat)
    forward_impacts = np.zeros(
        tuple([module_inputs.shape[0]] + list(module_weights.shape))
    )
    # Collapse the extra dimension(s)
    neuron_output_grads = module_outputs.grad
    # shape (batch_size, out_features), it's the output gradient vector that
    # has the output gradient for each out_feature/neuron
    neuron_norms = module_outputs - module.bias
    # shape (batch_size, out_features), it's the whole averaged output that
    # reaches each out_neuron in vector form

    # each o_i * w_ij. To get the oj, we should sum them up, but we don't want to.
    # We want to do FI for each w_ij.
    each_weight_output = torch.multiply(
        module_weights, module_inputs.reshape(module_inputs.shape[0], 1, -1)
    )
    # result has shape (batch_size, out_feat, in_feat), it multiplied each row of
    # module_weights[out_feat] elementwise with mean_inputs

    normalized = torch.div(
        each_weight_output, neuron_norms.reshape(neuron_norms.shape[0], -1, 1)
    )
    # divide each_weight_output[:, i] elementwise with neuron_norms

    forward_impacts = (
        (normalized * neuron_output_grads.reshape(neuron_output_grads.shape[0], -1, 1))
        .detach()
        .cpu()
    )

    # it's important to normalize the values. We don't care if the gradient was positive
    # or negative, we just want the magnitude of the FI.
    forward_impacts = torch.abs(forward_impacts)

    # TODO remove this old slower implementation once we grow confident in the newer one
    # For each "neuron" oj, we do all the weights i,j at the same time
    # forward_impact_aux = np.zeros(module_weights.shape)
    # for j in range(module_weights.shape[0]):
    #     neuron_weights = module_weights[j, :]

    # each o_i * w_ij. To get the oj, we should sum them up, but we don't want to.
    # We want to do FI for each w_ij.
    #     weight_output = torch.multiply(mean_inputs_aux, neuron_weights)
    #     normalized_aux = weight_output / neuron_norms[j]
    #     forward_impact_aux[j, :] = (
    #           normalized_aux * neuron_output_grads[j]
    #     ).detach().cpu()
    # for j in range(module_weights.shape[0]):
    #     for i in range(module_weights.shape[1]):
    #         assert forward_impact[j][i] == forward_impact_aux[j][i]

    return forward_impacts


def _get_forward_impact_conv2d_filters(
    module: torch.nn.Conv2d, forward_trace: Tuple
) -> np.ndarray:
    """Compute forward impact of all in and out filters/features for a Conv2D module.

    Follows formula in https://doi.org/10.1145/3563210 to compute a Linear layer's
    weights' forward impact. For each filter in the module, we build a kernel that
    only has that filter, and we compute the output. This way we can know how much
    each filter affected the output compared to the other values of the kernel.
    That value is multiplied by the output gradient, which tells us how much each
    value of the layer output affects the model output.

    :param module: the module under consideration
    :param forward_trace: inputs and outputs of the layers, with the recorded gradients
    :return: An nparray with the forward impact of all filters in the module.
    """
    module_input, module_output = forward_trace
    if module_output.grad is None:
        # no gradient of the output for this layer so can not compute forward impact.
        return None

    # # <<<REMOVE>>>
    # grad_values = torch.mean(module_output.grad)
    # # <<<>>>

    module_weights = module.weight
    filters_forward_impact = np.empty(
        (module_weights.shape[0], module_weights.shape[1]), dtype=np.longfloat
    )

    # for each filter of the kernel, get an output of the conv layer that says how
    # important that filter is for the out
    only_curr_filter_krnl = torch.nn.Conv2d(
        1,
        module.out_channels,
        module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
        padding_mode=module.padding_mode,
        device=module_input.device,
        dtype=module.weight.dtype,
    )

    current_weights = torch.nn.Parameter(
        torch.zeros(
            (module.out_channels, 1, module_weights.size()[2], module_weights.size()[3]),
            device=module_input.device,
        )
    )

    module_output_no_zeros = torch.where(
        module_output == 0, sys.float_info.epsilon, module_output
    )

    with torch.no_grad():  # we don`t want to record gradients for this.
        for in_filter in range(module_weights.size()[1]):
            # compute the left part of the FI, how much effect the current filter has
            # on the output of this layer we divide the output if this was the only
            # weight in the kernel by the absolute value of the actual output
            current_weights[:, 0] = module.weight[:, in_filter]
            only_curr_filter_krnl.weight = current_weights
            current_weight_output = torch.divide(
                only_curr_filter_krnl(module_input[:, in_filter : in_filter + 1]),
                module_output_no_zeros,
            )
            # it is necessary to do in_filter:in_filter+1 instead of just in_filter to
            # avoid the dimension being squeezed

            importance = torch.mul(current_weight_output, module_output.grad)
            scalar_importance = torch.sum(torch.abs(importance), dim=(2, 3))
            filters_forward_impact[:, in_filter] = scalar_importance[0].detach().cpu()

    filters_forward_impact = np.abs(filters_forward_impact)

    # # <<<REMOVE>>>
    # return filters_forward_impact, grad_values
    # # <<<>>>
    
    return filters_forward_impact


def _get_forward_impact_conv2d_weights(
    module: torch.nn.Conv2d,
    module_name: str,
    forward_trace: Tuple,
    suspicious_filters: List = None,
) -> np.ndarray:
    """Compute forward impact of all weights for a Conv2D module.

    Follows formula in https://doi.org/10.1145/3563210 to compute a Linear layer's
    weights' forward impact. For each weight in the module, we build a kernel that
    only has that weight, and we compute the output. This way we can know how much
    each weight affected the output compared to the other values of the kernel.
    That value is multiplied by the output gradient, which tells us how much each
    value of the layer output affects the model output.

    :param module: the module under consideration
    :param forward_trace: inputs and outputs of the layers, with the recorded gradients
    :param suspicious_filters: list of filters for which to computer the FI. Computed for
    all filters if None
    :return: An array of the same size as the layer's weights, containing their forward
    impact
    """
    module_weights = module.weight
    module_input, module_output = forward_trace
    if (
        module_output.grad is None
        or suspicious_filters is not None
        and module_name not in list(map(lambda x: x[0], suspicious_filters))
    ):
        # no gradient of the output for this layer so can not compute forward impact or
        # layer's filters are not suspicious.
        return None

    forward_impact_module = np.zeros(module_weights.shape, dtype=np.longfloat)

    if suspicious_filters is not None:
        suspicious_filters_coords = list(
            map(lambda x: x[1], filter(lambda x: x[0] == module_name, suspicious_filters))
        )
    else:  # No suspicious filters passed, using all filters
        suspicious_filters_coords = [
            (out_filter, in_filter)
            for out_filter in enumerate(module_weights.shape[0])
            for in_filter in enumerate(module_weights.shape[1])
        ]

    # for each weight of the kernel, get an output of the conv layer that says how
    # important that weight is for the out
    only_curr_weight_krnl = torch.nn.Conv2d(
        1,
        module.out_channels,
        module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
        padding_mode=module.padding_mode,
        device=module_input.device,
        dtype=module.weight.dtype,
    )

    current_weights = torch.nn.Parameter(
        torch.zeros(
            (module.out_channels, 1, module_weights.size()[2], module_weights.size()[3]),
            device=module_input.device,
        )
    )

    module_output_no_zeros = torch.where(
        module_output == 0, sys.float_info.epsilon, module_output
    )

    with torch.no_grad():  # we don`t want to record gradients for this.
        for out_filter, in_filter in suspicious_filters_coords:
            for kernel_weight_i in range(module_weights.size()[2]):
                for kernel_weight_j in range(module_weights.size()[3]):
                    # compute the left part of the FI, how much effect the current weight
                    # has on the output of this layer we divide the output if this was the
                    # only weight in the kernel by the absolute value of the actual output
                    current_weights[
                        :, 0, kernel_weight_i, kernel_weight_j
                    ] = module.weight[:, in_filter, kernel_weight_i, kernel_weight_j]
                    only_curr_weight_krnl.weight = current_weights
                    current_weight_output = torch.divide(
                        only_curr_weight_krnl(module_input[:, in_filter]),
                        module_output_no_zeros,
                    )
                    # reset weights to zero for next weight pass
                    current_weights[:, 0, kernel_weight_i, kernel_weight_j] = 0

                    importance = torch.mul(
                        current_weight_output[:, out_filter],
                        module_output.grad[:, out_filter],
                    )

                    # each weights affects all pixels of the output, so we sum all the
                    # pixels importances
                    scalar_importance = torch.sum(torch.abs(importance))
                    forward_impact_module[
                        out_filter, in_filter, kernel_weight_j, kernel_weight_i
                    ] = scalar_importance

    forward_impact_module = np.abs(forward_impact_module)
    return forward_impact_module


def compute_backward_loss(
    model: torch.nn.Module,
    loss_function: Callable[[Any, Any], torch.Tensor],
    model_output,
    target_output,
    device,
    img_score: float,
    suspicious_filters: List = None,
    per_filter: bool = False,
) -> dict:
    """Compute backward loss of all weights for a module.

    Computes the loss of the model for a given output and target output (ground truth)
    and propagates the gradient of the loss back to the model's weights.

    :param model: the model under consideration
    :param loss_function: the loss function to consider. Must be differentiable .
    :param model_output: output from the model for a given input
    :param target_output: ground truth corresponding to the output
    :param device: device that processes the tensors
     with weights' back-propagated gradient for each layer in the model with a gradient
    :param img_score: How well the model detected object in the current image
    :param per_filter: whether to computer BL per filter (for convolutional layers) or
    per weight
    :param suspicious_filters: list of convolutional layer filters in which to compute
    the BL. All filters used if None
    :return: A dictionary of the BL (and squared and multiplied by score for correlation
    computation) for each module
    """
    result = dict()

    model.zero_grad()
    model_output.requires_grad_()
    model_output.retain_grad()
    if model_output.grad is not None:
        model_output.grad.zero_()
    loss = loss_function(model_output, target_output)
    loss.backward(torch.ones(loss.shape, device=device), retain_graph=True)

    for module_name, module in model.named_modules():
        if hasattr(module, "weight"):
            # TODO: check why we would have NaNs in the gradient and how to deal with them
            if (
                module.weight is not None
                and module.weight.grad is not None
                and not torch.any(torch.isnan(module.weight.grad).any())
            ):
                BLs = module.weight.grad.clone().detach().cpu().numpy()

                if isinstance(module, torch.nn.Conv2d):
                    if suspicious_filters is not None:
                        # Mask all Backwards Loss Values
                        BLs_mask = np.full_like(BLs, True, dtype=bool)
                        for _, coord, _, _ in filter(
                            lambda x: x[0] == module_name, suspicious_filters
                        ):
                            out_filter, in_filter = coord
                            BLs_mask[out_filter, in_filter, :] = False
                        BLs[BLs_mask] = 0

                    if per_filter:
                        BLs = np.mean(BLs, axis=(2, 3))

                # it's important to normalize the values. We don't care if the gradient
                # was positive or negative, we just want the magnitude of the BL.
                BLs = np.abs(BLs)

                result[module_name] = (BLs, np.square(BLs), BLs * img_score)
                # if doning classic arachne instead of correlation based scoring,
                # assigning all zeroes instead of the actual squared value and FI_score
                # can speed up experiments
                # result[module_name] = (BLs, np.zeros((1,1)), np.zeros((1,1)))

    return result


def matrix_compute_correlation(xis, xis_sqrd, xis_with_yis, yis, yis_sqrd, inputs_amount):
    """Follows the single-pass algorithm suggested in https://en.wikipedia.org/wiki/Pearson_correlation_coefficient."""
    x_std_dev = np.sqrt(xis_sqrd - (np.square(xis) / inputs_amount))
    y_std_dev = np.sqrt(yis_sqrd - (np.square(yis) / inputs_amount))
    denominator = x_std_dev * y_std_dev
    result = xis_with_yis - (xis * yis / inputs_amount)
    result = np.squeeze(result / denominator)
    x_std_dev = np.squeeze(x_std_dev)
    y_std_dev = np.squeeze(y_std_dev)
    # If FI or BL has a std dev = 0, not suspicious but we need to replace the nan that
    # dominates the pareto front
    result[x_std_dev == 0] = 0
    result[y_std_dev == 0] = 0
    return result


def is_pareto_efficient(costs, return_mask=True):
    """Find the pareto-efficient points.

    Thanks to https://stackoverflow.com/a/40239615

    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

def get_suspicious_objects_relative(
    fi_bl_data_positive: Dict,
    fi_bl_data_negative: Dict,
    mode="relative",
    selection="pareto",
    per_layer=5,
) -> List:
    """Computes Set and Group of Suspicious Objects given FI and BL.

    :param mode: Use Relative FI and BL or absolute FI and BL
    :param selection: Take "pareto" for Pareto Front
        or return top x Weights otherwise
    :param per_layer: Maximum Number of returned Weights for selection
        type != "pareto"
    :param fi_bl_data_positive: dictionary containing for each module
        a dictionary of the FI and BL of its objects for each model inp
    :param fi_bl_data_negative: dictionary containing for each module
        a dictionary of the FI and BL of its objects for each model inp
    :param input_scores: dictionary of output score for each module inp
    :return: list of suspicious object and Dict with Layerwise Groups
    """
    suspicious_objects = []
    # Collect per_layer Suspicious Weights for each Layer
    grouped_suspicious_objects = dict()

    for module_name in tqdm.tqdm(
        fi_bl_data_positive, desc="Suspicious objects for each layer."
    ):
        inputs_with_data_fi = fi_bl_data_positive[module_name].get(
            "inputs_with_data_fi", list()
        )
        inputs_amount_fi = len(inputs_with_data_fi)
        inputs_with_data_bl = fi_bl_data_positive[module_name].get(
            "inputs_with_data_bl", list()
        )
        inputs_amount_bl = len(inputs_with_data_bl)
        if inputs_amount_fi == 0 or inputs_amount_bl == 0:
            print(
                f"{module_name} had no suspiciousness for any input,",
                f"skipping: {inputs_amount_fi}; {inputs_amount_bl}",
            )
            continue

        fis_sum_pos = fi_bl_data_positive[module_name]["FI"][0]
        if isinstance(fis_sum_pos, torch.Tensor):
            fis_sum_pos = fis_sum_pos.numpy()
        bls_sum_pos = fi_bl_data_positive[module_name]["BL"][0]

        fis_sum_neg = fi_bl_data_negative[module_name]["FI"][0]
        if isinstance(fis_sum_neg, torch.Tensor):
            fis_sum_neg = fis_sum_neg.numpy()
        bls_sum_neg = fi_bl_data_negative[module_name]["BL"][0]
        fis = None
        bls = None
        if mode == "relative":
            # Compute the relative FI and BL from positive to negative as Factor
            fis = np.abs(np.nan_to_num(np.divide(fis_sum_neg, np.add(fis_sum_pos, 1))))
            bls = np.abs(np.nan_to_num(np.divide(bls_sum_neg, np.add(bls_sum_pos, 1))))
        elif mode == "absolute":
            # Compute the Sum as the deciding Factor
            fis = np.abs(fis_sum_pos + fis_sum_neg)
            bls = np.abs(bls_sum_pos + bls_sum_neg)

        if selection == "pareto":
            # Invert FI and BL, as Pareto Front Minimzes
            fis = -fis
            bls = -bls
        suspicious_objects_metrics_module = np.array([fis.flatten(), bls.flatten()]).T
        if selection == "pareto":
            # Select the Pareto Front for FI and BL as the Suspicious Weights
            front_index = is_pareto_efficient(
                suspicious_objects_metrics_module, return_mask=False
            )
        else:
            # Select the highest per_layer Weights from each Layer
            scores = [
                (fi * bl, i)
                for i, (fi, bl) in enumerate(suspicious_objects_metrics_module)
            ]
            scores = sorted(scores, key=lambda x: x[0], reverse=True)
            scores = scores[0:per_layer]
            front_index = [i for score, i in scores]

        suspicious_objects_layer = [
            (
                module_name,
                np.unravel_index(weight_index, fis.shape),
                fis[np.unravel_index(weight_index, fis.shape)],
                bls[np.unravel_index(weight_index, bls.shape)],
            )
            for weight_index in front_index
        ]

        grouped_suspicious_objects[module_name] = suspicious_objects_layer
        suspicious_objects.extend(suspicious_objects_layer)
        logger.info(
            f"Computed pareto front for {module_name},"
            f"current module has {len(front_index)} suspicious objects,"
            f"{len(suspicious_objects)} in total for all modules"
        )
    if selection == "pareto":
        # Select the Pareto Front for FI and BL as the Suspicious Weights
        front_index = is_pareto_efficient(
            np.array([[w[2], w[3]] for w in suspicious_objects]), return_mask=False
        )
    else:
        # Select all of the Suspicious Weights
        front_index = [i for i in range(len(suspicious_objects))]

    suspicious_objects = [suspicious_objects[f_i] for f_i in front_index]
    return suspicious_objects, grouped_suspicious_objects
