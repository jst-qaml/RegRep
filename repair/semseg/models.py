"""Model config definition."""

import copy
import functools
import os
from typing import Any, Callable, List, Tuple

from repair.semseg.precompute import PrecomputeDataloader

import torch
# from mmseg.apis import inference_model, init_model
from torch.utils.data import DataLoader


class EIARepairModel:
    """Wrapper class for unifying interface and usage of torch modules.
    
    This class is just an example, please build your own adapted to your model.
    """

    # public interface
    # may require override by the the subclass
    def __init__(
        self,
        model_path: str,
    ) -> None:
        """Sample of how to init an EAIRepairModel.

        Please build your own custom init function
        """
        # initialize variables

        # load model
        torch_model: torch.nn.Module = None

        checkpoint = torch.load(model_path)
        torch_model.load_state_dict(checkpoint["model_state_dict"])
        self.inner_model: torch.nn.Module = torch_model
        self.precomputing_enabled = False

    def get_repair_and_test_dataloaders(
        self,
        repair_data_path: str,
        test_data_path: str,
        batch_size: int = 2,
        transforms: List[Callable] = [],  # noqa B006
        shuffle: bool = True,
        collate_function: Callable = collate_fn,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create the repair and test datasets."""
        # separate repair data path to data dir and target
        path = repair_data_path
        repair_data_dir = os.path.dirname(path)  # noqa PTH 120
        repair_target = os.path.basename(path)  # noqa PTH 119

        repair_dataset = load_dataset(
            repair_data_dir,
            repair_target,
            transforms=transforms,
        )

        repair_data_loader = torch.utils.data.DataLoader(
            repair_dataset.inner_dataset,
            batch_size=batch_size,
            collate_fn=collate_function,
            shuffle=shuffle,
        )

        ## same for test
        path = test_data_path
        test_data_dir = os.path.dirname(path)  # noqa PTH120
        test_target = os.path.basename(path)  # noqa PTH119

        test_dataset = load_dataset(
            test_data_dir,
            test_target,
            transforms=transforms,
        )

        test_data_loader = torch.utils.data.DataLoader(
            test_dataset.inner_dataset,
            batch_size=batch_size,
            collate_fn=collate_function,
            shuffle=shuffle,
        )

        return repair_data_loader, test_data_loader

    def predict(self, data: torch.Tensor) -> Any:
        """Run the model over the given data.

        This methods should make all necessaries pre and post processing
        """
        return self.inner_model(data)

    def move_gt_to_device(self, gt, device: str):
        """Move ground truth data to device."""
        return gt.to(device)

    def deepcopy_inner_model(self, memo):
        """Properly copy the inner model of the wrapper."""
        return copy.deepcopy(self.inner_model)

    def is_elegible_layer(self, layer: Tuple[str, torch.nn.Module]) -> bool:
        """Checks wether a layer is elegible to be repaired."""
        return True

    def save_inner_model(self, saving_path: str):
        """Save the inner model for future loading."""
        torch.save(
            {
                "model": self.inner_model.state_dict(),
                "optimizer": None,
            },
            saving_path,
        )

    # private interface
    def __call__(self, data: torch.Tensor, use_postprocessing=True) -> Any:
        """Override forward/call function to unify api to torch modules."""
        return self.predict(data, use_postprocessing)

    def enable_precomputing(self) -> None:
        """Enable precomputing is the model is compaatible."""
        if self.is_inner_model_precomputing_compatible():
            precomputer = PrecomputeDataloader(
                ".results/precomputed_outputs", self.inner_model
            )
            self.inner_model.enable_precomputing(precomputer)
            self.precomputing_enabled = True
            print("Precomputing enabled")
        else:
            print("Precomputing not available")

    def disable_precomputing(self):
        """Disables precomputing."""
        if self.is_inner_model_precomputing_compatible():
            self.inner_model.disable_precomputing()
            self.precomputing_enabled = False
            print("Precomputing disabled")

    def is_inner_model_precomputing_compatible(self) -> bool:
        """Indicates if the inner model has the ability to precompute."""
        return hasattr(self.inner_model, "enable_precomputing")

    def to(self, device):
        """Move inner model to the specified device."""
        self.inner_model = self.inner_model.to(device)
        return self

    def __getattr__(self, name):
        """Overrides attributes not assigned to the Model with those of the inner model.

        This allows to retrieve arguments like named parameters from within the wrapper
        """
        # If the attribute doesn't exist in the Model class, delegate it to inner_model
        return_value = None
        if name == "inner_model" or not hasattr(self, "inner_model"):
            raise AttributeError(f"{self} does not have inner_model")
        inner_attr = getattr(self.inner_model, name, None)
        if inner_attr is not None:
            if callable(inner_attr):
                # If it's a method, bind it to inner_model to maintain proper 'self'
                @functools.wraps(inner_attr)
                def wrapper(*args, **kwargs):
                    return inner_attr(*args, **kwargs)

                return_value = wrapper
            else:
                return_value = inner_attr

        if return_value is not None:
            return return_value
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __deepcopy__(self, memo):
        """Custom implementation for deep copying the model."""
        # Create a new instance of the class
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy each attribute, handling the PyTorch model separately
        for k, v in self.__dict__.items():
            if k == "inner_model":
                result.__dict__[k] = self.deepcopy_inner_model(memo)
            else:
                result.__dict__[k] = copy.deepcopy(v, memo)

        return result

class SemsegModel(EIARepairModel):
    """Model config class for the semantic segmentation model.
    
    Please build in accordance to your model and the EIARepairModel"""

    def __init__(
        self,
        model_path: str,
        device: str,
    ) -> None:
        """Initialized msegmentation and loads Weights."""
        # include your pytorch model here as needed by the class EIARepairModel
        pass
