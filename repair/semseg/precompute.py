"""Precomputation assistant class to speed up repair times."""
import hashlib
import os
import pickle
from enum import Enum
from typing import Any, Callable, Optional

import torch


class PrecomputeSavingMode(Enum):
    """Precomputation saving mode."""

    ram = "ram"
    disk = "disk"


class PrecomputeDataloader:
    """Precomputation manager class."""

    def __init__(
        self,
        saving_path: str,
        model: torch.nn.Module,
        saving_mode: PrecomputeSavingMode = PrecomputeSavingMode.ram,
    ) -> None:
        """
        Initializes the PrecomputeDataloader with a given saving path.

        Args:
            saving_path (str): The directory path where precomputed outputs will be saved.
        """
        self.mode = saving_mode
        if self.disk_mode:
            self._init_for_disk(saving_path, model)
        else:
            self._init_for_ram()

    def _init_for_disk(
        self,
        saving_path: str,
        model: torch.nn.Module,
    ) -> None:
        """
        Initializes the PrecomputeDataloader with a given saving path.

        Args:
            saving_path (str): The directory path where precomputed outputs will be saved.
        """
        self.model_hash = self._hash_model(model)
        self.saving_path = os.path.join(saving_path, self.model_hash)  # noqa PTH118
        os.makedirs(self.saving_path, exist_ok=True)  # noqa PTH103
        self.index_file = os.path.join(self.saving_path, "index.pkl")  # noqa PTH118
        if os.path.exists(self.index_file):  # noqa PTH110
            with open(self.index_file, "rb") as f:
                self.index = pickle.load(f)
        else:
            self.index = {}

    def _init_for_ram(
        self,
    ) -> None:
        """
        Initializes the PrecomputeDataloader with a given saving path.

        Args:
            saving_path (str): The directory path where precomputed outputs will be saved.
        """
        self.index = {}

    @property
    def disk_mode(self) -> bool:
        """Evaluates if the precomputer is in disk mode."""
        return self.mode == PrecomputeSavingMode.disk

    def _hash_tensor(self, tensor: torch.Tensor) -> str:
        """
        Computes a SHA-256 hash for a given tensor.

        Args:
            tensor (torch.Tensor): The tensor to be hashed.

        Returns:
            str: The SHA-256 hash of the tensor.
        """
        array = tensor.cpu().numpy()
        return hashlib.sha256(array).hexdigest()

    def _hash_model(self, model: torch.nn.Module) -> str:
        """
        Computes a SHA-256 hash for a given model's state_dict.

        Args:
            model (torch.nn.Module): The model to be hashed.

        Returns:
            str: The SHA-256 hash of the model's state_dict.
        """
        state_dict = model.state_dict()
        # Ensure the state_dict is ordered
        ordered_state_dict = {
            k: state_dict[k].cpu().numpy() for k in sorted(state_dict.keys())
        }
        model_bytes = pickle.dumps(ordered_state_dict)
        return hashlib.sha256(model_bytes).hexdigest()

    def save_precomputed_output(self, tensor: torch.Tensor, output: torch.Tensor) -> None:
        """
        Saves a precomputed output tensor indexed by the hash of the input tensor.

        Args:
            tensor (torch.Tensor): The input tensor to be hashed and used as a key.
            output (torch.Tensor): The precomputed output tensor to be saved.
        """
        tensor_hash = self._hash_tensor(tensor)
        if self.disk_mode:
            output_path = os.path.join(  # noqa PTH110
                self.saving_path, f"{tensor_hash}.pt"
            )
            torch.save(output, output_path)
            self.index[tensor_hash] = output_path
            self._update_index()
        else:
            self.index[tensor_hash] = (self._move_output_to_cpu(output), tensor.device)

    def _apply_for_all_tensors(self, obj: Any, action: Callable):
        """Apply the action over all tensors in the iterable object."""
        if isinstance(obj, torch.Tensor):
            return action(obj)
        elif isinstance(obj, list):
            return [self._apply_for_all_tensors(elem, action) for elem in obj]
        elif isinstance(obj, set):
            return set([self._apply_for_all_tensors(elem, action) for elem in obj])
        elif isinstance(obj, dict):
            return {
                key: self._apply_for_all_tensors(elem, action)
                for key, elem in obj.items()
            }
        else:
            return obj

    def _move_output_to_cpu(self, output: Any) -> Any:
        """Move any output tensors to cpu memory."""

        def action(x):
            return x.detach().cpu()

        return self._apply_for_all_tensors(output, action)

    def _move_output_to_device(self, output: Any, device: str) -> Any:
        """Move any output tensors to device."""

        def action(x):
            return x.to(device)

        return self._apply_for_all_tensors(output, action)

    def get_precomputed_output(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Retrieves a precomputed output tensor for a given input tensor if it exists.

        Args:
            tensor (torch.Tensor): The input tensor whose precomputed output is to
                be retrieved.

        Returns:
            Optional[torch.Tensor]: The precomputed output tensor if found, else None.
        """
        tensor_hash = self._hash_tensor(tensor)
        if tensor_hash in self.index:
            if self.disk_mode:
                output_path = self.index[tensor_hash]
                return torch.load(output_path, tensor.device)
            else:
                saved_output, device = self.index[tensor_hash]
                return self._move_output_to_device(saved_output, device)
        else:
            return None

    def _update_index(self) -> None:
        """Updates the index file with the current index data."""
        with open(self.index_file, "wb") as f:
            pickle.dump(self.index, f)
