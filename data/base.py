from typing import Any, List

import torch
from torch.utils.data import Dataset


class GeoDatasetWrapper:
    """
    A superclass wrapper for a particular dataset and the transforms applied to each of its splits.
    """
    def __init__(self, splits: List[str], **kwargs) -> None:
        self._splits = splits
        self._load_all_splits(**kwargs)

    def get_split(self, split: str) -> Dataset:
        if split not in self._splits:
            raise ValueError(f"Invalid split: {split}. Must be one of {', '.join(self._splits)}")
        return self.__getattribute__(f"_{split}_data")

    def _load_all_splits(self, **kwargs) -> None:
        """
        Apply transforms and return train, validation, and test dataset splits.
        """
        for split in self._splits:
            setattr(self, f"_{split}_data", self._load_split(split, **kwargs))

    def _load_split(self, split: str, **kwargs) -> Dataset:
        raise NotImplementedError

    def _load_transform(self, split: str) -> Any:
        """
        Return the data transform for the given split.
        """
        raise NotImplementedError

    def _load_inverse_transform(self, split: str) -> Any:
        """
        Return the inverse transform for the given split (for plotting purposes).
        """
        raise NotImplementedError

    def get_lon_lat(self, metadata) -> torch.Tensor:
        """
        Return [lon lat] tensor given a batch of data from the underlying dataset.
        """
        raise NotImplementedError
    
    def get_metadata_fields(self) -> List[str]:
        """
        Return the metadata fields of the dataset.
        """
        raise NotImplementedError

    def add_to_metadata(self, name, metadata, indices, default_value):
        """
        Add metadata to the dataset.

        Args:
            - name (str): The name of the metadata field.
            - metadata (Tensor): The metadata to add.
            - indices (Tensor): The indices of the samples to add the metadata to.
            - default_value (Any): The default value to use for samples not in indices.
        """
        raise NotImplementedError


def denormalize(tensor: torch.Tensor, mean: tuple, std: tuple) -> torch.Tensor:
    return tensor * torch.Tensor(std).to(tensor.device).view(3, 1, 1) + torch.Tensor(mean).to(tensor.device).view(3, 1, 1)