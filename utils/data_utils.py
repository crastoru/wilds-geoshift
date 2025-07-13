import copy
import os
import tarfile
from typing import Any, Dict, List

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from wilds import get_dataset
from wilds.datasets.poverty_dataset import _MEANS_2009_17, _STD_DEVS_2009_17
from wilds.datasets.wilds_dataset import WILDSSubset

from utils.transforms import FIX_MATCH_AUGMENTATION_POOL, RandAugment

POVERTY_RGB_MEAN = np.array([_MEANS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1))
POVERTY_RGB_STD = np.array([_STD_DEVS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1))

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def denormalize(tensor: torch.Tensor, mean: tuple, std: tuple) -> torch.Tensor:
    return tensor * torch.Tensor(std).to(tensor.device).view(3, 1, 1) + torch.Tensor(mean).to(tensor.device).view(3, 1, 1)


class GeoDatasetWrapper:
    """
    A superclass wrapper for a particular dataset and the transforms applied to each of its splits.
    """
    def __init__(self, dataset: Dataset, splits: List[str], **kwargs) -> None:
        self._dataset = dataset
        self._splits = splits
        self._load_all_splits(**kwargs)

    def get_split(self, split: str) -> Dataset:
        if split not in self._splits:
            raise ValueError(f"Invalid split: {split}. Must be one of {', '.join(self._splits)}")
        return self.__getattribute__(f"_{split}_data")

    @property
    def dataset(self):
        return self._dataset

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

    def _load_split(self, split: str, **kwargs) -> Dataset:
        raise NotImplementedError

    def _load_all_splits(self, **kwargs) -> None:
        """
        Apply transforms and return train, validation, and test dataset splits.
        """
        for split in self._splits:
            setattr(self, f"_{split}_data", self._load_split(split, **kwargs))
    
    def get_lon_lat(self, batch):
        """
        Return [lon lat] tensor given a batch of data from the underlying dataset.
        """
        raise NotImplementedError


class WILDSDatasetWrapper(GeoDatasetWrapper):
    def __init__(self, 
                 data_dir: str, 
                 dataset_name: str, 
                 version_str: str = "v1.1", 
                 splits: List[str] = None, 
                 group_field: str = None,
                 split_groups: Dict[str, List[str]] = None,
                 combine_id_ood: bool = False,
                 remove_split: str = None,
                 **dataset_kwargs: Any
                 ) -> None:
        
        # Load raw data from pre-downloaded folder (compressed or uncompressed)
        data_folder_name = f"{dataset_name}_{version_str}"
        if os.path.exists(os.path.join(data_dir, data_folder_name)):
            root_dir = data_dir
        elif os.path.exists(os.path.join(data_dir, f"{data_folder_name}.tar.gz")):
            tar = tarfile.open(os.path.join(data_dir, f"{data_folder_name}.tar.gz"), "r:gz")
            tar.extractall(path="data")
            tar.close()
            root_dir = "data"
        else:
            raise ValueError(f"Could not find {data_folder_name} or {data_folder_name}.tar.gz at {data_dir}")

        # Load the full WILDSDataset
        dataset = get_dataset(dataset=dataset_name, root_dir=root_dir, download=False, **dataset_kwargs)

        valid_splits = list(dataset._split_names.keys())
        if splits is not None:
            for split in splits:
                if split not in valid_splits:
                    raise ValueError(f"Invalid split: {split}. Must be one of {', '.join(valid_splits)}")
        else:
            splits = valid_splits

        if combine_id_ood and 'id_val' in splits:
            splits.remove('id_val')
        if combine_id_ood and 'id_test' in splits:
            splits.remove('id_test')
            
        # Load and transform the splits
        super().__init__(dataset, splits, group_field=group_field, split_groups=split_groups, combine_id_ood=combine_id_ood)

        # Add lat/lon to metadata (not included by default)
        if 'lat' not in self._dataset.metadata.keys() or 'lon' not in self._dataset.metadata.keys():
            raise ValueError("Metadata does not contain 'lat' and 'lon' fields. Cannot initialize GeoDatasetWrapper.")

        if remove_split is not None and 'split' in self._dataset.metadata.keys():
            mask = np.asarray(self._dataset.metadata['split'] == remove_split)
        else:
            mask = np.zeros(len(self._dataset.metadata['lat']), dtype=bool)

        lat_array = np.asarray(list((self._dataset.metadata['lat'])))[~mask]
        self.add_to_metadata('lat', lat_array, np.arange(len(lat_array)), 0)
        lon_array = np.asarray(list((self._dataset.metadata['lon'])))[~mask]
        self.add_to_metadata('lon', lon_array, np.arange(len(lon_array)), 0)

    def _load_split(self, split: str, **kwargs) -> Dataset:
        t = self._load_transform(split)
        data = self._dataset.get_subset(split, transform=t)
        
        if kwargs.get('combine_id_ood', False) and split in ['val', 'test']:
            id_data = self._dataset.get_subset(f"id_{split}", transform=t)
            data = WILDSSubset(data.dataset, np.concatenate([data.indices, id_data.indices]), data.transform)
        
        group_field = kwargs.get('group_field', None)
        split_groups = kwargs.get('split_groups', None)

        if group_field is not None and split_groups is not None and split in split_groups:
            return self.filter_by_metadata_field(data, group_field, split_groups[split])
        return data

    def get_lon_lat(self, metadata):
        """
        Return batch of [lon lat] from a batch of metadata (an n x d matrix containing 
        d metadata fields for n different points).
        """
        lon = metadata[:, self._dataset.metadata_fields.index('lon')]
        lat = metadata[:, self._dataset.metadata_fields.index('lat')]
        return torch.stack([lon, lat], dim=1)

    def add_to_metadata(self, name, metadata, indices, default_value):
        """
        Add another field to this dataset's metadata.

        Args:
            - name (str): name of the new metadata field
            - metadata: n-dimensional array of metadata to add (n can be less than the size of the full dataset)
            - indices: n-dimensional array of indices in the full dataset for which the metadata should be added
            - default_value: value for the new metadata field to use for data points not included in `indices` 
        """
        assert len(metadata) == len(indices)
        all_metadata_len = self._dataset.metadata_array.shape[0]
        assert len(indices) <= all_metadata_len

        all_metadata = torch.zeros(all_metadata_len) + default_value
        all_metadata[indices] = torch.tensor(metadata, dtype=all_metadata.dtype)
        all_metadata = all_metadata.unsqueeze(-1)

        self._dataset._metadata_array = torch.cat([self._dataset.metadata_array, all_metadata], dim=1)
        self._dataset._metadata_fields.append(name)

        for split in self._splits:
            data = self.get_split(split)
            data.dataset._metadata_array = self._dataset.metadata_array

    def filter_by_metadata_field(self, data, field_name, values):
        if field_name not in data.metadata_map:
            raise ValueError(f"Cannot filter by field {field_name} as it was not found in the dataset metadata.")
        
        values = [metadata_val for metadata_val in values if metadata_val in data.metadata_map[field_name]]
        if len(values) == 0:
            raise ValueError(f"None of the values {', '.join(values)} were found in the dataset metadata.")

        idx_to_keep = [data.metadata_map[field_name].index(metadata_val) for metadata_val in values]
        mask = np.zeros(len(data))
        for metadata_val in idx_to_keep:
            mask[np.where(data.metadata_array[:, 0] == metadata_val)[0]] = 1
        
        indices = data.indices[mask.astype(bool)]
        return WILDSSubset(data.dataset, indices, data.transform)


class FMoWDataset(WILDSDatasetWrapper):
    def __init__(self, 
                 data_dir: str, 
                 randaug: bool, 
                 clip_processor: str = None, 
                 splits: List[str] = None, 
                 group_field: str = None,
                 split_groups: Dict[str, List[str]] = None,
                 combine_id_ood: bool = False,
                 eval_only_transforms: bool = False
                 ) -> None:

        self.image_size = 224
        self.randaug = randaug
        self.clip_processor = clip_processor
        self.eval_only_transforms = eval_only_transforms

        # Call superclass init to get the splits
        super().__init__(data_dir, dataset_name="fmow", version_str="v1.1", splits=splits, group_field=group_field,
                         split_groups=split_groups, combine_id_ood=combine_id_ood, remove_split='seq')
        
        self.n_classes = self._dataset._n_classes

    def _load_transform(self, split: str):
        if self.clip_processor is not None:
            size = 224 if 'base' in self.clip_processor else int(self.clip_processor.split('-')[-1])
            default_transforms = [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ]
        else:
            default_transforms = [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ]

        if split != 'train' or self.eval_only_transforms:
            return transforms.Compose(default_transforms)

        default_train_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(self.image_size)
        ]
        if self.randaug:
            default_train_transforms.append(RandAugment(2, FIX_MATCH_AUGMENTATION_POOL))
        
        return transforms.Compose([
            *default_train_transforms,
            *default_transforms
            ])


    def _load_inverse_transform(self, split):
        if split == 'train':
            raise ValueError("Cannot perform inverse transform on training data.")
        else:
            return lambda t: denormalize(t, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD).permute(1, 2, 0).numpy()
    

class PovertyMapDataset(WILDSDatasetWrapper):
    def __init__(self, 
                 data_dir: str,
                 splits: List[str] = None, 
                 group_field: str = None,
                 split_groups: Dict[str, List[str]] = None,
                 combine_id_ood: bool = False,
                 fold: str = 'A'
                ) -> None:
        self.image_size = 224
        self.n_classes = 1  # regression

        # Call superclass init to get the splits
        super().__init__(data_dir, dataset_name="poverty", version_str="v1.1", splits=splits, 
                         group_field=group_field, split_groups=split_groups, combine_id_ood=combine_id_ood, fold=fold)

    # Taken from: https://github.com/p-lambda/wilds/blob/472677590de351857197a9bf24958838c39c272b/examples/transforms.py#L246
    def poverty_rgb_color_transform(self, ms_img, transform):
        poverty_rgb_means = np.array([_MEANS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1))
        poverty_rgb_stds = np.array([_STD_DEVS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1))

        def unnormalize_rgb_in_poverty_ms_img(ms_img):
            result = ms_img.detach().clone()
            result[:3] = (result[:3] * poverty_rgb_stds) + poverty_rgb_means
            return result

        def normalize_rgb_in_poverty_ms_img(ms_img):
            result = ms_img.detach().clone()
            result[:3] = (result[:3] - poverty_rgb_means) / poverty_rgb_stds
            return ms_img

        color_transform = transforms.Compose([
            transforms.Lambda(lambda ms_img: unnormalize_rgb_in_poverty_ms_img(ms_img)),
            transform,
            transforms.Lambda(lambda ms_img: normalize_rgb_in_poverty_ms_img(ms_img)),
        ])
        # The first three channels of the Poverty MS images are BGR
        # So we shuffle them to the standard RGB to do the ColorJitter
        # Before shuffling them back
        ms_img[:3] = color_transform(ms_img[[2,1,0]])[[2,1,0]] # bgr to rgb to bgr
        return ms_img

    def _load_transform(self, split: str):
        return None
        if split != 'train':
            return None  # No transforms for validation and test splits
        
        # PovertyMap comes normalized by default
        transform = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1)

        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Lambda(lambda ms_img: self.poverty_rgb_color_transform(ms_img, transform)),
        ])
    
    def _load_inverse_transform(self, split):
        if split == 'train':
            raise ValueError("Cannot perform inverse transform on training data.")
        else:
            return lambda t: denormalize(t, POVERTY_RGB_MEAN, 2 * POVERTY_RGB_STD).permute(1, 2, 0).numpy()