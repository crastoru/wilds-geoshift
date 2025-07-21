import os
import tarfile
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from wilds import get_dataset
from wilds.datasets.wilds_dataset import WILDSSubset

from data.base import GeoDatasetWrapper

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