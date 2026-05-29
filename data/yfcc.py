import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.base import GeoDatasetWrapper
from data.torchspatial_base import TorchSpatialDataset


class TorchSpatialYfccDataset(GeoDatasetWrapper):
    def __init__(self, data_dir: str, splits: List[str] = None, groupby_field: str = None, **kwargs) -> None:
        self.default_splits = ['train', 'val', 'test']
        self.valid_groupby_fields = ['country', 'continent', 'urban_class']
        if splits is None:
            splits = self.default_splits
        else:
            if not set(splits).issubset(set(self.default_splits)):
                raise ValueError(f"Invalid split: {splits}. Must be one of {self.default_splits}")

        if groupby_field is not None and groupby_field not in self.valid_groupby_fields:
            raise ValueError(f"Invalid groupby_field: {groupby_field}. Must be one of {self.valid_groupby_fields}.")
        self.groupby_field = groupby_field
        
        super().__init__(splits, data_dir=data_dir, **kwargs)

        self.n_classes = 100
        self.image_embed_dim = 2048

    def _load_split(self, split: str, **kwargs) -> Dataset:
        if split not in self.default_splits:
            raise ValueError(f"Invalid split: {split}. Must be either 'train' or 'val'.")

        split_groups = kwargs.get('split_groups', None)
        keep_groups = split_groups[split] if split_groups and split in split_groups else None

        data_dir = kwargs.pop('data_dir')
        feats_file = os.path.join(data_dir, "features_inception", "YFCC_{split}_net_feats.npy".format(split=split))
        image_feats = torch.tensor(np.load(feats_file), dtype=torch.float32) 

        preds_file = os.path.join(data_dir, "features_inception", "YFCC_{split}_preds.npy".format(split=split))
        image_preds = torch.tensor(np.load(preds_file), dtype=torch.float32)

        csv_locs_file = os.path.join(data_dir, "train_test_split_with_continents_with_urban.csv")
        drop_ids, loc_feats, labels, groupby_field_values = self.load_from_csv(csv_locs_file, split, keep_groups)

        mask = [i for i in range(len(labels)) if i not in drop_ids]
        image_feats = image_feats[mask, :]
        image_preds = image_preds[mask, :]

        metadata, metadata_fields = [], []
        if len(groupby_field_values) > 0:
            metadata_fields.append(self.groupby_field)
            metadata.append(groupby_field_values)

        return_image_preds = kwargs.get('return_image_preds', False)

        return TorchSpatialDataset(
            image_feats, 
            image_preds,
            loc_feats,
            labels, 
            metadata=metadata, 
            metadata_fields=metadata_fields, 
            return_image_preds=return_image_preds
        )

    def load_from_csv(self, csv_file: str, split: str, keep_groups: List[str] = None) -> torch.Tensor:
        """
        Load longitude and latitude from a CSV file and return as a tensor.
        The CSV file should contain an array of [lon, lat] pairs.
        """
        columns = ['path', 'lat', 'lon', 'split', 'class']
        if self.groupby_field is not None:
            columns.append(self.groupby_field)
            
        df = pd.read_csv(csv_file)[columns]
        df = df[df['split'] == split]
        
        drop_idx = []
        if keep_groups is not None and self.groupby_field is not None:
            df = df[df[self.groupby_field].isin(keep_groups)]
            drop_idx = df.index[df[self.groupby_field].isin(keep_groups) == False].tolist()
            
        locs = torch.tensor(df[['lon', 'lat']].values, dtype=torch.float32)
        labels = torch.tensor(df['class'].values, dtype=torch.long)
        if self.groupby_field is not None:
            groupby_field_values = df[self.groupby_field].tolist()
        else:
            groupby_field_values = []
            
        return drop_idx, locs, labels, groupby_field_values

    def get_lon_lat(self, metadata) -> torch.Tensor:
        return metadata[:, -2:]

    def get_metadata_fields(self) -> List[str]:
        return self.get_split(self._splits[0]).metadata_fields

    def get_metadata_map(self) -> Dict[str, List[Any]]:
        return self.get_split(self._splits[0]).metadata_map

    def add_to_metadata(self, name, metadata, indices, default_value):
        raise NotImplementedError

    def compute_groupby_metrics(self, y_true, y_pred, metadata, prefix=''):
        if self.groupby_field is None:
            return {}
        
        field_idx = self.get_metadata_fields().index(self.groupby_field)
        groupby_values = metadata[:, field_idx].numpy()
        unique_groups = np.unique(groupby_values)

        if prefix != '' and not prefix.endswith('_'):
            prefix += '_'

        group_name_to_acc = {}
        for g in unique_groups:
            mask = groupby_values == g
            if np.sum(mask) > 0:
                group_acc = (y_true[mask] == y_pred[mask]).float().mean().item()
                group_name = self.get_metadata_map()[self.groupby_field][int(g)] if self.groupby_field in self.get_metadata_map() else str(int(g))
                group_name_to_acc[f'{prefix}acc_{self.groupby_field}_{group_name}'] = group_acc

        result = {}
        if len(group_name_to_acc) > 0:
            group_metrics = {
                **group_name_to_acc,
                f'{prefix}acc_avg_{self.groupby_field}': np.mean(list(group_name_to_acc.values())),
                f'{prefix}acc_worst_{self.groupby_field}': np.min(list(group_name_to_acc.values()))
            }
            result.update(group_metrics)
        
        return result