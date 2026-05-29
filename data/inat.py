import json
import os
from typing import Any, Dict, List

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset

from data.base import GeoDatasetWrapper
from data.torchspatial_base import TorchSpatialDataset


class TorchSpatialiNat2018Dataset(GeoDatasetWrapper):
    def __init__(self, data_dir: str, splits: List[str] = None, groupby_field: str = None, **kwargs) -> None:
        self.default_splits = ['train', 'val']
        self.valid_groupby_fields = ['country', 'continent', 'ecoregion_class_index', 'biome_class_index', 'realmbiome_class_index']
        if splits is None:
            splits = self.default_splits
        else:
            if not set(splits).issubset(set(self.default_splits)):
                raise ValueError(f"Invalid split: {splits}. Must be one of {self.default_splits}")

        if groupby_field is not None and groupby_field not in self.valid_groupby_fields:
            raise ValueError(f"Invalid groupby_field: {groupby_field}. Must be one of {self.valid_groupby_fields}.")
        self.groupby_field = groupby_field
        
        super().__init__(splits, data_dir=data_dir, **kwargs)

        self.n_classes = 8142
        self.image_embed_dim = 2048

    def _load_split(self, split: str, **kwargs) -> Dataset:
        if split not in self.default_splits:
            raise ValueError(f"Invalid split: {split}. Must be either 'train' or 'val'.")

        split_groups = kwargs.get('split_groups', None)
        keep_groups = split_groups[split] if split_groups and split in split_groups else None

        data_dir = kwargs.pop('data_dir')
        feats_file = os.path.join(data_dir, "features_inception", "inat2018_{split}_net_feats.npy".format(split=split))
        image_feats = torch.tensor(np.load(feats_file), dtype=torch.float32)

        preds_file = os.path.join(data_dir, "features_inception", "inat2018_{split}_preds_sparse.npz".format(split=split))
        image_preds = torch.tensor(sparse.load_npz(preds_file).toarray(), dtype=torch.float32)

        suffix = ""  # "_full" if split == "val" else ""
        json_locs_file = os.path.join(data_dir, "{split}2018_locations_revgc_ecoregions{suffix}.json".format(split=split, suffix=suffix))
        drop_image_ids, loc_feats, groupby_field_values = self.load_locs_from_json(json_locs_file, keep_groups)

        image_annos_file = os.path.join(data_dir, "{split}2018.json".format(split=split))
        with open(image_annos_file, 'r') as f:
            image_annos_json = json.load(f)

        all_labels = [annos['category_id'] for annos in image_annos_json['annotations']]
        all_image_ids = [annos['id'] for annos in image_annos_json['images']]
        
        mask = [i for i in range(len(all_image_ids)) if all_image_ids[i] not in drop_image_ids]
        image_feats = image_feats[mask, :]
        image_preds = image_preds[mask, :]
        labels = [all_labels[i] for i in mask]

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

    def load_locs_from_json(self, json_file: str, keep_groups: List[str] = None) -> torch.Tensor:
        """
        Load longitude and latitude from a JSON file and return as a tensor.
        The JSON file should contain an array of [lon, lat] pairs.
        """
        with open(json_file, 'r') as f:
            locs_json = json.load(f)
        drop_ids = []
        locs, groupby_field_values = [], []
        for l in locs_json:
            if l['lon'] is None or l['lat'] is None:
                drop_ids.append(l['id'])
            else:
                if self.groupby_field in self.valid_groupby_fields:
                    if self.groupby_field not in l:
                        drop_ids.append(l['id'])
                    else:
                        if keep_groups is not None and l[self.groupby_field] not in keep_groups:
                            drop_ids.append(l['id'])
                        else:
                            locs.append([l['lon'], l['lat']])
                            groupby_field_values.append(str(l[self.groupby_field]))
                else:
                    locs.append([l['lon'], l['lat']])
                    
        return drop_ids, torch.tensor(locs, dtype=torch.float32), groupby_field_values

    def get_lon_lat(self, metadata) -> torch.Tensor:
        return metadata[:, -2:]

    def get_metadata_fields(self) -> List[str]:
        return self.get_split(self._splits[0]).metadata_fields

    def get_metadata_map(self) -> Dict[str, List[Any]]:
        return self.get_split(self._splits[0]).metadata_map

    def add_to_metadata(self, name, metadata, indices, default_value):
        raise NotImplementedError("TODO")

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
            sorted_accs = sorted(list(group_name_to_acc.values()))
            group_metrics = {
                **group_name_to_acc,
                f'{prefix}acc_avg_{self.groupby_field}': np.mean(sorted_accs),
                f'{prefix}acc_worst_{self.groupby_field}': sorted_accs[0],
                f"{prefix}acc_second_worst_{self.groupby_field}": sorted_accs[1],
                f"{prefix}acc_third_worst_{self.groupby_field}": sorted_accs[2],
            }
            result.update(group_metrics)
        
        return result