import torch
from typing import List, Any
from torch.utils.data import Dataset


class TorchSpatialDataset(Dataset):
    def __init__(
            self,
            image_features: torch.Tensor,
            image_preds: torch.Tensor,
            loc_feats: torch.Tensor,
            labels: torch.Tensor,
            metadata: List[List[Any]] = None,
            metadata_fields: List[str] = None,
            return_image_preds: bool = False
        ) -> None:
        assert len(image_features) == len(image_preds) == len(loc_feats) == len(labels)

        self.image_features = image_features
        self.image_preds = image_preds
        self.loc_feats = loc_feats
        self.labels = labels
        self.return_image_preds = return_image_preds

        self.metadata_map = {}
        self.metadata_fields = [] if metadata_fields is None else metadata_fields
        self.metadata_array = self.process_metadata(metadata)
        self.metadata_fields.extend(['lon', 'lat'])

    def process_metadata(self, metadata: List[List[Any]]) -> torch.Tensor:
        if isinstance(metadata, List):
            if len(metadata) == 0:
                return None
            
            # if any of the metadata fields are strings, convert to categorical integers
            processed_metadata = []
            for i, field in enumerate(metadata):
                if isinstance(field[0], str):
                    unique_vals = list(set(field))
                    val_to_int = {val: j for j, val in enumerate(unique_vals)}
                    field = [val_to_int[val] for val in field]
                    # save the mapping from int to original value
                    self.metadata_map[self.metadata_fields[i]] = unique_vals #list(val_to_int.keys())
                processed_metadata.append(torch.tensor(field).unsqueeze(1))
            return torch.cat(processed_metadata, dim=1)
        return metadata

    def __len__(self):
        return len(self.loc_feats)

    def __getitem__(self, index):
        loc_feat  = self.loc_feats[index, :]
        label = self.labels[index]
        if self.metadata_array is not None:
            metadata = self.metadata_array[index, :]
            metadata = torch.cat([metadata, loc_feat])
        else:
            metadata = loc_feat

        if self.return_image_preds:
            # image predictions: (n_classes)
            image_preds = self.image_preds[index, :]
            return image_preds, label, metadata

        # cnn image features: (2048)
        image_features = self.image_features[index, :]
        return image_features, label, metadata