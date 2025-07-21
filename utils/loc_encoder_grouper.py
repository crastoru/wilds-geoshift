import numpy as np
import torch
from sklearn.cluster import KMeans
from wilds.common.data_loaders import get_eval_loader
from wilds.common.grouper import Grouper

from data.base import GeoDatasetWrapper
from models.location_encoder import LocationEncoder


class LocEncoderGrouper(Grouper):
    def __init__(
            self, 
            data_wrapper: GeoDatasetWrapper,
            loc_encoder: str,
            n_groups: int,
            seed: int,
            device: str,
            num_workers: int,
            loc_encoder_weights=None
        ) -> None:

        assert isinstance(data_wrapper, GeoDatasetWrapper)
        self.data_wrapper = data_wrapper
        self.num_metadata_fields = data_wrapper.get_split('train').metadata_array.shape[-1]
        self._n_groups = n_groups
        self.k_means = KMeans(n_clusters=self._n_groups, random_state=seed)
        self.unassigned_value = -1
        self.device = torch.device(device)

        self.loc_encoder = LocationEncoder(loc_encoder, loc_encoder_weights, device=self.device, frozen=True)
        self.loc_encoder.requires_grad_(False)
        self.loc_encoder = self.loc_encoder.to(self.device)

        self.metadata_field_name = f'{self.loc_encoder.name}_group'
        self.cluster(num_workers)

    @property
    def n_groups(self):
        """
        The number of groups defined by this Grouper.
        """
        return self._n_groups

    def metadata_to_group(self, metadata, return_counts=False):
        """
        Args:
            - metadata (Tensor): An n x d matrix containing d metadata fields
                                 for n different points.
            - return_counts (bool): If True, return group counts as well.
        Output:
            - group (Tensor): An n-length vector of groups.
            - group_counts (Tensor): Optional, depending on return_counts.
                                     An n_group-length vector of integers containing the
                                     numbers of data points in each group in the metadata.
        """
        if self.has_assigned_loc_encoder_group(metadata):
            # use group assignments cached in metadata if they are there
            metadata_fields = self.data_wrapper.dataset.metadata_fields
            groups = metadata[:, metadata_fields.index(self.metadata_field_name)]
        else: 
            ll = self.data_wrapper.get_lon_lat(metadata).to(self.device)
            with torch.no_grad():
                embeds = self.loc_encoder(ll)
                embeds = (1. / embeds.norm(dim=1))[:, None] * embeds
            groups = self.k_means.predict(embeds.cpu())  # move to CPU for k-means
            groups = torch.tensor(groups).to(metadata.device)

        if not return_counts:
            return groups.long()
        
        group_counts = torch.tensor([len(np.where(groups == i)[0]) for i in range(self._n_groups)]).to(metadata.device)
        return groups.long(), group_counts

    def has_assigned_loc_encoder_group(self, metadata):
        metadata_fields = self.data_wrapper.dataset.metadata_fields
        return self.metadata_field_name in metadata_fields and metadata[0, metadata_fields.index(self.metadata_field_name)] != self.unassigned_value

    def group_str(self, group):
        return f"{self.loc_encoder.name} location encoder cluster {group}"

    def group_field_str(self, group):
        return self.group_str(group).replace(' ', '')

    def cluster(self, workers=1):
        dataloader = get_eval_loader("standard", self.data_wrapper.get_split('train'), batch_size=16, num_workers=workers)
        print(f"Clustering for {self.loc_encoder.name} location encoder...")
        embeddings = []
        for i, batch in enumerate(dataloader):
            _, _, metadata = batch
            
            with torch.no_grad():
                ll = self.data_wrapper.get_lon_lat(metadata).to(self.device)
                batch_embeds = self.loc_encoder(ll)
                batch_embeds = (1. / batch_embeds.norm(dim=1))[:, None] * batch_embeds  # normalize
            embeddings.append(batch_embeds.cpu())
        embed_matrix = torch.cat(embeddings, dim=0)
        groups = self.k_means.fit_predict(embed_matrix)

        self.data_wrapper.add_to_metadata(self.metadata_field_name, groups, self.data_wrapper.get_split('train').indices, self.unassigned_value)