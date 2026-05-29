import torch
import torch.nn as nn

from models.location_encoder import LocationEncoder, Wrap


class GroupEncoder(nn.Module):
    def __init__(self, n_groups, embed_dim):
        super().__init__()
        assert n_groups > 0, "Number of groups must be positive"
        self.embeddings = nn.Parameter(torch.randn(n_groups, embed_dim))
        self.out_dim = embed_dim

    def has_trainable_params(self) -> bool:
        return True

    def set_device(self, device):
        self.embeddings = self.embeddings.to(device)

    def forward(self, groups: torch.Tensor) -> torch.Tensor:
        return self.embeddings[groups]


class IndicatorDomainRelation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, queries, key):
        return torch.eq(queries, key).float().unsqueeze(-1)
    

class D3GRelation(nn.Module):
    def __init__(self, beta, out_dim, loc_encoder_name: str = "none", loc_encoder_weights=None, n_groups=0, freeze_loc_encoder=False) -> None:
        super().__init__()
        self.fixed_relations = IndicatorDomainRelation()
        self.beta = beta
        self.loc_encoder = None
        self.nnet = None
        self.nnet_latlon = None

        if self.beta < 1:
            self.nnet = nn.Sequential(
                nn.Linear(1, out_dim // 2),
                nn.ReLU(),
                nn.Linear(out_dim // 2, out_dim)
            )

            if loc_encoder_name != 'none':
                print("Using location encoder {} in D3G relation with frozen={}".format(loc_encoder_name, freeze_loc_encoder))
                if loc_encoder_name == "groups":
                    self.loc_encoder = GroupEncoder(n_groups, 256)
                else:
                    self.loc_encoder = LocationEncoder(loc_encoder_name, loc_encoder_weights, "cpu", frozen=freeze_loc_encoder)

                self.nnet_latlon = nn.Sequential(
                    nn.Linear(self.loc_encoder.out_dim, out_dim // 2),
                    nn.ReLU(),
                    nn.Linear(out_dim // 2, out_dim)
                )

            self.weights = nn.Parameter(torch.rand(out_dim))
            self.cos = nn.CosineSimilarity()
    
    def set_device(self, device):
        if self.loc_encoder is not None:
            self.loc_encoder.set_device(device)
        if self.nnet is not None:
            self.nnet = self.nnet.to(device)
            self.weights = self.weights.to(device)
        if self.nnet_latlon is not None:
            self.nnet_latlon = self.nnet_latlon.to(device)

    def forward(self, queries, queries_latlon, key):
        fixed_relations = self.fixed_relations(queries, key)
        if self.beta == 1:
            return fixed_relations
        keys = key.repeat(queries.shape[0]).unsqueeze(-1).float()
        
        if self.loc_encoder is not None:
            if isinstance(self.loc_encoder, GroupEncoder):
                loc_embeds = self.loc_encoder(queries)
            else:
                loc_embeds = self.loc_encoder(queries_latlon)
            learned_relations = self.cos(self.weights * self.nnet_latlon(loc_embeds), self.weights * self.nnet(keys)).unsqueeze(-1)
        else:
            loc_embeds = None
            queries = queries.unsqueeze(-1).float()
            learned_relations = self.cos(self.weights * self.nnet(queries), self.weights * self.nnet(keys)).unsqueeze(-1)
        
        return {
            'weights': self.beta * fixed_relations + (1. - self.beta) * learned_relations,
            'loc_embed': loc_embeds
        }