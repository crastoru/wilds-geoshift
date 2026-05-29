import os
import sys
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from geoclip import LocationEncoder as GeoClipLocationEncoder

sys.path.append(os.path.join(os.getcwd(), "satclip"))
from satclip.load_lightweight import get_satclip_loc_encoder


class RandomFourierFeatures(nn.Module):
    def __init__(self, input_dim: int, mapping_size: int = 256, scale: float = 1.0) -> None:
        super().__init__()
        self.B = nn.Parameter(torch.randn((input_dim, mapping_size)) * scale, requires_grad=False)
        self.out_dim = mapping_size * 2

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        # Ensure input and B are the same dtype (float32)
        x = torch.deg2rad(lonlat).to(self.B.dtype)  # (N, 2)
        x_proj = 2.0 * torch.pi * x @ self.B  # (N, mapping_size)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # (N, mapping_size * 2)


class Wrap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.out_dim = 4

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        cos = torch.cos(torch.deg2rad(lonlat))
        sin = torch.sin(torch.deg2rad(lonlat))
        return torch.cat((cos, sin), axis=1).to(torch.float32)


class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y

        return out


class FCNet(nn.Module):
    def __init__(self, num_inputs, num_filts):
        super(FCNet, self).__init__()
        self.num_filts = num_filts
        self.feats = nn.Sequential(nn.Linear(num_inputs, num_filts),
                                    nn.ReLU(inplace=True),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts))

    def forward(self, x):
        return self.feats(x)
    

class LocationEncoder(nn.Module):
    def __init__(self, name, weights=None, device="cpu", frozen=True) -> None:
        super().__init__()
        if name.startswith("wrap"):
            self.model = Wrap()
            self.out_dim = self.model.out_dim
        elif name.startswith("rff"):
            self.model = RandomFourierFeatures(input_dim=2, mapping_size=256, scale=16.0)
            self.out_dim = self.model.out_dim
        elif name.startswith("satclip"):
            if weights is None:
                raise ValueError("Must provide SatCLIP model weights")
            self.model = get_satclip_loc_encoder(weights, device)
            self.out_dim = 256
        elif name.startswith("geoclip"):
            self.model = GeoClipLocationEncoder()
            self.out_dim = 512
        else:
            raise ValueError(f"Location encoder {name} is not supported.")
    
        if name.endswith('fcnet'):
            self.fcnet = FCNet(self.out_dim, 256)
            self.out_dim = 256
        elif name.endswith('linear'):
            self.fcnet = nn.Linear(self.out_dim, 256)
            self.out_dim = 256
        else:
            self.fcnet = None
    
        self.name = name
        self.frozen = frozen
        if self.frozen:
            self.model.requires_grad_(False)

    def has_trainable_params(self) -> bool:
        return self.fcnet is not None or not self.frozen

    def set_device(self, device):
        """
        Set the device for the model.
        """
        if self.fcnet is not None:
            self.fcnet = self.fcnet.to(device)
        self.model = self.model.to(device)
        self._device = device

    def forward(self, lonlat):
        if self.name.startswith('geoclip') or self.name.startswith('rff'):
            coords = lonlat.flip(1).to(torch.float32)  # GeoCLIP expects lat, lon order
        elif self.name.startswith('satclip'):
            coords = lonlat.to(torch.float64) 
        else:
            coords = lonlat 
            
        if not self.frozen:
            embeds = self.model(coords)
        else:
            with torch.no_grad():
                embeds = self.model(coords)

        if self.name.startswith('satclip') or self.name.startswith('geoclip'):
            embeds = (1. / embeds.norm(dim=1))[:, None] * embeds  # normalize
            embeds = embeds.to(torch.float32)
        
        if self.fcnet is not None:
            embeds = self.fcnet(embeds)
        return embeds
    

class FiLM(nn.Module):
    def __init__(self, x_dim: int, z_dim: int) -> None:
        super().__init__()
        self.film_add = nn.Sequential(
                nn.Linear(z_dim, x_dim // 2),
                nn.ReLU(),
                nn.Linear(x_dim // 2, x_dim)
            )
        self.film_mul = nn.Sequential(
                nn.Linear(z_dim, x_dim // 2),
                nn.ReLU(),
                nn.Linear(x_dim // 2, x_dim)
            )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        mul = self.film_mul(z)
        add = self.film_add(z)
        return mul * x + add
    

class ConditionalLinear(nn.Module):
    def __init__(self, x_dim: int, out_dim: int, z_dim=0, dropout=0.1, use_relu=True) -> None:
        super().__init__()
        if z_dim > 0:
            self.film = FiLM(x_dim, z_dim)
        else:
            self.film = None
        self.linear = nn.Linear(x_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_relu = use_relu

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert 'x' in inputs, "x must be a key in inputs dict"
        x = inputs['x']
        z = inputs['z'] if 'z' in inputs else None

        if self.film and z is not None:
            x = self.film(x, z)
        out = self.linear(x)
        out = self.dropout(out)
        if self.use_relu:
            out = F.relu(out)
        return {'x':out, 'z':z, 'film_out':x} if z is not None else {'x':out}