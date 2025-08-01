import os
import sys
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from geoclip import LocationEncoder as GeoClipLocationEncoder

sys.path.append(os.path.join(os.getcwd(), "satclip"))
from satclip.load_lightweight import get_satclip_loc_encoder


class Wrap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.out_dim = 4

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        cos = torch.cos(torch.deg2rad(lonlat))
        sin = torch.sin(torch.deg2rad(lonlat))
        return torch.cat((cos, sin), axis=1).to(torch.float32)


class WrapLinear(nn.Module):
    def __init__(self, out_dim=256) -> None:
        super().__init__()
        self.wrap = Wrap()
        self.linear = nn.Linear(4, out_dim)
        self.out_dim = out_dim

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        return self.linear(self.wrap(lonlat))


class LocationEncoder(nn.Module):
    def __init__(self, name, weights=None, device="cpu", frozen=True) -> None:
        super().__init__()
        if name == "wrap":
            self.model = Wrap()
            self.out_dim = self.model.out_dim
        elif name == "wrap-linear":
            self.model = WrapLinear()
            self.out_dim = self.model.out_dim
        elif name == "satclip":
            if weights is None:
                raise ValueError("Must provide SatCLIP model weights")
            self.model = get_satclip_loc_encoder(weights, device)
            self.out_dim = 256
        elif name == "geoclip":
            self.model = GeoClipLocationEncoder()
            self.out_dim = 512
        else:
            raise ValueError(f"Location encoder {name} is not supported.")
    
        self.name = name
        self.frozen = frozen
        if self.frozen:
            self.model.requires_grad_(False)

    def set_device(self, device):
        """
        Set the device for the model.
        """
        self.model = self.model.to(device)
        self._device = device

    def forward(self, lonlat):
        if self.name == 'geoclip':
            coords = lonlat.flip(1).to(torch.float32)  # GeoCLIP expects lat, lon order
        else:
            coords = lonlat.to(torch.float64) 
        if not self.frozen:
            embeds = self.model(coords)
        else:
            with torch.no_grad():
                embeds = self.model(coords)
        
        if self.name == 'satclip' or self.name == 'geoclip':
            embeds = (1. / embeds.norm(dim=1))[:, None] * embeds  # normalize
        return embeds.to(torch.float32)

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