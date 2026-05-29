import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121
from transformers import CLIPImageProcessor, CLIPModel
from models.prithvi import PrithviMAE, config

from models.resnet import ResNet18


class DenseNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet = densenet121(DenseNet121_Weights.IMAGENET1K_V1)

        # densenet is a linear classifier on top of a CNN
        # in_features is the output dimension of the CNN after flattening/averaging W x H dimensions
        self.out_dim = self.densenet.classifier.in_features
    
    def forward(self, x: torch.Tensor):
        embed = self.densenet.features(x)
        embed = F.relu(embed, inplace=True)
        embed = F.adaptive_avg_pool2d(embed, (1, 1))
        return torch.flatten(embed, 1)
    
    
class ResNet18Encoder(nn.Module):
    def __init__(self, num_channels=3):
        super().__init__()
        self.resnet = ResNet18(num_channels=num_channels)
        self.out_dim = self.resnet.fc.in_features
    
    def forward(self, x: torch.Tensor):
        _, feats = self.resnet(x, with_feats=True)
        return feats


class CLIPEncoder(nn.Module):
    def __init__(self, model_id='openai/clip-vit-large-patch14-336'):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPImageProcessor.from_pretrained(model_id)
        self.out_dim = self.model.visual_projection.out_features

    def forward(self, x: torch.Tensor):
        embed = self.model.get_image_features(pixel_values=x)
        return embed
    

class Prithvi(nn.Module):
    def __init__(self, checkpoint_path, bands=[2, 1, 0]):
        """
        Args:
            checkpoint_path: path to the pretrained PrithviMAE checkpoint
            bands: specifies ordered selection of bands to use as input
                (e.g. if input is RGB, specify [2, 1, 0])

        PrithviMAE expects 6-channel input in this order:
            1. Blue
            2. Green
            3. Red
            4. Narrow NIR (Near-Infrared)
            5. SWIR 1 (Short-Wave Infrared 1)
            6. SWIR 2 (Short-Wave Infrared 2) 
        """
        super().__init__()
        assert len(bands) <= 6, "PrithviMAE only supports up to 6 input channels."

        # Load Prithvi model
        self.model = PrithviMAE(**config['pretrained_cfg'])
        state_dict = torch.load(checkpoint_path, weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.train()
        
        self.out_dim = config['pretrained_cfg']['embed_dim']
        self.bands = bands

        if len(bands) < 6:
            # Map input bands to remaining channels if necessary
            self.adapter = nn.Conv3d(len(bands), 6 - len(bands), kernel_size=1)

    def forward(self, x: torch.Tensor):
        if x.ndim == 4:
            x = x[:, self.bands, :, :]
        else:
            x = x[self.bands, :, :]
            x = x.unsqueeze(0)  # Add batch dimension

        x = x.unsqueeze(2)  # Add time dimension

        if len(self.bands) < 6:
            x_missing_channels = self.adapter(x)
            x_all_channels = torch.cat((x, x_missing_channels), dim=1)  # (B, 6, 1, H, W)
        else:
            x_all_channels = x

        out = self.model.encoder(x_all_channels)
        return out[0][:, 0, :]  # CLS pooling