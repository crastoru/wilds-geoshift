import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121
from transformers import CLIPImageProcessor, CLIPModel

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