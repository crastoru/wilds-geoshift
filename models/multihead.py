import torch
import torch.nn as nn

from models.domain_relations import D3GRelation
from models.location_encoder import ConditionalLinear, LocationEncoder


class MultiHeadClassifier(nn.Module):
    def __init__(self, num_heads, encoder_dim, num_classes, dropout=0.1, beta=0, use_film=False, loc_encoder="none", loc_encoder_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.use_film = use_film
        self._device = "cpu"  # device can be set later

        if loc_encoder != "none" and (self.use_film or self.num_heads > 1):
            print("Using location encoder {} in multi-head classifier".format(loc_encoder))
            frozen = not self.use_film  # if using FiLM, we want to train the location encoder
            self.loc_encoder = LocationEncoder(loc_encoder, loc_encoder_weights, self._device, frozen=frozen)
            self.z_dim = self.loc_encoder.out_dim
        else:
            self.loc_encoder = None
            self.z_dim = 0

        self.classifiers = nn.ModuleList([
            nn.Sequential(
                ConditionalLinear(encoder_dim, self.num_classes, self.z_dim if self.use_film else 0, dropout, use_relu=False)
            ) for _ in range(self.num_heads)])
        
        if self.num_heads > 1:
            self.domain_relation = D3GRelation(beta, 256, self.loc_encoder)

    def set_device(self, device):
        self._device = device
        if self.loc_encoder is not None:
            self.loc_encoder.set_device(device)

    def forward(self, embed, lonlat, groups=None):
        if self.loc_encoder and self.use_film:
            z = self.loc_encoder(lonlat)
            inputs = {'x':embed, 'z':z.to(embed.dtype)}
        else:
            inputs = {'x':embed} 

        result = {'features': embed}

        if self.num_heads > 1:
            # Get D3G domain weights
            domain_weights = torch.cat([self.domain_relation(groups, lonlat, torch.tensor(i, device=torch.device(self._device))) for i in range(self.num_heads)], dim=1)
            domain_weights = domain_weights.unsqueeze(-1)  # batch_size x num_heads x 1
            
            # Get outputs from each prediction head
            outputs = [self.classifiers[i](inputs) for i in range(self.num_heads)]
            head_outputs = torch.stack([outputs[i]['x'] for i in range(self.num_heads)], dim=1)  # batch_size x num_heads x num_classes 
            
            if self.training:
                # for training, use predictions from one head only
                result['logits'] = head_outputs[torch.arange(0,len(groups)), groups]
                # enforce consistency loss on all other heads
                domain_weights[torch.arange(0,len(groups)), groups] = 0  
            
            result['rel_logits'] = torch.sum(domain_weights * head_outputs, dim=1) / torch.sum(domain_weights, dim=1)  # batch_size x num_classes
            
            if not self.training:
                # for inference, use weighted average of predictions from all heads
                result['logits'] = result['rel_logits']
            
            if self.loc_encoder and self.use_film:
                result['film_out'] = torch.mean([outputs[i]['film_out'] for i in range(self.num_heads)], dim=1)
        else:    
            outputs = self.classifiers[0](inputs)
            result['logits'] = outputs['x']

            if self.loc_encoder and self.use_film:
                result['film_out'] = outputs['film_out']
                result['z'] = outputs['z']

        return result


