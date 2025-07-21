import os
import random
import uuid

import matplotlib.pyplot as plt
import mlflow
import torch
from azure.core.exceptions import ResourceExistsError
from PIL import Image
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from data.poverty import PovertyMapDataset
from models.image_encoder import ResNet18Encoder
from models.lightning.base import WILDSLightningBase


class PovertyLightning(WILDSLightningBase):
    def __init__(
            self, 
            data_wrapper, 
            grouper=None,
            loc_encoder_weights=None,
            loc_encoder="none", 
            multihead=True, 
            lr_init=1e-3,
            weight_decay=1e-2,
            lr_scheduler={},
            coral_weight=0,
            irm_weight=0,
            use_film=False,
            film_penalty=0,
            film_lambda=0,
            d3g_beta=1.,
            dropout=0.1,
            d3g_loss_coeff=0.,
            group_dro=False,
            n_grad_accumulation=1
        ) -> None:
        assert isinstance(data_wrapper, PovertyMapDataset)
        image_encoder = ResNet18Encoder(num_channels=8)
        image_encoder_out_dim = image_encoder.out_dim

        super().__init__(data_wrapper, image_encoder_out_dim, grouper, loc_encoder_weights, loc_encoder, multihead, lr_init, weight_decay, lr_scheduler, 
                         coral_weight, irm_weight, use_film, film_penalty, film_lambda, d3g_beta, dropout, d3g_loss_coeff, group_dro, n_grad_accumulation)

        self.image_encoder = image_encoder
        self.loss = torch.nn.MSELoss(reduction='none')
        
    def forward(self, batch):
        x, _, metadata = batch
        embed = self.image_encoder(x)
        lonlat = self.data_wrapper.get_lon_lat(metadata)
        if self.loc_encoder is not None:
            loc_embed = self.loc_encoder(lonlat)
            embed = torch.cat([embed, loc_embed], dim=1).to(embed.dtype)
        if self.grouper is not None and self.classifier.num_heads > 1:
            groups = self.metadata_to_group(metadata)
        else:
            groups = None
        result = self.classifier(embed, lonlat, groups)
        return {'y_preds': result['logits'].float(), **result}

    @rank_zero_only
    def plot(self, batch, batch_idx, y_preds, stage, n=5):
        # plot n random predictions from the batch
        x, y, _ = batch
        fig, ax = plt.subplots(1, n, figsize=(20, 5))
        inverse_t = self.data_wrapper._load_inverse_transform(stage)

        for j in range(n):
            i = random.randint(0, len(batch))
            ax[j].set_title(f"True: {y[i].detach().cpu().item()}\nPred: {y_preds[i].detach().cpu().item()}")
            # PovertyMap data has 8 channels - first three are B,G,R. Convert to RGB before normalizing.
            ax[j].imshow(inverse_t(x[i][[2,1,0]].detach().cpu()))
            ax[j].xaxis.set_visible(False)
            ax[j].yaxis.set_visible(False)
        
        out_dir = f'outputs/{stage}/{self.global_step}'
        out_f = f'{out_dir}/batch_{batch_idx}_{str(uuid.uuid4())[:8]}.png'
        os.makedirs(out_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_f)
        plt.close()
        try:
            with Image.open(out_f) as img:
                mlflow.log_image(img, out_f)
        except (ResourceExistsError, Exception):
            pass