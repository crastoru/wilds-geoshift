import os
import random
import uuid

import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn.functional as F
from azure.core.exceptions import ResourceExistsError
from PIL import Image
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from wilds.datasets.fmow_dataset import categories as fmow_categories

from data.fmow import FMoWDataset
from models.domain_relations import GroupEncoder
from models.image_encoder import CLIPEncoder, Prithvi
from models.lightning.base import LightningBase
from utils.train_utils import get_class_by_name


class FMoWLightning(LightningBase):
    def __init__(
            self, 
            data_wrapper, 
            image_encoder='clip',
            prithvi_weights=None,
            grouper=None,
            loc_encoder_weights=None,
            loc_encoder="none", 
            freeze_loc_encoder=True,
            use_loc_encoder_as_prior=False,
            domain_predictor_weight=0.,
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
            d3g_use_loc_encoder=False,
            dropout=0.1,
            d3g_loss_coeff=0.,
            group_dro=False,
            n_grad_accumulation=1,
            group_wise_logging=False
        ) -> None:
        assert isinstance(data_wrapper, FMoWDataset)
        
        if image_encoder == 'clip':
            image_encoder = CLIPEncoder(data_wrapper.clip_processor)
        elif image_encoder == 'prithvi':
            image_encoder = Prithvi(prithvi_weights)

        image_encoder_out_dim = image_encoder.out_dim

        super().__init__(data_wrapper, image_encoder_out_dim, grouper, loc_encoder_weights, loc_encoder, freeze_loc_encoder, use_loc_encoder_as_prior, domain_predictor_weight, multihead, lr_init, weight_decay, lr_scheduler, 
                         coral_weight, irm_weight, use_film, film_penalty, film_lambda, d3g_beta, d3g_use_loc_encoder, dropout, d3g_loss_coeff, group_dro, n_grad_accumulation, group_wise_logging)

        self.image_encoder = image_encoder
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, batch):
        x, _, metadata = batch
        embed = self.image_encoder(x)
        lonlat = self.data_wrapper.get_lon_lat(metadata)
        if self.grouper is not None:
            groups = self.metadata_to_group(metadata)
        else:
            groups = None

        if self.loc_encoder is not None:
            if isinstance(self.loc_encoder, GroupEncoder):
                loc_embed = self.loc_encoder(groups)
            else:
                loc_embed = self.loc_encoder(lonlat)
            if self.loc_encoder_classifier is None:
                embed = torch.cat([embed, loc_embed], dim=1).to(embed.dtype)
        else:
            loc_embed = None

        result = self.classifier(embed, lonlat, groups)
        y_preds = torch.argmax(F.softmax(result['logits'], dim=1), dim=1)

        if self.loc_encoder_classifier is not None:
            image_probs = F.softmax(result['logits'], dim=1)
            image_logits = result['logits']
            result = self.loc_encoder_classifier(loc_embed, lonlat, groups)
            result['loc_probs'] = F.sigmoid(result['logits'])
            result['image_logits'] = image_logits
            y_preds = torch.argmax(image_probs * result['loc_probs'], dim=1)

        return {'y_preds': y_preds, 'loc_embed': loc_embed, **result}

    def shared_epoch_end(self, stage: str):
        outputs = self.shared_step_metrics[stage]
        all_y = torch.cat([o['y'] for o in outputs]).detach().cpu()
        all_y_preds = torch.cat([o['y_preds'] for o in outputs]).detach().cpu()
        all_metadata = torch.cat([o['metadata'] for o in outputs]).detach().cpu()
        all_losses = torch.cat([o['losses'] for o in outputs]).detach().cpu()

        if stage == "train":
            acc = (all_y == all_y_preds).float().sum() / all_y.shape[0]
            metrics = {f"{stage}_acc_avg": acc}
        else:
            wilds_metrics, _ = self.data_wrapper.get_split(stage).eval(all_y_preds, all_y, all_metadata) 
            metrics = {f"{stage}_{m}":wilds_metrics[m] for m in wilds_metrics}
            metrics[f'{stage}_loss_avg'] = (all_losses.float().sum() / all_losses.shape[0]).item()

        self.log_dict(metrics, prog_bar=False, sync_dist=True)
        self.shared_step_metrics[stage] = []

    @rank_zero_only
    def plot(self, batch, batch_idx, y_preds, stage, n=5):
        # plot n random predictions from the batch
        x, y, _ = batch
        if not isinstance(x, torch.Tensor):
            x = x['pixel_values'][0]
        fig, ax = plt.subplots(1, n, figsize=(20, 5))
        inverse_t = self.data_wrapper._load_inverse_transform(stage)

        for j in range(n):
            i = random.randint(0, len(batch))
            ax[j].set_title(f"True: {fmow_categories[y[i].detach().cpu().item()]}\nPred: {fmow_categories[y_preds[i].detach().cpu().item()]}")
            ax[j].imshow(inverse_t(x[i].detach().cpu()))
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

    def configure_optimizers(self):
        classifier_optimizer = torch.optim.AdamW(list(self.classifier.parameters()), lr=self.lr_init, weight_decay=self.weight_decay)
        classifier_scheduler = get_class_by_name(self.lr_scheduler_config.class_reference)(
            classifier_optimizer,
            **self.lr_scheduler_config.params
        )

        encoder_optimizer = torch.optim.AdamW(list(self.image_encoder.parameters()), lr=self.lr_init / 10, weight_decay=self.weight_decay)
        encoder_scheduler = get_class_by_name(self.lr_scheduler_config.class_reference)(
            encoder_optimizer,
            **self.lr_scheduler_config.params
        )

        optimizers, schedulers = [classifier_optimizer, encoder_optimizer], [classifier_scheduler, encoder_scheduler]

        if self.loc_encoder_classifier is not None:
            loc_encoder_head_optimizer = torch.optim.AdamW(list(self.loc_encoder_classifier.parameters()), lr=self.lr_init, weight_decay=self.weight_decay)
            loc_encoder_head_scheduler = get_class_by_name(self.lr_scheduler_config.class_reference)(
                loc_encoder_head_optimizer,
                **self.lr_scheduler_config.params
            )
            optimizers.append(loc_encoder_head_optimizer)
            schedulers.append(loc_encoder_head_scheduler)

        if self.domain_predictor is not None:
            domain_optimizer = torch.optim.AdamW(list(self.domain_predictor.parameters()), lr=self.lr_init / 10, weight_decay=self.weight_decay)
            domain_scheduler = get_class_by_name(self.lr_scheduler_config.class_reference)(
                domain_optimizer,
                **self.lr_scheduler_config.params
            )
            optimizers.append(domain_optimizer)
            schedulers.append(domain_scheduler)

        if self.loc_encoder is not None and self.loc_encoder.has_trainable_params():
            loc_encoder_optimizer = torch.optim.AdamW(list(self.loc_encoder.parameters()), lr=self.lr_init, weight_decay=self.weight_decay)
            loc_encoder_scheduler = get_class_by_name(self.lr_scheduler_config.class_reference)(
                loc_encoder_optimizer,
                **self.lr_scheduler_config.params
            )
            optimizers.append(loc_encoder_optimizer)
            schedulers.append(loc_encoder_scheduler)

        return optimizers, schedulers
    

    def shared_epoch_end(self, stage: str):
        if stage != 'train':
            outputs = self.shared_step_metrics[stage]
            all_y = torch.cat([o['y'] for o in outputs]).detach().cpu()
            all_y_preds = torch.cat([o['y_preds'] for o in outputs]).detach().cpu()
            all_metadata = torch.cat([o['metadata'] for o in outputs]).detach().cpu()
            all_losses = torch.cat([o['losses'] for o in outputs]).detach().cpu()
            
            wilds_metrics, _ = self.data_wrapper.get_split(stage).eval(all_y, all_y_preds, all_metadata) 
            metrics = {f"{stage}_{m}":wilds_metrics[m] for m in wilds_metrics}
            metrics[f'{stage}_loss_avg'] = (all_losses.float().sum() / all_losses.shape[0]).item()

            self.log_dict(metrics, prog_bar=False, sync_dist=True)

        self.shared_step_metrics[stage] = []