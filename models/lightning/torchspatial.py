import os
import random

import mlflow
import pandas as pd
import torch
import torch.nn.functional as F
from azure.core.exceptions import ResourceExistsError
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from data.inat import TorchSpatialiNat2018Dataset
from data.yfcc import TorchSpatialYfccDataset
from models.domain_relations import GroupEncoder
from models.lightning.base import LightningBase
from models.location_encoder import FCNet
from utils.train_utils import get_class_by_name


class TorchSpatialClassification(LightningBase):
    def __init__(
            self, 
            data_wrapper,
            grouper=None,
            loc_encoder_weights=None,
            loc_encoder="none", 
            use_loc_encoder_as_prior=False,
            domain_predictor_weight=0.,
            freeze_loc_encoder=True,
            multihead=False, 
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
        assert isinstance(data_wrapper, TorchSpatialiNat2018Dataset) or isinstance(data_wrapper, TorchSpatialYfccDataset)
        image_encoder_out_dim = data_wrapper.image_embed_dim

        super().__init__(data_wrapper, image_encoder_out_dim, grouper, loc_encoder_weights, loc_encoder, freeze_loc_encoder, use_loc_encoder_as_prior, domain_predictor_weight, multihead, lr_init, weight_decay, lr_scheduler, 
                         coral_weight, irm_weight, use_film, film_penalty, film_lambda, d3g_beta, d3g_use_loc_encoder, dropout, d3g_loss_coeff, group_dro, n_grad_accumulation, group_wise_logging)

        self.image_encoder = None

        if self.loc_encoder is not None and use_loc_encoder_as_prior:
            self.classifier = None

        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, batch):
        image_embed, _, metadata = batch
        lonlat = self.data_wrapper.get_lon_lat(metadata)
        if self.grouper is not None:
            groups = self.metadata_to_group(metadata)
        else:
            groups = None

        if self.image_encoder is not None:
            image_embed = self.image_encoder(image_embed)

        if self.loc_encoder is not None:
            if isinstance(self.loc_encoder, GroupEncoder):
                loc_embed = self.loc_encoder(groups)
            else:
                loc_embed = self.loc_encoder(lonlat)
            
            if self.loc_encoder_classifier is None:
                embed = torch.cat([image_embed, loc_embed], dim=1).to(image_embed.dtype)
            else:
                embed = image_embed
        else:
            embed = image_embed
            loc_embed = None

        if self.loc_encoder_classifier is not None:
            result = self.loc_encoder_classifier(loc_embed, lonlat, groups)
            result['loc_probs'] = F.sigmoid(result['logits'])
            y_preds = torch.argmax(embed * result['loc_probs'], dim=1)
        else:
            result = self.classifier(embed, lonlat, groups)
            y_preds = torch.argmax(F.softmax(result['logits'], dim=1), dim=1)

        return {'y_preds': y_preds, 'loc_embed': loc_embed, **result}

    def shared_epoch_end(self, stage: str):
        outputs = self.shared_step_metrics[stage]
        all_y = torch.cat([o['y'] for o in outputs]).detach().cpu()
        all_y_preds = torch.cat([o['y_preds'] for o in outputs]).detach().cpu()
        acc = (all_y == all_y_preds).float().sum() / all_y.shape[0]
        
        if 'd_preds' in outputs[0]:
            all_d = torch.cat([self.metadata_to_group(o['metadata']) for o in outputs]).detach().cpu()
            all_d_preds = torch.cat([o['d_preds'] for o in outputs]).detach().cpu()
            d_acc = (all_d == all_d_preds).float().sum() / all_d.shape[0]
        all_losses = torch.cat([o['losses'] for o in outputs]).detach().cpu()
        loss_avg = (all_losses.float().sum() / all_losses.shape[0]).item()

        metadata = torch.cat([o['metadata'] for o in outputs]).detach().cpu()
        groupby_metrics = self.data_wrapper.compute_groupby_metrics(all_y, all_y_preds, metadata, stage)
        d_pred_metrics = {f'{stage}_d_acc_avg': d_acc, } if 'd_preds' in outputs[0] else {}

        metrics = {f"{stage}_acc_avg": acc, f'{stage}_loss_avg': loss_avg, **groupby_metrics, **d_pred_metrics}
        self.log_dict(metrics, prog_bar=False, sync_dist=True)

        self.shared_step_metrics[stage] = []

    @rank_zero_only
    def plot(self, batch, batch_idx, y_preds, stage, n=20):
        _, y, _ = batch
        indices = random.sample(range(len(y)), min(n, len(y)))

        # Prepare lines to write
        lines = []
        for idx in indices:
            pred = y_preds[idx].item()
            true = y[idx].item()
            lines.append(f"predicted: {pred}, true: {true}\n")

        # Save to txt file
        filename = f'outputs/{stage}/{self.global_step}/sample_preds.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.writelines(lines)
        try:
            mlflow.log_artifact(filename, os.path.dirname(filename))
        except (ResourceExistsError, Exception):
            pass

    def configure_optimizers(self):
        if self.loc_encoder_classifier is not None:
            loc_encoder_head_optimizer = torch.optim.AdamW(list(self.loc_encoder_classifier.parameters()), lr=self.lr_init, weight_decay=self.weight_decay)
            loc_encoder_head_scheduler = get_class_by_name(self.lr_scheduler_config.class_reference)(
                loc_encoder_head_optimizer,
                **self.lr_scheduler_config.params
            )
            optimizers, schedulers = [loc_encoder_head_optimizer], [loc_encoder_head_scheduler]
        else:
            classifier_optimizer = torch.optim.AdamW(list(self.classifier.parameters()), lr=self.lr_init, weight_decay=self.weight_decay)
            classifier_scheduler = get_class_by_name(self.lr_scheduler_config.class_reference)(
                classifier_optimizer,
                **self.lr_scheduler_config.params
            )
            optimizers, schedulers = [classifier_optimizer], [classifier_scheduler]

        if self.image_encoder is not None:
            encoder_optimizer = torch.optim.AdamW(list(self.image_encoder.parameters()), lr=self.lr_init, weight_decay=self.weight_decay)
            encoder_scheduler = get_class_by_name(self.lr_scheduler_config.class_reference)(
                encoder_optimizer,
                **self.lr_scheduler_config.params
            )
            optimizers.append(encoder_optimizer)
            schedulers.append(encoder_scheduler)

        if self.domain_predictor is not None:
            domain_optimizer = torch.optim.AdamW(list(self.domain_predictor.parameters()), lr=self.lr_init / 10, weight_decay=0.)
            domain_scheduler = get_class_by_name(self.lr_scheduler_config.class_reference)(
                domain_optimizer,
                **self.lr_scheduler_config.params
            )
            optimizers.append(domain_optimizer)
            schedulers.append(domain_scheduler)

        if self.loc_encoder is not None:
            loc_encoder_optimizer = torch.optim.AdamW(list(self.loc_encoder.parameters()), lr=self.lr_init, weight_decay=0.)
            loc_encoder_scheduler = get_class_by_name(self.lr_scheduler_config.class_reference)(
                loc_encoder_optimizer,
                **self.lr_scheduler_config.params
            )
            optimizers.append(loc_encoder_optimizer)
            schedulers.append(loc_encoder_scheduler)

        return optimizers, schedulers