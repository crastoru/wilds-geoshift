from collections import defaultdict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from models.location_encoder import LocationEncoder
from models.multihead import MultiHeadClassifier
from utils.domain_adaptation import GroupDRO, coral_loss, irm_penalty
from utils.satclip_grouper import SatCLIPGrouper
from utils.train_utils import get_class_by_name


class WILDSLightningBase(pl.LightningModule):
    def __init__(
            self, 
            data_wrapper, 
            image_encoder_out_dim,
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

        super().__init__()
        self.data_wrapper = data_wrapper
        self.grouper = grouper
        
        if multihead:
            assert self.grouper is not None  # one head per group
            n_heads = grouper.n_groups
        else:
            n_heads = 1

        self.loc_encoder = None
        encoder_out_dim = image_encoder_out_dim

        # If loc_encoder was provided but we're not using any other method, use it for input features
        if not use_film and not multihead and not group_dro and coral_weight == 0 and irm_weight == 0:
            if loc_encoder != "none":
                print("Including loctaion encoder {} as input features...".format(loc_encoder))
                self.loc_encoder = LocationEncoder(loc_encoder, loc_encoder_weights, device="cpu")
                encoder_out_dim += self.loc_encoder.out_dim
        
        self.classifier = MultiHeadClassifier(n_heads, encoder_out_dim, self.data_wrapper.n_classes, dropout, d3g_beta, use_film, loc_encoder, loc_encoder_weights)

        if use_film and film_lambda > 0:
            assert self.grouper is not None
            self.domain_predictor = torch.nn.Linear(self.classifier.z_dim, self.grouper.n_groups)
        else:
            self.domain_predictor = None

        if group_dro:
            assert self.grouper is not None
            self.group_dro = GroupDRO(self.grouper.n_groups)
        else:
            self.group_dro = None

        self.coral_weight = coral_weight
        self.irm_weight = irm_weight
        self.film_penalty = film_penalty
        self.film_lambda = film_lambda
        self.d3g_loss_coeff = d3g_loss_coeff
        self.lr_init = lr_init
        self.weight_decay = weight_decay
        self.lr_scheduler_config = lr_scheduler
        self.N_grad_accumulation = n_grad_accumulation

        # set to False so we can use separate hyperparams for encoder/decoder training
        self.automatic_optimization = False

        # use these to calculate metrics
        self.shared_step_metrics = defaultdict(list)
        self.save_hyperparameters(ignore=['data_wrapper'])

    def forward(self, batch):
        raise NotImplementedError

    def plot(self, batch, batch_idx, y_preds, stage, n=5):
        raise NotImplementedError

    def metadata_to_group(self, metadata):
        if isinstance(self.grouper, SatCLIPGrouper):
            return self.grouper.metadata_to_group(metadata)
        
        # wilds Grouper uses CPU, so move metadata off GPU
        data_device = metadata.device
        return self.grouper.metadata_to_group(metadata.cpu()).to(data_device)

    def add_consistency_loss(self, loss, result, y):
        if self.film_penalty > 0 and 'film_out' in result.keys():
            loss = loss + self.film_penalty * F.mse_loss(result['features'], result['film_out'])
        if self.d3g_loss_coeff > 0 and 'rel_logits' in result.keys():
            loss = loss + self.d3g_loss_coeff * torch.mean(self.loss(result['rel_logits'], y))
        return loss

    def shared_step(self, batch, batch_idx, stage):
        results = self.forward(batch)
        logits = results['logits']
        features = results['features']
        y_preds = results['y_preds']
        _, y, metadata = batch

        # optimization step
        if stage == 'train':
            if batch_idx % self.N_grad_accumulation == 0:
                for opt in self.optimizers():
                    opt.zero_grad()

            if self.group_dro is not None or self.coral_weight > 0 or self.irm_weight > 0 or self.film_lambda > 0:
                groups = self.metadata_to_group(metadata)

            # calculate loss
            losses = self.loss(logits, y) 
            if self.group_dro is not None:
                loss, group_losses = self.group_dro.reweighted_loss(groups, losses)
            else:
                loss = torch.mean(losses)

            loss = self.add_consistency_loss(loss, results, y)

            if self.film_lambda > 0:
                loss = loss + self.film_lambda * F.cross_entropy(self.domain_predictor(results['z']), groups)

            if self.irm_weight > 0:
                loss = loss + self.irm_weight * irm_penalty(groups, logits, y, self.loss, self.device)

            if self.coral_weight > 0:
                loss = loss + self.coral_weight * coral_loss(groups, features, self.device)

            loss = loss / self.N_grad_accumulation
            self.manual_backward(loss)
            
            if self.group_dro is not None:
                self.group_dro.update_group_weights(group_losses)

            if (batch_idx + 1) % self.N_grad_accumulation == 0:
                for opt in self.optimizers():
                    opt.step()
        else:
            losses = self.loss(logits, y) 
            loss = torch.mean(losses)        

        batch_metrics = {
            "y": y,
            "y_preds": y_preds,
            "metadata": metadata,
            "losses": losses
        }

        if stage != 'train' and batch_idx < 2:
            self.plot(batch, batch_idx, y_preds, stage)

        self.log_dict({f'{stage}_loss': loss})
        self.shared_step_metrics[stage].append(batch_metrics)
    
    def shared_epoch_end(self, stage: str):
        if stage != 'train':
            outputs = self.shared_step_metrics[stage]
            all_y = torch.cat([o['y'] for o in outputs]).detach().cpu()
            all_y_preds = torch.cat([o['y_preds'] for o in outputs]).detach().cpu()
            all_metadata = torch.cat([o['metadata'] for o in outputs]).detach().cpu()
            all_losses = torch.cat([o['losses'] for o in outputs]).detach().cpu()
            
            wilds_metrics, _ = self.data_wrapper.dataset.eval(all_y, all_y_preds, all_metadata) 
            metrics = {f"{stage}_{m}":wilds_metrics[m] for m in wilds_metrics}
            metrics[f'{stage}_loss_avg'] = (all_losses.float().sum() / all_losses.shape[0]).item()

            self.log_dict(metrics, prog_bar=False, sync_dist=True)

        # restart metrics tracking
        self.shared_step_metrics[stage] = []

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'test')

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
        if self.domain_predictor is not None:
            domain_optimizer = torch.optim.AdamW(list(self.domain_predictor.parameters()), lr=self.lr_init / 10, weight_decay=self.weight_decay)
            domain_scheduler = get_class_by_name(self.lr_scheduler_config.class_reference)(
                domain_optimizer,
                **self.lr_scheduler_config.params
            )
            return [classifier_optimizer, encoder_optimizer, domain_optimizer], [classifier_scheduler, encoder_scheduler, domain_scheduler]

        return [classifier_optimizer, encoder_optimizer], [classifier_scheduler, encoder_scheduler]
    
    def on_train_epoch_end(self):
        for sch in self.lr_schedulers():
            sch.step()
        self.shared_epoch_end('train')

    def on_validation_epoch_end(self):
        self.shared_epoch_end('val')

    def on_test_epoch_end(self):
        self.shared_epoch_end('test')

    def on_fit_start(self):
        self.classifier.set_device(self.device)
        if self.group_dro is not None:
            self.group_dro.set_device(self.device)
        if self.loc_encoder is not None:
            self.loc_encoder.set_device(self.device)