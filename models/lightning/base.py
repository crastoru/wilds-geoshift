from collections import defaultdict
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from models.domain_relations import GroupEncoder
from models.location_encoder import LocationEncoder
from models.multihead import MultiHeadClassifier
from utils.domain_adaptation import (GroupDRO, compute_per_group_losses,
                                     coral_loss, irm_penalty)


class LightningBase(pl.LightningModule):
    def __init__(
            self, 
            data_wrapper,
            image_encoder_out_dim,
            grouper=None,
            loc_encoder_weights=None,
            loc_encoder="none", 
            freeze_loc_encoder=True,
            use_loc_encoder_as_prior=False,
            domain_predictor_weight=False,
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

        super().__init__()
        self.data_wrapper = data_wrapper
        self.grouper = grouper
        self.group_wise_logging = group_wise_logging
        
        if multihead:
            assert self.grouper is not None  # one head per group
            n_heads = grouper.n_groups
        else:
            n_heads = 1

        if loc_encoder == "groups":
            assert self.grouper is not None

        self.loc_encoder = None
        self.loc_encoder_classifier = None  # for geo prior
        self.domain_predictor = None
        self.classifier = None
        self.freeze_loc_encoder = freeze_loc_encoder
        encoder_out_dim = image_encoder_out_dim

        # If loc_encoder was provided but we're not using any other method, use it for input features
        if use_loc_encoder_as_prior or (not use_film and not multihead and not group_dro and coral_weight == 0 and irm_weight == 0):
            if loc_encoder != "none":
                print("Including location encoder {} as input features...".format(loc_encoder))
                if loc_encoder == "groups":
                    self.loc_encoder = GroupEncoder(self.grouper.n_groups, 256)
                else:
                    self.loc_encoder = LocationEncoder(loc_encoder, loc_encoder_weights, device="cpu", frozen=self.freeze_loc_encoder)
                if not use_loc_encoder_as_prior:  # concatenation
                    encoder_out_dim += self.loc_encoder.out_dim
                    
                if domain_predictor_weight > 0.:
                    print("Using domain predictor with location encoder with dims: {} x {}".format(self.loc_encoder.out_dim, grouper.n_groups if grouper is not None else self.data_wrapper.n_classes))
                    self.domain_predictor = torch.nn.Linear(self.loc_encoder.out_dim, grouper.n_groups if grouper is not None else self.data_wrapper.n_classes)
        
        if use_loc_encoder_as_prior:
            assert loc_encoder != "none", "To use location encoder as prior, must specify a location encoder"
            self.loc_encoder_classifier = MultiHeadClassifier(
                num_heads=n_heads,
                encoder_dim=self.loc_encoder.out_dim,
                num_classes=self.data_wrapper.n_classes,
                dropout=dropout,
                beta=d3g_beta,
                d3g_use_loc_encoder=d3g_use_loc_encoder,
                use_film=use_film,
                loc_encoder=loc_encoder,
                loc_encoder_weights=loc_encoder_weights,
                freeze_loc_encoder=freeze_loc_encoder
            )

            if use_film and film_lambda > 0:
                n_groups = self.grouper.n_groups if self.grouper is not None else self.data_wrapper.n_classes
                print("Using FiLM and domain predictor in geo prior classifier with dims: {} x {}".format(self.loc_encoder_classifier.z_dim, n_groups))
                self.domain_predictor = torch.nn.Linear(self.loc_encoder_classifier.z_dim, n_groups)

        self.classifier = MultiHeadClassifier(
            num_heads=n_heads, 
            encoder_dim=encoder_out_dim, 
            num_classes=self.data_wrapper.n_classes, 
            dropout=dropout, 
            beta=d3g_beta, 
            n_groups=self.grouper.n_groups if self.grouper is not None else 0,
            d3g_use_loc_encoder=d3g_use_loc_encoder, 
            use_film=use_film, 
            loc_encoder=loc_encoder, 
            loc_encoder_weights=loc_encoder_weights,
            freeze_loc_encoder=freeze_loc_encoder
        )

        if multihead and d3g_use_loc_encoder and domain_predictor_weight > 0.:
            loc_encoder_out_dim = self.classifier.domain_relation.loc_encoder.out_dim
            print("Using domain predictor with location encoder with dims: {} x {}".format(loc_encoder_out_dim, grouper.n_groups if grouper is not None else self.data_wrapper.n_classes))
            self.domain_predictor = torch.nn.Linear(loc_encoder_out_dim, grouper.n_groups if grouper is not None else self.data_wrapper.n_classes)

        if use_film and film_lambda > 0:
            n_groups = self.grouper.n_groups if self.grouper is not None else self.data_wrapper.n_classes
            print("Using FiLM and domain predictor in classifier with dims: {} x {}".format(self.classifier.z_dim, n_groups))
            self.domain_predictor = torch.nn.Linear(self.classifier.z_dim, n_groups)

        if group_dro:
            assert self.grouper is not None
            self.group_dro = GroupDRO(self.grouper.n_groups)
        else:
            self.group_dro = None

        self.domain_predictor_weight = domain_predictor_weight
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
        y_preds = results['y_preds']
        _, y, metadata = batch
        d_preds = None

        # optimization step
        if stage == 'train':
            if batch_idx % self.N_grad_accumulation == 0:
                optimizer = self.optimizers()
                for opt in optimizer if isinstance(optimizer, list) else [optimizer]:
                    opt.zero_grad()

            if self.film_lambda > 0 and self.grouper is None:
                groups = y
            elif self.group_dro is not None or self.domain_predictor_weight > 0 or self.coral_weight > 0 or self.irm_weight > 0 or self.film_lambda > 0:
                groups = self.metadata_to_group(metadata)

            # calculate loss
            if 'loc_probs' in results.keys():  # loc encoder as prior
                loc_prior_loss = -torch.log((1 - results['loc_probs']) + 1e-5)
                loc_prior_loss[torch.arange(len(loc_prior_loss)), y] = -torch.log(results['loc_probs'][torch.arange(len(loc_prior_loss)), y] + 1e-5) * self.data_wrapper.n_classes
                losses = loc_prior_loss.mean(dim=1)
                if 'image_logits' in results.keys():  # also have image logits
                    image_loss = self.loss(results['image_logits'], y)
                    losses = (losses + image_loss) / 2
            else:
                losses = self.loss(results['logits'], y) 
            
            if self.group_dro is not None:
                loss, group_losses = self.group_dro.reweighted_loss(groups, losses)
            else:
                loss = torch.mean(losses)

            loss = self.add_consistency_loss(loss, results, y)

            if self.film_lambda > 0 and self.domain_predictor is not None:
                d_logits = self.domain_predictor(results['z'])
                loss = loss + self.film_lambda * F.cross_entropy(d_logits, groups)
                d_preds = torch.argmax(F.softmax(d_logits, dim=1), dim=1)
                results['d_preds'] = d_preds

            if self.irm_weight > 0:
                loss = loss + self.irm_weight * irm_penalty(groups, results['logits'], y, self.loss, self.device)

            if self.coral_weight > 0:
                loss = loss + self.coral_weight * coral_loss(groups, results['features'], self.device)

            if self.domain_predictor_weight > 0. and 'loc_embed' in results.keys() and results['loc_embed'] is not None:
                groups = self.metadata_to_group(metadata)
                d_logits = self.domain_predictor(results['loc_embed'])
                d_preds = torch.argmax(F.softmax(d_logits, dim=1), dim=1)
                results['d_preds'] = d_preds
                domain_loss = F.cross_entropy(d_logits, groups)
                loss = loss + self.domain_predictor_weight * domain_loss

            loss = loss / self.N_grad_accumulation
            self.manual_backward(loss)

            if self.group_dro is not None:
                self.group_dro.update_group_weights(group_losses)

            if (batch_idx + 1) % self.N_grad_accumulation == 0:
                optimizer = self.optimizers()
                for opt in optimizer if isinstance(optimizer, list) else [optimizer]:
                    opt.step()
        else:
            if self.domain_predictor_weight > 0. and 'loc_embed' in results.keys() and results['loc_embed'] is not None:
                d_logits = self.domain_predictor(results['loc_embed'])
                d_preds = torch.argmax(F.softmax(d_logits, dim=1), dim=1)
                results['d_preds'] = d_preds

            if self.film_lambda > 0. and 'z' in results.keys() and results['z'] is not None:
                d_logits = self.domain_predictor(results['z'])
                d_preds = torch.argmax(F.softmax(d_logits, dim=1), dim=1)
                results['d_preds'] = d_preds

            if 'loc_probs' in results.keys():  # loc encoder as prior
                loc_prior_loss = -torch.log((1 - results['loc_probs']) + 1e-5)
                loc_prior_loss[torch.arange(len(loc_prior_loss)), y] = -torch.log(results['loc_probs'][torch.arange(len(loc_prior_loss)), y] + 1e-5) * self.data_wrapper.n_classes
                losses = loc_prior_loss.mean(dim=1)
                if 'image_logits' in results.keys():  # also have image logits
                    image_loss = self.loss(results['image_logits'], y)
                    losses = (losses + image_loss) / 2
            else:
                losses = self.loss(results['logits'], y) 
            loss = torch.mean(losses)        

        batch_metrics = {
            "y": y,
            "y_preds": y_preds,
            "metadata": metadata,
            "losses": losses,
        }

        if 'd_preds' in results.keys() and self.grouper is not None:
            batch_metrics['d_preds'] = results['d_preds']
            groups = self.metadata_to_group(metadata)

        if stage != 'train' and batch_idx < 2:
            self.plot(batch, batch_idx, y_preds, stage)     

        self.log_dict({
            f'{stage}_loss': loss, 
            f'{stage}_acc': torch.sum((y_preds == y).float()) / len(y), 
            f'{stage}_d_acc': torch.sum((d_preds == groups).float()) / len(groups) if 'd_preds' in results.keys() else 0
            }, on_step=(stage == 'train'))

        if self.grouper is not None and self.group_wise_logging:
            groups = self.metadata_to_group(metadata)
            group_losses = compute_per_group_losses(groups, losses)
            for i, group_idx in enumerate(group_losses.keys()):
                group_name = self.grouper.group_field_str(group_idx)
                self.log(f'{stage}_loss_{group_name}', group_losses[group_idx], on_step=(stage == 'train'))
                group_acc = torch.sum((y_preds[groups == group_idx] == y[groups == group_idx]).float()) / torch.sum((groups == group_idx).float())
                if stage == 'train':
                    self.log(f'{stage}_acc_{group_name}', group_acc, on_step=(stage == 'train'))
                batch_metrics["acc_{}".format(group_name)] = group_acc

        self.shared_step_metrics[stage].append(batch_metrics)
    
    def shared_epoch_end(self, stage: str):
        # restart metrics tracking
        self.shared_step_metrics[stage] = []

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, 'test')
		
    def on_train_epoch_end(self):
        lr_scheduler = self.lr_schedulers()
        if isinstance(lr_scheduler, List):
            for sch in lr_scheduler:
                sch.step()
        else:
            lr_scheduler.step()

        self.shared_epoch_end('train')

    def on_validation_epoch_end(self):
        self.shared_epoch_end('val')

    def on_test_epoch_end(self):
        self.shared_epoch_end('test')

    def on_fit_start(self):
        if self.classifier is not None:
            self.classifier.set_device(self.device)
        if self.group_dro is not None:
            self.group_dro.set_device(self.device)
        if self.loc_encoder is not None:
            self.loc_encoder.set_device(self.device)
        if self.loc_encoder_classifier is not None:
            self.loc_encoder_classifier.set_device(self.device)