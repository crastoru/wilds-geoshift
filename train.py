import argparse
import logging
import os
from math import floor

import mlflow
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.seed import seed_everything
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.strategies import DDPStrategy
from wilds.common.data_loaders import get_eval_loader, get_train_loader

from utils.loc_encoder_grouper import LocEncoderGrouper
from utils.train_utils import (find_best_checkpoint, get_class_by_name,
                               load_config)

logging.basicConfig(
    format='%(asctime)s :: %(name)s :: %(levelname)-8s :: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("wilds-geoshift")
logger.setLevel(logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the parent directory of data folder (compressed or uncompressed)")
    parser.add_argument("--config", type=str, required=True, help="List of config files, e.g. fmow,wraplinear,film")
    parser.add_argument("--loc-encoder-weights", type=str, required=False, help="Path to .ckpt file containing SatCLIP location encoder weights.") 
    parser.add_argument("--checkpoint", type=str, required=False, default='none', help="Path to checkpoint file. If 'none', train a new model.") 
    parser.add_argument("--experiment-name", type=str, default="wilds-geoshift")

    # need this for aml
    args = parser.parse_known_args()[0]
    os.makedirs("outputs", exist_ok=True)

    # read yaml config
    config = load_config(os.path.join(os.getcwd(), "configs"), args.config)
    print("-" * 50)
    print(OmegaConf.to_yaml(config, resolve=True, sort_keys=True))
    print("-" * 50)

    seed_everything(config.training.seed, workers=True)

    dev_count = torch.cuda.device_count()
    n_cpus = os.cpu_count()
    n_gpus = 1 if dev_count == 0 else dev_count # avoid divide by zero
    cpus_per_gpu = floor(n_cpus/n_gpus)
    logger.info(f"found {n_cpus} cpus and {dev_count} gpus. assigning {cpus_per_gpu} cpus per gpu")
    valid_and_test_workers = max(1, floor(cpus_per_gpu * .15))
    train_loader_workers = cpus_per_gpu - (valid_and_test_workers * 2)

    # Extract and load the dataset
    logger.info("Preparing dataset...")
    data_wrapper = get_class_by_name(config.dataset.class_reference)(args.data_dir, **config.dataset.params)
    
    train_data = data_wrapper.get_split('train')
    val_data = data_wrapper.get_split('val')
    test_data = data_wrapper.get_split('test')

    logger.info(f"Splits: train {len(train_data)} | val {len(val_data)} | test {len(test_data)}")
        
    if 'grouper' in config.keys():
        grouper_cls = get_class_by_name(config.grouper.class_reference)
        
        if grouper_cls == LocEncoderGrouper:
            grouper = grouper_cls(
                data_wrapper, 
                config.grouper.params.loc_encoder,
                config.grouper.params.n_groups,
                config.training.seed, 
                device="cuda" if dev_count > 0 else "cpu", 
                num_workers=cpus_per_gpu,
                loc_encoder_weights=args.loc_encoder_weights if args.loc_encoder_weights else None
            )
        else:
            grouper = grouper_cls(
                data_wrapper.dataset,
                **config.grouper.params
            )
    else:
        grouper = None

    train_loader = get_train_loader(
        "standard", 
        train_data, 
        grouper=grouper,
        uniform_over_groups=config.data_loader.uniform_over_groups,
        batch_size=config.data_loader.batch_size, 
        num_workers=train_loader_workers
    )
    
    val_loader = get_eval_loader(
        "standard", 
        val_data, 
        batch_size=config.data_loader.batch_size, 
        num_workers=valid_and_test_workers
    )

    test_loader = get_eval_loader(
        "standard",
         test_data,
         batch_size=config.data_loader.batch_size,
         num_workers=valid_and_test_workers
    )

    model = get_class_by_name(config.model.class_reference)(
          data_wrapper,
          grouper=grouper,
          loc_encoder_weights=args.loc_encoder_weights,
          **config.model.params
    )

    mflow_logger = MLFlowLogger(
        tracking_uri=mlflow.get_tracking_uri(), 
        experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", args.experiment_name),
        run_id=os.environ.get("AZUREML_RUN_ID")
    )

    if args.checkpoint == 'none':
        checkpoint_dir = os.path.join("outputs", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        ckpt_callback = ModelCheckpoint(
                save_top_k=config.checkpoint.save_top_k,
                monitor=config.checkpoint.monitor,
                mode=config.checkpoint.mode,
                dirpath=str(checkpoint_dir),
                filename=config.checkpoint.filename
            )
    
        callbacks = [
            ckpt_callback,
            LearningRateMonitor(logging_interval='step')
        ]
        epochs = config.training.epochs
        
    else:
        checkpoint_dir = args.checkpoint
        callbacks = []
        epochs = 0

    default_root_dir = os.path.join("outputs", "trainer")
    os.makedirs(default_root_dir, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=mflow_logger,
        default_root_dir=str(default_root_dir),
        log_every_n_steps=50,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="16-mixed",
        devices=1
    )

    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )

    # hack to ensure hyperparameters are not re-logged at test time (MLFlow throws an error)
    trainer.lightning_module._log_hyperparams = False  

    if args.checkpoint == 'none':
        logger.info("Testing best model...")
        best_model_path = find_best_checkpoint(checkpoint_dir, config.checkpoint.mode == 'max')
        logger.info(f"Loading best model {best_model_path}...")
    else:
        best_model_path = args.checkpoint
        logger.info(f"Loading checkpoint {best_model_path}...")

    test_metrics = trainer.test(ckpt_path=best_model_path, dataloaders=test_loader, verbose=True)
    logger.info(f"Test metrics: {test_metrics}")


if __name__ == '__main__':
    main()