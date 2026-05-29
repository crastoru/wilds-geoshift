import argparse
import logging
import os
from math import floor

import mlflow
import pytorch_lightning as pl
import tabulate
import torch
from lightning_fabric.utilities.seed import seed_everything
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

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
    parser.add_argument("--prithvi-weights", type=str, required=False, help="Path to .pt file containing Prithvi image encoder weights.")
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
    dataset_params = config.dataset.params if 'params' in config.dataset else {}
    if 'TorchSpatial' not in config.dataset.class_reference and 'return_image_preds' in dataset_params:
        dataset_params.pop('return_image_preds')
    data_wrapper = get_class_by_name(config.dataset.class_reference)(args.data_dir, **dataset_params)

    train_data = data_wrapper.get_split(config.data_loader.get('train_split', 'train'))
    val_data = data_wrapper.get_split(config.data_loader.get('val_split', 'val'))
    test_data = data_wrapper.get_split(config.data_loader.get('test_split', 'test'))

    logger.info(f"Splits: train {len(train_data)} | val {len(val_data)} | test {len(test_data)}")
        
    if 'grouper' not in config.keys():
        grouper = None
    else:
        grouper = get_class_by_name(config.grouper.class_reference)(train_data.dataset, **config.grouper.params)

        # log counts per group in each split
        _, train_counts = grouper.metadata_to_group(train_data.metadata_array, return_counts=True)
        _, val_counts = grouper.metadata_to_group(val_data.metadata_array, return_counts=True)
        _, test_counts = grouper.metadata_to_group(test_data.metadata_array, return_counts=True)
        table = [
            [grouper.group_str(i), train_counts[i], val_counts[i], test_counts[i]]
            for i in range(grouper.n_groups)
        ]

        logger.info("\n" + tabulate.tabulate(
            table,
            headers=["Group", "Train", "Val", "Test"],
            tablefmt="github"
        ))

    train_loader = DataLoader(
        train_data,
        batch_size=config.data_loader.batch_size,
        shuffle=True,
        num_workers=train_loader_workers
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config.data_loader.batch_size,
        shuffle=False,
        num_workers=valid_and_test_workers
    )

    test_loader = DataLoader(
        test_data,
        batch_size=config.data_loader.batch_size,
        shuffle=False,
        num_workers=valid_and_test_workers
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
                filename=config.checkpoint.filename,
            )
    
        callbacks = [
            ckpt_callback,
            LearningRateMonitor(logging_interval='step')
        ]
        epochs = config.training.epochs
        
    else:
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

    if epochs > 0:
        if 'image_encoder' in config.model.params and config.model.params.image_encoder == 'prithvi':
            logger.info("Using Prithvi image encoder with weights from {}".format(args.prithvi_weights))
            config.model.params['prithvi_weights'] = args.prithvi_weights

        model = get_class_by_name(config.model.class_reference)(
            data_wrapper,
            grouper=grouper,
            loc_encoder_weights=args.loc_encoder_weights,
            **config.model.params
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

    if epochs > 0:
        test_metrics = trainer.test(ckpt_path=best_model_path, dataloaders=test_loader, verbose=True, weights_only=False)
    else:
        model = get_class_by_name(config.model.class_reference).load_from_checkpoint(
            best_model_path,
            map_location="cuda" if dev_count > 0 else "cpu",
            data_wrapper=data_wrapper,
            grouper=grouper,
            loc_encoder_weights=args.loc_encoder_weights,
            **config.model.params
        )
        model.eval()
        test_metrics = trainer.test(model, dataloaders=test_loader, verbose=True)

    logger.info(f"Test metrics: {test_metrics}")


if __name__ == '__main__':
    main()