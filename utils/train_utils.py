import importlib
import os
from pathlib import Path

from omegaconf import OmegaConf


def find_best_checkpoint(output_dir: str, higher_is_better=True):
    checkpoints = list(Path(output_dir).glob("*.ckpt"))
    if len(checkpoints) == 0:
        raise ValueError(f"no checkpoints found in {output_dir}")
    checkpoints.sort(key=lambda x: float(x.stem.split("=")[-1]))
    return checkpoints[-1] if higher_is_better else checkpoints[0]

def load_config(config_dir: str, config_names: str):
    configs = []
    for c in config_names.split(","):
        found = False
        for root, dirs, files in os.walk(config_dir):
            if f"{c}.yaml" in files:
                config_path = os.path.join(root, f"{c}.yaml")
                configs.append(OmegaConf.load(config_path))
                found = True
                break
        if not found:
            raise ValueError(f"Could not find config file {c}.yaml in {config_dir}")

    return OmegaConf.merge(*configs)

def get_class_by_name(cls_reference: str):
    parts = cls_reference.split(".")
    module_name = ".".join(parts[:-1])
    class_name = parts[-1]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)