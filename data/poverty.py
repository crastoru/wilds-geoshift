from typing import Dict, List

import numpy as np
import torchvision.transforms as transforms
from wilds.datasets.poverty_dataset import _MEANS_2009_17, _STD_DEVS_2009_17

from data.base import denormalize
from data.wilds_base import WILDSDatasetWrapper

POVERTY_RGB_MEAN = np.array([_MEANS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1))
POVERTY_RGB_STD = np.array([_STD_DEVS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1))


class PovertyMapDataset(WILDSDatasetWrapper):
    def __init__(self, 
                 data_dir: str,
                 splits: List[str] = None, 
                 group_field: str = None,
                 split_groups: Dict[str, List[str]] = None,
                 combine_id_ood: bool = False,
                 fold: str = 'A'
                ) -> None:
        self.image_size = 224
        self.n_classes = 1  # regression

        # Call superclass init to get the splits
        super().__init__(data_dir, dataset_name="poverty", version_str="v1.1", splits=splits, 
                         group_field=group_field, split_groups=split_groups, combine_id_ood=combine_id_ood, fold=fold)

    # Taken from: https://github.com/p-lambda/wilds/blob/472677590de351857197a9bf24958838c39c272b/examples/transforms.py#L246
    def poverty_rgb_color_transform(self, ms_img, transform):
        poverty_rgb_means = np.array([_MEANS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1))
        poverty_rgb_stds = np.array([_STD_DEVS_2009_17[c] for c in ['RED', 'GREEN', 'BLUE']]).reshape((-1, 1, 1))

        def unnormalize_rgb_in_poverty_ms_img(ms_img):
            result = ms_img.detach().clone()
            result[:3] = (result[:3] * poverty_rgb_stds) + poverty_rgb_means
            return result

        def normalize_rgb_in_poverty_ms_img(ms_img):
            result = ms_img.detach().clone()
            result[:3] = (result[:3] - poverty_rgb_means) / poverty_rgb_stds
            return result

        color_transform = transforms.Compose([
            transforms.Lambda(lambda ms_img: unnormalize_rgb_in_poverty_ms_img(ms_img)),
            transform,
            transforms.Lambda(lambda ms_img: normalize_rgb_in_poverty_ms_img(ms_img)),
        ])
        # The first three channels of the Poverty MS images are BGR
        # So we shuffle them to the standard RGB to do the ColorJitter
        # Before shuffling them back
        ms_img[:3] = color_transform(ms_img[[2,1,0]])[[2,1,0]] # bgr to rgb to bgr
        return ms_img

    def _load_transform(self, split: str):
        return None
    
    def _load_inverse_transform(self, split):
        return lambda t: denormalize(t, POVERTY_RGB_MEAN, 2 * POVERTY_RGB_STD).permute(1, 2, 0).numpy()