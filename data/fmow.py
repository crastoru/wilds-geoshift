from typing import Dict, List

import torchvision.transforms as transforms

from data.base import denormalize
from data.wilds_base import WILDSDatasetWrapper
from utils.transforms import FIX_MATCH_AUGMENTATION_POOL, RandAugment

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class FMoWDataset(WILDSDatasetWrapper):
    def __init__(self, 
                 data_dir: str, 
                 randaug: bool, 
                 clip_processor: str = None, 
                 splits: List[str] = None, 
                 group_field: str = None,
                 split_groups: Dict[str, List[str]] = None,
                 combine_id_ood: bool = False,
                 eval_only_transforms: bool = False
                 ) -> None:

        self.image_size = 224
        self.randaug = randaug
        self.clip_processor = clip_processor
        self.eval_only_transforms = eval_only_transforms

        # Call superclass init to get the splits
        super().__init__(data_dir, dataset_name="fmow", version_str="v1.1", splits=splits, group_field=group_field,
                         split_groups=split_groups, combine_id_ood=combine_id_ood, remove_split='seq')
        
        self.n_classes = self._dataset._n_classes

    def _load_transform(self, split: str):
        if self.clip_processor is not None:
            size = 224 if 'base' in self.clip_processor else int(self.clip_processor.split('-')[-1])
            default_transforms = [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ]
        else:
            default_transforms = [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ]

        if split != 'train' or self.eval_only_transforms:
            return transforms.Compose(default_transforms)

        default_train_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(self.image_size)
        ]
        if self.randaug:
            default_train_transforms.append(RandAugment(2, FIX_MATCH_AUGMENTATION_POOL))
        
        return transforms.Compose([
            *default_train_transforms,
            *default_transforms
            ])


    def _load_inverse_transform(self, split):
        if split == 'train':
            raise ValueError("Cannot perform inverse transform on training data.")
        else:
            return lambda t: denormalize(t, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD).permute(1, 2, 0).numpy()