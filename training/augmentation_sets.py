import random
from abc import ABC, abstractmethod
from typing import Dict

import albumentations as A


class WarmingAugmentations(ABC):
    @abstractmethod
    def get_train_augmentations(self, config: Dict) -> A.Compose:
        pass

    @abstractmethod
    def get_val_augmentations(self, config: Dict) -> A.Compose:
        pass


class Rotate90_270(A.RandomRotate90):
    def get_params(self):
        return {"factor": random.choice([1, 3])}


class FlipsAugs(WarmingAugmentations):

    def get_train_augmentations(self, config: Dict) -> A.Compose:
        size = config.get("size", 512)

        return A.ReplayCompose([
            A.Resize(size, size),
            A.OneOf([
                A.HorizontalFlip(p=0.33),
                Rotate90_270(p=0.67)], p=0.75)
        ])

    def get_val_augmentations(self, config: Dict) -> A.Compose:
        size = config.get("size", 512)

        return A.ReplayCompose([
            A.Resize(size, size),
        ])
