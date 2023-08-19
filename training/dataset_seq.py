import os.path
import random

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import ReplayCompose
from torch.utils.data import Dataset

_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)


def normalize_range(data, bounds):
    """Maps data to the range [0, 1]."""
    return (data - bounds[0]) / (bounds[1] - bounds[0])


class DatasetSeq(Dataset):
    def __init__(
            self,
            mode: str,
            dataset_dir: str,
            transforms: A.Compose,
    ):
        self.mode = mode
        self.dataset_dir = os.path.join(dataset_dir, "validation" if mode == "val" else "train")
        self.transforms = transforms
        self.ids = os.listdir(self.dataset_dir)

    def __getitem__(self, i):
        try:
            return self.getitem(i)
        except Exception as e:
            print(e)
            return self.getitem(random.randint(0, len(self.ids) - 1))

    def getitem(self, i):
        file_id = self.ids[i]
        band11 = np.load(os.path.join(self.dataset_dir, file_id, 'band_11.npy'))
        band14 = np.load(os.path.join(self.dataset_dir, file_id, 'band_14.npy'))
        band15 = np.load(os.path.join(self.dataset_dir, file_id, 'band_15.npy'))

        r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
        g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
        b = normalize_range(band14, _T11_BOUNDS)

        imgs = np.array([r, g, b])

        imgs = np.transpose(imgs, (3, 1, 2, 0))
        mask = np.load(os.path.join(self.dataset_dir, file_id, "human_pixel_masks.npy")).astype(np.uint8)

        replay = None
        image_crops = []
        mask_crops = []
        for i in range(len(imgs)):
            image = imgs[i]
            if replay is None:
                sample = self.transforms(image=image, mask=mask)
                replay = sample["replay"]
            else:
                sample = ReplayCompose.replay(replay, image=image, mask=mask)
            image_ = sample["image"]
            image_crops.append(image_)
            mask_crops.append(sample["mask"])
        images = np.array(image_crops)
        masks = mask_crops[0]
        sample = {}
        sample['mask'] = torch.from_numpy(np.moveaxis(masks, -1, 0)).float()
        sample['image'] = torch.from_numpy(np.moveaxis(images, -1, 1)).float()

        return sample

    @property
    def is_train(self):
        return self.mode == "train"

    def __len__(self):
        return len(self.ids)

class DatasetSingleFolds(Dataset):
    def __init__(
            self,
            fold: int,
            folds_csv: str,
            mode: str,
            dataset_dir: str,
            transforms: A.Compose,
            limit: int = 999999
    ):
        self.mode = mode
        self.dataset_dir = os.path.join(dataset_dir, "train")
        self.transforms = transforms
        df = pd.read_csv(folds_csv, dtype={"file_id": str, "fold": int})
        if mode == "train":
            df = df[df.fold != fold]
        else:
            df = df[df.fold == fold]
        self.df = df
        self.ids = df.file_id.values
        self.limit = limit

    def __getitem__(self, i):
        try:
            return self.getitem(i)
        except Exception as e:
            print(e)
            return self.getitem(random.randint(0, len(self.ids) - 1))

    def getitem(self, i):
        file_id = self.ids[i]

        band11 = np.load(os.path.join(self.dataset_dir, file_id, 'band_11.npy'))
        band14 = np.load(os.path.join(self.dataset_dir, file_id, 'band_14.npy'))
        band15 = np.load(os.path.join(self.dataset_dir, file_id, 'band_15.npy'))

        r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
        g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
        b = normalize_range(band14, _T11_BOUNDS)

        imgs = np.array([r, g, b])

        imgs = np.transpose(imgs, (3, 1, 2, 0))[4]
        mask = np.load(os.path.join(self.dataset_dir, file_id, "human_pixel_masks.npy")).astype(np.uint8)

        sample = self.transforms(image=imgs, mask=mask)
        images = sample["image"]
        masks = sample["mask"]
        sample = {}
        sample['mask'] = torch.from_numpy(np.moveaxis(masks, -1, 0)).float()
        sample['image'] = torch.from_numpy(np.moveaxis(images, -1, 0)).float()

        return sample

    @property
    def is_train(self):
        return self.mode == "train"

    def __len__(self):
        return min(len(self.ids), self.limit)


class DatasetSinglePseudoRound(Dataset):
    def __init__(
            self,
            fold: int,
            folds_csv: str,
            mode: str,
            dataset_dir: str,
            transforms: A.Compose,
            limit: int = 999999
    ):
        self.mode = mode
        self.root_dir = dataset_dir
        self.dataset_dir = os.path.join(dataset_dir, "train")
        self.transforms = transforms
        df = pd.read_csv(folds_csv, dtype={"file_id": str, "fold": int})
        if mode == "train":
            df = df[df.fold != fold]
        else:
            df = df[df.fold == fold]
        self.df = df
        self.ids = df.file_id.values
        self.limit = limit

    def __getitem__(self, i):
        try:
            return self.getitem(i)
        except Exception as e:
            print(e)
            return self.getitem(random.randint(0, len(self.ids) - 1))

    def getitem(self, i):
        file_id = self.ids[i]

        band11 = np.load(os.path.join(self.dataset_dir, file_id, 'band_11.npy'))
        band14 = np.load(os.path.join(self.dataset_dir, file_id, 'band_14.npy'))
        band15 = np.load(os.path.join(self.dataset_dir, file_id, 'band_15.npy'))

        r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
        g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
        b = normalize_range(band14, _T11_BOUNDS)

        imgs = np.array([r, g, b])

        imgs = np.transpose(imgs, (3, 1, 2, 0))
        if self.is_train:
            pseudo_mask = np.load(os.path.join(f"{self.root_dir}/seg_preds", f"{file_id}.npy")).astype(
                np.float32)
            pseudo_mask[np.isnan(pseudo_mask)] = 0
        mask = np.load(os.path.join(self.dataset_dir, file_id, "human_pixel_masks.npy")).astype(np.uint8)

        if self.is_train and random.random() < 0.4:
            idx = random.randint(0, 7)
            pseudo_mask = pseudo_mask[idx]
            imgs = imgs[idx]
            if idx != 4:
                mask = np.expand_dims(pseudo_mask.astype(np.float32), -1)
        else:
            imgs = imgs[4]

        sample = self.transforms(image=imgs, mask=mask)
        images = sample["image"]
        masks = sample["mask"]
        sample = {}
        sample['mask'] = torch.from_numpy(np.moveaxis(masks, -1, 0)).float()
        sample['image'] = torch.from_numpy(np.moveaxis(images, -1, 0)).float()

        return sample

    @property
    def is_train(self):
        return self.mode == "train"

    def __len__(self):
        return min(len(self.ids), self.limit)


class DatasetSinglePseudoRoundAll(Dataset):
    def __init__(
            self,
            mode: str,
            dataset_dir: str,
            transforms: A.Compose,
    ):
        self.mode = mode
        self.dataset_dir = os.path.join(dataset_dir, "validation" if mode == "val" else "train")
        self.transforms = transforms
        self.ids = os.listdir(self.dataset_dir)

    def __getitem__(self, i):
        try:
            return self.getitem(i)
        except Exception as e:
            print(e)
            return self.getitem(random.randint(0, len(self.ids) - 1))

    def getitem(self, i):
        file_id = self.ids[i]

        band11 = np.load(os.path.join(self.dataset_dir, file_id, 'band_11.npy'))
        band14 = np.load(os.path.join(self.dataset_dir, file_id, 'band_14.npy'))
        band15 = np.load(os.path.join(self.dataset_dir, file_id, 'band_15.npy'))

        r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
        g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
        b = normalize_range(band14, _T11_BOUNDS)

        imgs = np.array([r, g, b])

        imgs = np.transpose(imgs, (3, 1, 2, 0))
        if self.is_train:
            pseudo_mask = np.load(os.path.join("/mnt/md0/datasets/warming/seg_preds", f"{file_id}.npy"))
            pseudo_mask = (pseudo_mask / 255).astype(np.float32)
            pseudo_mask[np.isnan(pseudo_mask)] = 0
        mask = np.load(os.path.join(self.dataset_dir, file_id, "human_pixel_masks.npy")).astype(np.uint8)
        if self.is_train and random.random() < 0.4:
            idx = random.randint(0, 7)
            pseudo_mask = pseudo_mask[idx]
            imgs = imgs[idx]
            if idx != 4:
                mask = np.expand_dims((cv2.resize(pseudo_mask, (256, 256))).astype(np.float32), -1)
        else:
            imgs = imgs[4]
        mask = mask.astype(np.float32)

        sample = self.transforms(image=imgs, mask=mask)
        images = sample["image"]
        masks = sample["mask"]
        sample = {}
        sample['mask'] = torch.from_numpy(np.moveaxis(masks, -1, 0)).float()
        sample['image'] = torch.from_numpy(np.moveaxis(images, -1, 0)).float()
        sample['file_id'] = str(file_id)

        return sample

    @property
    def is_train(self):
        return self.mode == "train"

    def __len__(self):
        return len(self.ids)
