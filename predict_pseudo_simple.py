import argparse
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import zoo
from training.config import load_config
from training.dataset_seq import normalize_range, _TDIFF_BOUNDS, _CLOUD_TOP_TDIFF_BOUNDS, _T11_BOUNDS
from training.utils import load_checkpoint


class DatasetSinglePseudo(Dataset):
    def __init__(
            self,
            fold: int,
            folds_csv: str,
            dataset_dir: str,
    ):
        self.dataset_dir = os.path.join(dataset_dir, "train")
        df = pd.read_csv(folds_csv, dtype={"file_id": str, "fold": int})
        df = df[df.fold == fold]
        self.df = df
        self.ids = df.file_id.values

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
        images = np.array([cv2.resize(img, (512, 512)) for img in imgs])
        sample = {}
        sample['image'] = torch.from_numpy(np.moveaxis(images, -1, 1)).float()
        sample['file_id'] = file_id

        return sample

    def __len__(self):
        return len(self.ids)


def load_model(config_path, checkpoint):
    conf = load_config(config_path)
    model = zoo.__dict__[conf['network']](**conf["encoder_params"])
    model = model.cuda()
    load_checkpoint(model, checkpoint)
    channels_last = conf["encoder_params"].get("channels_last", False)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model.eval()


def init_gpu(args):
    if args.distributed:
        dist.init_process_group(backend="nccl",
                                rank=args.local_rank,
                                world_size=args.world_size)
        torch.cuda.set_device(args.local_rank)
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def process_data(models, args):
    ds = DatasetSinglePseudo(fold=args.fold, folds_csv="folds.csv", dataset_dir=args.data_dir)
    sampler = None
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=False)
    loader = DataLoader(ds, batch_size=1, sampler=sampler, shuffle=False, num_workers=8)
    for sample in tqdm(loader):
        file_id = sample["file_id"][0]
        img = sample["image"].cuda().float()[0]

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                preds = None
                for model in models:
                    if preds is None:
                        preds = model(img)["mask"].sigmoid().cpu().float()
                    else:
                        preds += model(img)["mask"].sigmoid().cpu().float()

                    preds += torch.flip(model(torch.flip(img, dims=(3,)))["mask"].sigmoid().cpu().float(), dims=(3,))
                    preds += torch.rot90(model(torch.rot90(img, k=1, dims=(2, 3)))["mask"].sigmoid().cpu().float(), k=-1,
                                        dims=(2, 3))
                    preds += torch.rot90(model(torch.rot90(img, k=-1, dims=(2, 3)))["mask"].sigmoid().cpu().float(), k=1,
                                        dims=(2, 3))
                preds /= 4
                preds /= len(models)
                preds = preds.numpy()
            preds[np.isnan(preds)] = 0
            preds = (np.moveaxis(preds, 1, -1) * 255).astype(np.uint8)
            preds = np.array([cv2.resize(p, (256, 256)) for p in preds])
            os.makedirs(args.out_dir, exist_ok=True)
            np.save(os.path.join(args.out_dir, file_id), preds)


def main():
    args = parse_args()
    init_gpu(args)
    checkpoint_paths = args.checkpoint.split(",")
    config_paths = args.config.split(",")
    models = [
        load_model(os.path.join("configs", f"{config_path}.json"), os.path.join(args.weights_path, checkpoint_path)) for
        checkpoint_path, config_path in zip(checkpoint_paths, config_paths)]
    if args.distributed:
        models = [DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                          find_unused_parameters=True) for model in models]
    process_data(models, args)


def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--config', type=str)
    arg('--workers', type=int, default=16, help='number of cpu threads to use')
    arg('--gpu', type=str, default='1', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--checkpoint', type=str, required=True)
    arg('--weights-path', type=str, default="weights")
    arg('--data-dir', type=str, default="/mnt/md0/datasets/warming/")
    arg('--out-dir', type=str, default="/mnt/md0/datasets/warming/seg_preds")
    arg('--fp16', action='store_true', default=False)
    arg('--fold', type=int, default=0)
    arg('--distributed', action='store_true', default=False)
    arg("--local-rank", default=0, type=int)
    arg("--world-size", default=1, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
