import argparse
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.losses import soft_dice_loss
from training.trainer import Evaluator
from training.utils import all_gather


class SegEvaluator(Evaluator):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args

    def init_metrics(self) -> Dict:
        return {"dice": 0, "best_t": 0}

    def validate(self, dataloader: DataLoader, model: torch.nn.Module, distributed: bool = False, local_rank: int = 0,
                 snapshot_name: str = "") -> Dict:

        sum_preds = []
        sum_targets = []
        for sample in tqdm(dataloader):
            masks = sample["mask"].cpu().float()
            img = sample["image"].cuda().float()

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    preds = model(img)["mask"].sigmoid().cpu().float()
                    preds += torch.flip(model(torch.flip(img, dims=(3,)))["mask"].sigmoid().cpu().float(), dims=(3,))
                    preds += torch.rot90(model(torch.rot90(img, k=1, dims=(2, 3)))["mask"].sigmoid().cpu().float(),
                                         k=-1,
                                         dims=(2, 3))
                    preds += torch.rot90(model(torch.rot90(img, k=-1, dims=(2, 3)))["mask"].sigmoid().cpu().float(),
                                         k=1,
                                         dims=(2, 3))
                    preds /= 4
                sum_preds.extend(preds.numpy())
                sum_targets.extend(masks.numpy())
        sum_targets = np.array(sum_targets)
        sum_preds = np.array(sum_preds)

        if distributed:
            sum_preds = all_gather(sum_preds)
            sum_preds = np.concatenate(sum_preds, 0)

            sum_targets = all_gather(sum_targets)
            sum_targets = np.concatenate(sum_targets, 0)

        result = 0
        best_t = 0
        if local_rank == 0:
            for t in tqdm(np.linspace(0.1, 0.6, 21)):
                dice = float(1 - soft_dice_loss(torch.from_numpy(sum_preds > t).cuda(),
                                                torch.from_numpy(sum_targets).cuda()).item())
                print(t, dice)
                if dice > result:
                    result = dice
                    best_t = t

        if distributed:
            dist.barrier()
        torch.cuda.empty_cache()
        return {"dice": result, "best_t": best_t}

    def get_improved_metrics(self, prev_metrics: Dict, current_metrics: Dict) -> Dict:
        improved = {}
        best_dice = prev_metrics["dice"]
        if current_metrics["dice"] > prev_metrics["dice"]:
            print("Dice improved from {:.4f} to {:.4f}".format(prev_metrics["dice"], current_metrics["dice"]))
            improved["dice"] = current_metrics["dice"]
            best_dice = current_metrics["dice"]
        print("Best Dice {:.4f} current {:.4f}@{:.2f}".format(best_dice, current_metrics["dice"],
                                                              current_metrics["best_t"]))
        return improved
