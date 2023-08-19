from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss


class LossCalculator(ABC):

    @abstractmethod
    def calculate_loss(self, outputs, sample):
        pass


class DiceLossCalculator(LossCalculator):
    def __init__(self, field: str):
        super().__init__()
        self.field = field
        self.loss = DiceLoss()

    def calculate_loss(self, outputs, sample):
        with torch.cuda.amp.autocast(enabled=False):
            outputs = outputs[self.field].float().sigmoid()
            targets = sample[self.field].cuda().float()
            return self.loss(outputs, targets)


class BCELossCalculator(LossCalculator):
    def __init__(self, field: str):
        super().__init__()
        self.field = field
        self.loss = BCEWithLogitsLoss()

    def calculate_loss(self, outputs, sample):
        with torch.cuda.amp.autocast(enabled=False):
            outputs = outputs[self.field].float()
            targets = sample[self.field].cuda().float()
            return self.loss(outputs, targets)


def dice_round(preds, trues, t=0.5):
    preds = (preds > t).float()
    return 1 - soft_dice_loss(preds, trues)


def soft_dice_loss(outputs, targets, eps=1e-5):
    dice_target = targets.view(-1).contiguous().float()
    dice_output = outputs.view(-1).contiguous().float()
    dim = (-1,)
    intersection = torch.sum(dice_output * dice_target, dim=dim)
    union = torch.sum(dice_output, dim=dim) + torch.sum(dice_target, dim=dim) + eps
    loss = (1 - (2 * intersection + eps) / union)
    return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return soft_dice_loss(input, target)


class BceDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = BCEWithLogitsLoss()

    def forward(self, input, target):
        sigmoid_input = torch.sigmoid(input)
        return self.bce(input, target) + soft_dice_loss(sigmoid_input, target)
