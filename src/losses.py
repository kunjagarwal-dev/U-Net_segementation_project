from typing import Any

import torch
import torch.nn as nn

def dice_loss(logits, target):
    smooth = 1e-6
    probs = torch.softmax(logits, dim=1)

    one_hot_target = nn.functional.one_hot(target, num_classes=3).permute(0,3,1,2).float()

    intersection = (torch.mul(probs, one_hot_target)).sum(dim=(0,2,3))
    prob_sum = probs.sum(dim=(0,2,3))

    target_sum = one_hot_target.sum(dim=(0,2,3))

    dice_per_class = (2 * intersection + smooth) / (prob_sum + target_sum + smooth)
    loss = (1 - dice_per_class).mean()

    return loss

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        dice = dice_loss(logits, target)

        return self.ce_weight * ce + self.dice_weight * dice