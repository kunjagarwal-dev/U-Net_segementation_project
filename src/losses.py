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