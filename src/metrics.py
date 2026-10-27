import math
import numpy as np

def pixel_accuracy(target, prediction):
    # Compare all pixel at once
    grid = target == prediction
    true_count = grid.sum()

    accuracy = true_count / target.numel()
    return accuracy.item()

def class_iou(target, prediction, class_id):
    target_id = target == class_id
    prediction_id = prediction == class_id

    intersection = (target_id & prediction_id).sum()
    union = (target_id | prediction_id).sum()

    # Class absent, skip when averaging
    if union == 0:
        return float("nan")

    iou = intersection / union
    return iou.item()

def class_dice(target, prediction, class_id):
    target_count = target == class_id
    prediction_count = prediction == class_id

    intersection = (target_count & prediction_count).sum()
    total = target_count.sum() + prediction_count.sum()

    # Class absent from both
    if total == 0:
        return float("nan")

    dice = 2*intersection / total
    return dice.item()

def mean_iou(target, prediction, num_classes=3):
    iou_lst = []

    for i in range(num_classes):
        result = class_iou(target, prediction, i)
        if math.isnan(result):
            continue

        iou_lst.append(result)

    if iou_lst == []:
        return float("nan")

    return sum(iou_lst) / len(iou_lst)

def _nan_mean(values):
    # Mean of the non-NaN values; NaN if nothing is valid
    valid = [v for v in values if not math.isnan(v)]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)

class SegmentationMetrics():
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)
        self.total = np.zeros(self.num_classes)

        self.correct = 0
        self.pixels = 0

    def update(self, target, prediction):
        for class_id in range(self.num_classes):
            target_count = target == class_id
            prediction_count = prediction == class_id

            self.intersection[class_id] += ((target_count & prediction_count).sum()).item()
            self.union[class_id] += ((target_count | prediction_count).sum()).item()
            self.total[class_id] += (target_count.sum() + prediction_count.sum()).item()

        self.correct += (target==prediction).sum().item()
        self.pixels += target.numel()

    def compute(self):
        iou = []
        dice = []
        for i in range(self.num_classes):
            # NaN = class absent from both target and prediction
            if self.union[i] == 0:
                iou.append(float("nan"))
            else:
                iou.append(float(self.intersection[i] / self.union[i]))
            if self.total[i] == 0:
                dice.append(float("nan"))
            else:
                dice.append(float(2 * self.intersection[i] / self.total[i]))
        if self.pixels == 0:
            pixel_acc = float("nan")
        else:
            pixel_acc = self.correct / self.pixels
        return {
            "pixel_acc": pixel_acc,
            "iou": iou,
            "dice": dice,
            "miou": _nan_mean(iou),
            "mdice": _nan_mean(dice),
        }



        
    