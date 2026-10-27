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

def class_mIou():
    pass