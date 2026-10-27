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

