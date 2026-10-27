def pixel_accuracy(target, prediction):
    
    # Compare all pixel at once
    grid = target == prediction
    true_count = grid.sum()

    accuracy = true_count / target.numel()
    return accuracy.item()
