import torch
from src.metrics import SegmentationMetrics


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader, loss_fn, device):
    model.eval()
    metrics = SegmentationMetrics()
    total_loss = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = loss_fn(logits, masks)
            total_loss += loss.item()

            predictions = logits.argmax(dim=1)
            metrics.update(masks, predictions)

    results = metrics.compute()
    results["loss"] = total_loss / len(loader)
    return results