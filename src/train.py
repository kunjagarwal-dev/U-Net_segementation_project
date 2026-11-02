import torch
import csv
import time
from pathlib import Path
from src.dataset import get_dataloaders
from src.metrics import SegmentationMetrics
from src.model import UNet
from src.losses import CombinedLoss


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

def main():
    device = "cuda"
    epochs = 30
    patience = 7

    train_loader, val_loader, _ = get_dataloaders(
        csv_path="data/processed/split.csv",
        image_folder="data/raw/oxford-iiit-pet/images",
        trimap_folder="data/raw/oxford-iiit-pet/annotations/trimaps",
        image_size=256,
        batch_size=8,
        num_workers=2,
    )

    model = UNet().to(device)
    loss_fn = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    best_miou = 0.0
    epochs_without_improvement = 0
    log_rows = []

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val = validate(model, val_loader, loss_fn, device)
        scheduler.step(val["miou"])

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val["loss"], 4),
            "val_miou": round(val["miou"], 4),
            "iou_pet": round(val["iou"][0], 4),
            "iou_bg": round(val["iou"][1], 4),
            "iou_border": round(val["iou"][2], 4),
            "lr": optimizer.param_groups[0]["lr"],
        }
        log_rows.append(row)
        print(row, f"{time.time() - start:.0f}s")

        with open("outputs/unet_scratch_log.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(log_rows)

        if val["miou"] > best_miou:
            best_miou = val["miou"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "models/unet_scratch_best.pth")
            print("  saved new best model")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping")
                break


if __name__ == "__main__":
    main()