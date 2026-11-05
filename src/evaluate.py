import torch

from src.dataset import get_dataloaders
from src.losses import CombinedLoss
from src.model import UNet, PretrainedUNet
from src.train import validate


def main():
    device = "cuda"
    _, _, test_loader = get_dataloaders(
        csv_path="data/processed/split.csv",
        image_folder="data/raw/oxford-iiit-pet/images",
        trimap_folder="data/raw/oxford-iiit-pet/annotations/trimaps",
        image_size=256,
        batch_size=8,
        num_workers=0,
    )
    loss_fn = CombinedLoss()

    for name, model in [("scratch", UNet()), ("pretrained", PretrainedUNet(pretrained=False))]:
        model.load_state_dict(torch.load(f"models/unet_{name}_best.pth", map_location=device))
        model.to(device)
        r = validate(model, test_loader, loss_fn, device)
        print(f"\n{name}")
        print(f"  mIoU {r['miou']:.4f} | mDice {r['mdice']:.4f} | pixel acc {r['pixel_acc']:.4f}")
        print(f"  IoU pet {r['iou'][0]:.4f} | bg {r['iou'][1]:.4f} | border {r['iou'][2]:.4f}")


if __name__ == "__main__":
    main()