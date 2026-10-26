import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
import albumentations as A
import cv2

class PetSegmentation(Dataset):
    def __init__(self, path, split, image_folder, trimap_folder, target_size, augmentation=False):
        self.path = path
        self.split = split
        self.image_folder = Path(image_folder)
        self.trimap_folder = Path(trimap_folder)
        self.target_size = target_size

        if augmentation:
            self.transform = A.Compose([
                # 1. Horizontal flip, probability 0.5
                A.HorizontalFlip(p=0.5),
                
                # 2. Small rotation (up to 15°) plus mild scale/shift
                A.ShiftScaleRotate(
                    shift_limit=0.05, 
                    scale_limit=0.05, 
                    rotate_limit=15, 
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ),
                
                # 3. Brightness and contrast jitter (automatically skips masks)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.5
                ),
            ])

        else:
            self.transform = None

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.df = pd.read_csv(self.path)
        self.stems = self.df.loc[self.df["split"] == self.split, "stem"].tolist()

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem_at_pos = self.stems[idx]

        image_path = self.image_folder / f"{stem_at_pos}.jpg"
        mask_path = self.trimap_folder / f"{stem_at_pos}.png"

        # 1. load the image, convert to RGB, resize (bilinear), keep as array
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            resized_image = image.resize(
                (self.target_size, self.target_size),
                resample=Image.Resampling.BILINEAR,
            )
            image_array = np.array(resized_image)       # (H, W, 3) uint8

        # 2. load the mask, resize (nearest), keep as array
        with Image.open(mask_path) as mask:
            resized_mask = mask.resize(
                (self.target_size, self.target_size),
                resample=Image.Resampling.NEAREST,
            )
            mask_array = np.array(resized_mask)          # (H, W) uint8, values 1-3

        # 3. augment image and mask together (train only)
        if self.transform is not None:
            augmented = self.transform(image=image_array, mask=mask_array)
            image_array = augmented["image"]
            mask_array = augmented["mask"]
            # empty corners from rotation/shift are filled with 0, which is not
            # a valid trimap value (valid values are 1, 2, 3), so make them background
            mask_array[mask_array == 0] = 2

        # 4. to tensors
        image_tensor = self.to_tensor(image_array)       # (3, H, W) float, 0-1
        normalised_tensor = self.normalize(image_tensor)

        mask_tensor = torch.tensor(mask_array, dtype=torch.long) - 1   # 0, 1, 2

        return normalised_tensor, mask_tensor

def get_dataloaders(csv_path, image_folder, trimap_folder,
                    image_size=256, batch_size=16, num_workers=0):
    train_ds = PetSegmentation(csv_path, "train", image_folder, trimap_folder,
                               image_size, augmentation=True)
    val_ds = PetSegmentation(csv_path, "val", image_folder, trimap_folder, image_size)
    test_ds = PetSegmentation(csv_path, "test", image_folder, trimap_folder, image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader






        