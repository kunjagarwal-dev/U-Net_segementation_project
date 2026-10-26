import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms

class PetSegmentation(Dataset):
    def __init__(self, path, split, image_folder, trimap_folder, target_size):
        self.path = path
        self.split = split
        self.image_folder = Path(image_folder)
        self.trimap_folder = Path(trimap_folder)
        self.target_size = target_size

        self.df = pd.read_csv(self.path)
        self.stems = self.df.loc[self.df["split"] == self.split, "stem"].tolist()

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem_at_pos = self.stems[idx]

        image_path = f"{self.image_folder}/{stem_at_pos}.jpg"
        mask_path = f"{self.trimap_folder}/{stem_at_pos}.png"


        with Image.open(image_path) as img:
            image = img.convert("RGB")
            resized_image = image.resize((self.target_size, self.target_size), resample=Image.Resampling.BILINEAR)
            image_tensor = transforms.ToTensor()(resized_image)
            normalised_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)

        with Image.open(mask_path) as mask:
            resized_mask = mask.resize((self.target_size, self.target_size), resample=Image.Resampling.NEAREST)
            numpy_mask = np.array(resized_mask)
            torch_tensor = torch.tensor(numpy_mask, dtype=torch.long)
    
            mask_tensor = torch_tensor - 1

        return normalised_tensor, mask_tensor

def get_dataloaders(csv_path, image_folder, trimap_folder,
                    image_size=256, batch_size=16, num_workers=0):
    train_ds = PetSegmentation(csv_path, "train", image_folder, trimap_folder, image_size)
    val_ds = PetSegmentation(csv_path, "val", image_folder, trimap_folder, image_size)
    test_ds = PetSegmentation(csv_path, "test", image_folder, trimap_folder, image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader






        