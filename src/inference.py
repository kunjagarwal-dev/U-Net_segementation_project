import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.model import PretrainedUNet

IMAGE_SIZE = 256

_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model(checkpoint_path, device):
    model = PretrainedUNet(pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_mask(model, image, device):
    # Handle RGBA / grayscale / palette uploads
    image = image.convert("RGB")
    original_size = image.size  # (width, height)

    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    tensor = _preprocess(resized).unsqueeze(0).to(device)  # [1, 3, 256, 256]

    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)  # [256, 256]

    # Nearest resampling so class labels are never blended
    mask = Image.fromarray(pred).resize(original_size, Image.Resampling.NEAREST)
    return np.array(mask)  # [height, width], values 0 = pet, 1 = background, 2 = border