import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.inference import load_model, predict_mask

CHECKPOINT = "models/unet_pretrained_best.pth"

# Class colours: 0 = pet, 1 = background, 2 = border
COLOURS = {
    0: (128, 0, 255),    # purple
    2: (255, 215, 0),    # yellow
}


@st.cache_resource
def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_model(CHECKPOINT, device), device


def make_overlay(image, mask, alpha):
    base = np.array(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    for class_id, colour in COLOURS.items():
        region = mask == class_id
        overlay[region] = (1 - alpha) * base[region] + alpha * np.array(colour)
    return overlay.astype(np.uint8)


st.title("Pet Segmentation")
st.write("Upload a photo of a cat or dog. The model finds the pet and its outline.")

model, device = get_model()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
alpha = st.slider("Overlay strength", 0.0, 1.0, 0.5)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    mask = predict_mask(model, image, device)

    pet_percent = 100 * (mask == 0).mean()

    col1, col2 = st.columns(2)
    col1.image(image, caption="Original", use_container_width=True)
    col2.image(make_overlay(image, mask, alpha),
               caption=f"Prediction (pet covers {pet_percent:.0f}% of the image)",
               use_container_width=True)

    st.caption("Purple = pet, yellow = border band, no colour = background.")