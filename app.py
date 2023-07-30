import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the pretrained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN', model_name='celeba')

# Define a function to generate an image from text
def generate_image_from_text(text, truncation=0.4):
    with torch.no_grad():
        image = model.sample(1, seed=text, truncation=truncation).squeeze(0)
    image = transforms.ToPILImage()(image)
    return image

# Streamlit app
st.title("Image Generation from Text")
st.write("Enter a text to generate an image:")

user_input = st.text_input("Text", "A beautiful landscape")

if st.button("Generate Image"):
    try:
        generated_image = generate_image_from_text(user_input)
        st.image(generated_image, caption="Generated Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error: {e}")
