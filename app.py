import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# same order as image_dataset_from_directory (alphabetical)
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("intel_cnn.keras")


model = load_model()

st.title("Intel Image Classifier")
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)

    x = np.array(img.resize((150, 150)), dtype="float32") / 255.0
    probs = model.predict(np.expand_dims(x, 0))[0]

    st.subheader(f"{CLASS_NAMES[np.argmax(probs)]} ({probs.max():.2%})")
    st.bar_chart(dict(zip(CLASS_NAMES, probs.tolist())))
