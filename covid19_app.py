#covid19_app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# Define class names
class_names = ['Covid', 'Normal', 'Viral Pneumonia']

# Load trained model
model = load_model('model_vgg16_pretrained_data_aug.keras') 

# Set title
st.title("Chest X-ray Classification (COVID Detection)")

# File uploader
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)

    # Preprocess image
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = preprocess_input(img_array)  # Normalize if needed
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Show result
    st.markdown(f"### Prediction: `{predicted_class}`")
    st.markdown(f"### Confidence: `{confidence:.2f}%`")
