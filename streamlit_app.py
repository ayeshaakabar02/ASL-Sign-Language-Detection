import streamlit as st
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Set title
st.title("ASL Alphabet Recognition")
st.markdown("Upload an image of a hand sign to predict the corresponding alphabet.")

# Load your trained model
model_path = "EfficientNetB0_asl_model.h5"  # Change to the model you want to use
model = load_model(model_path)

# Define class labels (must match your dataset folders)
class_labels = sorted(os.listdir("data/train"))  # This will list all class names

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image for prediction
    img_resized = img.resize((224, 224))  # ✅ Resize to match training input
    img_array = image.img_to_array(img_resized)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction[0])
    predicted_label = class_labels[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100

    # Show result
    st.markdown(f"### 🧠 Predicted Sign: `{predicted_label.upper()}`")
    st.markdown(f"### 🔍 Confidence: `{confidence:.2f}%`")
