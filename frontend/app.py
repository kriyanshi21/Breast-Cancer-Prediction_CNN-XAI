import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io
import base64

# Streamlit app configuration
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

# Title and description
st.title("Breast Cancer Prediction with Explainable AI")
st.markdown("""
Upload a breast ultrasound image to get a prediction (Benign or Malignant) 
and a Grad-CAM visualization to understand the model's decision.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Prepare image for API
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (128, 128))
    img_array = img_array.astype('float32') / 255.0
    img_bytes = cv2.imencode('.png', img_array)[1].tobytes()

    # Send request to FastAPI backend
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            files={"file": ("image.png", img_bytes, "image/png")}
        )
        response.raise_for_status()
        result = response.json()

        # Display prediction
        prediction = result["prediction"]
        probability = result["probability"]
        st.subheader("Prediction Result")
        st.write(f"**Prediction**: {'Malignant' if prediction == 1 else 'Benign'}")
        st.write(f"**Probability**: {probability:.4f}")

        # Display Grad-CAM
        gradcam_base64 = result["gradcam"]
        gradcam_img = base64.b64decode(gradcam_base64)
        st.subheader("Grad-CAM Explanation")
        st.image(gradcam_img, caption="Grad-CAM Heatmap (Red areas indicate regions of interest)", width=300)

    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with the API: {e}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and FastAPI | Model: CNN with Grad-CAM")