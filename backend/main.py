from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import cv2
import numpy as np
import base64
from pydantic import BaseModel
from typing import Dict
import io
from PIL import Image

app = FastAPI(title="Breast Cancer Prediction API")

# Load the model
model = tf.keras.models.load_model("breast_cancer_cnn_model.h5")
img_size = (128, 128)
last_conv_layer_name = "conv2d_2"  # Update if different in your model

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_gradcam_image(img_array, heatmap, alpha=0.4):
    img = (img_array[0] * 255).astype(np.uint8)
    heatmap = cv2.resize(heatmap, img_size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    # Convert to base64
    _, buffer = cv2.imencode('.png', superimposed_img)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(img)
    img_array = cv2.resize(img_array, img_size)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    probability = float(prediction[0][0])
    pred_class = 1 if probability > 0.5 else 0

    # Generate Grad-CAM
    heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name)
    gradcam_base64 = generate_gradcam_image(img_array, heatmap)

    return JSONResponse({
        "prediction": pred_class,
        "probability": probability,
        "gradcam": gradcam_base64
    })

@app.get("/")
async def root():
    return {"message": "Breast Cancer Prediction API"}