# TODO: Import your package, replace this by explicit imports of what you need

# import packages
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shap
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

# import functions of alzheimers_detection_tool
from alzheimers_detection_tool.registry import load_my_model
from alzheimers_detection_tool.data import load_data, image_to_array
from alzheimers_detection_tool.preprocess import preprocess

# Initialize API
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# load model once and store in memory
app.state.model = load_my_model()

# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {
        'message': "Hi, The API is running!"
    }

# Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
@app.post("/upload_image")
async def receive_image(img: UploadFile = File(...)):

    # Read the image file
    image = await img.read()

    # 1) Load the image and preprocess it
    image = load_data(image)

    # 2) Convert the image to a numpy array and preprocess it
    image = image_to_array(image)
    image = preprocess(image)

    # 3) Load trained model
    model1 = app.state.model

    # 4) Make the prediction
    prediction = model1.predict(image)
    print(prediction)
    return {
    'mild': float(prediction[0][0]),
    'none': float(prediction[0][1]),
    'very_mild': float(prediction[0][2])
    }

@app.post("/shap")
async def explain_image(img: UploadFile = File(...)):

        # Read the image file
        image_bytes = await img.read()
        print("Image bytes read successfully")

        # Load the image and preprocess it
        image = load_data(image_bytes)

        # Convert the image to a numpy array and preprocess it
        image_array = image_to_array(image)
        preprocessed_image = preprocess(image_array)

        # Load trained model
        model1 = app.state.model
        print("Model loaded successfully")

        class_names = ['mild_demented', 'non_demented', 'very_mild_demented']
        masker = shap.maskers.Image("blur(128,128)", preprocessed_image[0].shape)
        explainer = shap.Explainer(model1, masker, output_names=class_names)
        print("SHAP explainer created successfully")

        shap_values = explainer(preprocessed_image, max_evals=200, batch_size=20, silent=True)
        print("SHAP values computed successfully")

        # Create SHAP plot
        shap_img = io.BytesIO()
        shap.image_plot(shap_values, preprocessed_image)
        plt.savefig(shap_img, format='png')
        shap_img.seek(0)
        print("SHAP plot created successfully")

        # Convert SHAP plot to base64
        shap_img_base64 = base64.b64encode(shap_img.read()).decode('utf-8')
        print("SHAP plot converted to base64 successfully")

        response2 = {"shap_image": shap_img_base64}
        return JSONResponse(content=response2)
