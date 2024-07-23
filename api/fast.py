# TODO: Import your package, replace this by explicit imports of what you need

# import packages
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
import cv2


# import functions of alzheimers_detection_tool
from alzheimers_detection_tool.registry import load_my_model, save_shap_plot
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

        shap_values = explainer(preprocessed_image, max_evals=800, batch_size=30, silent=True)
        print("SHAP values computed successfully")

        #Create and saving SHAP plot
        save_shap_plot(shap_values, preprocessed_image)
        print("SHAP plot created successfully")


        #Opening the saved shapley explaination to send it as response to the http request
        img_plot = Image.open(os.path.join("shap", "img_plot.png"))
        img_plot = np.array(img_plot)
        response2 = cv2.imencode('.png', img_plot)[1]
        return Response(content=response2.tobytes(), media_type="image/png")
