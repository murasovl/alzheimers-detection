# TODO: Import your package, replace this by explicit imports of what you need
#from alzheimers_detection_tool.main import predict

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


from alzheimers_detection_tool.registry import load_my_model
from alzheimers_detection_tool.data import load_data, image_to_array
from alzheimers_detection_tool.preprocess import preprocess


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
app.state.model = load_my_model()

# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {
        'message': "Hi, The API is running!"
    }

# Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
# our func 2
@app.post("/upload_image")
async def receive_image(img: UploadFile = File(...)):
    # Read the image file
    image = await img.read()
    #img = Image.open(io.BytesIO(contents))

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
