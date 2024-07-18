# TODO: Import your package, replace this by explicit imports of what you need
from packagename.main import predict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from registry import load_model
from data import load_data
from preprocess import preprocess


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {
        'message': "Hi, The API is running!"
    }

# Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
@app.get("/predict")
# POST not GET for images
def get_predict(input_one: array):
    # TODO: Do something with your input
    # i.e. feed it to your model.predict, and return the output
    # For a dummy version, just return the sum of the two inputs and the original inputs

    # call load_model()
    model = load_model()
    # call load_data()
    test_data = load_data()
    # call preprocess()
    test_data = preprocess(test_data)
    # call model.predict()
    prediction = model.predict(test_data)

    #prediction = float(input_one) + float(input_two)
    return {
        'prediction': prediction,
        'inputs': {
            'input_one': input_one
        }
    }
