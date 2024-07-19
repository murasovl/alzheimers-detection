from tensorflow.keras.models import load_model
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_name = "alzheimer_balanced_v4_recall659_prec710.keras"

def load_my_model():
    model = load_model(os.path.join(root_dir, "models", model_name))
    return model
