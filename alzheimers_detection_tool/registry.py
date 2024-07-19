from tensorflow.keras.models import load_model
import os

filepath = "/Users/linamurasov/code/murasovl/alzheimers-detection/models/alzheimer_first_draft.keras"

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_name = "alzheimer_balanced_v4_recall659_prec710.keras"

def load_my_model():
    model = load_model(os.path.join(root_dir, "model", model_name))
    return model
