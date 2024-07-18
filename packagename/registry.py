from tensorflow.keras.models import load_model


# load_model from local directory

#tf.keras.models.load_model(
#    filepath, custom_objects=None, compile=True, safe_mode=True
#)

filepath = "/Users/linamurasov/code/murasovl/alzheimers-detection/models/alzheimer_first_draft.keras"

def load_model(filepath):
    model = load_model(filepath)
    return model
