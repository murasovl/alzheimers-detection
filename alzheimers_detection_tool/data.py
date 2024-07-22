import numpy as np
from PIL import Image
import io


def load_data(image_path):

    # open jpg file as image
    image = Image.open(io.BytesIO(image_path))
    print('image loaded')
    # Convert to RGB format
    image_rgb = image.convert('RGB')
    # Resize
    resized_image = image_rgb.resize((224, 224))
    # Normalizing inside model.

    return resized_image


def image_to_array(resized_image):

    # Converting the image data to numpy array
    image_array = np.array(resized_image)
    image_array = image_array.reshape((-1, 224, 224, 3))

    return image_array
