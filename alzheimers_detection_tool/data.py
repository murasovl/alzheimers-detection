import numpy as np
import cv2
from PIL import Image
import io

# load_data() preprocessing/resizing/convert if the input test image

#root_dir = ""
#image_path = os.path.join(root_dir, image_file)

def load_data(image_path):

    # Intepret as image
    image = Image.open(io.BytesIO(image_path))
    print('image loaded')
    # Convert to RGB
    image_rgb = image.convert('RGB')  # Convert to RGB format
    # Resize
    resized_image = image_rgb.resize((224, 224))
    # Normalizing inside model.

    return resized_image


def image_to_array(resized_image):

    # Converting the image data to numpy array
    image_array = np.array(resized_image)
    image_array = image_array.reshape((-1, 224, 224, 3))

    return image_array
