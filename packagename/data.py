import numpy as np
import cv2

# load_data() preprocessing/resizing/convert if the input test image

#root_dir = ""
#image_path = os.path.join(root_dir, image_file)

def load_data(image_path):

    # Intepret as image
    image = cv2.imread(image_path)
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    # Resize
    resized_image = cv2.resize(image_rgb, (224, 224))
    # Normalizing inside model.

    return resized_image


def imgage_to_array(resized_image):

    # Converting the image data to numpy array
    image_array = np.array(resized_image)

    return image_array
