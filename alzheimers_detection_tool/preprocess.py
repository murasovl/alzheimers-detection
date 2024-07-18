from tensorflow.keras.applications.vgg19 import preprocess_input

# preprocess() model specific preprocessing (for example for VGG19)
def preprocess(img_resized):
    img_preproc = preprocess_input(img_resized) # preprocessing for VGG-19
    return img_preproc
