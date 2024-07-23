from tensorflow.keras.models import load_model
import os
import shap
import matplotlib.pyplot as plt

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_name = "alzheimer_balanced_v4_recall659_prec710.keras"

def load_my_model():
    model = load_model(os.path.join(root_dir, "models", model_name))
    return model


# Saving the shapley explaination as a .png file to send it to the frontend
def save_shap_plot(shap_values, img):
    shap_path = os.path.join("shap", "img_plot.png")

    #Removing the previous .png file
    if os.path.exists(shap_path):
        os.remove(shap_path)
        print("Old image removed")

    shap.plots.image(shap_values, img, show=False)
    figure = plt.gcf()

    plt.savefig(shap_path)


    print("Shap plot saved")
