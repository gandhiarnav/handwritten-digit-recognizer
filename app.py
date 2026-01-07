from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = "model/digit_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    """
    Preprocess uploaded image to match MNIST format
    """
    image = image.convert("L")            # Convert to grayscale
    image = image.resize((28, 28))         # Resize to 28x28
    image = np.array(image)                # Convert to NumPy array
    image = image / 255.0                  # Normalize
    image = image.reshape(1, 28, 28, 1)    # Reshape for CNN
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        if "image" in request.files:
            file = request.files["image"]

            if file.filename != "":
                image = Image.open(file)
                processed_image = preprocess_image(image)

                pred = model.predict(processed_image)
                prediction = np.argmax(pred)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
