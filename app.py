import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request
from tensorflow import keras
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename


app = Flask(__name__)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = keras.models.load_model("models/pneumonia_model.keras")


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize((224, 224))

    image = np.array(image)

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)[0][0]

    if prediction >= 0.5:
        result = "Pneumonia"
        probability = float(prediction)
    else:
        result = "Normal"
        probability = float(1 - prediction)

    return result, probability


# --------------------------------------------------
# Home route
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    probability = None
    image_path = None
    error = None

    if request.method == "POST":

        if "file" not in request.files:
            error = "Please select an X-ray image."

            return render_template(
                "index.html",
                prediction=prediction,
                probability=probability,
                image_path=image_path,
                error=error
            )

        file = request.files["file"]

        if file.filename == "":
            error = "Please select an image."

            return render_template(
                "index.html",
                prediction=prediction,
                probability=probability,
                image_path=image_path,
                error=error
            )

        if not allowed_file(file.filename):
            error = "Only JPG, JPEG and PNG images are allowed."

            return render_template(
                "index.html",
                prediction=prediction,
                probability=probability,
                image_path=image_path,
                error=error
            )

        filename = secure_filename(file.filename)

        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(file_path)

        prediction, probability = predict_image(file_path)

        image_path = "/" + file_path.replace("\\", "/")

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        image_path=image_path,
        error=error
    )


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)