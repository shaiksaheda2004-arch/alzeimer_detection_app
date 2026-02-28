import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    BatchNormalization, Flatten,
    Dense, Dropout
)

# ---------------- APP CONFIG ----------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_PATH = "model/alzheimer_model.h5"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ---------------- MODEL ----------------
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(128,128,3)),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(64, (3,3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(128, (3,3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(4, activation="softmax")
    ])
    return model


print("🔄 Loading model...")
model = build_model()
model.load_weights(MODEL_PATH)
print("✅ Model loaded successfully")


# ---------------- ROUTES ----------------
@app.route("/")
def login():
    return render_template("login.html")


@app.route("/signup")
def signup():
    return render_template("signup.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ---------------- PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return redirect(url_for("upload"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("upload"))

    # Save image
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Preprocess image
    img = cv2.imread(filepath)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    pred = model.predict(img)

    classes = [
        "Normal",
        "Mild Demented",
        "Moderate Demented",
        "Very Mild Demented"
    ]

    pred_index = int(np.argmax(pred))
    diagnosis = classes[pred_index]

    # 🔥 CONFIDENCE (%)
    confidence = round(float(np.max(pred)) * 100, 2)

    # 🧾 MEDICAL REPORT DATA
    report = {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "study_type": "MRI Brain Scan",
        "model_name": "CNN-based Alzheimer Detection Model",
        "input_resolution": "128 × 128 RGB",
        "clinical_note": (
            "The AI model analyzed structural brain features "
            "associated with neurodegenerative disorders."
        )
    }

    print("Diagnosis:", diagnosis)
    print("Confidence:", confidence)

    return render_template(
        "result.html",
        report=report,
        image_path=url_for("uploaded_file", filename=file.filename)
    )





# ---------------- RUN SERVER ----------------
if __name__ == "_main_":
    print("🚀 Flask server starting...")
 
    app.run(host="0.0.0.0", port=10000)


