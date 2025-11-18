import cv2
from flask import Flask, render_template, request
import numpy as np
import json

app = Flask(__name__)

def load_manifest():
    with open("MANIFEST.json", "r") as f:
        return json.load(f)["models"]

MODELS_LIST = load_manifest()

@app.route("/")
def home():
    return render_template("index.html", models=MODELS_LIST)

@app.route("/analyse", methods=["POST"])
def analyse():
    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Template
    return f"{img.shape}"
