import cv2
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", models=["modelA", "modelB"])

@app.route("/analyse", methods=["POST"])
def analyse():
    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Template
    return f"{img.shape}"
