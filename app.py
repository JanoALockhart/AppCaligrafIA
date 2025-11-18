from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", models=["modelA", "modelB"])

@app.route("/analyse", methods=["POST"])
def analyse():
    return "Making Analysis"
