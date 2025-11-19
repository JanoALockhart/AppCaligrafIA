from flask import Flask, render_template, request, send_file
import json
import src.settings as settings
import src.caligraphy_analysis as ca
import src.sheet_building as sb

app = Flask(__name__)

def load_manifest():
    models = {}
    with open("MANIFEST.json", "r") as f:
        models_json = json.load(f)["models"]

    for model_object in models_json:
        models[model_object["id"]] = model_object

    return models

MODELS_MAP = load_manifest()

@app.route("/")
def home():
    return render_template("index.html", models=MODELS_MAP.values())

@app.route("/analyse", methods=["POST"])
def analyse():
    file = request.files["image"]
    model_id = request.form.get("model")
    selected_model = MODELS_MAP[model_id]
    sorted_recommendations, image_rows, predictions = ca.process_image_form(file, selected_model)

    return render_template("index.html", 
                           models=MODELS_MAP.values(), 
                           img_rows=image_rows, 
                           recomendations=sorted_recommendations,
                           predictions=predictions
                           )

@app.route("/download-template")
def download_sheet():
    return send_file(
        sb.build_template_sheet(),
        as_attachment=True,
        download_name="template.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.route("/download-recommendations", methods=["POST"])
def download_recommendations():
    recommendations = json.loads(request.form["recomendations"])
    top_letters = int(request.form.get("top_letters"))
    rows_per_letters = int(request.form.get("rows_per_letter"))

    return send_file(
        sb.build_recommendation_sheet(recomendations=recommendations, amount=top_letters, rows_per_letter=rows_per_letters),
        as_attachment=True,
        download_name="activity.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )