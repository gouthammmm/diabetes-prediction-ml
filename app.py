"""Local-only Flask interface for the educational diabetes classifier."""
from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

from diabetes_prediction import fit_demo_model

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("DIABETES_CSV", BASE_DIR / "data" / "diabetes.csv"))

FIELDS = [
    {"name": "Pregnancies", "label": "Pregnancies", "min": 0, "max": 20, "step": "1", "default": 2, "unit": "count"},
    {"name": "Glucose", "label": "Glucose", "min": 1, "max": 300, "step": "1", "default": 120, "unit": "mg/dL"},
    {"name": "BloodPressure", "label": "Diastolic blood pressure", "min": 1, "max": 200, "step": "1", "default": 70, "unit": "mmHg"},
    {"name": "SkinThickness", "label": "Skin thickness", "min": 1, "max": 100, "step": "1", "default": 20, "unit": "mm"},
    {"name": "Insulin", "label": "Insulin", "min": 1, "max": 1000, "step": "1", "default": 80, "unit": "mu U/mL"},
    {"name": "BMI", "label": "BMI", "min": 1, "max": 100, "step": "0.1", "default": 28, "unit": "kg/m²"},
    {"name": "DiabetesPedigreeFunction", "label": "Diabetes pedigree function", "min": 0, "max": 3, "step": "0.001", "default": 0.4, "unit": "dataset score"},
    {"name": "Age", "label": "Age", "min": 18, "max": 120, "step": "1", "default": 35, "unit": "years"},
]

app = Flask(__name__)
model, model_features, model_name, cv_auc = fit_demo_model(DATA_PATH)


@app.get("/")
def home():
    return render_template("index.html", fields=FIELDS)


@app.post("/predict")
def predict():
    values: dict[str, float] = {}
    errors: list[str] = []
    for field in FIELDS:
        raw = request.form.get(field["name"], "").strip()
        try:
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError
            if not field["min"] <= value <= field["max"]:
                errors.append(f"{field['label']} must be between {field['min']} and {field['max']}.")
            values[field["name"]] = value
        except ValueError:
            errors.append(f"Enter a valid number for {field['label']}.")

    if errors:
        return render_template("index.html", fields=FIELDS, values=request.form, errors=errors), 400

    sample = pd.DataFrame([[values[name] for name in model_features]], columns=model_features)
    probability = float(model.predict_proba(sample)[0, 1])
    predicted_class = int(model.predict(sample)[0])
    return render_template(
        "index.html",
        fields=FIELDS,
        values=request.form,
        result={
            "probability": probability,
            "class": predicted_class,
            "model": model_name,
            "cv_auc": cv_auc,
        },
    )


if __name__ == "__main__":
    print(f"Loaded {model_name}; 5-fold training-data ROC AUC: {cv_auc:.3f}")
    print("Open http://127.0.0.1:5000 in your browser. This demo is local-only.")
    app.run(host="127.0.0.1", port=5000, debug=False)
