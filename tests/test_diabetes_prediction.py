"""Basic checks for the educational data pipeline and local web form."""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from diabetes_prediction import (  # noqa: E402
    ZERO_AS_MISSING,
    load_data,
    make_pipeline,
    replace_invalid_zeros,
)
from sklearn.ensemble import RandomForestClassifier  # noqa: E402


class DataPipelineTests(unittest.TestCase):
    def test_included_data_has_expected_shape_and_binary_target(self):
        features, labels = load_data(PROJECT_DIR / "data" / "diabetes.csv")
        self.assertEqual(features.shape, (768, 8))
        self.assertEqual(set(labels.unique()), {0, 1})
        self.assertTrue(set(ZERO_AS_MISSING).issubset(features.columns))

    def test_measurement_zeros_become_missing_but_valid_zero_is_preserved(self):
        sample = np.array([[0.0, 0.0, 25.0]])
        cleaned = replace_invalid_zeros(sample, zero_positions=(1,))
        self.assertEqual(cleaned[0, 0], 0.0)
        self.assertTrue(np.isnan(cleaned[0, 1]))

    def test_pipeline_imputes_missing_values_and_predicts(self):
        features, labels = load_data(PROJECT_DIR / "data" / "diabetes.csv")
        positions = tuple(features.columns.get_loc(name) for name in ZERO_AS_MISSING)
        model = make_pipeline(
            RandomForestClassifier(n_estimators=10, random_state=7),
            scale=False,
            zero_positions=positions,
        ).fit(features, labels)
        prediction = model.predict(features.iloc[[0]])
        self.assertIn(int(prediction[0]), (0, 1))
        self.assertTrue(np.isfinite(model.predict_proba(features.iloc[[0]])).all())


class _StubModel:
    def predict(self, frame):
        return np.array([1])

    def predict_proba(self, frame):
        return np.array([[0.49, 0.51]])


class FlaskFormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        patcher = patch(
            "diabetes_prediction.fit_demo_model",
            return_value=(
                _StubModel(),
                [
                    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
                ],
                "Test Model",
                0.80,
            ),
        )
        patcher.start()
        cls.addClassCleanup(patcher.stop)
        sys.modules.pop("app", None)
        cls.app_module = importlib.import_module("app")
        cls.client = cls.app_module.app.test_client()

    def valid_form(self):
        return {field["name"]: str(field["default"]) for field in self.app_module.FIELDS}

    def test_homepage_loads_with_all_eight_inputs(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        for field in self.app_module.FIELDS:
            self.assertIn(field["label"].encode(), response.data)

    def test_valid_submission_renders_demo_result_and_disclaimer(self):
        response = self.client.post("/predict", data=self.valid_form())
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Class 1", response.data)
        self.assertIn(b"51.0%", response.data)
        self.assertIn(b"not an individual risk estimate or a medical diagnosis", response.data)

    def test_non_numeric_missing_and_out_of_range_values_are_rejected(self):
        for field_name, value in (("Glucose", "not-a-number"), ("Age", ""), ("BMI", "101")):
            with self.subTest(field=field_name, value=value):
                form = self.valid_form()
                form[field_name] = value
                self.assertEqual(self.client.post("/predict", data=form).status_code, 400)

    def test_nan_and_infinity_are_rejected(self):
        for value in ("NaN", "Infinity"):
            with self.subTest(value=value):
                form = self.valid_form()
                form["Glucose"] = value
                self.assertEqual(self.client.post("/predict", data=form).status_code, 400)


if __name__ == "__main__":
    unittest.main()
