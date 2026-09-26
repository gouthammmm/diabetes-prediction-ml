"""Train and evaluate diabetes classifiers on the Pima Indians dataset.

Run: python diabetes_prediction.py --data path/to/diabetes.csv
The CSV must contain the eight numeric predictors and a binary ``Outcome`` column.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

TARGET = "Outcome"
SEED = 12345
N_JOBS = 2  # Keep local evaluation responsive on typical laptops.
ZERO_AS_MISSING = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")


def replace_invalid_zeros(values: np.ndarray, *, zero_positions: tuple[int, ...]) -> np.ndarray:
    """Treat physiologically invalid zero measurements as missing."""
    cleaned = np.asarray(values, dtype=float).copy()
    for position in zero_positions:
        cleaned[cleaned[:, position] == 0, position] = np.nan
    return cleaned


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Pass --data <csv-path> or set the DIABETES_CSV environment variable."
        )
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"CSV must include a '{TARGET}' target column.")
    features = frame.drop(columns=[TARGET]).select_dtypes(include=[np.number])
    if features.empty or features.shape[1] != frame.shape[1] - 1:
        raise ValueError("All predictor columns must be numeric.")
    labels = frame[TARGET]
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Outcome must contain only non-missing binary values: 0 or 1.")
    if labels.nunique() != 2:
        raise ValueError("Dataset must contain examples from both Outcome classes.")
    return features, labels.astype(int)


def make_pipeline(estimator, *, scale: bool, zero_positions: tuple[int, ...]) -> Pipeline:
    steps = [
        (
            "invalid_zero_to_missing",
            FunctionTransformer(
                replace_invalid_zeros,
                kw_args={"zero_positions": zero_positions},
                validate=False,
            ),
        ),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def make_candidates(zero_positions: tuple[int, ...]) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
            scale=True,
            zero_positions=zero_positions,
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=SEED),
            scale=False,
            zero_positions=zero_positions,
        ),
        "Gradient Boosting": make_pipeline(
            GradientBoostingClassifier(random_state=SEED),
            scale=False,
            zero_positions=zero_positions,
        ),
    }


def fit_demo_model(data_path: Path) -> tuple[Pipeline, list[str], str, float]:
    """Select by stratified CV and fit the selected educational demo model."""
    X, y = load_data(data_path)
    if not set(ZERO_AS_MISSING).issubset(X.columns):
        absent = sorted(set(ZERO_AS_MISSING) - set(X.columns))
        raise ValueError(f"Expected Pima dataset columns are missing: {', '.join(absent)}")
    zero_positions = tuple(X.columns.get_loc(name) for name in ZERO_AS_MISSING)
    candidates = make_candidates(zero_positions)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {
        name: float(
            cross_validate(model, X, y, cv=cv, scoring="roc_auc", n_jobs=N_JOBS)["test_score"].mean()
        )
        for name, model in candidates.items()
    }
    best_name = max(scores, key=scores.get)
    best_model = candidates[best_name].fit(X, y)
    return best_model, list(X.columns), best_name, scores[best_name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("DIABETES_CSV", Path(__file__).parent / "data" / "diabetes.csv")),
        help="CSV path (default: ./diabetes.csv, or DIABETES_CSV).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(__file__).parent / "reports",
        help="Directory for generated held-out evaluation charts and metrics.",
    )
    return parser.parse_args()


def write_evaluation_report(report_dir: Path, cv_results: dict, test_results: dict) -> None:
    """Save model-comparison plots and metrics derived from this run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    report_dir.mkdir(parents=True, exist_ok=True)

    names = list(cv_results)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(names))
    for index, name in enumerate(names):
        values = cv_results[name]["roc_auc_folds"]
        jitter = np.linspace(-0.08, 0.08, len(values))
        ax.scatter(index + jitter, values, color="#14735d", alpha=0.8, zorder=3)
        ax.hlines(np.mean(values), index - 0.18, index + 0.18, color="#102c29", linewidth=2)
    ax.set_xticks(positions, names, rotation=12, ha="right")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("ROC AUC (training cross-validation fold)")
    ax.set_title("Model comparison · training data only")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(report_dir / "cross-validation-roc-auc.png", dpi=170)
    plt.close(fig)

    selected = test_results["selected_model"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(test_results["confusion_matrix"]), display_labels=["Class 0", "Class 1"]
    ).plot(ax=axes[0], colorbar=False, cmap="Greens", values_format="d")
    axes[0].set_title(f"Held-out test · {selected}")
    for name, result in test_results["models"].items():
        fpr, tpr, _ = roc_curve(test_results["y_true"], result["probabilities"])
        axes[1].plot(fpr, tpr, label=f"{name} (AUC {result['roc_auc']:.2f})")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#888", linewidth=1)
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC · held-out test")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(report_dir / "held-out-confusion-and-roc.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, result in test_results["models"].items():
        precision, recall, _ = precision_recall_curve(
            test_results["y_true"], result["probabilities"]
        )
        ax.plot(recall, precision, label=f"{name} (AP {result['pr_auc']:.2f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall · held-out test")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(report_dir / "held-out-precision-recall.png", dpi=170)
    plt.close(fig)

    serializable = {
        "selection_rule": "Highest mean 5-fold training ROC AUC; test split used only for final evaluation.",
        "cross_validation": {
            name: {
                metric: {
                    "folds": result[f"{metric}_folds"],
                    "mean": float(np.mean(result[f"{metric}_folds"])),
                    "std": float(np.std(result[f"{metric}_folds"])),
                }
                for metric in ("roc_auc", "pr_auc", "precision", "recall", "f1")
            }
            for name, result in cv_results.items()
        },
        "held_out_test": {
            "selected_model": selected,
            "models": {
                name: {"metrics": result["metrics"]}
                for name, result in test_results["models"].items()
            },
            "confusion_matrix_rows_actual_columns_predicted": test_results["confusion_matrix"],
            "note": "Educational dataset evaluation only; scores are not medical risk estimates.",
        },
    }
    (report_dir / "metrics.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    X, y = load_data(args.data)
    if not set(ZERO_AS_MISSING).issubset(X.columns):
        absent = sorted(set(ZERO_AS_MISSING) - set(X.columns))
        raise ValueError(f"Expected Pima dataset columns are missing: {', '.join(absent)}")
    zero_positions = tuple(X.columns.get_loc(name) for name in ZERO_AS_MISSING)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    candidates = make_candidates(zero_positions)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print(f"Dataset: {args.data.resolve()} ({len(X)} rows; {len(X.columns)} features)")
    print("Cross-validation (training split only; ROC AUC):")
    scores: dict[str, float] = {}
    cv_results: dict[str, dict] = {}
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }
    for name, pipeline in candidates.items():
        result = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=N_JOBS,
        )
        scores[name] = float(result["test_roc_auc"].mean())
        cv_results[name] = {
            "roc_auc_folds": [float(value) for value in result["test_roc_auc"]],
            "pr_auc_folds": [float(value) for value in result["test_pr_auc"]],
            "f1_folds": [float(value) for value in result["test_f1"]],
            "precision_folds": [float(value) for value in result["test_precision"]],
            "recall_folds": [float(value) for value in result["test_recall"]],
        }
        print(
            f"  {name}: ROC AUC {scores[name]:.3f} +/- "
            f"{result['test_roc_auc'].std():.3f}; "
            f"PR AUC {result['test_pr_auc'].mean():.3f}; "
            f"precision {result['test_precision'].mean():.3f}; "
            f"recall {result['test_recall'].mean():.3f}; F1 {result['test_f1'].mean():.3f}"
        )

    best_name = max(scores, key=scores.get)
    test_models: dict[str, dict] = {}
    for name, candidate in candidates.items():
        fitted = clone(candidate).fit(X_train, y_train)
        predicted = fitted.predict(X_test)
        probabilities = fitted.predict_proba(X_test)[:, 1]
        test_models[name] = {
            "predicted": predicted,
            "probabilities": probabilities,
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "pr_auc": float(average_precision_score(y_test, probabilities)),
            "metrics": {
                "accuracy": float(accuracy_score(y_test, predicted)),
                "precision": float(precision_score(y_test, predicted, zero_division=0)),
                "recall": float(recall_score(y_test, predicted, zero_division=0)),
                "f1": float(f1_score(y_test, predicted, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
                "pr_auc": float(average_precision_score(y_test, probabilities)),
            },
        }
    best_result = test_models[best_name]
    predicted = best_result["predicted"]
    print(f"\nSelected by training CV ROC AUC: {best_name}")
    print("Held-out test metrics (final evaluation):")
    for metric, value in best_result["metrics"].items():
        print(f"  {metric.replace('_', ' ').title():<10} {value:.3f}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
    print(matrix)
    print("\nClassification report:")
    print(classification_report(y_test, predicted, zero_division=0))
    test_results = {
        "selected_model": best_name,
        "models": test_models,
        "confusion_matrix": matrix.tolist(),
        "y_true": y_test.to_numpy(),
    }
    write_evaluation_report(args.report_dir, cv_results, test_results)
    print(f"Evaluation charts and metrics saved to: {args.report_dir.resolve()}")
    print("\nEducational demonstration only; this model is not a medical diagnostic tool.")


if __name__ == "__main__":
    main()
