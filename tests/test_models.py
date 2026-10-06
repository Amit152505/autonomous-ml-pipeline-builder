"""
test_models.py
Tests for end-to-end ML pipeline: preprocessing, model training, evaluation, and prediction.
"""

import os
import tempfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.preprocessing import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.model_selector import ModelSelector
from src.predictor import Predictor


def test_end_to_end_classification_pipeline():
    # Synthetic classification data
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, size=n),
        "income": np.random.uniform(20000, 100000, size=n),
        "department": np.random.choice(["Sales", "Engineering", "Marketing"], size=n),
        "target": np.random.choice(["Stay", "Leave"], size=n, p=[0.7, 0.3]),
    })

    # Train / test split FIRST (strict data leakage prevention)
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing
    preprocessor = DataPreprocessor(
        numerical_features=["age", "income"],
        categorical_features=["department"],
    )
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    y_train_trans, y_test_trans = preprocessor.prepare_target(
        y_train, y_test, is_classification=True
    )

    assert X_train_trans.shape[0] == 80
    assert X_test_trans.shape[0] == 20
    assert X_train_trans.shape[1] > 2  # Scaled nums + onehot cats

    # Model training
    trainer = ModelTrainer(problem_type="Classification", random_state=42)
    results = trainer.train_and_evaluate_all(
        X_train=X_train_trans,
        y_train=y_train_trans,
        X_test=X_test_trans,
        y_test=y_test_trans,
    )

    # Verify models trained
    assert "Baseline (Dummy)" in results
    assert "Logistic Regression" in results
    assert "Random Forest" in results
    assert results["Random Forest"]["status"] == "success"
    assert "F1" in results["Random Forest"]["metrics"]

    # Model selection
    selector = ModelSelector("Classification", results)
    best_info = selector.select_best_model()
    best_name = best_info["best_model_name"]
    best_model = trainer.trained_models[best_name]

    assert best_name in results
    assert best_info["best_metric_value"] >= 0.0

    # Save and predict
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model.joblib")
        predictor = Predictor(
            model=best_model,
            preprocessor=preprocessor,
            target_encoder=preprocessor.target_encoder,
            problem_type="Classification",
        )
        predictor.save(save_path)

        loaded_predictor = Predictor.load(save_path)
        sample_input = {"age": 30, "income": 50000.0, "department": "Sales"}
        pred_res = loaded_predictor.predict(sample_input)

        assert "prediction" in pred_res
        assert pred_res["prediction"] in ["Stay", "Leave"]


def test_end_to_end_regression_pipeline():
    np.random.seed(42)
    n = 80
    df = pd.DataFrame({
        "sqft": np.random.uniform(500, 3000, size=n),
        "rooms": np.random.randint(1, 5, size=n),
        "location": np.random.choice(["Urban", "Suburban"], size=n),
    })
    df["price"] = df["sqft"] * 150 + df["rooms"] * 10000 + np.random.normal(0, 5000, size=n)

    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = DataPreprocessor(
        numerical_features=["sqft", "rooms"],
        categorical_features=["location"],
    )
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    y_train_trans, y_test_trans = preprocessor.prepare_target(
        y_train, y_test, is_classification=False
    )

    trainer = ModelTrainer(problem_type="Regression", random_state=42)
    results = trainer.train_and_evaluate_all(
        X_train=X_train_trans,
        y_train=y_train_trans,
        X_test=X_test_trans,
        y_test=y_test_trans,
    )

    assert "Linear Regression" in results
    assert results["Linear Regression"]["status"] == "success"
    assert "RMSE" in results["Linear Regression"]["metrics"]
    assert "R2" in results["Linear Regression"]["metrics"]

    selector = ModelSelector("Regression", results)
    best_info = selector.select_best_model()
    best_name = best_info["best_model_name"]
    best_model = trainer.trained_models[best_name]

    predictor = Predictor(
        model=best_model,
        preprocessor=preprocessor,
        target_encoder=None,
        problem_type="Regression",
    )
    sample_input = {"sqft": 1500, "rooms": 3, "location": "Suburban"}
    pred_res = predictor.predict(sample_input)

    assert "prediction" in pred_res
    assert isinstance(pred_res["prediction"], float)
    assert pred_res["prediction"] > 0
