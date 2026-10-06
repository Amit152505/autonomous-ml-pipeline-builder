"""
model_trainer.py
Trains baseline and candidate machine learning algorithms for classification and regression tasks.
Includes DummyClassifier/DummyRegressor baselines, linear/logistic models, Decision Trees,
Random Forests, and XGBoost.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from .evaluator import ModelEvaluator


class ModelTrainer:
    """
    Trains and evaluates a curated suite of standard ML models against a baseline.
    """

    def __init__(
        self,
        problem_type: str,
        is_imbalanced: bool = False,
        random_state: int = 42,
    ):
        self.problem_type = problem_type
        self.is_imbalanced = is_imbalanced
        self.random_state = random_state
        self.trained_models: Dict[str, Any] = {}
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}

    def get_candidate_models(self) -> Dict[str, Any]:
        """
        Instantiates appropriate models based on problem type and class balance.
        """
        if self.problem_type == "Classification":
            cw = "balanced" if self.is_imbalanced else None
            return {
                "Baseline (Dummy)": DummyClassifier(strategy="most_frequent"),
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state,
                    class_weight=cw,
                ),
                "Decision Tree": DecisionTreeClassifier(
                    max_depth=5,
                    random_state=self.random_state,
                    class_weight=cw,
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=self.random_state,
                    class_weight=cw,
                ),
                "XGBoost": XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    eval_metric="logloss",
                ),
            }
        else:
            return {
                "Baseline (Dummy)": DummyRegressor(strategy="mean"),
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(
                    max_depth=5,
                    random_state=self.random_state,
                ),
                "Random Forest": RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    random_state=self.random_state,
                ),
                "XGBoost": XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=self.random_state,
                ),
            }

    def train_and_evaluate_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Iterates through candidate models, fits each on training data, evaluates on test data.
        Calls optional progress_callback(model_name, status) for UI updates.
        """
        candidate_models = self.get_candidate_models()
        self.evaluation_results = {}
        self.trained_models = {}

        for name, model in candidate_models.items():
            if progress_callback:
                progress_callback(name, "training")

            try:
                # Train model strictly on training split
                model.fit(X_train, y_train)
                self.trained_models[name] = model

                # Predict on test split
                y_pred = model.predict(X_test)

                # Predict probabilities if classification and supported
                y_prob = None
                if self.problem_type == "Classification" and hasattr(model, "predict_proba"):
                    try:
                        y_prob = model.predict_proba(X_test)
                    except Exception:
                        y_prob = None

                # Compute metrics
                if self.problem_type == "Classification":
                    metrics = ModelEvaluator.evaluate_classification(
                        y_true=y_test,
                        y_pred=y_pred,
                    )
                else:
                    metrics = ModelEvaluator.evaluate_regression(
                        y_true=y_test,
                        y_pred=y_pred,
                    )

                self.evaluation_results[name] = {
                    "metrics": metrics,
                    "y_pred": y_pred,
                    "y_prob": y_prob,
                    "status": "success",
                }

                if progress_callback:
                    progress_callback(name, "completed")

            except Exception as e:
                self.evaluation_results[name] = {
                    "metrics": {},
                    "error": str(e),
                    "status": "failed",
                }
                if progress_callback:
                    progress_callback(name, "failed")

        return self.evaluation_results

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Builds a tidy comparison DataFrame summarizing all model performances.
        """
        rows = []
        for name, res in self.evaluation_results.items():
            if res.get("status") == "success":
                row = {"Model": name}
                row.update(res["metrics"])
                # Remove confusion_matrix and labels from tabular comparison
                row.pop("confusion_matrix", None)
                row.pop("labels", None)
                rows.append(row)

        df = pd.DataFrame(rows)
        return df
