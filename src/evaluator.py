"""
evaluator.py
Computes standard performance metrics for classification and regression models.
Calculates Accuracy, Precision, Recall, F1, Confusion Matrix for classification,
and MAE, RMSE, R² for regression.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class ModelEvaluator:
    """
    Evaluates trained models on unseen test sets and returns structured metrics.
    """

    @staticmethod
    def evaluate_classification(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Computes Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
        Uses 'weighted' averaging to handle multi-class and imbalanced distributions fairly.
        """
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

        return {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "confusion_matrix": cm,
            "labels": labels or sorted(list(set(y_true))),
        }

    @staticmethod
    def evaluate_regression(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Computes MAE, RMSE, and R² for regression models.
        """
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))

        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
        }
