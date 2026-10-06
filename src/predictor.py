"""
predictor.py
Manages model persistence (save/load via Joblib) and inference on new inputs.
Ensures identical preprocessing transforms are executed on new test points.
"""

from typing import Dict, Any, Union, Optional
import os
import joblib
import numpy as np
import pandas as pd


class Predictor:
    """
    Handles saving, loading, and predicting with trained pipeline artifacts.
    """

    def __init__(
        self,
        model: Any = None,
        preprocessor: Any = None,
        target_encoder: Any = None,
        problem_type: str = "Classification",
        feature_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.target_encoder = target_encoder
        self.problem_type = problem_type
        self.feature_metadata = feature_metadata or {}

    def save(self, filepath: str) -> str:
        """
        Saves the complete inference bundle into a single joblib archive.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        bundle = {
            "model": self.model,
            "preprocessor": self.preprocessor,
            "target_encoder": self.target_encoder,
            "problem_type": self.problem_type,
            "feature_metadata": self.feature_metadata,
        }
        joblib.dump(bundle, filepath)
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "Predictor":
        """
        Loads an exported inference bundle from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")

        bundle = joblib.load(filepath)
        return cls(
            model=bundle.get("model"),
            preprocessor=bundle.get("preprocessor"),
            target_encoder=bundle.get("target_encoder"),
            problem_type=bundle.get("problem_type", "Classification"),
            feature_metadata=bundle.get("feature_metadata", {}),
        )

    def predict(self, input_data: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Performs inference on a single row or dataframe.
        Returns predicted label/value and probabilities if classification.
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Predictor is not initialized with a valid model and preprocessor.")

        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()

        # Transform inputs using the saved preprocessor pipeline
        X_trans = self.preprocessor.transform(input_df)

        # Generate raw predictions
        raw_pred = self.model.predict(X_trans)

        result: Dict[str, Any] = {
            "raw_prediction": raw_pred.tolist(),
        }

        if self.problem_type == "Classification":
            # Map back through LabelEncoder if available
            if self.target_encoder is not None:
                decoded_pred = self.target_encoder.inverse_transform(raw_pred.astype(int))
                result["prediction"] = decoded_pred[0] if len(decoded_pred) == 1 else decoded_pred.tolist()
            else:
                result["prediction"] = raw_pred[0] if len(raw_pred) == 1 else raw_pred.tolist()

            # Predict probabilities if supported
            if hasattr(self.model, "predict_proba"):
                try:
                    probs = self.model.predict_proba(X_trans)[0]
                    classes = (
                        self.target_encoder.classes_
                        if self.target_encoder is not None
                        else getattr(self.model, "classes_", [f"Class {i}" for i in range(len(probs))])
                    )
                    prob_dict = {
                        str(cls_name): round(float(p) * 100, 2)
                        for cls_name, p in zip(classes, probs)
                    }
                    result["probabilities"] = prob_dict
                except Exception:
                    result["probabilities"] = None
        else:
            # Regression prediction
            val = float(raw_pred[0]) if len(raw_pred) == 1 else [float(x) for x in raw_pred]
            result["prediction"] = val

        return result
