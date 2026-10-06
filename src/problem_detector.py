"""
problem_detector.py
Automatically infers whether a machine learning task is Classification or Regression,
validates the target variable, detects class imbalance, and provides human-readable explanations.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


class ProblemDetector:
    """
    Analyzes target column properties to recommend Classification vs Regression,
    with explainable decision rules and class balance checks.
    """

    def __init__(self, target_series: pd.Series):
        self.target = target_series.dropna()
        self.raw_target = target_series

    def validate_target(self) -> Tuple[bool, str]:
        """
        Validates whether the target column can be used for modeling.
        Returns (is_valid, error_message).
        """
        if self.raw_target is None:
            return False, "Target column is missing."

        if len(self.raw_target) == 0:
            return False, "Target column is empty."

        if self.target.empty:
            return False, "Target column contains only null/missing values."

        n_unique = self.target.nunique()
        if n_unique <= 1:
            return False, f"Target column contains only {n_unique} unique value. Supervised learning requires at least 2 distinct classes or values."

        return True, ""

    def detect_problem_type(self) -> Dict[str, Any]:
        """
        Infers problem type using transparent heuristics.
        """
        valid, msg = self.validate_target()
        if not valid:
            return {
                "problem_type": "Unknown",
                "reason": msg,
                "is_valid": False,
                "confidence": "Low",
                "unique_values_count": int(self.raw_target.nunique(dropna=True)),
                "is_imbalanced": False,
                "class_distribution": {},
            }

        n_unique = self.target.nunique()
        dtype = self.target.dtype
        is_numeric = pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)

        # Rule 1: Non-numeric (strings, categories, booleans) -> Classification
        if not is_numeric:
            dist = self.get_class_distribution()
            imbalanced, imb_msg = self.check_imbalance(dist)
            return {
                "problem_type": "Classification",
                "reason": f"The target column has categorical/text type '{dtype}' with {n_unique} unique classes.",
                "is_valid": True,
                "confidence": "High",
                "unique_values_count": int(n_unique),
                "is_imbalanced": imbalanced,
                "imbalance_message": imb_msg,
                "class_distribution": dist,
                "needs_confirmation": False,
            }

        # Rule 2: Numeric with small number of unique discrete integers (<= 10)
        # e.g., binary 0/1 or ratings 1, 2, 3, 4, 5
        if n_unique <= 10:
            dist = self.get_class_distribution()
            imbalanced, imb_msg = self.check_imbalance(dist)
            return {
                "problem_type": "Classification",
                "reason": f"The target is numeric but contains only {n_unique} distinct discrete values, suggesting discrete classes.",
                "is_valid": True,
                "confidence": "Medium",
                "unique_values_count": int(n_unique),
                "is_imbalanced": imbalanced,
                "imbalance_message": imb_msg,
                "class_distribution": dist,
                "needs_confirmation": True,  # Allow user toggle if they prefer regression
            }

        # Rule 3: Numeric with many unique values -> Regression
        return {
            "problem_type": "Regression",
            "reason": f"The target column is continuous numeric with {n_unique} distinct numerical values.",
            "is_valid": True,
            "confidence": "High",
            "unique_values_count": int(n_unique),
            "is_imbalanced": False,
            "imbalance_message": "",
            "class_distribution": {},
            "needs_confirmation": False,
        }

    def get_class_distribution(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculates counts and percentages for each class in classification.
        """
        counts = self.target.value_counts()
        total = len(self.target)
        dist = {}
        for cls_val, cnt in counts.items():
            pct = round((cnt / total) * 100, 2) if total > 0 else 0
            dist[str(cls_val)] = {
                "count": int(cnt),
                "percentage": float(pct),
            }
        return dist

    def check_imbalance(self, dist: Dict[str, Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Detects if any class is significantly underrepresented (< 20% in binary or ratio > 3:1).
        """
        if len(dist) < 2:
            return False, ""

        percentages = [v["percentage"] for v in dist.values()]
        min_pct = min(percentages)
        max_pct = max(percentages)

        # Flag imbalance if minority class is under 20% or max/min ratio exceeds 3:1
        if min_pct < 20.0 or (min_pct > 0 and (max_pct / min_pct) >= 3.0):
            return True, f"Significant class imbalance detected: minority class represents only {min_pct:.1f}% of samples."

        return False, "Classes are reasonably balanced."
