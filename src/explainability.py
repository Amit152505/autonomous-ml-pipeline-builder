"""
explainability.py
Extracts feature importances and coefficients from trained estimators,
mapping them back accurately to preprocessed column names.
Also generates human-readable "Autonomous Pipeline" decision summaries.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


class ModelExplainer:
    """
    Provides explainability reports, feature importance rankings,
    and end-to-end pipeline decision logs.
    """

    @staticmethod
    def get_feature_importance(
        model: Any,
        feature_names: List[str],
        top_n: int = 15,
    ) -> Dict[str, Any]:
        """
        Extracts feature importances from tree-based models or coefficients from linear models.
        Safely falls back if the model does not support feature rankings.
        """
        if not feature_names:
            return {
                "available": False,
                "importance_df": pd.DataFrame(),
                "explanation": "No feature names were provided from the preprocessing stage.",
            }

        importances = None
        method_name = ""

        # Check tree-based feature_importances_
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            method_name = "Tree-based Gini / Gain Importance"

        # Check linear model coefficients
        elif hasattr(model, "coef_"):
            coef = model.coef_
            # For multi-class classification or 2D array, take mean absolute magnitude across classes
            if coef.ndim > 1:
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)
            method_name = "Normalized Absolute Coefficient Magnitude"

        if importances is None:
            return {
                "available": False,
                "importance_df": pd.DataFrame(),
                "explanation": (
                    f"Feature importance is not directly available for '{model.__class__.__name__}'. "
                    f"Baseline models or non-parametric estimators do not compute individual feature weights."
                ),
            }

        # Handle potential length mismatch safely
        n_features = min(len(feature_names), len(importances))
        feat_subset = feature_names[:n_features]
        imp_subset = np.array(importances[:n_features], dtype=float)

        # Normalize sum to 1.0 (or 100%) if total > 0
        total_imp = np.sum(imp_subset)
        if total_imp > 0:
            norm_imp = imp_subset / total_imp
        else:
            norm_imp = imp_subset

        df = pd.DataFrame({
            "Feature": feat_subset,
            "Importance": imp_subset,
            "RelativeWeight": norm_imp,
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        return {
            "available": True,
            "method": method_name,
            "importance_df": df.head(top_n),
            "explanation": f"Calculated using {method_name}. Values indicate the relative contribution of each feature.",
        }

    @staticmethod
    def generate_pipeline_summary(
        problem_type: str,
        problem_reason: str,
        num_features: List[str],
        cat_features: List[str],
        dropped_ids: List[str],
        best_model_name: str,
        metric_name: str,
        best_score: float,
        best_reason: str,
        models_trained: List[str],
    ) -> str:
        """
        Creates a clean, formatted plain-text autonomous decision breakdown.
        """
        summary_lines = [
            "===========================================================",
            "             AUTONOMOUS ML PIPELINE SUMMARY                ",
            "===========================================================",
            f"1. PROBLEM FORMULATION:",
            f"   - Detected Type : {problem_type}",
            f"   - Justification : {problem_reason}",
            "",
            f"2. FEATURE ENGINEERING & PREPROCESSING:",
            f"   - Numerical Features   ({len(num_features)}): {', '.join(num_features) if num_features else 'None'}",
            f"     -> Strategy: Median Imputation + StandardScaler",
            f"   - Categorical Features ({len(cat_features)}): {', '.join(cat_features) if cat_features else 'None'}",
            f"     -> Strategy: Most Frequent Imputation + OneHotEncoder",
        ]

        if dropped_ids:
            summary_lines.append(
                f"   - Flagged High-Cardinality/ID ({len(dropped_ids)}): {', '.join(dropped_ids)}"
            )

        summary_lines.extend([
            f"   - Data Leakage Guard: Fitted strictly on training split only.",
            "",
            f"3. MODEL BENCHMARKING:",
            f"   - Candidate Models Evaluated : {', '.join(models_trained)}",
            f"   - Selection Criterion        : {metric_name}",
            "",
            f"4. BEST MODEL SELECTION:",
            f"   - Chosen Model               : {best_model_name}",
            f"   - Test {metric_name:<19} : {best_score:.4f}",
            f"   - Selection Rationale        : {best_reason}",
            "===========================================================",
        ])

        return "\n".join(summary_lines)
