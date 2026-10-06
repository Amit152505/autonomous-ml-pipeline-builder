"""
model_selector.py
Selects the best performing model from evaluation results using standard objective criteria:
F1-Score for classification (higher is better) and RMSE for regression (lower is better).
Generates concise technical explanations for why the model was chosen.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd


class ModelSelector:
    """
    Selects the optimal model based on task-appropriate evaluation criteria.
    """

    def __init__(self, problem_type: str, evaluation_results: Dict[str, Dict[str, Any]]):
        self.problem_type = problem_type
        self.results = evaluation_results

    def select_best_model(self) -> Dict[str, Any]:
        """
        Determines the winning model based on F1 (classification) or RMSE (regression).
        Excludes failed runs and prioritizes non-dummy models when performance is comparable.
        """
        successful_models = {
            name: res
            for name, res in self.results.items()
            if res.get("status") == "success" and res.get("metrics")
        }

        if not successful_models:
            raise ValueError("No models trained successfully to select from.")

        if self.problem_type == "Classification":
            metric_name = "F1"
            # Sort by F1 descending (higher is better)
            # When tie or baseline, prefer non-baseline if score is equal or close
            ranked = sorted(
                successful_models.items(),
                key=lambda item: (
                    item[1]["metrics"].get("F1", 0.0),
                    item[1]["metrics"].get("Accuracy", 0.0),
                    0 if "Baseline" in item[0] else 1,
                ),
                reverse=True,
            )
            best_name, best_data = ranked[0]
            best_score = best_data["metrics"].get("F1", 0.0)

            # Compare against baseline
            baseline_data = self.results.get("Baseline (Dummy)")
            baseline_score = baseline_data["metrics"].get("F1", 0.0) if baseline_data else 0.0
            improvement = (
                f" This represents an improvement of {round((best_score - baseline_score) * 100, 1)}% over the Dummy baseline ({baseline_score:.4f})."
                if baseline_data and "Baseline" not in best_name
                else ""
            )

            explanation = (
                f"Selected {best_name} because it achieved the highest F1 score ({best_score:.4f}) "
                f"on the unseen test set among all candidate models evaluated.{improvement} "
                f"Note: This is the best among the models evaluated on this dataset, not a universally optimal guarantee."
            )

        else:
            metric_name = "RMSE"
            # Sort by RMSE ascending (lower is better)
            ranked = sorted(
                successful_models.items(),
                key=lambda item: (
                    item[1]["metrics"].get("RMSE", float("inf")),
                    -item[1]["metrics"].get("R2", -float("inf")),
                    1 if "Baseline" in item[0] else 0,
                ),
            )
            best_name, best_data = ranked[0]
            best_score = best_data["metrics"].get("RMSE", 0.0)

            baseline_data = self.results.get("Baseline (Dummy)")
            baseline_score = baseline_data["metrics"].get("RMSE", 0.0) if baseline_data else 0.0
            improvement = (
                f" This reduces root-mean-squared error compared to the Dummy baseline ({baseline_score:.4f})."
                if baseline_data and "Baseline" not in best_name
                else ""
            )

            explanation = (
                f"Selected {best_name} because it achieved the lowest RMSE ({best_score:.4f}) "
                f"and an R² of {best_data['metrics'].get('R2', 0.0):.4f} on the unseen test set.{improvement} "
                f"Note: This is the best among the models evaluated on this dataset, not a universally optimal guarantee."
            )

        return {
            "best_model_name": best_name,
            "metric_name": metric_name,
            "best_metric_value": best_score,
            "metrics": best_data["metrics"],
            "explanation": explanation,
        }
