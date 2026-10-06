"""
preprocessing.py
Constructs leak-free scikit-learn preprocessing pipelines using ColumnTransformer,
SimpleImputer, StandardScaler, and OneHotEncoder.
Enforces strict train-only fitting to guarantee zero data leakage.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


class DataPreprocessor:
    """
    Builds and manages preprocessing pipelines for numerical and categorical features.
    Guarantees no data leakage by fitting transformers strictly on training splits.
    """

    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
    ):
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.pipeline: Optional[ColumnTransformer] = None
        self.target_encoder: Optional[LabelEncoder] = None
        self.feature_names_out: List[str] = []

    def build_transformer(self) -> ColumnTransformer:
        """
        Builds a ColumnTransformer with:
        - Numerical: Median Imputation + StandardScaler
        - Categorical: Most Frequent Imputation + OneHotEncoder(handle_unknown='ignore')
        """
        transformers = []

        if self.numerical_features:
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", num_pipeline, self.numerical_features))

        if self.categorical_features:
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat", cat_pipeline, self.categorical_features))

        self.pipeline = ColumnTransformer(
            transformers=transformers,
            remainder="drop",  # Drop unused features such as ID columns
        )
        return self.pipeline

    def fit(self, X_train: pd.DataFrame) -> "DataPreprocessor":
        """
        Fits the ColumnTransformer strictly on the training dataset.
        Prevents data leakage.
        """
        if self.pipeline is None:
            self.build_transformer()

        self.pipeline.fit(X_train)

        # Retrieve transformed feature names
        try:
            self.feature_names_out = list(self.pipeline.get_feature_names_out())
            # Clean up default prefixes like 'num__' and 'cat__' for nicer presentation
            self.feature_names_out = [
                name.replace("num__", "").replace("cat__", "")
                for name in self.feature_names_out
            ]
        except Exception:
            # Fallback for older scikit-learn or edge cases
            self.feature_names_out = [f"feature_{i}" for i in range(self.transform(X_train).shape[1])]

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms input feature DataFrame using the fitted preprocessor.
        """
        if self.pipeline is None:
            raise ValueError("Preprocessor has not been fitted yet. Call fit() first.")
        return self.pipeline.transform(X)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """
        Fits on training features and transforms them.
        """
        self.fit(X_train)
        return self.transform(X_train)

    def prepare_target(
        self,
        y_train: pd.Series,
        y_test: Optional[pd.Series] = None,
        is_classification: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Encodes target variables if needed.
        For classification: uses LabelEncoder if targets are strings/objects.
        For regression: converts to float numpy array.
        """
        if is_classification:
            # Check if non-numeric or labels require 0..N-1 encoding
            is_non_numeric = not pd.api.types.is_numeric_dtype(y_train.dtype)
            unique_train_vals = np.sort(y_train.dropna().unique())

            # If labels are non-numeric or not standard integers 0..K-1, encode them
            needs_encoding = is_non_numeric or not np.array_equal(
                unique_train_vals, np.arange(len(unique_train_vals))
            )

            if needs_encoding:
                self.target_encoder = LabelEncoder()
                y_train_arr = self.target_encoder.fit_transform(y_train.astype(str))
                y_test_arr = (
                    self.target_encoder.transform(y_test.astype(str))
                    if y_test is not None
                    else None
                )
                return y_train_arr, y_test_arr

            # Already clean numeric classes
            y_train_arr = y_train.to_numpy().astype(int)
            y_test_arr = y_test.to_numpy().astype(int) if y_test is not None else None
            return y_train_arr, y_test_arr

        # Regression
        y_train_arr = y_train.to_numpy().astype(float)
        y_test_arr = y_test.to_numpy().astype(float) if y_test is not None else None
        return y_train_arr, y_test_arr

    def get_summary(self) -> Dict[str, Any]:
        """
        Returns a human-readable summary of preprocessing decisions.
        """
        return {
            "numerical_features_count": len(self.numerical_features),
            "categorical_features_count": len(self.categorical_features),
            "numerical_strategy": "Median Imputation + StandardScaler",
            "categorical_strategy": "Most Frequent Imputation + OneHotEncoder",
            "total_output_features": len(self.feature_names_out),
            "output_features": self.feature_names_out,
        }
