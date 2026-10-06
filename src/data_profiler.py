"""
data_profiler.py
Analyzes and profiles input tabular datasets.
Identifies numerical, categorical, and potential ID columns,
calculates missing value statistics, duplicate counts, and summary metrics.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np


class DatasetProfiler:
    """
    Analyzes a pandas DataFrame to generate statistical summaries,
    detect column types, identify potential ID columns, and calculate missingness.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_basic_info(self) -> Dict[str, Any]:
        """
        Returns basic dataset dimensions and counts.
        """
        return {
            "num_rows": int(self.df.shape[0]),
            "num_columns": int(self.df.shape[1]),
            "columns": list(self.df.columns),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "total_cells": int(self.df.size),
            "missing_cells": int(self.df.isna().sum().sum()),
        }

    def detect_column_types(self) -> Dict[str, List[str]]:
        """
        Categorizes columns into numerical, categorical, and potential ID columns.
        """
        numerical_cols: List[str] = []
        categorical_cols: List[str] = []
        potential_id_cols: List[str] = []

        total_rows = len(self.df)

        for col in self.df.columns:
            dtype = self.df[col].dtype
            n_unique = self.df[col].nunique(dropna=True)
            col_name_lower = str(col).lower()

            # ID heuristic: high cardinality relative to rows or matching ID naming pattern
            has_id_keyword = any(
                tag in col_name_lower for tag in ["id", "uuid", "identifier", "key"]
            )
            is_id_name = has_id_keyword and (
                (total_rows <= 10 and n_unique == total_rows)
                or (total_rows > 10 and (n_unique / total_rows) >= 0.6)
            )

            is_id_cardinality = (total_rows > 10) and (n_unique / total_rows >= 0.95)

            if is_id_name or is_id_cardinality:
                potential_id_cols.append(col)

            # Determine numerical vs categorical
            if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)

        return {
            "numerical": numerical_cols,
            "categorical": categorical_cols,
            "potential_ids": potential_id_cols,
        }

    def get_missing_value_summary(self) -> pd.DataFrame:
        """
        Returns a DataFrame showing missing value count and percentage per column.
        """
        missing_count = self.df.isna().sum()
        total_rows = len(self.df)
        missing_pct = (missing_count / total_rows * 100).round(2) if total_rows > 0 else 0

        summary = pd.DataFrame({
            "Column": self.df.columns,
            "Missing Values": missing_count.values,
            "Missing %": missing_pct.values,
            "Data Type": [str(self.df[c].dtype) for c in self.df.columns],
            "Unique Values": [self.df[c].nunique(dropna=True) for c in self.df.columns]
        })

        return summary.sort_values(by="Missing Values", ascending=False).reset_index(drop=True)

    def get_column_profile(self) -> List[Dict[str, Any]]:
        """
        Generates detailed profiling information for each column.
        """
        types = self.detect_column_types()
        total_rows = len(self.df)
        profiles = []

        for col in self.df.columns:
            s = self.df[col]
            n_missing = int(s.isna().sum())
            missing_pct = round((n_missing / total_rows) * 100, 2) if total_rows > 0 else 0.0
            n_unique = int(s.nunique(dropna=True))

            is_id = col in types["potential_ids"]
            col_type = "Numerical" if col in types["numerical"] else "Categorical"

            warning = None
            if is_id:
                warning = "This column appears to be an ID and may not be useful for prediction."
            elif missing_pct > 30:
                warning = f"High missingness: {missing_pct}% of values are missing."

            profiles.append({
                "name": col,
                "type": col_type,
                "dtype": str(s.dtype),
                "unique_count": n_unique,
                "missing_count": n_missing,
                "missing_pct": missing_pct,
                "is_potential_id": is_id,
                "warning": warning,
            })

        return profiles
