"""
test_profiler.py
Tests for DatasetProfiler: column type detection, missing values, ID detection.
"""

import pandas as pd
import numpy as np
from src.data_profiler import DatasetProfiler


def test_detect_column_types():
    df = pd.DataFrame({
        "age": [25, 30, 35, 40],
        "salary": [50000.0, 60000.5, 75000.0, 80000.0],
        "department": ["Sales", "HR", "IT", "Sales"],
        "is_active": [True, False, True, True],
        "user_id": [101, 102, 103, 104],
    })
    profiler = DatasetProfiler(df)
    types = profiler.detect_column_types()

    assert "age" in types["numerical"]
    assert "salary" in types["numerical"]
    assert "department" in types["categorical"]
    assert "is_active" in types["categorical"]
    assert "user_id" in types["potential_ids"]


def test_missing_values_summary():
    df = pd.DataFrame({
        "col1": [1, np.nan, 3, np.nan],
        "col2": ["A", "B", "C", "D"],
    })
    profiler = DatasetProfiler(df)
    summary = profiler.get_missing_value_summary()

    assert len(summary) == 2
    col1_row = summary[summary["Column"] == "col1"].iloc[0]
    assert col1_row["Missing Values"] == 2
    assert col1_row["Missing %"] == 50.0

    col2_row = summary[summary["Column"] == "col2"].iloc[0]
    assert col2_row["Missing Values"] == 0
    assert col2_row["Missing %"] == 0.0


def test_basic_info():
    df = pd.DataFrame({
        "a": [1, 2, 2],
        "b": [3, 4, 4],
    })
    profiler = DatasetProfiler(df)
    info = profiler.get_basic_info()

    assert info["num_rows"] == 3
    assert info["num_columns"] == 2
    assert info["duplicate_rows"] == 1
