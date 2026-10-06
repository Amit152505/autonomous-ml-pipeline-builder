"""
test_detector.py
Tests for ProblemDetector: classification vs regression detection and target validation.
"""

import pandas as pd
from src.problem_detector import ProblemDetector


def test_categorical_classification_detection():
    target = pd.Series(["cat", "dog", "bird", "cat", "dog", "cat"])
    detector = ProblemDetector(target)
    result = detector.detect_problem_type()

    assert result["problem_type"] == "Classification"
    assert result["is_valid"] is True
    assert result["unique_values_count"] == 3


def test_continuous_regression_detection():
    # 50 unique continuous float values
    target = pd.Series([10.5 + i * 1.7 for i in range(50)])
    detector = ProblemDetector(target)
    result = detector.detect_problem_type()

    assert result["problem_type"] == "Regression"
    assert result["is_valid"] is True


def test_discrete_numeric_classification():
    # Binary 0/1 integers
    target = pd.Series([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
    detector = ProblemDetector(target)
    result = detector.detect_problem_type()

    assert result["problem_type"] == "Classification"
    assert result["unique_values_count"] == 2


def test_invalid_single_value_target():
    target = pd.Series([1, 1, 1, 1, 1])
    detector = ProblemDetector(target)
    result = detector.detect_problem_type()

    assert result["is_valid"] is False
    assert "at least 2 distinct" in result["reason"]
