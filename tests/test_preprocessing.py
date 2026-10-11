"""
Unit and integration tests for preprocessing module.
Tests data validation, schema checking, NaN/Inf handling, scaling, and sequence construction.
"""

import io
import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from preprocessing import (
    load_feature_columns,
    load_scaler,
    validate_csv_data,
    preprocess_features,
    create_sequences,
    full_preprocessing_pipeline,
    ValidationResult
)


@pytest.fixture(scope="module")
def feature_columns():
    return load_feature_columns("models/feature_columns.json")


@pytest.fixture(scope="module")
def scaler():
    return load_scaler("models/minmax_scaler.joblib")


def test_scaler_properties(scaler, feature_columns):
    """Test that the loaded scaler is compatible with the 78 feature columns."""
    assert isinstance(scaler, MinMaxScaler)
    assert scaler.n_features_in_ == 78
    assert len(feature_columns) == 78
    assert scaler.data_min_.shape == (78,)
    assert scaler.data_max_.shape == (78,)
    assert scaler.feature_range == (0, 1)


def test_valid_csv_input(feature_columns, scaler):
    """Test preprocessing with valid synthetic network flow data."""
    n_rows = 250
    # Create valid dataframe with exact 78 columns plus an extra Label column
    data = {col: np.random.uniform(0, 1000, size=n_rows) for col in feature_columns}
    data["Label"] = ["BENIGN"] * n_rows
    df = pd.DataFrame(data)

    seqs, ranges, val_res, clean_df = full_preprocessing_pipeline(
        data=df,
        feature_columns=feature_columns,
        scaler=scaler,
        time_steps=100,
        stride=100
    )

    assert val_res.is_valid is True
    assert len(val_res.errors) == 0
    assert val_res.total_rows == 250
    assert val_res.total_columns == 79
    assert val_res.num_sequences == 2  # (250 - 100) // 100 + 1 = 2
    assert val_res.incomplete_rows == 50  # 250 - 200 = 50
    assert seqs.shape == (2, 100, 78)
    assert len(ranges) == 2
    assert ranges[0] == (0, 99)
    assert ranges[1] == (100, 199)


def test_empty_csv(feature_columns):
    """Test that empty CSV raises appropriate validation error without crashing."""
    empty_csv = ""
    buf = io.StringIO(empty_csv)
    df, val_res = validate_csv_data(buf, feature_columns, time_steps=100)
    assert val_res.is_valid is False
    assert any("empty" in err.lower() for err in val_res.errors)
    assert df is None


def test_missing_required_features(feature_columns):
    """Test that missing required feature columns produces a clear error."""
    # Create dataframe with only 10 features
    partial_data = {col: np.random.rand(120) for col in feature_columns[:10]}
    df = pd.DataFrame(partial_data)

    clean_df, val_res = validate_csv_data(df, feature_columns, time_steps=100)
    assert val_res.is_valid is False
    assert len(val_res.missing_features) == 68
    assert any("missing 68 required" in err.lower() for err in val_res.errors)
    assert clean_df is None


def test_extra_columns_safe_handling(feature_columns, scaler):
    """Test that extra non-model columns are safely ignored with a warning."""
    data = {col: np.random.rand(120) for col in feature_columns}
    # Add non-model metadata columns
    data["Flow ID"] = ["192.168.1.1-10.0.0.1"] * 120
    data["Source IP"] = ["192.168.1.1"] * 120
    data["Destination IP"] = ["10.0.0.1"] * 120
    data["Timestamp"] = ["07/07/2017 8:30:00"] * 120
    data["Label"] = ["DDoS"] * 120
    df = pd.DataFrame(data)

    seqs, ranges, val_res, clean_df = full_preprocessing_pipeline(
        data=df,
        feature_columns=feature_columns,
        scaler=scaler,
        time_steps=100,
        stride=100
    )

    assert val_res.is_valid is True
    assert len(val_res.unexpected_columns) == 5
    assert any("non-model column" in w.lower() for w in val_res.warnings)
    assert seqs.shape == (1, 100, 78)


def test_incorrect_feature_ordering(feature_columns, scaler):
    """Test that shuffled feature columns are strictly reordered to the original training sequence."""
    shuffled_cols = list(feature_columns)
    np.random.seed(42)
    np.random.shuffle(shuffled_cols)
    assert shuffled_cols != feature_columns

    # Fill distinct column values
    data = {col: np.full(100, idx, dtype=np.float32) for idx, col in enumerate(feature_columns)}
    shuffled_df = pd.DataFrame(data)[shuffled_cols]

    # Preprocess
    scaled = preprocess_features(shuffled_df, feature_columns, scaler)
    assert scaled.shape == (100, 78)


def test_nan_and_infinite_values(feature_columns, scaler):
    """Test that NaN and Inf values are safely handled and imputed with 0.0."""
    data = {col: np.zeros(150, dtype=np.float32) for col in feature_columns}
    df = pd.DataFrame(data).astype(object)

    # Insert NaN, Inf, -Inf, and invalid string
    df.loc[5, feature_columns[0]] = np.nan
    df.loc[10, feature_columns[1]] = np.inf
    df.loc[15, feature_columns[2]] = -np.inf
    df.loc[20, feature_columns[3]] = "invalid_string"

    clean_df, val_res = validate_csv_data(df, feature_columns, time_steps=100)
    assert val_res.is_valid is True
    assert val_res.nan_count > 0
    assert val_res.inf_count > 0

    scaled = preprocess_features(clean_df, feature_columns, scaler)
    assert not np.isnan(scaled).any()
    assert not np.isinf(scaled).any()


def test_insufficient_sequence_length(feature_columns):
    """Test that datasets with fewer than 100 rows trigger an insufficient rows error."""
    data = {col: np.random.rand(50) for col in feature_columns}
    df = pd.DataFrame(data)

    clean_df, val_res = validate_csv_data(df, feature_columns, time_steps=100)
    assert val_res.is_valid is False
    assert any("insufficient rows" in err.lower() for err in val_res.errors)


def test_sequence_windowing_sliding_vs_nonoverlapping(feature_columns, scaler):
    """Test sequence construction with non-overlapping (stride=100) vs sliding (stride=20)."""
    raw = np.random.uniform(0, 1, size=(200, 78)).astype(np.float32)

    # Non-overlapping
    seqs_non, ranges_non = create_sequences(raw, time_steps=100, stride=100)
    assert seqs_non.shape == (2, 100, 78)
    assert ranges_non == [(0, 99), (100, 199)]

    # Sliding window with stride=20: (200 - 100) // 20 + 1 = 6 sequences
    seqs_slide, ranges_slide = create_sequences(raw, time_steps=100, stride=20)
    assert seqs_slide.shape == (6, 100, 78)
    assert ranges_slide[0] == (0, 99)
    assert ranges_slide[1] == (20, 119)
    assert ranges_slide[-1] == (100, 199)
