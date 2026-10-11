"""
Unit and integration tests for Multi-Model NIDS extension.
Tests loading, inference, thresholding, metrics, and ground-truth evaluation
across all 5 models: CNN-LSTM, CNN-only, LSTM-only, Random Forest, and XGBoost.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from model_registry import MODEL_REGISTRY, get_model_metadata
from preprocessing import (
    load_feature_columns,
    load_scaler,
    validate_csv_data,
    preprocess_features,
    preprocess_tabular_features,
    create_sequences,
    extract_ground_truth_labels,
    aggregate_sequence_ground_truth
)
from inference import get_multi_model_service, MultiModelInferenceService


@pytest.fixture(scope="module")
def multi_service():
    return get_multi_model_service()


@pytest.fixture(scope="module")
def feature_columns():
    return load_feature_columns("models/cnn_lstm/feature_columns.json")


@pytest.fixture(scope="module")
def scaler():
    return load_scaler("models/cnn_lstm/minmax_scaler.joblib")


def test_registry_contains_all_five_models():
    """Verify that all five models are properly registered with valid metadata."""
    expected_ids = {"cnn_lstm", "cnn_only", "lstm_only", "random_forest", "xgboost"}
    assert set(MODEL_REGISTRY.keys()) == expected_ids

    for m_id, meta in MODEL_REGISTRY.items():
        assert meta.model_path.exists(), f"Model checkpoint missing for {m_id} at {meta.model_path}"
        assert meta.config_path.exists(), f"Config file missing for {m_id} at {meta.config_path}"
        assert meta.default_threshold > 0.0
        assert meta.n_features == 78


@pytest.mark.parametrize("model_id", ["cnn_lstm", "cnn_only", "lstm_only", "random_forest", "xgboost"])
def test_all_models_load_successfully(multi_service, model_id):
    """Verify that each model adapter loads without error."""
    adapter = multi_service.get_adapter(model_id)
    assert adapter.model is not None


def test_sequence_models_inference(multi_service, feature_columns, scaler):
    """Test inference on CNN-only and LSTM-only sequence models."""
    raw = np.random.uniform(0, 1, size=(200, 78)).astype(np.float32)
    seqs, ranges = create_sequences(raw, time_steps=100, stride=100)

    for m_id in ["cnn_only", "lstm_only"]:
        df_res, summary = multi_service.predict(m_id, seqs, row_ranges=ranges)
        assert len(df_res) == 2
        assert summary.total_sequences == 2
        assert summary.model_type == "sequence"
        assert (df_res["prediction_score"] >= 0.0).all()
        assert (df_res["prediction_score"] <= 1.0).all()


def test_tabular_models_inference(multi_service, feature_columns):
    """Test inference on Random Forest and XGBoost tabular models."""
    raw = np.random.uniform(0, 100, size=(50, 78)).astype(np.float32)

    for m_id in ["random_forest", "xgboost"]:
        df_res, summary = multi_service.predict(m_id, raw)
        assert len(df_res) == 50
        assert summary.total_sequences == 50
        assert summary.model_type == "tabular"
        assert summary.unit_name == "records"
        assert (df_res["prediction_score"] >= 0.0).all()
        assert (df_res["prediction_score"] <= 1.0).all()


def test_ground_truth_evaluation(multi_service):
    """Verify calculation of accuracy, precision, recall, f1, and confusion matrix."""
    raw = np.random.uniform(0, 100, size=(20, 78)).astype(np.float32)
    # Synthetic ground truth: 10 Benign, 10 Attack
    gt = np.array([0] * 10 + [1] * 10)

    df_res, summary = multi_service.predict("random_forest", raw, ground_truth=gt)
    assert summary.evaluation is not None
    assert 0.0 <= summary.evaluation.accuracy <= 1.0
    assert 0.0 <= summary.evaluation.precision <= 1.0
    assert 0.0 <= summary.evaluation.recall <= 1.0
    assert 0.0 <= summary.evaluation.f1_score <= 1.0
    cm_total = (
        summary.evaluation.true_positives +
        summary.evaluation.true_negatives +
        summary.evaluation.false_positives +
        summary.evaluation.false_negatives
    )
    assert cm_total == 20


@pytest.mark.parametrize("csv_file, expected_pred", [
    ("test_benign.csv", "BENIGN"),
    ("test_ddos.csv", "ATTACK")
])
def test_all_models_on_benchmark_files(multi_service, feature_columns, scaler, csv_file, expected_pred):
    """Verify all five models classify pure benign and pure ddos benchmark datasets correctly."""
    path = Path(csv_file)
    if not path.exists():
        pytest.skip(f"{csv_file} not found")

    df = pd.read_csv(path)
    clean_cols = [c.strip().replace('\ufeff', '') for c in df.columns]
    df.columns = clean_cols

    X_tabular = preprocess_tabular_features(df, feature_columns)
    X_scaled = scaler.transform(X_tabular)
    seqs, ranges = create_sequences(X_scaled, time_steps=100, stride=100)

    # 1. Sequence models
    for m_id in ["cnn_lstm", "cnn_only", "lstm_only"]:
        df_res, summary = multi_service.predict(m_id, seqs, row_ranges=ranges)
        pred_label = "ATTACK" if summary.attack_count > summary.benign_count else "BENIGN"
        assert pred_label == expected_pred, f"Model {m_id} expected {expected_pred} on {csv_file}"

    # 2. Tabular models
    for m_id in ["random_forest", "xgboost"]:
        df_res, summary = multi_service.predict(m_id, X_tabular)
        pred_label = "ATTACK" if summary.attack_count > summary.benign_count else "BENIGN"
        assert pred_label == expected_pred, f"Model {m_id} expected {expected_pred} on {csv_file}"
