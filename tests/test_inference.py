"""
Unit and integration tests for inference module.
Tests real CNN-LSTM checkpoint loading, input/output dimensions, thresholding,
metrics calculation, export formatting, and deterministic reproducibility.
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from inference import NIDSInferenceEngine, get_inference_engine
from preprocessing import full_preprocessing_pipeline, load_feature_columns, load_scaler


@pytest.fixture(scope="module")
def inference_engine():
    """Initializes the real inference engine with actual model artifacts."""
    return get_inference_engine(
        model_path="models/best_cnn_lstm_checkpoint.keras",
        config_path="models/model_config.json"
    )


def test_model_loading_and_architecture(inference_engine):
    """Verify that the model checkpoint loads properly and has the expected dimensions."""
    assert inference_engine.model is not None
    assert inference_engine.model.input_shape == (None, 100, 78)
    assert inference_engine.model.output_shape == (None, 1)
    assert inference_engine.time_steps == 100
    assert inference_engine.n_features == 78
    assert np.isclose(inference_engine.threshold, 0.46, atol=1e-3)
    assert inference_engine.target_mapping == {"BENIGN": 0, "ATTACK": 1}


def test_output_shape_and_probability_range(inference_engine):
    """Verify that predictions are valid probabilities in [0.0, 1.0]."""
    np.random.seed(42)
    dummy_seqs = np.random.uniform(0.0, 1.0, size=(5, 100, 78)).astype(np.float32)
    results_df, summary = inference_engine.predict(dummy_seqs)

    assert len(results_df) == 5
    assert (results_df["prediction_score"] >= 0.0).all()
    assert (results_df["prediction_score"] <= 1.0).all()
    assert set(results_df["predicted_label"].unique()).issubset({"BENIGN", "ATTACK"})
    assert summary.total_sequences == 5
    assert summary.benign_count + summary.attack_count == 5


def test_threshold_behavior(inference_engine):
    """Test custom thresholding behavior and label assignment."""
    dummy_seqs = np.random.uniform(0.0, 1.0, size=(4, 100, 78)).astype(np.float32)

    # With threshold 0.0, every prediction >= 0.0 must be ATTACK
    df_attack, summary_attack = inference_engine.predict(dummy_seqs, custom_threshold=0.0)
    assert summary_attack.attack_count == 4
    assert summary_attack.benign_count == 0
    assert (df_attack["predicted_label"] == "ATTACK").all()

    # With threshold 1.0, predictions < 1.0 must be BENIGN
    df_benign, summary_benign = inference_engine.predict(dummy_seqs, custom_threshold=1.0)
    assert summary_benign.benign_count == 4
    assert summary_benign.attack_count == 0
    assert (df_benign["predicted_label"] == "BENIGN").all()


def test_inference_metrics_and_summary(inference_engine):
    """Verify that inference duration, throughput, and latency are calculated accurately."""
    dummy_seqs = np.random.uniform(0.0, 1.0, size=(10, 100, 78)).astype(np.float32)
    results_df, summary = inference_engine.predict(dummy_seqs)

    assert summary.inference_time_seconds > 0
    assert summary.throughput_seq_per_sec > 0
    assert summary.latency_ms_per_seq > 0
    assert summary.total_sequences == 10
    assert np.isclose(summary.benign_percentage + summary.attack_percentage, 100.0, atol=0.1)

    # Test JSON serialization of summary
    summary_dict = summary.to_dict()
    json_str = json.dumps(summary_dict)
    loaded_dict = json.loads(json_str)
    assert loaded_dict["total_sequences"] == 10
    assert "throughput_seq_per_sec" in loaded_dict


def test_repeated_inference_consistency(inference_engine):
    """Verify that repeated inference runs on the same input yield identical deterministic results."""
    np.random.seed(123)
    dummy_seqs = np.random.uniform(0.0, 1.0, size=(8, 100, 78)).astype(np.float32)

    df1, _ = inference_engine.predict(dummy_seqs)
    df2, _ = inference_engine.predict(dummy_seqs)

    np.testing.assert_allclose(
        df1["prediction_score"].to_numpy(),
        df2["prediction_score"].to_numpy(),
        rtol=1e-6,
        err_msg="Inference predictions must be deterministic and identical across repeated runs."
    )
    assert (df1["predicted_label"] == df2["predicted_label"]).all()


def test_real_pipeline_end_to_end(inference_engine):
    """End-to-end integration test with sample_data.csv."""
    sample_path = Path("sample_data.csv")
    if not sample_path.exists():
        pytest.skip("sample_data.csv not generated yet")

    feat_cols = load_feature_columns("models/feature_columns.json")
    scaler = load_scaler("models/minmax_scaler.joblib")

    seqs, ranges, val_res, clean_df = full_preprocessing_pipeline(
        data=sample_path,
        feature_columns=feat_cols,
        scaler=scaler,
        time_steps=100,
        stride=100
    )

    assert val_res.is_valid is True
    assert seqs.shape[0] == 3  # 300 rows // 100 = 3 sequences
    assert seqs.shape[1:] == (100, 78)

    results_df, summary = inference_engine.predict(seqs, row_ranges=ranges)

    assert len(results_df) == 3
    assert summary.total_sequences == 3
    # Check that sequence results contain valid labels and score
    assert all(label in ["BENIGN", "ATTACK"] for label in results_df["predicted_label"])
    assert results_df["row_range"].iloc[0] == "Rows 0–99"
    assert results_df["row_range"].iloc[1] == "Rows 100–199"
    assert results_df["row_range"].iloc[2] == "Rows 200–299"
