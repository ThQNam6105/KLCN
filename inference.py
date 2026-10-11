"""
Modular Multi-Model Inference Architecture for Network Intrusion Detection System.
Supports CNN-LSTM, Pure CNN, Pure LSTM, Random Forest, and XGBoost with
deterministic inference, threshold tuning, telemetry, and ground-truth evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from model_registry import ModelMetadata, MODEL_REGISTRY, get_model_metadata


@dataclass
class EvaluationMetrics:
    """Supervised evaluation metrics against ground truth."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceSummary:
    """Summary metrics of an inference run for any model."""
    model_name: str
    total_sequences: int                 # Also represents total evaluated units (samples)
    benign_count: int
    attack_count: int
    benign_percentage: float
    attack_percentage: float
    avg_prediction_score: float
    min_prediction_score: float
    max_prediction_score: float
    threshold_applied: float
    inference_time_seconds: float
    throughput_seq_per_sec: float
    latency_ms_per_seq: float
    batch_size_used: int
    timestamp_utc: str
    model_id: str = "cnn_lstm"
    model_type: str = "sequence"         # 'sequence' or 'tabular'
    unit_name: str = "sequences"         # 'sequences' or 'records'
    evaluation: Optional[EvaluationMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.evaluation:
            d["evaluation"] = self.evaluation.to_dict()
        return d


# -----------------------------------------------------------------------------
# Model Adapters (Modular inference for each architecture)
# -----------------------------------------------------------------------------

class BaseModelAdapter(ABC):
    """Abstract interface for all model adapters."""

    def __init__(self, metadata: ModelMetadata):
        self.metadata = metadata
        self.model = None

    @abstractmethod
    def load(self) -> None:
        """Loads model weights/artifacts from disk."""
        pass

    @abstractmethod
    def predict_scores(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Computes continuous attack probabilities in [0.0, 1.0]."""
        pass


class KerasSequenceAdapter(BaseModelAdapter):
    """Adapter for sequence models (CNN-LSTM, CNN-only, LSTM-only)."""

    def load(self) -> None:
        import keras
        if not self.metadata.model_path.exists():
            raise FileNotFoundError(f"Keras checkpoint not found at: {self.metadata.model_path}")
        self.model = keras.models.load_model(self.metadata.model_path, compile=False)

    def predict_scores(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        if self.model is None:
            self.load()
        preds = self.model.predict(X, batch_size=batch_size, verbose=0)
        return preds.flatten().astype(np.float64)


class SklearnTabularAdapter(BaseModelAdapter):
    """Adapter for Scikit-Learn tree models (Random Forest)."""

    def load(self) -> None:
        if not self.metadata.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.metadata.model_path}")
        self.model = joblib.load(self.metadata.model_path)

    def predict_scores(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        if self.model is None:
            self.load()
        # predict_proba returns [P(BENIGN), P(ATTACK)]
        probs = self.model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1].astype(np.float64)
        return probs.flatten().astype(np.float64)


class XGBoostTabularAdapter(BaseModelAdapter):
    """Adapter for XGBoost gradient boosted tree models."""

    def load(self) -> None:
        if not self.metadata.model_path.exists():
            raise FileNotFoundError(f"XGBoost model file not found at: {self.metadata.model_path}")
        self.model = joblib.load(self.metadata.model_path)

    def predict_scores(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        if self.model is None:
            self.load()
        probs = self.model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1].astype(np.float64)
        return probs.flatten().astype(np.float64)


# -----------------------------------------------------------------------------
# Multi-Model Inference Service
# -----------------------------------------------------------------------------

class MultiModelInferenceService:
    """
    Central service managing model lifecycle, predictions, telemetry,
    and multi-model comparison across all 5 models.
    """

    def __init__(self):
        self._adapters: Dict[str, BaseModelAdapter] = {}

    def get_adapter(self, model_id: str) -> BaseModelAdapter:
        """Retrieves or creates a cached model adapter."""
        if model_id in self._adapters:
            return self._adapters[model_id]

        meta = get_model_metadata(model_id)
        if meta.framework.startswith("TensorFlow") or "Keras" in meta.framework:
            adapter = KerasSequenceAdapter(meta)
        elif meta.framework == "Scikit-Learn":
            adapter = SklearnTabularAdapter(meta)
        elif meta.framework == "XGBoost":
            adapter = XGBoostTabularAdapter(meta)
        else:
            raise ValueError(f"Unsupported framework '{meta.framework}' for model '{model_id}'")

        adapter.load()
        self._adapters[model_id] = adapter
        return adapter

    def predict(
        self,
        model_id: str,
        input_data: np.ndarray,
        row_ranges: Optional[List[Tuple[int, int]]] = None,
        custom_threshold: Optional[float] = None,
        batch_size: int = 1024,
        ground_truth: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, InferenceSummary]:
        """
        Executes inference using the specified model.

        Parameters:
        - model_id: 'cnn_lstm', 'cnn_only', 'lstm_only', 'random_forest', 'xgboost'
        - input_data: 3D sequence array (N, 100, 78) OR 2D tabular array (N, 78)
        - row_ranges: List of (start_row, end_row) index tuples for sequence models
        - custom_threshold: Optional decision boundary override
        - batch_size: Mini-batch size
        - ground_truth: Optional binary ground truth array (0: BENIGN, 1: ATTACK) aligned with input_data
        """
        meta = get_model_metadata(model_id)
        adapter = self.get_adapter(model_id)

        n_units = len(input_data)
        threshold = custom_threshold if custom_threshold is not None else meta.default_threshold
        unit_name = "sequences" if meta.model_type == "sequence" else "records"

        if n_units == 0:
            empty_df = pd.DataFrame(columns=[
                "sequence_id", "row_range", "prediction_score", "predicted_label", "detection_status"
            ])
            empty_summary = InferenceSummary(
                model_name=meta.display_name,
                total_sequences=0,
                benign_count=0,
                attack_count=0,
                benign_percentage=0.0,
                attack_percentage=0.0,
                avg_prediction_score=0.0,
                min_prediction_score=0.0,
                max_prediction_score=0.0,
                threshold_applied=threshold,
                inference_time_seconds=0.0,
                throughput_seq_per_sec=0.0,
                latency_ms_per_seq=0.0,
                batch_size_used=batch_size,
                timestamp_utc=datetime.utcnow().isoformat() + "Z",
                model_id=model_id,
                model_type=meta.model_type,
                unit_name=unit_name
            )
            return empty_df, empty_summary

        # Measure high-precision inference duration
        t_start = time.perf_counter()
        scores = adapter.predict_scores(input_data, batch_size=batch_size)
        t_end = time.perf_counter()

        inference_time = max(t_end - t_start, 1e-6)
        throughput = n_units / inference_time
        latency_ms = (inference_time / n_units) * 1000.0

        is_attack = scores >= threshold
        labels = np.where(is_attack, "ATTACK", "BENIGN")
        statuses = np.where(is_attack, "CẢNH BÁO: Phát hiện xâm nhập", "BÌNH THƯỜNG: Lưu lượng an toàn")

        # Format observation labels (Sequence ID vs Row Index)
        if meta.model_type == "sequence":
            unit_ids = np.arange(n_units)
            if row_ranges and len(row_ranges) == n_units:
                row_range_strs = [f"Rows {r[0]}–{r[1]}" for r in row_ranges]
            else:
                row_range_strs = ["N/A"] * n_units
            col_id_name = "sequence_id"
        else:
            unit_ids = np.arange(n_units)
            row_range_strs = [f"Row {i}" for i in range(n_units)]
            col_id_name = "record_id"

        results_df = pd.DataFrame({
            col_id_name: unit_ids,
            "row_range": row_range_strs,
            "prediction_score": np.round(scores, 6),
            "predicted_label": labels,
            "detection_status": statuses
        })
        # Standardize for backward compatibility
        if col_id_name != "sequence_id":
            results_df["sequence_id"] = results_df[col_id_name]

        attack_count = int(np.sum(is_attack))
        benign_count = n_units - attack_count
        attack_pct = round((attack_count / n_units) * 100.0, 2)
        benign_pct = round((benign_count / n_units) * 100.0, 2)

        # Calculate optional ground truth evaluation metrics
        eval_metrics = None
        if ground_truth is not None and len(ground_truth) == n_units:
            pred_binary = is_attack.astype(int)
            gt_binary = np.asarray(ground_truth, dtype=int)
            acc = float(accuracy_score(gt_binary, pred_binary))
            prec = float(precision_score(gt_binary, pred_binary, zero_division=0))
            rec = float(recall_score(gt_binary, pred_binary, zero_division=0))
            f1 = float(f1_score(gt_binary, pred_binary, zero_division=0))
            cm = confusion_matrix(gt_binary, pred_binary, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            eval_metrics = EvaluationMetrics(
                accuracy=round(acc, 4),
                precision=round(prec, 4),
                recall=round(rec, 4),
                f1_score=round(f1, 4),
                true_positives=int(tp),
                true_negatives=int(tn),
                false_positives=int(fp),
                false_negatives=int(fn)
            )

        summary = InferenceSummary(
            model_name=meta.display_name,
            total_sequences=n_units,
            benign_count=benign_count,
            attack_count=attack_count,
            benign_percentage=benign_pct,
            attack_percentage=attack_pct,
            avg_prediction_score=float(np.round(np.mean(scores), 6)),
            min_prediction_score=float(np.round(np.min(scores), 6)),
            max_prediction_score=float(np.round(np.max(scores), 6)),
            threshold_applied=float(np.round(threshold, 4)),
            inference_time_seconds=float(np.round(inference_time, 4)),
            throughput_seq_per_sec=float(np.round(throughput, 2)),
            latency_ms_per_seq=float(np.round(latency_ms, 4)),
            batch_size_used=batch_size,
            timestamp_utc=datetime.utcnow().isoformat() + "Z",
            model_id=model_id,
            model_type=meta.model_type,
            unit_name=unit_name,
            evaluation=eval_metrics
        )

        return results_df, summary


# -----------------------------------------------------------------------------
# Backward-Compatible NIDSInferenceEngine (Preserves 100% of existing API)
# -----------------------------------------------------------------------------

class NIDSInferenceEngine:
    """
    Backward-compatible wrapper preserving the existing CNN-LSTM interface.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/cnn_lstm/best_cnn_lstm_checkpoint.keras",
        config_path: Union[str, Path] = "models/cnn_lstm/model_config.json"
    ):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.threshold: float = 0.46
        self.target_mapping: Dict[str, int] = {"BENIGN": 0, "ATTACK": 1}
        self.time_steps: int = 100
        self.n_features: int = 78
        self._service = MultiModelInferenceService()
        self._load_config()
        # Ensure adapter is preloaded
        self._adapter = self._service.get_adapter("cnn_lstm")
        self.model = self._adapter.model

    def _load_config(self) -> None:
        p = self.config_path
        if not p.exists():
            alt = Path("models/model_config.json")
            if alt.exists():
                p = alt
            else:
                p = Path("models/cnn_lstm/model_config.json")
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            self.threshold = float(self.config.get("best_threshold", 0.46))
            self.target_mapping = self.config.get("target_mapping", {"BENIGN": 0, "ATTACK": 1})
            self.time_steps = int(self.config.get("time_steps", 100))
            self.n_features = int(self.config.get("n_features", 78))

    def predict(
        self,
        sequences: np.ndarray,
        row_ranges: Optional[List[Tuple[int, int]]] = None,
        custom_threshold: Optional[float] = None,
        batch_size: int = 1024
    ) -> Tuple[pd.DataFrame, InferenceSummary]:
        return self._service.predict(
            model_id="cnn_lstm",
            input_data=sequences,
            row_ranges=row_ranges,
            custom_threshold=custom_threshold,
            batch_size=batch_size
        )


_cached_service: Optional[MultiModelInferenceService] = None
_cached_legacy_engine: Optional[NIDSInferenceEngine] = None


def get_multi_model_service() -> MultiModelInferenceService:
    """Singleton getter for the multi-model inference service."""
    global _cached_service
    if _cached_service is None:
        _cached_service = MultiModelInferenceService()
    return _cached_service


def get_inference_engine(
    model_path: Union[str, Path] = "models/cnn_lstm/best_cnn_lstm_checkpoint.keras",
    config_path: Union[str, Path] = "models/cnn_lstm/model_config.json"
) -> NIDSInferenceEngine:
    """Singleton getter preserving legacy CNN-LSTM inference engine."""
    global _cached_legacy_engine
    if _cached_legacy_engine is None:
        _cached_legacy_engine = NIDSInferenceEngine(model_path=model_path, config_path=config_path)
    return _cached_legacy_engine
