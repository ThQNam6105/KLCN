"""
CNN-LSTM Network Intrusion Detection System - Preprocessing Module
Preserves exact training feature ordering, MinMaxScaler scaling, sequence construction,
and tabular feature extraction for all 5 evaluation models.
"""

from dataclasses import dataclass, field
import io
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class ValidationResult:
    """Encapsulates the outcome of dataset validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    total_rows: int = 0
    total_columns: int = 0
    matched_features: int = 0
    missing_features: List[str] = field(default_factory=list)
    unexpected_columns: List[str] = field(default_factory=list)
    nan_count: int = 0
    inf_count: int = 0
    num_sequences: int = 0
    incomplete_rows: int = 0
    has_ground_truth: bool = False
    ground_truth_counts: dict = field(default_factory=dict)


def normalize_column_name(col: str) -> str:
    """
    Normalizes a column name by stripping BOM and whitespace,
    mirroring the data cleaning convention of the CIC-IDS2017 pipeline.
    """
    return str(col).replace("\ufeff", "").strip()


def resolve_existing_path(*candidate_paths: Union[str, Path]) -> Path:
    """Returns the first existing Path from the candidates."""
    for p in candidate_paths:
        path_obj = Path(p)
        if path_obj.exists():
            return path_obj
    return Path(candidate_paths[0])


def load_feature_columns(filepath: Union[str, Path] = "models/feature_columns.json") -> List[str]:
    """Loads the ordered list of 78 feature columns required by the model."""
    path = resolve_existing_path(
        filepath,
        "models/cnn_lstm/feature_columns.json",
        "models/feature_columns.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"Feature columns file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        columns = json.load(f)
    if not isinstance(columns, list) or len(columns) == 0:
        raise ValueError(f"Invalid feature columns structure in {path}")
    return columns


def load_scaler(filepath: Union[str, Path] = "models/minmax_scaler.joblib") -> MinMaxScaler:
    """Loads the pre-fitted MinMaxScaler artifact."""
    path = resolve_existing_path(
        filepath,
        "models/cnn_lstm/minmax_scaler.joblib",
        "models/minmax_scaler.joblib"
    )
    if not path.exists():
        raise FileNotFoundError(f"Scaler artifact not found at: {path}")
    scaler = joblib.load(path)
    if not isinstance(scaler, MinMaxScaler):
        raise TypeError(f"Loaded scaler is not a MinMaxScaler, got {type(scaler)}")
    return scaler


def extract_ground_truth_labels(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Extracts binary ground-truth labels if a 'Label' or 'label' column is present.
    Maps 'BENIGN' (case-insensitive) to 0, and any attack label to 1.
    """
    label_col = next((c for c in df.columns if normalize_column_name(c).lower() == "label"), None)
    if label_col is None:
        return None

    raw_labels = df[label_col].astype(str).str.strip()
    is_attack = (raw_labels.str.upper() != "BENIGN").to_numpy(dtype=int)
    return is_attack


def aggregate_sequence_ground_truth(
    row_labels: np.ndarray,
    row_ranges: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Aggregates flow-level ground-truth labels to sequence-level ground truth.
    A sequence window is labeled ATTACK (1) if ANY flow within the window is an attack.
    """
    seq_labels = np.zeros(len(row_ranges), dtype=int)
    for i, (start_idx, end_idx) in enumerate(row_ranges):
        # Window inclusive [start_idx, end_idx]
        window_labels = row_labels[start_idx:end_idx + 1]
        seq_labels[i] = 1 if np.any(window_labels == 1) else 0
    return seq_labels


def validate_csv_data(
    data: Union[str, Path, pd.DataFrame, io.BytesIO, io.StringIO],
    feature_columns: List[str],
    time_steps: int = 100,
    stride: int = 100,
    is_sequence: bool = True
) -> Tuple[Optional[pd.DataFrame], ValidationResult]:
    """
    Performs comprehensive schema and data validation on an uploaded Network Flow CSV.

    Validates:
    1. Empty file/data.
    2. CSV formatting / parsability.
    3. Missing required features (fails validation if any are missing).
    4. Unexpected / extra columns (flags as warning, safely ignored).
    5. Duplicate column names.
    6. Non-numeric / NaN / Inf values (flags count, will be safely cleaned).
    7. Row count sufficiency relative to `time_steps`.
    8. Presence of ground-truth Label column.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Load Data
    df: Optional[pd.DataFrame] = None
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        try:
            if isinstance(data, (str, Path)):
                path = Path(data)
                if not path.exists():
                    return None, ValidationResult(is_valid=False, errors=[f"File not found: {path}"])
                if path.stat().st_size == 0:
                    return None, ValidationResult(is_valid=False, errors=["Uploaded file is empty (0 bytes)."])
                # Try common encodings
                for enc in ["utf-8", "utf-8-sig", "latin1"]:
                    try:
                        df = pd.read_csv(path, encoding=enc, low_memory=False)
                        break
                    except Exception:
                        continue
                if df is None:
                    df = pd.read_csv(path, low_memory=False)
            else:
                # BytesIO or StringIO
                content = data.getvalue() if hasattr(data, "getvalue") else data.read()
                if not content or len(content) == 0:
                    return None, ValidationResult(is_valid=False, errors=["Uploaded file is empty."])
                if isinstance(data, io.BytesIO):
                    data.seek(0)
                df = pd.read_csv(data, low_memory=False)
        except Exception as exc:
            return None, ValidationResult(is_valid=False, errors=[f"Malformed CSV file or parsing error: {str(exc)}"])

    if df is None or df.empty:
        return None, ValidationResult(is_valid=False, errors=["CSV file contains no data rows."])

    total_rows, total_cols = df.shape

    # 2. Duplicate column names check
    normalized_cols = [normalize_column_name(c) for c in df.columns]
    seen_cols = set()
    duplicate_cols = []
    for c in normalized_cols:
        if c in seen_cols:
            duplicate_cols.append(c)
        seen_cols.add(c)
    if duplicate_cols:
        warnings.append(f"Detected duplicate column names (normalized): {list(set(duplicate_cols))}. The first occurrence will be retained.")
        df.columns = normalized_cols
        df = df.loc[:, ~df.columns.duplicated()]
    else:
        df.columns = normalized_cols

    # 3. Required feature presence
    curr_cols = set(df.columns)
    missing = [col for col in feature_columns if col not in curr_cols]
    unexpected = [col for col in df.columns if col not in set(feature_columns)]

    if missing:
        errors.append(
            f"Missing {len(missing)} required model feature(s): {missing[:5]}"
            + (f" and {len(missing) - 5} more." if len(missing) > 5 else ".")
        )

    if unexpected:
        warnings.append(
            f"Found {len(unexpected)} non-model column(s) (e.g. {unexpected[:3]}). "
            "These will be safely excluded from model inference."
        )

    # 4. Check for ground truth
    has_ground_truth = "Label" in df.columns
    gt_counts = {}
    if has_ground_truth:
        counts = df["Label"].astype(str).str.strip().value_counts().to_dict()
        gt_counts = {str(k): int(v) for k, v in counts.items()}

    # 5. Row count sufficiency
    effective_min_rows = time_steps if is_sequence else 1
    if total_rows < effective_min_rows:
        errors.append(
            f"Insufficient rows: CSV has {total_rows} row(s), but the model requires at least {effective_min_rows} "
            f"flow record(s) for inference."
        )

    # Calculate sequence counts and tail rows
    if is_sequence:
        if total_rows >= time_steps:
            num_sequences = (total_rows - time_steps) // stride + 1
            last_covered_idx = (num_sequences - 1) * stride + time_steps
            incomplete_rows = total_rows - last_covered_idx
        else:
            num_sequences = 0
            incomplete_rows = total_rows
    else:
        num_sequences = total_rows
        incomplete_rows = 0

    if is_sequence and incomplete_rows > 0 and total_rows >= time_steps:
        warnings.append(
            f"{incomplete_rows} trailing row(s) cannot form a complete {time_steps}-step sequence "
            f"with stride={stride} and will be excluded from sequence inference."
        )

    # 6. Check numerical / NaN / Inf presence across required features
    nan_count = 0
    inf_count = 0
    available_model_cols = [c for c in feature_columns if c in df.columns]
    if available_model_cols:
        subset = df[available_model_cols]
        num_subset = subset.apply(pd.to_numeric, errors="coerce")
        nan_count = int(num_subset.isna().sum().sum())
        num_values = num_subset.to_numpy()
        inf_count = int(np.isinf(num_values).sum())

        if nan_count > 0:
            warnings.append(f"Detected {nan_count} missing (NaN) or non-numeric value(s) in required features. These will be imputed with 0.0.")
        if inf_count > 0:
            warnings.append(f"Detected {inf_count} infinite value(s) (e.g. Flow Bytes/s). These will be safely replaced with 0.0.")

    is_valid = len(errors) == 0

    result = ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        total_rows=total_rows,
        total_columns=total_cols,
        matched_features=len(feature_columns) - len(missing),
        missing_features=missing,
        unexpected_columns=unexpected,
        nan_count=nan_count,
        inf_count=inf_count,
        num_sequences=num_sequences,
        incomplete_rows=incomplete_rows,
        has_ground_truth=has_ground_truth,
        ground_truth_counts=gt_counts
    )

    return df if is_valid else None, result


def preprocess_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    scaler: MinMaxScaler
) -> np.ndarray:
    """
    Aligns features to the exact training column order, replaces NaN/Inf,
    and applies the pre-fitted MinMaxScaler for Deep Learning models.

    Never calls fit() or fit_transform().
    """
    X = df[feature_columns].copy()
    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    raw_array = X.to_numpy(dtype=np.float32)
    scaled_array = scaler.transform(raw_array)
    return scaled_array.astype(np.float32)


def preprocess_tabular_features(
    df: pd.DataFrame,
    feature_columns: List[str]
) -> np.ndarray:
    """
    Aligns features to the exact training column order, replaces NaN/Inf with 0.0,
    and returns unscaled raw 2D numerical array (N_rows, 78) for Tree-based models (RF, XGBoost).
    """
    X = df[feature_columns].copy()
    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    return X.to_numpy(dtype=np.float32)


def create_sequences(
    scaled_features: np.ndarray,
    time_steps: int = 100,
    stride: int = 100
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Constructs 3D sequence tensors (num_sequences, time_steps, n_features)
    using sliding or non-overlapping windowing over consecutive flow records.
    """
    n_rows, n_features = scaled_features.shape

    if n_rows < time_steps:
        return np.empty((0, time_steps, n_features), dtype=np.float32), []

    num_sequences = (n_rows - time_steps) // stride + 1
    sequences = np.zeros((num_sequences, time_steps, n_features), dtype=np.float32)
    row_ranges: List[Tuple[int, int]] = []

    for i in range(num_sequences):
        start_idx = i * stride
        end_idx = start_idx + time_steps
        sequences[i] = scaled_features[start_idx:end_idx]
        row_ranges.append((start_idx, end_idx - 1))

    return sequences, row_ranges


def full_preprocessing_pipeline(
    data: Union[str, Path, pd.DataFrame, io.BytesIO, io.StringIO],
    feature_columns: List[str],
    scaler: MinMaxScaler,
    time_steps: int = 100,
    stride: int = 100
) -> Tuple[np.ndarray, List[Tuple[int, int]], ValidationResult, Optional[pd.DataFrame]]:
    """
    Executes the complete sequence preprocessing pipeline from raw CSV input to CNN-LSTM tensors.
    """
    df, validation_result = validate_csv_data(
        data=data,
        feature_columns=feature_columns,
        time_steps=time_steps,
        stride=stride,
        is_sequence=True
    )

    if not validation_result.is_valid or df is None:
        return np.empty((0, time_steps, len(feature_columns)), dtype=np.float32), [], validation_result, None

    scaled_data = preprocess_features(df, feature_columns, scaler)
    sequences, row_ranges = create_sequences(scaled_data, time_steps=time_steps, stride=stride)

    return sequences, row_ranges, validation_result, df
