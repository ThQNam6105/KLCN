"""
Đăng ký mô hình phát hiện xâm nhập mạng (NIDS).
Định nghĩa metadata, kiến trúc, ngưỡng phân loại và đường dẫn artifact
cho tất cả 5 mô hình: CNN-LSTM, CNN, LSTM, Random Forest và XGBoost.
Toàn bộ nội dung sử dụng Tiếng Việt có dấu chuẩn xác và không sử dụng emoji.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelMetadata:
    """Siêu dữ liệu và cấu hình thực thi cho từng mô hình NIDS."""
    model_id: str
    display_name: str
    model_type: str                  # 'sequence' (chuỗi) hoặc 'tabular' (dạng bảng)
    framework: str                   # 'Keras 3 / TensorFlow', 'Scikit-Learn', 'XGBoost'
    model_path: Path
    config_path: Path
    scaler_path: Optional[Path]      # None nếu là mô hình cây không dùng scaler
    feature_columns_path: Path
    default_threshold: float
    time_steps: int                  # 100 cho mô hình chuỗi, 1 cho mô hình bảng
    n_features: int = 78
    description: str = ""
    short_name: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    historical_metrics: Dict[str, float] = field(default_factory=dict)
    training_time_seconds: Optional[float] = None


def resolve_existing_path(*candidate_paths: str) -> Path:
    """Trả về đường dẫn hợp lệ đầu tiên tồn tại trên ổ đĩa."""
    for p in candidate_paths:
        path_obj = Path(p)
        if path_obj.exists():
            return path_obj
    return Path(candidate_paths[0])


def get_registered_models() -> Dict[str, ModelMetadata]:
    """Khởi tạo và trả về danh mục 5 mô hình đã huấn luyện."""
    return {
        "cnn_lstm": ModelMetadata(
            model_id="cnn_lstm",
            display_name="Mô hình đề xuất CNN-LSTM",
            short_name="CNN-LSTM",
            model_type="sequence",
            framework="TensorFlow / Keras 3",
            model_path=resolve_existing_path(
                "models/cnn_lstm/best_cnn_lstm_checkpoint.keras",
                "models/best_cnn_lstm_checkpoint.keras"
            ),
            config_path=resolve_existing_path(
                "models/cnn_lstm/model_config.json",
                "models/model_config.json"
            ),
            scaler_path=resolve_existing_path(
                "models/cnn_lstm/minmax_scaler.joblib",
                "models/minmax_scaler.joblib"
            ),
            feature_columns_path=resolve_existing_path(
                "models/cnn_lstm/feature_columns.json",
                "models/feature_columns.json"
            ),
            default_threshold=0.46,
            time_steps=100,
            n_features=78,
            description="Mô hình học sâu kết hợp trích xuất đặc trưng không gian 1D-CNN và quan hệ thời gian tuần tự LSTM trên các chuỗi gồm 100 luồng mạng liên tiếp.",
            hyperparameters={
                "Tối ưu hóa (Optimizer)": "Nadam(lr=0.0005, weight_decay=1e-4)",
                "Hàm mất mát (Loss)": "BinaryFocalCrossentropy(alpha=0.25, gamma=2.0)",
                "Kích thước batch": 1024,
                "Hạt giống ngẫu nhiên (Seed)": 42
            },
            historical_metrics={
                "accuracy": 0.9972,
                "precision": 0.9972,
                "recall": 0.9972,
                "f1_score": 0.9972,
                "macro_f1": 0.7675,
                "roc_auc": 0.9961,
                "fpr": 0.0005
            },
            training_time_seconds=107.07
        ),
        "cnn_only": ModelMetadata(
            model_id="cnn_only",
            display_name="Mô hình đối chứng CNN (Ablation)",
            short_name="CNN",
            model_type="sequence",
            framework="TensorFlow / Keras 3",
            model_path=resolve_existing_path("models/cnn_only/best_cnn_only_checkpoint.keras"),
            config_path=resolve_existing_path("models/cnn_only/cnn_only_metrics.json"),
            scaler_path=resolve_existing_path(
                "models/cnn_only/ablation_minmax_scaler.joblib",
                "models/cnn_lstm/minmax_scaler.joblib"
            ),
            feature_columns_path=resolve_existing_path(
                "models/cnn_lstm/feature_columns.json",
                "models/feature_columns.json"
            ),
            default_threshold=0.475,
            time_steps=100,
            n_features=78,
            description="Mô hình đối chứng thực nghiệm gồm 2 lớp tích chập 1D-CNN, Batch Normalization, Flatten và Dense, loại bỏ thành phần tuần tự LSTM.",
            hyperparameters={
                "Số bộ lọc (Filters)": [128, 128],
                "Kích thước kernel": 3,
                "Các lớp Dense": [64, 1],
                "Tỉ lệ Dropout": 0.2
            },
            historical_metrics={
                "accuracy": 0.9800,
                "precision": 0.9423,
                "recall": 0.9408,
                "f1_score": 0.9415,
                "macro_f1": 0.9647,
                "roc_auc": 0.9873,
                "pr_auc": 0.9757,
                "fpr": 0.0119
            },
            training_time_seconds=75.43
        ),
        "lstm_only": ModelMetadata(
            model_id="lstm_only",
            display_name="Mô hình đối chứng LSTM (Ablation)",
            short_name="LSTM",
            model_type="sequence",
            framework="TensorFlow / Keras 3",
            model_path=resolve_existing_path("models/lstm_only/best_lstm_only_checkpoint.keras"),
            config_path=resolve_existing_path("models/lstm_only/lstm_only_metrics.json"),
            scaler_path=resolve_existing_path(
                "models/lstm_only/ablation_minmax_scaler.joblib",
                "models/cnn_lstm/minmax_scaler.joblib"
            ),
            feature_columns_path=resolve_existing_path(
                "models/cnn_lstm/feature_columns.json",
                "models/feature_columns.json"
            ),
            default_threshold=0.53,
            time_steps=100,
            n_features=78,
            description="Mô hình đối chứng thực nghiệm gồm các lớp LSTM xếp chồng (128 và 64 units) kết hợp Dense, không sử dụng tích chập trích xuất đặc trưng.",
            hyperparameters={
                "Các lớp LSTM (Units)": [128, 64],
                "Các lớp Dense": [64, 1],
                "Tỉ lệ Dropout": 0.2
            },
            historical_metrics={
                "accuracy": 0.9797,
                "precision": 0.9464,
                "recall": 0.9346,
                "f1_score": 0.9404,
                "macro_f1": 0.9641,
                "roc_auc": 0.9844,
                "pr_auc": 0.9656,
                "fpr": 0.0110
            },
            training_time_seconds=68.85
        ),
        "random_forest": ModelMetadata(
            model_id="random_forest",
            display_name="Mô hình đối chứng Random Forest (Dạng bảng)",
            short_name="Random Forest",
            model_type="tabular",
            framework="Scikit-Learn",
            model_path=resolve_existing_path("models/random_forest/random_forest_model.joblib"),
            config_path=resolve_existing_path("models/random_forest/model_config.json"),
            scaler_path=None,
            feature_columns_path=resolve_existing_path(
                "models/random_forest/feature_columns.json",
                "models/cnn_lstm/feature_columns.json"
            ),
            default_threshold=0.30,
            time_steps=1,
            n_features=78,
            description="Mô hình học máy truyền thống gồm 300 cây quyết định, phân loại trực tiếp từng bản ghi luồng mạng độc lập với trọng số cân bằng lớp.",
            hyperparameters={
                "Số lượng cây (n_estimators)": 300,
                "Trọng số lớp (class_weight)": "balanced",
                "Hạt giống ngẫu nhiên (Seed)": 42
            },
            historical_metrics={
                "accuracy": 0.9926,
                "precision": 0.9925,
                "recall": 0.9926,
                "f1_score": 0.9924,
                "macro_f1": 0.7167,
                "roc_auc": 0.9976,
                "fpr": 0.0015
            },
            training_time_seconds=792.11
        ),
        "xgboost": ModelMetadata(
            model_id="xgboost",
            display_name="Mô hình đối chứng XGBoost (Dạng bảng)",
            short_name="XGBoost",
            model_type="tabular",
            framework="XGBoost",
            model_path=resolve_existing_path("models/xgboost/xgboost_model.joblib"),
            config_path=resolve_existing_path("models/xgboost/model_config.json"),
            scaler_path=None,
            feature_columns_path=resolve_existing_path(
                "models/xgboost/feature_columns.json",
                "models/cnn_lstm/feature_columns.json"
            ),
            default_threshold=0.845,
            time_steps=1,
            n_features=78,
            description="Mô hình tăng cường độ dốc Extreme Gradient Boosting, phân loại trực tiếp từng bản ghi luồng mạng độc lập sử dụng cây theo biểu đồ tần số.",
            hyperparameters={
                "Số lượng cây (n_estimators)": 500,
                "Độ sâu tối đa (max_depth)": 8,
                "Tốc độ học (learning_rate)": 0.08,
                "Tỉ lệ lấy mẫu (subsample)": 0.8,
                "Tỉ lệ đặc trưng (colsample_bytree)": 0.8,
                "Phương pháp cây (tree_method)": "hist",
                "Trọng số lớp dương (scale_pos_weight)": 4.8676,
                "Hạt giống ngẫu nhiên (Seed)": 42
            },
            historical_metrics={
                "accuracy": 0.9926,
                "precision": 0.9925,
                "recall": 0.9926,
                "f1_score": 0.9924,
                "macro_f1": 0.7167,
                "roc_auc": 0.9976,
                "fpr": 0.0015
            },
            training_time_seconds=141.27
        )
    }


MODEL_REGISTRY: Dict[str, ModelMetadata] = get_registered_models()


def get_model_metadata(model_id: str) -> ModelMetadata:
    """Lấy thông tin metadata của một mô hình theo ID."""
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"Không tìm thấy mô hình có ID '{model_id}'. Các mô hình khả dụng: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_id]
