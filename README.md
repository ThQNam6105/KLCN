# Hệ Thống Phát Hiện Xâm Nhập Mạng Đa Mô Hình (NIDS)

Nền tảng trình diễn và đối sánh dành cho khóa luận tốt nghiệp cử nhân ngành An toàn Thông tin:
**"Nghiên cứu ứng dụng mô hình CNN-LSTM trong phát hiện xâm nhập mạng dựa trên dữ liệu Network Flow"** (Mã đề tài: **ATTT-KLCN54**).

---

## 1. Tổng Quan Đề Tài

Ứng dụng cho phép kiểm thử, đánh giá thời gian thực và đối sánh hiệu năng của **5 mô hình học máy và học sâu** trên tập dữ liệu Network Flow **CIC-IDS2017**:

1. **CNN-LSTM (Mô hình đề xuất):** Kiến trúc lai kết hợp trích xuất đặc trưng không gian 1D-CNN và quan hệ thời gian tuần tự LSTM trên các chuỗi gồm 100 luồng mạng liên tiếp.
2. **CNN (Ablation Baseline):** Kiến trúc gồm 2 lớp tích chập 1D-CNN kết hợp Batch Normalization và Dense, loại bỏ thành phần tuần tự.
3. **LSTM (Ablation Baseline):** Kiến trúc gồm 2 lớp LSTM xếp chồng (128 và 64 units) kết hợp Dense, loại bỏ tích chập lọc đặc trưng.
4. **Random Forest (Baseline dạng bảng):** Mô hình tập hợp 300 cây quyết định phân loại trực tiếp trên từng bản ghi luồng mạng độc lập với trọng số cân bằng lớp.
5. **XGBoost (Baseline dạng bảng):** Mô hình tăng cường độ dốc Extreme Gradient Boosting phân loại trực tiếp trên từng bản ghi luồng mạng độc lập.

Các tính năng chính của hệ thống:
- **Phân tích đơn mô hình:** Thực thi suy luận với đầy đủ thông tin xác suất, nhãn dự đoán, ngưỡng quyết định và xuất kết quả chi tiết.
- **So sánh đa mô hình:** Đối sánh trực tiếp cả 5 mô hình trên cùng một tập dữ liệu CSV với bảng so sánh, biểu đồ tỷ lệ tấn công và ma trận đồng thuận.
- **Đánh giá nhãn thực tế:** Tính toán trực tiếp các độ đo học máy có giám sát (Accuracy, Precision, Recall, F1-Score, Ma trận nhầm lẫn Confusion Matrix) khi tệp dữ liệu có cột nhãn.
- **Tổng hợp Ablation Study:** Trình bày đầy đủ các thông số kiến trúc, thời gian huấn luyện và kết quả thực nghiệm của nghiên cứu thành phần.

---

## 2. Cấu Trúc Thư Mục Dự Án

```text
DEMO/
├── app.py                          # Giao diện web Streamlit đa mô hình (Việt hóa hoàn toàn có dấu)
├── inference.py                    # Module suy luận mô đun hóa và các adapter
├── model_registry.py               # Danh mục trung tâm quản lý 5 mô hình
├── preprocessing.py                # Pipeline tiền xử lý dữ liệu (Chuỗi và Dạng bảng)
├── run_demo.bat                    # Tệp khởi động nhanh 1-click trên Windows
├── requirements.txt                # Các thư viện phụ thuộc
├── README.md                       # Tài liệu hướng dẫn hệ thống bằng tiếng Việt
│
├── sample_data.csv                 # Tập dữ liệu mẫu 300 luồng
├── test_benign.csv                 # 200 luồng an toàn trích xuất từ Monday-WorkingHours
├── test_ddos.csv                   # 200 luồng tấn công DDoS từ Friday-DDos
├── test_portscan.csv               # 200 luồng tấn công PortScan từ Friday-PortScan
├── test_mixed.csv                  # 300 luồng hỗn hợp có nhãn thực tế
│
├── models/
│   ├── cnn_lstm/                   # Artifacts của mô hình đề xuất CNN-LSTM
│   │   ├── best_cnn_lstm_checkpoint.keras
│   │   ├── feature_columns.json
│   │   ├── minmax_scaler.joblib
│   │   └── model_config.json
│   │
│   ├── cnn_only/                   # Artifacts của mô hình đối chứng CNN
│   │   ├── best_cnn_only_checkpoint.keras
│   │   ├── ablation_minmax_scaler.joblib
│   │   └── cnn_only_metrics.json
│   │
│   ├── lstm_only/                  # Artifacts của mô hình đối chứng LSTM
│   │   ├── best_lstm_only_checkpoint.keras
│   │   ├── ablation_minmax_scaler.joblib
│   │   └── lstm_only_metrics.json
│   │
│   ├── random_forest/              # Artifacts của mô hình đối chứng Random Forest
│   │   ├── random_forest_model.joblib
│   │   ├── feature_columns.json
│   │   ├── label_mapping.json
│   │   └── model_config.json
│   │
│   └── xgboost/                    # Artifacts của mô hình đối chứng XGBoost
│       ├── xgboost_model.joblib
│       ├── feature_columns.json
│       ├── label_mapping.json
│       └── model_config.json
│
├── results/                        # Thư mục xuất báo cáo dự đoán và thống kê
│
└── tests/
    ├── test_preprocessing.py       # Kiểm thử tiền xử lý dữ liệu
    ├── test_inference.py           # Kiểm thử suy luận mô hình CNN-LSTM
    └── test_multi_model.py         # Kiểm thử suy luận đa mô hình và đánh giá
```

---

## 3. Đặc Tả Tiền Xử Lý Và Tính Tương Thích Của 5 Mô Hình

| Mô hình | Kiểu cấu trúc | Nền tảng | Kích thước đầu vào | Quy trình tiền xử lý | Ngưỡng quyết định | Định dạng đầu ra |
|---|---|---|---|---|---|---|
| **CNN-LSTM** | Chuỗi thời gian | Keras 3 / TF 2.21 | `(N, 100, 78)` | Xử lý NaN/Inf -> `MinMaxScaler` -> Cửa sổ trượt ($T=100$) | **0.4600** | Sigmoid $P(\text{ATTACK}) \in [0, 1]$ |
| **CNN** | Chuỗi thời gian | Keras 3 / TF 2.21 | `(N, 100, 78)` | Xử lý NaN/Inf -> `MinMaxScaler` -> Cửa sổ trượt ($T=100$) | **0.4750** | Sigmoid $P(\text{ATTACK}) \in [0, 1]$ |
| **LSTM** | Chuỗi thời gian | Keras 3 / TF 2.21 | `(N, 100, 78)` | Xử lý NaN/Inf -> `MinMaxScaler` -> Cửa sổ trượt ($T=100$) | **0.5300** | Sigmoid $P(\text{ATTACK}) \in [0, 1]$ |
| **Random Forest** | Dạng bảng | Scikit-Learn | `(N, 78)` | Xử lý NaN/Inf -> **Mảng 2D gốc (Không chuẩn hóa)** | **0.3000** | Xác suất cây $P(\text{ATTACK}) \in [0, 1]$ |
| **XGBoost** | Dạng bảng | XGBoost | `(N, 78)` | Xử lý NaN/Inf -> **Mảng 2D gốc (Không chuẩn hóa)** | **0.8450** | Xác suất cây $P(\text{ATTACK}) \in [0, 1]$ |

> **Lưu ý khoa học quan trọng:** Các mô hình cây (Random Forest, XGBoost) được huấn luyện trực tiếp trên đặc trưng số học gốc. Việc áp dụng MinMaxScaler làm sai lệch các điểm rẽ nhánh của cây quyết định. Ngược lại, các mô hình học sâu (CNN-LSTM, CNN, LSTM) bắt buộc phải sử dụng MinMaxScaler về đoạn $[0, 1]$. Hệ thống đã tự động điều phối pipeline phù hợp cho từng loại mô hình.

---

## 4. Kết Quả Kiểm Định Thực Nghiệm Trên 5 Mô Hình

| Tập kiểm thử | Đặc điểm dữ liệu | CNN-LSTM | CNN | LSTM | Random Forest | XGBoost |
|---|---|---|---|---|---|---|
| `test_benign.csv` | 200 luồng an toàn | **0% Tấn công** (0/2 chuỗi) | **0% Tấn công** (0/2 chuỗi) | **0% Tấn công** (0/2 chuỗi) | **0% Tấn công** (0/200 dòng) | **0% Tấn công** (0/200 dòng) |
| `test_ddos.csv` | 200 luồng DDoS | **100% Tấn công** (2/2 chuỗi) | **100% Tấn công** (2/2 chuỗi) | **100% Tấn công** (2/2 chuỗi) | **100% Tấn công** (200/200 dòng) | **100% Tấn công** (200/200 dòng) |
| `test_portscan.csv` | 200 luồng PortScan | **100% Tấn công** (2/2 chuỗi) | **100% Tấn công** (2/2 chuỗi) | **50% Tấn công** (1/2 chuỗi) | **100% Tấn công** (200/200 dòng) | **100% Tấn công** (200/200 dòng) |
| `test_mixed.csv` | 150 an toàn + 150 tấn công | **66.7% Tấn công** (2/3 chuỗi) | **66.7% Tấn công** (2/3 chuỗi) | **33.3% Tấn công** (1/3 chuỗi) | **50.0% Tấn công** (150/300 dòng) | **50.0% Tấn công** (150/300 dòng) |

---

## 5. Hướng Dẫn Khởi Động Ứng Dụng

### Cách 1: Chạy nhanh bằng 1-click trên Windows
Nhấn đúp chuột vào tệp [`run_demo.bat`](file:///d:/TÔI%20YÊU%20HUIT/KHÓA%20LUẬN%20CỬ%20NHÂN/DEMO/run_demo.bat) trong thư mục `DEMO`.

### Cách 2: Chạy từ dòng lệnh CMD hoặc PowerShell
```bash
cd "D:\TÔI YÊU HUIT\KHÓA LUẬN CỬ NHÂN\DEMO"
.\.venv\Scripts\streamlit run app.py
```

Trình duyệt web sẽ tự động mở ứng dụng tại địa chỉ: `http://localhost:8501`.

---

## 6. Bộ Kiểm Thử Tự Động

Chạy toàn bộ 26 bài kiểm thử tự động bằng pytest:

```bash
.\.venv\Scripts\python -m pytest -v tests/
```

Phạm vi kiểm thử bao gồm:
- Kiểm tra toàn vẹn tất cả các checkpoint và metadata của 5 mô hình.
- Kiểm thử lược đồ CSV, xử lý giá trị NaN/Inf và phân đoạn chuỗi thời gian.
- Kiểm thử adapter cho Keras, Scikit-Learn và XGBoost.
- Kiểm thử tính toán độ đo có giám sát (Accuracy, Precision, Recall, F1, Confusion Matrix).
- Kiểm thử tích hợp đầu-cuối trên các tập dữ liệu benchmark.
- Tính nhất quán và khả năng tái lặp xác định trên các lượt chạy lặp lại.
