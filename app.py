"""
Hệ Thống Phát Hiện Xâm Nhập Mạng Đa Mô Hình (NIDS) - Ứng Dụng Web
Hệ thống minh họa cho khóa luận tốt nghiệp cử nhân:
"Nghiên cứu ứng dụng mô hình CNN-LSTM trong phát hiện xâm nhập mạng dựa trên dữ liệu Network Flow"
Mã đề tài: ATTT-KLCN54
Hỗ trợ 5 mô hình: CNN-LSTM, Pure CNN, Pure LSTM, Random Forest và XGBoost.
Toàn bộ giao diện sử dụng Tiếng Việt có dấu chuẩn xác và tuyệt đối không sử dụng emoji.
"""

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model_registry import MODEL_REGISTRY, ModelMetadata, get_model_metadata
from preprocessing import (
    load_feature_columns,
    load_scaler,
    validate_csv_data,
    preprocess_features,
    preprocess_tabular_features,
    create_sequences,
    extract_ground_truth_labels,
    aggregate_sequence_ground_truth,
    ValidationResult
)
from inference import (
    get_multi_model_service,
    MultiModelInferenceService,
    InferenceSummary,
    EvaluationMetrics
)

# -----------------------------------------------------------------------------
# Cấu hình trang Streamlit và giao diện tối (Cyber Dark)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NIDS | Hệ Thống Phát Hiện Xâm Nhập Mạng",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyber Dark CSS Styling với Panel bên trái nền trắng chữ xanh
st.markdown("""
<style>
    /* Nền tổng thể và kiểu chữ phần nội dung chính */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Thanh tiêu đề trên cùng */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* ----------------------------------------------------------------- */
    /* XÓA NÚT THU GỌN PANEL (SIDEBAR TOGGLE/COLLAPSE BUTTON)           */
    /* ----------------------------------------------------------------- */
    [data-testid="stSidebarCollapseButton"],
    button[data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarHeader"],
    section[data-testid="stSidebar"] button[kind="headerNoPadding"],
    section[data-testid="stSidebar"] button[data-testid="baseButton-headerNoPadding"],
    header[data-testid="stHeader"] button[data-testid="baseButton-headerNoPadding"],
    header[data-testid="stHeader"] [data-testid="stSidebarCollapseButton"],
    header[data-testid="stHeader"] [data-testid="collapsedControl"],
    header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"],
    button[aria-label="Close sidebar"],
    button[aria-label="Open sidebar"],
    button[title="Close sidebar"],
    button[title="Open sidebar"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* ----------------------------------------------------------------- */
    /* PANEL BÊN TRÁI (SIDEBAR) - THIẾT KẾ TỐI GIẢN & CHUYÊN NGHIỆP      */
    /* ----------------------------------------------------------------- */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
        color: #172B4D !important;
    }

    /* Đảm bảo toàn bộ chữ trong sidebar mặc định là màu tối #172B4D, không bị dính màu trắng */
    section[data-testid="stSidebar"] * {
        color: #172B4D;
    }

    @media (min-width: 768px) {
        section[data-testid="stSidebar"] {
            width: 260px !important;
            min-width: 260px !important;
            max-width: 260px !important;
        }
    }

    section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
        padding: 20px 16px 24px 16px !important;
        background-color: #FFFFFF !important;
    }

    /* Khối Logo & Thương hiệu */
    section[data-testid="stSidebar"] div[data-testid="stImage"] {
        margin: 0 0 10px 0 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stImage"] img {
        max-width: 155px !important;
        width: 155px !important;
        height: auto !important;
        object-fit: contain !important;
        display: block !important;
    }

    .sidebar-brand-text {
        margin-top: 2px;
        margin-bottom: 2px;
    }
    .sidebar-brand-title {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #0056A6 !important;
        letter-spacing: -0.01em !important;
        line-height: 1.25 !important;
        margin: 0 !important;
    }
    .sidebar-brand-subtitle {
        font-size: 0.74rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin-top: 2px !important;
        line-height: 1.2 !important;
    }

    /* Tiêu đề phân mục trong sidebar */
    .sidebar-section-title {
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        color: #07539F !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        margin-top: 2px !important;
        margin-bottom: 8px !important;
        user-select: none !important;
    }

    /* Đường kẻ phân cách tinh tế */
    .sidebar-divider {
        height: 1px !important;
        background-color: #E2E8F0 !important;
        margin: 16px 0 !important;
        border: none !important;
    }

    /* ------------------------------------------------------------- */
    /* MENU ĐIỀU HƯỚNG TỐI GIẢN (COMPACT VERTICAL MENU)               */
    /* ------------------------------------------------------------- */
    /* Ẩn hoàn toàn nhãn mặc định của stRadio (tránh hiện chữ ĐIỀU HƯỚNG thành nút menu) */
    section[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] [data-testid="stRadio"] > label,
    section[data-testid="stSidebar"] [data-testid="stRadio"] > div:first-child:not([role="radiogroup"]):not([class*="e1dm8mpf0"]) {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
        pointer-events: none !important;
    }

    /* Ẩn triệt để nút tròn radio (radio circle) trên tất cả các phiên bản Streamlit */
    section[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"],
    section[data-testid="stSidebar"] [data-testid="stRadio"] div[data-testid="stRadioButton"],
    section[data-testid="stSidebar"] [data-testid="stRadioOption"] input[type="radio"],
    section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child,
    section[data-testid="stSidebar"] [class*="e1dm8mpf4"],
    section[data-testid="stSidebar"] [class*="e1dm8mpf5"],
    section[data-testid="stSidebar"] [data-testid="stRadio"] svg,
    section[data-testid="stSidebar"] [data-testid="stRadioOption"] svg {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        visibility: hidden !important;
        opacity: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Khung chứa danh sách điều hướng */
    section[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioGroup"],
    section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"],
    section[data-testid="stSidebar"] [class*="e1dm8mpf0"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 4px !important;
        width: 100% !important;
    }

    /* Từng dòng mục điều hướng */
    section[data-testid="stSidebar"] [data-testid="stRadioOption"],
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] label,
    section[data-testid="stSidebar"] div[role="radiogroup"] label,
    section[data-testid="stSidebar"] [class*="e1dm8mpf2"],
    section[data-testid="stSidebar"] .react-aria-RadioField,
    section[data-testid="stSidebar"] .react-aria-Radio {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        height: 40px !important;
        min-height: 40px !important;
        padding: 0 12px !important;
        border-radius: 8px !important;
        margin: 2px 0 !important;
        cursor: pointer !important;
        background-color: transparent !important;
        border-left: 3px solid transparent !important;
        transition: all 0.15s ease-in-out !important;
        text-decoration: none !important;
        box-sizing: border-box !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div,
    section[data-testid="stSidebar"] [data-testid="stRadioOption"] [class*="e1dm8mpf3"],
    section[data-testid="stSidebar"] [class*="e1dm8mpf2"] > div,
    section[data-testid="stSidebar"] [class*="e1dm8mpf3"],
    section[data-testid="stSidebar"] [data-testid="stRadioOption"] div[data-testid="stMarkdownContainer"] {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        gap: 0 !important;
    }

    /* MÀU CHỮ: Màu tối đậm #172B4D rõ ràng, sắc nét trên nền trắng */
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] p,
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] span,
    section[data-testid="stSidebar"] div[role="radiogroup"] p,
    section[data-testid="stSidebar"] div[role="radiogroup"] span,
    section[data-testid="stSidebar"] [class*="e1dm8mpf2"] p,
    section[data-testid="stSidebar"] [class*="e1dm8mpf2"] span,
    section[data-testid="stSidebar"] [data-testid="stRadioOption"] p,
    section[data-testid="stSidebar"] [data-testid="stRadioOption"] span,
    section[data-testid="stSidebar"] .react-aria-RadioField p,
    section[data-testid="stSidebar"] .react-aria-RadioField span,
    section[data-testid="stSidebar"] .react-aria-Radio p,
    section[data-testid="stSidebar"] .react-aria-Radio span {
        font-size: 0.90rem !important;
        font-weight: 500 !important;
        color: #172B4D !important;
        line-height: 1.2 !important;
        text-decoration: none !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 1 !important;
    }

    /* Biểu tượng outline đơn sắc (Monochrome outline icons) */
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]::before,
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] label::before,
    section[data-testid="stSidebar"] div[role="radiogroup"] label::before,
    section[data-testid="stSidebar"] [class*="e1dm8mpf2"]::before,
    section[data-testid="stSidebar"] .react-aria-RadioField::before,
    section[data-testid="stSidebar"] .react-aria-Radio::before {
        content: "";
        display: inline-block;
        width: 17px;
        height: 17px;
        min-width: 17px;
        margin-right: 10px;
        background-color: #64748B;
        -webkit-mask-size: contain;
        mask-size: contain;
        -webkit-mask-repeat: no-repeat;
        mask-repeat: no-repeat;
        -webkit-mask-position: center;
        mask-position: center;
        transition: background-color 0.15s ease-in-out;
    }

    /* 1. Bảng điều khiển: layout-dashboard */
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(1) [data-testid="stRadioOption"]::before,
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > label:nth-child(1)::before,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(1)::before,
    section[data-testid="stSidebar"] [class*="e1dm8mpf0"] > [class*="e1dm8mpf2"]:nth-of-type(1)::before {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect width='7' height='9' x='3' y='3' rx='1'/%3E%3Crect width='7' height='5' x='14' y='3' rx='1'/%3E%3Crect width='7' height='9' x='14' y='12' rx='1'/%3E%3Crect width='7' height='5' x='3' y='16' rx='1'/%3E%3C/svg%3E");
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect width='7' height='9' x='3' y='3' rx='1'/%3E%3Crect width='7' height='5' x='14' y='3' rx='1'/%3E%3Crect width='7' height='9' x='14' y='12' rx='1'/%3E%3Crect width='7' height='5' x='3' y='16' rx='1'/%3E%3C/svg%3E");
    }

    /* 2. Phân tích luồng mạng: activity */
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(2) [data-testid="stRadioOption"]::before,
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > label:nth-child(2)::before,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(2)::before,
    section[data-testid="stSidebar"] [class*="e1dm8mpf0"] > [class*="e1dm8mpf2"]:nth-of-type(2)::before {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M22 12h-4l-3 9L9 3l-3 9H2'/%3E%3C/svg%3E");
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M22 12h-4l-3 9L9 3l-3 9H2'/%3E%3C/svg%3E");
    }

    /* 3. So sánh đa mô hình: chart-no-axes-combined */
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(3) [data-testid="stRadioOption"]::before,
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > label:nth-child(3)::before,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(3)::before,
    section[data-testid="stSidebar"] [class*="e1dm8mpf0"] > [class*="e1dm8mpf2"]:nth-of-type(3)::before {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 20v-6'/%3E%3Cpath d='M6 20V10'/%3E%3Cpath d='M18 20V4'/%3E%3C/svg%3E");
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 20v-6'/%3E%3Cpath d='M6 20V10'/%3E%3Cpath d='M18 20V4'/%3E%3C/svg%3E");
    }

    /* 4. Đánh giá nhãn thực tế: shield-check */
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(4) [data-testid="stRadioOption"]::before,
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > label:nth-child(4)::before,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(4)::before,
    section[data-testid="stSidebar"] [class*="e1dm8mpf0"] > [class*="e1dm8mpf2"]:nth-of-type(4)::before {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z'/%3E%3Cpath d='m9 12 2 2 4-4'/%3E%3C/svg%3E");
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z'/%3E%3Cpath d='m9 12 2 2 4-4'/%3E%3C/svg%3E");
    }

    /* 5. Thông tin mô hình: brain-circuit / cpu */
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(5) [data-testid="stRadioOption"]::before,
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > label:nth-child(5)::before,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(5)::before,
    section[data-testid="stSidebar"] [class*="e1dm8mpf0"] > [class*="e1dm8mpf2"]:nth-of-type(5)::before {
        -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect width='16' height='16' x='4' y='4' rx='2'/%3E%3Crect width='6' height='6' x='9' y='9' rx='1'/%3E%3Cpath d='M15 2v2'/%3E%3Cpath d='M15 20v2'/%3E%3Cpath d='M2 15h2'/%3E%3Cpath d='M2 9h2'/%3E%3Cpath d='M20 15h2'/%3E%3Cpath d='M20 9h2'/%3E%3Cpath d='M9 2v2'/%3E%3Cpath d='M9 20v2'/%3E%3C/svg%3E");
        mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect width='16' height='16' x='4' y='4' rx='2'/%3E%3Crect width='6' height='6' x='9' y='9' rx='1'/%3E%3Cpath d='M15 2v2'/%3E%3Cpath d='M15 20v2'/%3E%3Cpath d='M2 15h2'/%3E%3Cpath d='M2 9h2'/%3E%3Cpath d='M20 15h2'/%3E%3Cpath d='M20 9h2'/%3E%3Cpath d='M9 2v2'/%3E%3Cpath d='M9 20v2'/%3E%3C/svg%3E");
    }

    /* Trạng thái di chuột (Hover) */
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover,
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] label:hover,
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover,
    section[data-testid="stSidebar"] [class*="e1dm8mpf2"]:hover,
    section[data-testid="stSidebar"] .react-aria-RadioField:hover,
    section[data-testid="stSidebar"] .react-aria-Radio:hover {
        background-color: #F1F5F9 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover p,
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] label:hover p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover p,
    section[data-testid="stSidebar"] [class*="e1dm8mpf2"]:hover p,
    section[data-testid="stSidebar"] .react-aria-RadioField:hover p,
    section[data-testid="stSidebar"] .react-aria-Radio:hover p {
        color: #172B4D !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover::before,
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] label:hover::before,
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover::before,
    section[data-testid="stSidebar"] [class*="e1dm8mpf2"]:hover::before,
    section[data-testid="stSidebar"] .react-aria-RadioField:hover::before,
    section[data-testid="stSidebar"] .react-aria-Radio:hover::before {
        background-color: #172B4D !important;
    }

    /* Trạng thái kích hoạt (Active Navigation) */
    section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"],
    section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected],
    section[data-testid="stSidebar"] [class*="e1dm8mpf2"][data-selected="true"],
    section[data-testid="stSidebar"] [class*="e1dm8mpf2"][data-selected],
    section[data-testid="stSidebar"] .react-aria-RadioField[data-selected="true"],
    section[data-testid="stSidebar"] .react-aria-RadioField[data-selected],
    section[data-testid="stSidebar"] .react-aria-Radio[data-selected="true"],
    section[data-testid="stSidebar"] .react-aria-Radio[data-selected],
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked),
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] label:has(input:checked),
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked),
    section[data-testid="stSidebar"] [class*="e1dm8mpf2"]:has(input:checked) {
        background-color: #EAF3FF !important;
        border-left: 3px solid #0056A6 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p,
    section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] span,
    section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected] p,
    section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected] span,
    section[data-testid="stSidebar"] [class*="e1dm8mpf2"][data-selected="true"] p,
    section[data-testid="stSidebar"] [class*="e1dm8mpf2"][data-selected="true"] span,
    section[data-testid="stSidebar"] [class*="e1dm8mpf2"][data-selected] p,
    section[data-testid="stSidebar"] [class*="e1dm8mpf2"][data-selected] span,
    section[data-testid="stSidebar"] .react-aria-RadioField[data-selected="true"] p,
    section[data-testid="stSidebar"] .react-aria-RadioField[data-selected] p,
    section[data-testid="stSidebar"] .react-aria-Radio[data-selected="true"] p,
    section[data-testid="stSidebar"] .react-aria-Radio[data-selected] p,
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) p,
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] label:has(input:checked) p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p,
    section[data-testid="stSidebar"] [class*="e1dm8mpf2"]:has(input:checked) p {
        color: #0056A6 !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"]::before,
    section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected]::before,
    section[data-testid="stSidebar"] [class*="e1dm8mpf2"][data-selected="true"]::before,
    section[data-testid="stSidebar"] [class*="e1dm8mpf2"][data-selected]::before,
    section[data-testid="stSidebar"] .react-aria-RadioField[data-selected="true"]::before,
    section[data-testid="stSidebar"] .react-aria-RadioField[data-selected]::before,
    section[data-testid="stSidebar"] .react-aria-Radio[data-selected="true"]::before,
    section[data-testid="stSidebar"] .react-aria-Radio[data-selected]::before,
    section[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked)::before,
    section[data-testid="stSidebar"] [data-testid="stRadioGroup"] label:has(input:checked)::before,
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked)::before,
    section[data-testid="stSidebar"] [class*="e1dm8mpf2"]:has(input:checked)::before {
        background-color: #0056A6 !important;
    }

    /* ------------------------------------------------------------- */
    /* HỘP CHỌN MÔ HÌNH PHÂN TÍCH (COMPACT MODEL SELECTOR)           */
    /* ------------------------------------------------------------- */
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }

    /* Ẩn widget label mặc định của stSelectbox nếu có */
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > label {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Khung hộp chọn: nền trắng sáng, viền #CBD5E1 rõ ràng, cao 40px */
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div,
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [class*="e1fp86qc0"],
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        height: 40px !important;
        min-height: 40px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
        transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div:hover,
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [class*="e1fp86qc0"]:hover,
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
        border-color: #07539F !important;
        box-shadow: 0 0 0 1px rgba(7, 83, 159, 0.2) !important;
    }

    /* MÀU CHỮ TRONG MÔ HÌNH PHÂN TÍCH: Xanh tối RGB(7, 83, 159) / #07539F đậm nét, tương phản hoàn hảo */
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] input,
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [class*="e1fp86qc1"],
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [class*="e1fp86qc3"],
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] p,
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] span,
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div {
        color: #07539F !important;
        -webkit-text-fill-color: #07539F !important;
        font-size: 0.90rem !important;
        font-weight: 600 !important;
    }

    /* Biểu tượng mũi tên dropdown */
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] svg,
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [class*="e1fp86qc2"] svg {
        fill: #07539F !important;
        color: #07539F !important;
        stroke: #07539F !important;
    }

    /* Danh sách tùy chọn khi mở menu dropdown */
    [class*="e1fp86qc4"],
    [class*="e1fp86qc5"],
    ul[role="listbox"],
    div[data-baseweb="popover"],
    div[data-baseweb="menu"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.15) !important;
    }

    [class*="e1fp86qc7"],
    [class*="e1fp86qc8"],
    li[role="option"],
    div[role="option"] {
        color: #172B4D !important;
        background-color: #FFFFFF !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
        border-radius: 4px !important;
    }

    [class*="e1fp86qc7"]:hover,
    [class*="e1fp86qc7"][data-hovered],
    [class*="e1fp86qc7"][aria-selected="true"],
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"],
    div[role="option"]:hover,
    div[role="option"][aria-selected="true"] {
        background-color: #EAF3FF !important;
        color: #07539F !important;
        font-weight: 600 !important;
    }

    /* Đèn báo trạng thái mô hình */
    .sidebar-status-container {
        display: flex;
        align-items: center;
        margin-top: 8px;
        padding-left: 2px;
    }
    .status-dot {
        width: 8px !important;
        height: 8px !important;
        border-radius: 50% !important;
        background-color: #16A34A !important;
        display: inline-block !important;
        box-shadow: 0 0 0 2px rgba(22, 163, 74, 0.2) !important;
    }
    .status-text {
        font-size: 0.80rem !important;
        font-weight: 500 !important;
        color: #16A34A !important;
        margin-left: 7px !important;
    }

    /* ------------------------------------------------------------- */
    /* CHÂN TRANG SIDEBAR (COMPACT ANCHORED FOOTER)                  */
    /* ------------------------------------------------------------- */
    .sidebar-footer {
        margin-top: 8px;
        padding-bottom: 4px;
    }
    .sidebar-footer-brand {
        font-size: 0.74rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        letter-spacing: 0.03em !important;
    }
    .sidebar-footer-copy {
        font-size: 0.70rem !important;
        color: #94A3B8 !important;
        margin-top: 2px !important;
    }

    /* ----------------------------------------------------------------- */
    /* CÁC THÀNH PHẦN NỘI DUNG CHÍNH (CYBER DARK)                        */
    /* ----------------------------------------------------------------- */
    /* Thẻ chỉ số Metric */
    div[data-testid="stMetric"] {
        background-color: #151d30;
        border: 1px solid #1f2d48;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #00d2ff;
        transition: border-color 0.2s ease-in-out;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 700;
    }

    /* Khung chứa thông tin */
    .dashboard-card {
        background-color: #131a2c;
        border: 1px solid #202b42;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }

    /* Bảng dữ liệu */
    div[data-testid="stDataFrame"] {
        border: 1px solid #1e293b;
        border-radius: 6px;
        background-color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Tải và lưu trữ tài nguyên mô hình
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_service_and_base_artifacts() -> Tuple[MultiModelInferenceService, List[str], object]:
    """Khởi tạo dịch vụ suy luận đa mô hình và nạp tập đặc trưng gốc."""
    svc = get_multi_model_service()
    feat_cols = load_feature_columns("models/cnn_lstm/feature_columns.json")
    scaler = load_scaler("models/cnn_lstm/minmax_scaler.joblib")
    return svc, feat_cols, scaler


service, FEATURE_COLUMNS, BASE_SCALER = get_service_and_base_artifacts()


# -----------------------------------------------------------------------------
# Khởi tạo trạng thái phiên làm việc (Session State)
# -----------------------------------------------------------------------------
if "active_model_id" not in st.session_state:
    st.session_state.active_model_id = "cnn_lstm"
if "analysis_results_df" not in st.session_state:
    st.session_state.analysis_results_df = None
if "analysis_summary" not in st.session_state:
    st.session_state.analysis_summary = None
if "multi_comparison_results" not in st.session_state:
    st.session_state.multi_comparison_results = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
if "uploaded_file_size_kb" not in st.session_state:
    st.session_state.uploaded_file_size_kb = 0.0
if "raw_preview_df" not in st.session_state:
    st.session_state.raw_preview_df = None
if "validation_result" not in st.session_state:
    st.session_state.validation_result = None
if "active_data_source" not in st.session_state:
    st.session_state.active_data_source = None


# -----------------------------------------------------------------------------
# Thanh điều hướng bên trái (Sidebar)
# -----------------------------------------------------------------------------
with st.sidebar:
    # 1. Branding Header
    logo_path = Path("assets/logo.jpg")
    if logo_path.exists():
        st.image(str(logo_path), width=155)

    st.markdown("""
    <div class="sidebar-brand-text">
        <div class="sidebar-brand-title">CNN-LSTM NIDS</div>
        <div class="sidebar-brand-subtitle">Network Intrusion Detection</div>
    </div>
    <div class="sidebar-divider"></div>
    """, unsafe_allow_html=True)

    # 2. Navigation Menu (Trực tiếp dưới dải phân cách)
    nav_options = [
        "Bảng điều khiển",
        "Phân tích luồng mạng",
        "So sánh đa mô hình",
        "Đánh giá nhãn thực tế",
        "Thông tin mô hình và thực nghiệm"
    ]

    navigation = st.radio(
        "menu_chuc_nang",
        options=nav_options,
        format_func=lambda x: "Thông tin mô hình" if x == "Thông tin mô hình và thực nghiệm" else x,
        index=0 if st.session_state.analysis_results_df is not None else 1,
        label_visibility="collapsed"
    )

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    # 3. Model Selector
    st.markdown("<div class='sidebar-section-title'>MÔ HÌNH PHÂN TÍCH</div>", unsafe_allow_html=True)

    model_options = {
        "cnn_lstm": "CNN-LSTM",
        "cnn_only": "CNN (Ablation)",
        "lstm_only": "LSTM (Ablation)",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost"
    }
    selected_model_id = st.selectbox(
        "Mô hình phân tích",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=list(model_options.keys()).index(st.session_state.active_model_id),
        label_visibility="collapsed"
    )
    st.session_state.active_model_id = selected_model_id
    curr_meta = get_model_metadata(selected_model_id)

    st.markdown("""
    <div class="sidebar-status-container">
        <span class="status-dot"></span>
        <span class="status-text">Sẵn sàng</span>
    </div>
    <div class="sidebar-divider"></div>
    """, unsafe_allow_html=True)

    # 4. Footer
    st.markdown("""
    <div class="sidebar-footer">
        <div class="sidebar-footer-brand">HUIT · ATTT-KLCN54</div>
        <div class="sidebar-footer-copy">CNN-LSTM NIDS © 2026</div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Hàm phụ trợ: Tiền xử lý và suy luận cho 1 mô hình
# -----------------------------------------------------------------------------
def run_single_model_inference(
    model_id: str,
    df: pd.DataFrame,
    stride: int = 100,
    threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, InferenceSummary]:
    meta = get_model_metadata(model_id)
    gt_row_labels = extract_ground_truth_labels(df)

    if meta.model_type == "sequence":
        scaled = preprocess_features(df, FEATURE_COLUMNS, BASE_SCALER)
        seqs, ranges = create_sequences(scaled, time_steps=meta.time_steps, stride=stride)
        gt_seq_labels = None
        if gt_row_labels is not None and len(ranges) > 0:
            gt_seq_labels = aggregate_sequence_ground_truth(gt_row_labels, ranges)
        return service.predict(
            model_id=model_id,
            input_data=seqs,
            row_ranges=ranges,
            custom_threshold=threshold,
            ground_truth=gt_seq_labels
        )
    else:
        tabular_X = preprocess_tabular_features(df, FEATURE_COLUMNS)
        return service.predict(
            model_id=model_id,
            input_data=tabular_X,
            custom_threshold=threshold,
            ground_truth=gt_row_labels
        )


# -----------------------------------------------------------------------------
# TRANG 1: BẢNG ĐIỀU KHIỂN
# -----------------------------------------------------------------------------
if navigation == "Bảng điều khiển":
    st.title("Bảng Điều Khiển Phát Hiện Xâm Nhập Mạng")
    st.markdown(f"Mô hình giám sát hiện tại: **{curr_meta.display_name}** (`{curr_meta.framework}`)")

    if st.session_state.analysis_summary is None or st.session_state.analysis_results_df is None:
        st.info(
            "Chưa có dữ liệu phân tích nào.\n\n"
            "Vui lòng chuyển sang mục 'Phân tích luồng mạng' trên thanh menu bên trái để tải lên tệp CSV "
            "hoặc chọn một trong các tập dữ liệu kiểm thử để bắt đầu phân tích."
        )
    else:
        summary: InferenceSummary = st.session_state.analysis_summary
        df_results: pd.DataFrame = st.session_state.analysis_results_df

        # Hàng 1: Các thẻ chỉ số KPI
        don_vi_str = "Chuỗi" if summary.model_type == "sequence" else "Bản ghi"
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.metric(f"Tổng số {don_vi_str}", f"{summary.total_sequences:,}")
        with m2:
            st.metric("Bình thường (BENIGN)", f"{summary.benign_count:,}", f"{summary.benign_percentage}%")
        with m3:
            st.metric("Tấn công (ATTACK)", f"{summary.attack_count:,}", f"{summary.attack_percentage}%", delta_color="inverse")
        with m4:
            st.metric("Tỷ lệ tấn công", f"{summary.attack_percentage}%")
        with m5:
            st.metric("Thời gian suy luận", f"{summary.inference_time_seconds:.4f}s")
        with m6:
            st.metric("Tốc độ xử lý", f"{summary.throughput_seq_per_sec:,.0f} {summary.unit_name}/s")

        st.markdown("<br>", unsafe_allow_html=True)

        # Hàng 2: Biểu đồ trực quan
        col_chart1, col_chart2 = st.columns([1, 1])

        with col_chart1:
            st.subheader("Phân bố kết quả phân loại")
            donut_data = pd.DataFrame({
                "Nhãn": ["BENIGN (Bình thường)", "ATTACK (Tấn công)"],
                "Số lượng": [summary.benign_count, summary.attack_count]
            })
            fig_donut = px.pie(
                donut_data,
                names="Nhãn",
                values="Số lượng",
                hole=0.6,
                color="Nhãn",
                color_discrete_map={"BENIGN (Bình thường)": "#10b981", "ATTACK (Tấn công)": "#ef4444"}
            )
            fig_donut.update_layout(
                template="plotly_dark",
                paper_bgcolor="#131a2c",
                plot_bgcolor="#131a2c",
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_chart2:
            st.subheader("Phân bố điểm xác suất tấn công")
            fig_hist = px.histogram(
                df_results,
                x="prediction_score",
                nbins=25,
                color="predicted_label",
                color_discrete_map={"BENIGN": "#10b981", "ATTACK": "#ef4444"},
                labels={"prediction_score": "Điểm xác suất tấn công", "count": "Số lượng"}
            )
            fig_hist.add_vline(
                x=summary.threshold_applied,
                line_dash="dash",
                line_color="#00d2ff",
                annotation_text=f"Ngưỡng ({summary.threshold_applied:.2f})",
                annotation_position="top left"
            )
            fig_hist.update_layout(
                template="plotly_dark",
                paper_bgcolor="#131a2c",
                plot_bgcolor="#131a2c",
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Hàng 3: Biểu đồ tiến trình chuỗi/bản ghi
        st.subheader("Diễn biến xác suất tấn công theo trình tự")
        fig_timeline = go.Figure()

        benign_mask = df_results["predicted_label"] == "BENIGN"
        attack_mask = df_results["predicted_label"] == "ATTACK"
        x_axis_col = "sequence_id" if "sequence_id" in df_results.columns else df_results.index

        fig_timeline.add_trace(go.Scatter(
            x=df_results.loc[benign_mask, x_axis_col],
            y=df_results.loc[benign_mask, "prediction_score"],
            mode="markers",
            name="BENIGN (Bình thường)",
            marker=dict(color="#10b981", size=6, symbol="circle"),
            text=df_results.loc[benign_mask, "row_range"],
            hovertemplate="Chỉ số: %{x}<br>%{text}<br>Xác suất: %{y:.4f}<extra></extra>"
        ))
        fig_timeline.add_trace(go.Scatter(
            x=df_results.loc[attack_mask, x_axis_col],
            y=df_results.loc[attack_mask, "prediction_score"],
            mode="markers",
            name="ATTACK (Tấn công)",
            marker=dict(color="#ef4444", size=7, symbol="diamond"),
            text=df_results.loc[attack_mask, "row_range"],
            hovertemplate="Chỉ số: %{x}<br>%{text}<br>Xác suất: %{y:.4f}<extra></extra>"
        ))
        fig_timeline.add_hline(
            y=summary.threshold_applied,
            line_dash="dash",
            line_color="#00d2ff",
            annotation_text=f"Ngưỡng phân định ({summary.threshold_applied:.2f})"
        )
        fig_timeline.update_layout(
            template="plotly_dark",
            paper_bgcolor="#131a2c",
            plot_bgcolor="#131a2c",
            xaxis_title=f"Chỉ số {don_vi_str}",
            yaxis_title="Điểm xác suất tấn công",
            margin=dict(t=30, b=30, l=30, r=30),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_timeline, use_container_width=True)


# -----------------------------------------------------------------------------
# TRANG 2: PHÂN TÍCH LUỒNG MẠNG
# -----------------------------------------------------------------------------
elif navigation == "Phân tích luồng mạng":
    st.title("Tải Lên Luồng Mạng Và Phân Tích Xâm Nhập")
    st.markdown(f"Mô hình đang sử dụng: **{curr_meta.display_name}** ({'Cấp độ chuỗi' if curr_meta.model_type == 'sequence' else 'Cấp độ bản ghi'})")

    col_input, col_params = st.columns([2, 1])

    with col_input:
        st.subheader("1. Nguồn dữ liệu đầu vào")
        uploaded_file = st.file_uploader(
            "Tải lên tệp CSV Network Flow",
            type=["csv"],
            help="Chọn tệp CSV chứa đầy đủ 78 đặc trưng network flow tương thích tập dữ liệu CIC-IDS2017."
        )

        st.markdown("**Hoặc chọn tập dữ liệu thực nghiệm sẵn có:**")
        b1, b2, b3, b4, b5 = st.columns(5)
        with b1:
            load_sample = st.button("Mẫu chung (300)", use_container_width=True)
        with b2:
            load_benign = st.button("An toàn (200)", use_container_width=True)
        with b3:
            load_ddos = st.button("DDoS (200)", use_container_width=True)
        with b4:
            load_portscan = st.button("PortScan (200)", use_container_width=True)
        with b5:
            load_mixed = st.button("Hỗn hợp (300)", use_container_width=True)

    with col_params:
        st.subheader("2. Cấu hình chuỗi và ngưỡng phân loại")
        if curr_meta.model_type == "sequence":
            window_stride = st.selectbox(
                "Bước trượt cửa sổ (Stride)",
                options=[100, 50, 20, 10],
                index=0,
                format_func=lambda x: f"{x} luồng ({'Không chồng lấn' if x == 100 else 'Cửa sổ trượt'})"
            )
        else:
            window_stride = 100
            st.info("Mô hình dạng bảng: Phân loại độc lập trực tiếp trên từng bản ghi luồng mạng.")

        use_custom_thresh = st.toggle("Điều chỉnh ngưỡng quyết định", value=False)
        if use_custom_thresh:
            selected_threshold = st.slider(
                "Ngưỡng phân loại tấn công",
                min_value=0.05,
                max_value=0.95,
                value=float(curr_meta.default_threshold),
                step=0.01
            )
        else:
            selected_threshold = float(curr_meta.default_threshold)
            st.info(f"Ngưỡng tối ưu của mô hình: **{selected_threshold:.4f}**")

    # Xử lý chọn nguồn dữ liệu
    if load_sample:
        st.session_state.active_data_source = Path("sample_data.csv")
        st.session_state.uploaded_filename = "sample_data.csv (Mẫu CIC-IDS2017 gồm 300 luồng)"
    elif load_benign:
        st.session_state.active_data_source = Path("test_benign.csv")
        st.session_state.uploaded_filename = "test_benign.csv (200 luồng an toàn - Monday)"
    elif load_ddos:
        st.session_state.active_data_source = Path("test_ddos.csv")
        st.session_state.uploaded_filename = "test_ddos.csv (200 luồng tấn công DDoS - Friday)"
    elif load_portscan:
        st.session_state.active_data_source = Path("test_portscan.csv")
        st.session_state.uploaded_filename = "test_portscan.csv (200 luồng tấn công PortScan - Friday)"
    elif load_mixed:
        st.session_state.active_data_source = Path("test_mixed.csv")
        st.session_state.uploaded_filename = "test_mixed.csv (300 luồng hỗn hợp có nhãn)"
    elif uploaded_file is not None:
        st.session_state.active_data_source = uploaded_file
        st.session_state.uploaded_filename = uploaded_file.name

    # Kiểm tra tính hợp lệ và hiển thị xem trước
    if st.session_state.active_data_source is not None:
        source = st.session_state.active_data_source
        file_size_kb = round(source.stat().st_size / 1024, 2) if isinstance(source, Path) else round(source.size / 1024, 2)
        st.session_state.uploaded_file_size_kb = file_size_kb

        st.markdown("---")
        st.subheader("3. Kiểm tra tính hợp lệ và xem trước dữ liệu")

        clean_df, val_res = validate_csv_data(
            source,
            feature_columns=FEATURE_COLUMNS,
            time_steps=curr_meta.time_steps,
            stride=window_stride,
            is_sequence=(curr_meta.model_type == "sequence")
        )
        st.session_state.validation_result = val_res

        v1, v2, v3, v4 = st.columns(4)
        with v1:
            st.metric("Tệp nguồn", st.session_state.uploaded_filename)
        with v2:
            st.metric("Dung lượng tệp", f"{file_size_kb:,.1f} KB")
        with v3:
            st.metric("Số dòng / Số cột", f"{val_res.total_rows:,} / {val_res.total_columns}")
        with v4:
            unit_count_str = f"{val_res.num_sequences:,} Chuỗi" if curr_meta.model_type == "sequence" else f"{val_res.total_rows:,} Bản ghi"
            st.metric("Số đơn vị phân tích", unit_count_str)

        if val_res.is_valid:
            st.success(f"Kiểm tra hợp lệ thành công: Lược đồ CSV phù hợp với yêu cầu của mô hình {curr_meta.display_name} ({val_res.matched_features}/78 đặc trưng khớp chuẩn).")
            if val_res.has_ground_truth:
                st.info(f"Phát hiện nhãn thực tế (Label): Phân bố nhãn là: {val_res.ground_truth_counts}")
        else:
            st.error("Tệp CSV không hợp lệ theo yêu cầu của mô hình.")
            for err in val_res.errors:
                st.markdown(f"- {err}")

        if val_res.warnings:
            with st.expander("Cảnh báo chẩn đoán tiền xử lý", expanded=False):
                for warn in val_res.warnings:
                    st.markdown(f"- {warn}")

        if clean_df is not None:
            with st.expander("Xem trước dữ liệu (10 dòng đầu tiên)", expanded=False):
                st.dataframe(clean_df.head(10), use_container_width=True)

        # Nút kích hoạt suy luận
        if val_res.is_valid and clean_df is not None:
            st.markdown("---")
            run_btn = st.button(f"Bắt đầu phát hiện xâm nhập với {curr_meta.display_name}", type="primary", use_container_width=True)

            if run_btn:
                with st.spinner(f"Đang thực thi suy luận với mô hình {curr_meta.display_name}..."):
                    res_df, summary = run_single_model_inference(
                        model_id=st.session_state.active_model_id,
                        df=clean_df,
                        stride=window_stride,
                        threshold=selected_threshold
                    )
                    st.session_state.analysis_results_df = res_df
                    st.session_state.analysis_summary = summary
                st.success(f"Phân tích hoàn tất trong {summary.inference_time_seconds:.4f} giây (Tốc độ: {summary.throughput_seq_per_sec:,.0f} {summary.unit_name}/giây).")

    # Bảng kết quả phân tích
    if st.session_state.analysis_results_df is not None:
        st.markdown("---")
        st.subheader("4. Kết quả phát hiện chi tiết")
        res_df: pd.DataFrame = st.session_state.analysis_results_df
        summary: InferenceSummary = st.session_state.analysis_summary

        r1, r2, r3, r4, r5 = st.columns(5)
        with r1:
            st.metric("Tổng số phân tích", f"{summary.total_sequences:,} {summary.unit_name}")
        with r2:
            st.metric("Bình thường (BENIGN)", f"{summary.benign_count:,}", f"{summary.benign_percentage}%")
        with r3:
            st.metric("Tấn công (ATTACK)", f"{summary.attack_count:,}", f"{summary.attack_percentage}%", delta_color="inverse")
        with r4:
            st.metric("Điểm xác suất TB", f"{summary.avg_prediction_score:.4f}")
        with r5:
            st.metric("Thời gian suy luận", f"{summary.inference_time_seconds:.4f}s")

        # Bộ lọc
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            filter_choice = st.radio("Bộ lọc kết quả", options=["Tất cả", "Chỉ BENIGN", "Chỉ ATTACK"], horizontal=True)
        with f_col2:
            search_query = st.text_input("Tìm kiếm theo mã định danh hoặc dải luồng", "")

        filtered_df = res_df.copy()
        if filter_choice == "Chỉ BENIGN":
            filtered_df = filtered_df[filtered_df["predicted_label"] == "BENIGN"]
        elif filter_choice == "Chỉ ATTACK":
            filtered_df = filtered_df[filtered_df["predicted_label"] == "ATTACK"]

        if search_query.strip():
            filtered_df = filtered_df[
                filtered_df["row_range"].str.contains(search_query, case=False) |
                filtered_df["sequence_id"].astype(str).str.contains(search_query)
            ]

        st.dataframe(
            filtered_df,
            use_container_width=True,
            column_config={
                "sequence_id": st.column_config.NumberColumn("Mã định danh", format="%d"),
                "row_range": st.column_config.TextColumn("Dải luồng gốc"),
                "prediction_score": st.column_config.ProgressColumn(
                    "Xác suất tấn công",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f"
                ),
                "predicted_label": st.column_config.TextColumn("Nhãn dự đoán"),
                "detection_status": st.column_config.TextColumn("Trạng thái cảnh báo")
            },
            hide_index=True
        )

        st.markdown("---")
        st.subheader("5. Xuất kết quả phân tích")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.download_button(
                label="Tải xuống kết quả chi tiết (CSV)",
                data=res_df.to_csv(index=False).encode("utf-8"),
                file_name=f"nids_{st.session_state.active_model_id}_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with exp_col2:
            st.download_button(
                label="Tải xuống báo cáo tổng hợp (JSON)",
                data=json.dumps(summary.to_dict(), indent=2, ensure_ascii=False).encode("utf-8"),
                file_name=f"nids_summary_{st.session_state.active_model_id}_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )


# -----------------------------------------------------------------------------
# TRANG 3: SO SÁNH ĐA MÔ HÌNH
# -----------------------------------------------------------------------------
elif navigation == "So sánh đa mô hình":
    st.title("So Sánh Hiệu Năng Đa Mô Hình Phát Hiện Xâm Nhập")
    st.markdown("Thực thi và đối sánh cả 5 mô hình trên cùng một tập dữ liệu Network Flow.")

    st.subheader("1. Chọn các mô hình cần so sánh")
    all_model_ids = list(MODEL_REGISTRY.keys())
    selected_bench_models = st.multiselect(
        "Chọn danh sách mô hình đối sánh",
        options=all_model_ids,
        default=all_model_ids,
        format_func=lambda x: f"{MODEL_REGISTRY[x].display_name} [{MODEL_REGISTRY[x].framework}]"
    )

    if not selected_bench_models:
        st.warning("Vui lòng chọn ít nhất một mô hình để thực hiện so sánh.")
        st.stop()

    if st.session_state.active_data_source is None:
        st.info("Vui lòng nạp một tập dữ liệu ở mục 'Phân tích luồng mạng' hoặc chọn nhanh dưới đây:")
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("Nạp dữ liệu hỗn hợp (300 luồng)"):
                st.session_state.active_data_source = Path("test_mixed.csv")
                st.session_state.uploaded_filename = "test_mixed.csv (300 luồng hỗn hợp)"
                st.rerun()
        with b_col2:
            if st.button("Nạp dữ liệu DDoS (200 luồng)"):
                st.session_state.active_data_source = Path("test_ddos.csv")
                st.session_state.uploaded_filename = "test_ddos.csv (200 luồng DDoS)"
                st.rerun()
        st.stop()

    source = st.session_state.active_data_source
    st.markdown(f"**Tập dữ liệu mục tiêu:** `{st.session_state.uploaded_filename}`")

    if st.button("Bắt đầu chạy so sánh đa mô hình", type="primary", use_container_width=True):
        clean_df, val_res = validate_csv_data(
            source,
            feature_columns=FEATURE_COLUMNS,
            time_steps=100,
            stride=100,
            is_sequence=True
        )

        if not val_res.is_valid or clean_df is None:
            st.error("Tập dữ liệu không hợp lệ cho các mô hình chuỗi.")
            st.stop()

        summaries: List[InferenceSummary] = []
        model_dfs: Dict[str, pd.DataFrame] = {}

        progress_bar = st.progress(0, text="Đang thực hiện so sánh...")
        for idx, m_id in enumerate(selected_bench_models):
            progress_bar.progress((idx) / len(selected_bench_models), text=f"Đang đánh giá {MODEL_REGISTRY[m_id].display_name}...")
            m_df, m_sum = run_single_model_inference(m_id, clean_df, stride=100)
            summaries.append(m_sum)
            model_dfs[m_id] = m_df
        progress_bar.progress(1.0, text="Hoàn tất so sánh!")

        st.session_state.multi_comparison_results = {
            "summaries": summaries,
            "dfs": model_dfs
        }

    # Hiển thị kết quả so sánh
    if st.session_state.multi_comparison_results is not None:
        results = st.session_state.multi_comparison_results
        summaries: List[InferenceSummary] = results["summaries"]
        model_dfs: Dict[str, pd.DataFrame] = results["dfs"]

        st.markdown("---")
        st.subheader("2. Bảng tổng hợp so sánh hiệu năng")

        comp_records = []
        for s in summaries:
            kieu_mo_hinh_vn = "Chuỗi thời gian" if s.model_type == "sequence" else "Dạng bảng"
            comp_records.append({
                "Mô hình": s.model_name,
                "Kiểu cấu trúc": kieu_mo_hinh_vn,
                "Đơn vị phân tích": f"{s.total_sequences:,} {s.unit_name}",
                "Số lượng BENIGN": s.benign_count,
                "Số lượng ATTACK": s.attack_count,
                "Tỷ lệ phát hiện tấn công": f"{s.attack_percentage}%",
                "Ngưỡng áp dụng": s.threshold_applied,
                "Thời gian thực thi": f"{s.inference_time_seconds:.4f}s",
                "Tốc độ xử lý": f"{s.throughput_seq_per_sec:,.0f} {s.unit_name}/s",
                "Điểm TB": f"{s.avg_prediction_score:.4f}"
            })

        st.dataframe(pd.DataFrame(comp_records), use_container_width=True, hide_index=True)

        st.info("""
        Lưu ý khoa học về mức độ phân giải:
        - Các mô hình chuỗi (CNN-LSTM, CNN, LSTM): Đánh giá trên các chuỗi thời gian gồm 100 luồng mạng liên tiếp.
        - Các mô hình dạng bảng (Random Forest, XGBoost): Đánh giá trực tiếp từng bản ghi luồng mạng độc lập.
        - Không thể đồng nhất 1 chuỗi với 1 bản ghi đơn lẻ, vì hai phương pháp hoạt động ở hai cấp độ phân giải thời gian khác nhau.
        """)

        # Biểu đồ trực quan so sánh
        st.markdown("---")
        st.subheader("3. Biểu đồ phân tích so sánh")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Tỷ lệ phát hiện tấn công (%)")
            chart_df = pd.DataFrame([
                {"Mô hình": s.model_name, "Tỷ lệ tấn công (%)": s.attack_percentage, "Kiểu": "Chuỗi" if s.model_type == "sequence" else "Dạng bảng"}
                for s in summaries
            ])
            fig_bar = px.bar(
                chart_df,
                x="Mô hình",
                y="Tỷ lệ tấn công (%)",
                color="Kiểu",
                color_discrete_map={"Chuỗi": "#00d2ff", "Dạng bảng": "#10b981"},
                text_auto=".1f"
            )
            fig_bar.update_layout(template="plotly_dark", paper_bgcolor="#131a2c", plot_bgcolor="#131a2c")
            st.plotly_chart(fig_bar, use_container_width=True)

        with c2:
            st.markdown("#### Thời gian thực thi suy luận (giây)")
            time_df = pd.DataFrame([
                {"Mô hình": s.model_name, "Thời gian (giây)": s.inference_time_seconds, "Kiểu": "Chuỗi" if s.model_type == "sequence" else "Dạng bảng"}
                for s in summaries
            ])
            fig_time = px.bar(
                time_df,
                x="Mô hình",
                y="Thời gian (giây)",
                color="Kiểu",
                color_discrete_map={"Chuỗi": "#00d2ff", "Dạng bảng": "#10b981"},
                text_auto=".4f"
            )
            fig_time.update_layout(template="plotly_dark", paper_bgcolor="#131a2c", plot_bgcolor="#131a2c")
            st.plotly_chart(fig_time, use_container_width=True)

        # Độ đồng thuận
        st.markdown("---")
        st.subheader("4. Mức độ đồng thuận giữa các mô hình cùng phân giải")

        seq_models_in_bench = [s.model_id for s in summaries if s.model_type == "sequence"]
        tab_models_in_bench = [s.model_id for s in summaries if s.model_type == "tabular"]

        agree_col1, agree_col2 = st.columns(2)

        with agree_col1:
            st.markdown("#### Đồng thuận giữa các mô hình chuỗi (100 luồng/chuỗi)")
            if len(seq_models_in_bench) >= 2:
                seq_preds = {m_id: model_dfs[m_id]["predicted_label"].values for m_id in seq_models_in_bench}
                matrix_data = []
                for m1 in seq_models_in_bench:
                    row = {"Mô hình": MODEL_REGISTRY[m1].short_name or MODEL_REGISTRY[m1].display_name.split()[0]}
                    for m2 in seq_models_in_bench:
                        agree_pct = np.mean(seq_preds[m1] == seq_preds[m2]) * 100.0
                        row[MODEL_REGISTRY[m2].short_name or MODEL_REGISTRY[m2].display_name.split()[0]] = f"{agree_pct:.1f}%"
                    matrix_data.append(row)
                st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)
            else:
                st.info("Vui lòng chọn ít nhất 2 mô hình chuỗi để xem tỷ lệ đồng thuận chuỗi.")

        with agree_col2:
            st.markdown("#### Đồng thuận giữa các mô hình dạng bảng (Từng luồng)")
            if len(tab_models_in_bench) >= 2:
                tab_preds = {m_id: model_dfs[m_id]["predicted_label"].values for m_id in tab_models_in_bench}
                matrix_tab = []
                for m1 in tab_models_in_bench:
                    row = {"Mô hình": MODEL_REGISTRY[m1].short_name or MODEL_REGISTRY[m1].display_name.split()[0]}
                    for m2 in tab_models_in_bench:
                        agree_pct = np.mean(tab_preds[m1] == tab_preds[m2]) * 100.0
                        row[MODEL_REGISTRY[m2].short_name or MODEL_REGISTRY[m2].display_name.split()[0]] = f"{agree_pct:.1f}%"
                    matrix_tab.append(row)
                st.dataframe(pd.DataFrame(matrix_tab), use_container_width=True, hide_index=True)
            else:
                st.info("Vui lòng chọn cả Random Forest và XGBoost để xem tỷ lệ đồng thuận dạng bảng.")


# -----------------------------------------------------------------------------
# TRANG 4: ĐÁNH GIÁ NHÃN THỰC TẾ
# -----------------------------------------------------------------------------
elif navigation == "Đánh giá nhãn thực tế":
    st.title("Đánh Giá Hiệu Năng Mô Hình Có Giám Sát")
    st.markdown("Tính toán các độ đo thực nghiệm (Accuracy, Precision, Recall, F1-Score, Ma trận nhầm lẫn) dựa trên cột nhãn thực tế (Label).")

    if st.session_state.active_data_source is None:
        st.info("Vui lòng tải lên tệp CSV có cột 'Label' hoặc chọn tập dữ liệu kiểm thử bên dưới:")
        if st.button("Nạp tập kiểm thử hỗn hợp (test_mixed.csv)"):
            st.session_state.active_data_source = Path("test_mixed.csv")
            st.session_state.uploaded_filename = "test_mixed.csv (300 luồng hỗn hợp có nhãn)"
            st.rerun()
        st.stop()

    source = st.session_state.active_data_source
    clean_df, val_res = validate_csv_data(source, feature_columns=FEATURE_COLUMNS, time_steps=100, stride=100)

    if clean_df is None or not val_res.has_ground_truth:
        st.warning("Tập dữ liệu hiện tại không có cột 'Label'. Vui lòng tải lên tập dữ liệu có nhãn thực tế để thực hiện đánh giá.")
        st.stop()

    st.success(f"Phát hiện nhãn thực tế: {val_res.ground_truth_counts}")

    eval_model_id = st.selectbox(
        "Chọn mô hình cần đánh giá",
        options=list(MODEL_REGISTRY.keys()),
        format_func=lambda x: MODEL_REGISTRY[x].display_name
    )

    if st.button(f"Thực hiện đánh giá cho {MODEL_REGISTRY[eval_model_id].display_name}", type="primary"):
        with st.spinner("Đang tính toán các độ đo hiệu năng..."):
            res_df, summary = run_single_model_inference(eval_model_id, clean_df)

        if summary.evaluation is not None:
            ev: EvaluationMetrics = summary.evaluation
            st.markdown("---")
            st.subheader(f"Kết quả đánh giá cho {summary.model_name}")

            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.metric("Độ chính xác (Accuracy)", f"{ev.accuracy * 100:.2f}%")
            with e2:
                st.metric("Độ chuẩn xác (Precision)", f"{ev.precision * 100:.2f}%")
            with e3:
                st.metric("Độ thu hồi (Recall)", f"{ev.recall * 100:.2f}%")
            with e4:
                st.metric("Điểm F1 (F1-Score)", f"{ev.f1_score * 100:.2f}%")

            # Ma trận nhầm lẫn
            st.markdown("---")
            c_col1, c_col2 = st.columns([1, 1])

            with c_col1:
                st.markdown("#### Biểu đồ ma trận nhầm lẫn (Confusion Matrix)")
                cm_matrix = np.array([
                    [ev.true_negatives, ev.false_positives],
                    [ev.false_negatives, ev.true_positives]
                ])
                fig_cm = px.imshow(
                    cm_matrix,
                    labels=dict(x="Nhãn dự đoán", y="Nhãn thực tế", color="Số lượng"),
                    x=["BENIGN", "ATTACK"],
                    y=["BENIGN", "ATTACK"],
                    text_auto=True,
                    color_continuous_scale="Blues"
                )
                fig_cm.update_layout(template="plotly_dark", paper_bgcolor="#131a2c", plot_bgcolor="#131a2c")
                st.plotly_chart(fig_cm, use_container_width=True)

            with c_col2:
                st.markdown("#### So sánh với kết quả thực nghiệm trong khóa luận")
                meta = get_model_metadata(eval_model_id)
                hist = meta.historical_metrics
                st.markdown("""
                Bảng dưới đây so sánh các độ đo kiểm thử thời gian thực hiện tại với kết quả thực nghiệm trên tập test đã được ghi nhận trong khóa luận.
                """)
                comp_table = [
                    {"Độ đo": "Accuracy", "Kết quả Demo hiện tại": f"{ev.accuracy * 100:.2f}%", "Kết quả Khóa luận": f"{hist.get('accuracy', 0)*100:.2f}%"},
                    {"Độ đo": "Precision", "Kết quả Demo hiện tại": f"{ev.precision * 100:.2f}%", "Kết quả Khóa luận": f"{hist.get('precision', 0)*100:.2f}%"},
                    {"Độ đo": "Recall", "Kết quả Demo hiện tại": f"{ev.recall * 100:.2f}%", "Kết quả Khóa luận": f"{hist.get('recall', 0)*100:.2f}%"},
                    {"Độ đo": "F1-Score", "Kết quả Demo hiện tại": f"{ev.f1_score * 100:.2f}%", "Kết quả Khóa luận": f"{hist.get('f1_score', 0)*100:.2f}%"},
                ]
                st.dataframe(pd.DataFrame(comp_table), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# TRANG 5: THÔNG TIN MÔ HÌNH VÀ THỰC NGHIỆM
# -----------------------------------------------------------------------------
elif navigation == "Thông tin mô hình và thực nghiệm":
    st.title("Kiến Trúc Mô Hình, Nghiên Cứu Ablation Và Phương Pháp Luận")
    st.markdown("Thông số kỹ thuật xác thực được trích xuất trực tiếp từ các artifact của 5 mô hình.")

    tab_titles = [meta.short_name or meta.display_name.split()[0] for meta in MODEL_REGISTRY.values()] + ["Tổng kết Ablation Study"]
    tabs = st.tabs(tab_titles)

    for idx, (m_id, meta) in enumerate(MODEL_REGISTRY.items()):
        with tabs[idx]:
            st.header(meta.display_name)
            st.markdown(f"*{meta.description}*")

            i_col1, i_col2 = st.columns(2)
            with i_col1:
                st.subheader("Cấu hình kỹ thuật")
                kieu_mo_hinh_vn = "Chuỗi thời gian (100 luồng/chuỗi)" if meta.model_type == "sequence" else "Dạng bảng (Từng luồng đơn lẻ)"
                st.markdown(f"""
                - **Nền tảng thực thi:** `{meta.framework}`
                - **Kiểu cấu trúc:** `{kieu_mo_hinh_vn}`
                - **Số đặc trưng yêu cầu:** `{meta.n_features}` (Tập đặc trưng CIC-IDS2017)
                - **Ngưỡng quyết định:** `{meta.default_threshold:.4f}`
                - **Tệp artifact:** `{meta.model_path.name}`
                - **Thời gian huấn luyện:** `{meta.training_time_seconds:.2f} giây`
                """)
            with i_col2:
                st.subheader("Siêu tham số huấn luyện")
                for k, v in meta.hyperparameters.items():
                    st.markdown(f"- **{k}:** `{v}`")

            st.subheader("Các độ đo thực nghiệm trên tập Test (Khóa luận)")
            m_cards = st.columns(len(meta.historical_metrics))
            for c_idx, (m_name, m_val) in enumerate(meta.historical_metrics.items()):
                with m_cards[c_idx]:
                    st.metric(m_name.upper().replace("_", " "), f"{m_val*100 if m_val <= 1.0 else m_val:.2f}%")

    with tabs[-1]:
        st.header("Thực nghiệm Ablation: So sánh CNN, LSTM và CNN-LSTM")
        st.markdown("""
        Khóa luận đã thực hiện so sánh hệ thống giữa kiến trúc đơn lẻ và mô hình đề xuất lai:
        - **1D-CNN (Ablation):** Hiệu quả cao trong việc trích xuất đặc trưng không gian cục bộ và tương quan giữa các trường thông tin của luồng mạng.
        - **LSTM (Ablation):** Học tốt quan hệ phụ thuộc tuần tự theo thời gian giữa các phiên mạng liên tiếp.
        - **CNN-LSTM (Mô hình đề xuất):** Tích hợp đồng thời khả năng trích xuất đặc trưng không gian và duy trì trạng thái bộ nhớ thời gian, đạt được điểm F1 cao nhất và tỉ lệ báo động giả (FPR) thấp nhất trong thực nghiệm.
        """)

        ablation_comparison = pd.DataFrame([
            {
                "Mô hình": MODEL_REGISTRY["cnn_lstm"].display_name,
                "Loại hình": "Lai không gian - thời gian (Đề xuất)",
                "Accuracy": f"{MODEL_REGISTRY['cnn_lstm'].historical_metrics.get('accuracy', 0)*100:.2f}%",
                "F1-Score": f"{MODEL_REGISTRY['cnn_lstm'].historical_metrics.get('f1_score', 0)*100:.2f}%",
                "FPR": f"{MODEL_REGISTRY['cnn_lstm'].historical_metrics.get('fpr', 0)*100:.4f}%",
                "Thời gian huấn luyện": f"{MODEL_REGISTRY['cnn_lstm'].training_time_seconds}s"
            },
            {
                "Mô hình": MODEL_REGISTRY["cnn_only"].display_name,
                "Loại hình": "Chỉ đặc trưng không gian (Ablation)",
                "Accuracy": f"{MODEL_REGISTRY['cnn_only'].historical_metrics.get('accuracy', 0)*100:.2f}%",
                "F1-Score": f"{MODEL_REGISTRY['cnn_only'].historical_metrics.get('f1_score', 0)*100:.2f}%",
                "FPR": f"{MODEL_REGISTRY['cnn_only'].historical_metrics.get('fpr', 0)*100:.4f}%",
                "Thời gian huấn luyện": f"{MODEL_REGISTRY['cnn_only'].training_time_seconds}s"
            },
            {
                "Mô hình": MODEL_REGISTRY["lstm_only"].display_name,
                "Loại hình": "Chỉ đặc trưng thời gian (Ablation)",
                "Accuracy": f"{MODEL_REGISTRY['lstm_only'].historical_metrics.get('accuracy', 0)*100:.2f}%",
                "F1-Score": f"{MODEL_REGISTRY['lstm_only'].historical_metrics.get('f1_score', 0)*100:.2f}%",
                "FPR": f"{MODEL_REGISTRY['lstm_only'].historical_metrics.get('fpr', 0)*100:.4f}%",
                "Thời gian huấn luyện": f"{MODEL_REGISTRY['lstm_only'].training_time_seconds}s"
            }
        ])
        st.dataframe(ablation_comparison, use_container_width=True, hide_index=True)
