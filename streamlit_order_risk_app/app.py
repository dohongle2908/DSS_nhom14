from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier

from preprocessing import RAW_FEATURES, TARGET_COLUMN


APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "catboost_order_risk.cbm"
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.joblib"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.json"

PROBA_COLUMN = "Xác suất lãi dày"
LABEL_COLUMN = "Nhãn dự đoán"
RESULT_COLUMN = "Kết quả dự đoán"

SUBCATEGORY_TO_CATEGORY = {
    "Binders": "Office Supplies",
    "Bookcases": "Furniture",
    "Chairs": "Furniture",
    "Electronic Games": "Electronics",
    "Laptops": "Electronics",
    "Markers": "Office Supplies",
    "Paper": "Office Supplies",
    "Pens": "Office Supplies",
    "Phones": "Electronics",
    "Printers": "Electronics",
    "Sofas": "Furniture",
    "Tables": "Furniture",
}

UPLOAD_REQUIRED_COLUMNS = [col for col in RAW_FEATURES if col != "Category"]


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        return None, None, None

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return model, preprocessor, metrics


def predict_dataframe(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    model, preprocessor, _ = load_artifacts()
    df = normalize_order_categories(df)
    X = preprocessor.transform(df)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    result = df.copy()
    result[PROBA_COLUMN] = proba
    result[LABEL_COLUMN] = pred
    result[RESULT_COLUMN] = result[LABEL_COLUMN].map(
        {
            1: "Lãi dày",
            0: "Lãi mỏng",
        }
    )
    return result


def infer_category(sub_category: str) -> str:
    return SUBCATEGORY_TO_CATEGORY.get(str(sub_category), "Unknown")


def normalize_order_categories(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "Sub-Category" not in normalized.columns:
        return normalized

    normalized["Category"] = normalized["Sub-Category"].map(infer_category)
    return normalized


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded_file)
    raise ValueError("Chỉ hỗ trợ file .csv, .xlsx hoặc .xls")


def csv_download_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")


def set_threshold(value: float) -> None:
    st.session_state["profit_threshold"] = value


st.set_page_config(
    page_title="Dự đoán mức lãi đơn hàng",
    layout="wide",
)

st.title("Dự đoán mức lãi đơn hàng")
st.caption("Ứng dụng dự đoán đơn hàng thuộc nhóm lãi mỏng hay lãi dày bằng mô hình CatBoost.")

model, preprocessor, metrics = load_artifacts()

if model is None or preprocessor is None:
    st.error(
        "Chưa tìm thấy model artifact. Hãy chạy lệnh `python train_model.py` "
        "trong folder `streamlit_order_risk_app` trước khi mở app."
    )
    st.stop()

if "profit_threshold" not in st.session_state:
    st.session_state["profit_threshold"] = 0.50

with st.sidebar:
    st.header("Cấu hình phân loại")
    st.write("Chọn mức cắt xác suất để phân loại giữa **Lãi mỏng** và **Lãi dày**.")

    col_easy, col_default, col_strict = st.columns(3)
    with col_easy:
        st.button("Dễ lãi dày", on_click=set_threshold, args=(0.40,), use_container_width=True)
    with col_default:
        st.button("Mặc định", on_click=set_threshold, args=(0.50,), use_container_width=True)
    with col_strict:
        st.button("Khắt khe", on_click=set_threshold, args=(0.60,), use_container_width=True)

    threshold = st.slider(
        "Ngưỡng phân loại Lãi dày",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        key="profit_threshold",
    )

    st.caption(
        f"Nếu {PROBA_COLUMN} >= {threshold:.2f} thì dự đoán **Lãi dày**, "
        "ngược lại là **Lãi mỏng**."
    )

    if metrics:
        st.divider()
        st.subheader("Model đang dùng")
        st.write(metrics.get("model", "CatBoostClassifier"))
        st.write(f"Số feature: {metrics.get('feature_count', len(preprocessor.feature_columns_))}")
        st.write(f"F1 trên tập test: {metrics.get('f1_score', 0):.4f}")
        st.write(f"Accuracy trên tập test: {metrics.get('accuracy', 0):.4f}")


payment_modes = preprocessor.categories_.get("PaymentMode", [])
categories = preprocessor.categories_.get("Category", [])
sub_categories = preprocessor.categories_.get("Sub-Category", [])
states = list(preprocessor.label_maps_.get("State", {}).keys())
cities = list(preprocessor.label_maps_.get("City", {}).keys())

tab_manual, tab_upload, tab_info = st.tabs(["Nhập tay", "Upload file", "Thông tin model"])

with tab_manual:
    col_left, col_right = st.columns(2)

    with col_left:
        sub_category = st.selectbox("Sub-Category", sub_categories)
        category = infer_category(sub_category)
        st.text_input("Category tự động", value=category, disabled=True)
        payment_mode = st.selectbox("PaymentMode", payment_modes)

    with col_right:
        quantity = st.number_input("Quantity", min_value=0.0, value=10.0, step=1.0)
        amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0)
        state = st.selectbox("State", states)
        city = st.selectbox("City", cities)

    input_row = pd.DataFrame(
        [
            {
                "Category": category,
                "Sub-Category": sub_category,
                "Quantity": quantity,
                "Amount": amount,
                "PaymentMode": payment_mode,
                "State": state,
                "City": city,
            }
        ]
    )

    if st.button("Dự đoán mức lãi", type="primary"):
        prediction = predict_dataframe(input_row, threshold)
        proba = float(prediction.loc[0, PROBA_COLUMN])
        label = int(prediction.loc[0, LABEL_COLUMN])
        result_text = prediction.loc[0, RESULT_COLUMN]

        metric_col_1, metric_col_2 = st.columns(2)
        metric_col_1.metric(PROBA_COLUMN, f"{proba:.2%}")
        metric_col_2.metric("Nhãn dự đoán", f"{label} - {result_text}")
        st.dataframe(prediction, use_container_width=True)

with tab_upload:
    st.write("File cần có các cột đầu vào:")
    st.code(", ".join(UPLOAD_REQUIRED_COLUMNS))
    st.caption("Cột Category sẽ được app tự suy ra từ Sub-Category.")

    uploaded_file = st.file_uploader(
        "Upload CSV/Excel",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file is not None:
        try:
            batch_df = read_uploaded_file(uploaded_file)
            missing_cols = [col for col in UPLOAD_REQUIRED_COLUMNS if col not in batch_df.columns]
            if missing_cols:
                st.error(f"File thiếu cột: {missing_cols}")
            else:
                batch_df = normalize_order_categories(batch_df)
                prediction_df = predict_dataframe(batch_df, threshold)
                st.success(f"Đã dự đoán {len(prediction_df)} dòng.")
                st.dataframe(prediction_df, use_container_width=True)
                st.download_button(
                    label="Tải file kết quả CSV",
                    data=csv_download_bytes(prediction_df),
                    file_name="du_doan_muc_lai_don_hang.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.exception(exc)

with tab_info:
    st.subheader("Biến đầu vào gốc")
    st.write(RAW_FEATURES)

    st.subheader("Feature model-ready sau xử lý")
    st.write(preprocessor.feature_columns_)

    if metrics:
        st.subheader("Chỉ số đánh giá khi train")
        st.json(metrics)

    st.subheader("Ý nghĩa nhãn")
    st.write(f"`{TARGET_COLUMN}=1`: Lãi dày")
    st.write(f"`{TARGET_COLUMN}=0`: Lãi mỏng")
    st.write(
        "Lưu ý: ngưỡng tạo nhãn khi train là `Profit_Margin >= 0.25`. "
        "Còn ngưỡng trong sidebar là ngưỡng cắt xác suất dự đoán của model."
    )
