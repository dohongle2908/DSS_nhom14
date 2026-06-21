# Ứng dụng Streamlit dự đoán mức lãi đơn hàng

Ứng dụng dùng để dự đoán đơn hàng thuộc nhóm **lãi mỏng** hay **lãi dày**.

Model chính được chọn là `CatBoostClassifier`, vì trong bảng đánh giá hiện tại model này có F1-score cao nhất.

## 1. Cài thư viện

```bash
pip install -r requirements.txt
```

## 2. Train và lưu model

Chạy lệnh này trong folder `streamlit_order_risk_app`:

```bash
python train_model.py
```

Lệnh này sẽ tạo các file trong folder `artifacts`:

- `catboost_order_risk.cbm`
- `preprocessor.joblib`
- `model_metrics.json`

## 3. Chạy app

```bash
streamlit run app.py
```

## 4. Cột đầu vào khi upload file

File CSV/Excel cần có 6 cột:

```text
Sub-Category, Quantity, Amount, PaymentMode, State, City
```

`Category` sẽ được app tự suy ra từ `Sub-Category`.

Có thể dùng file `sample_input.csv` để test nhanh.

## 5. Ghi chú quan trọng

- Threshold tạo nhãn lúc train data là `Profit_Margin >= 0.25`.
- Class `1` là đơn hàng lãi dày.
- Class `0` là đơn hàng lãi mỏng.
- Ngưỡng trong sidebar của app là ngưỡng cắt xác suất dự đoán.
- Nếu `Xác suất lãi dày >= ngưỡng` thì app dự đoán `Lãi dày`.
- Nếu `Xác suất lãi dày < ngưỡng` thì app dự đoán `Lãi mỏng`.
