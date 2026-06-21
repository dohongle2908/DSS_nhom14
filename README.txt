# FOLDER KẾT QUẢ DỰ ÁN DSS

Dự án Decision Support System (DSS) sử dụng các mô hình Machine Learning để phân loại đơn hàng theo mức độ lợi nhuận và triển khai ứng dụng dự đoán bằng Streamlit.

---

## 1. File Colab
**Đường dẫn:**
ket_qua/colab_notebook/DSS_ensemble_optuna_classification_models.ipynb
**Mô tả:**

File notebook được sử dụng để chạy trên Google Colab. Notebook bao gồm quá trình huấn luyện các mô hình classification, tối ưu siêu tham số bằng Optuna và đánh giá hiệu năng mô hình.

**Nội dung chính:**
- Huấn luyện các mô hình classification
- Tối ưu siêu tham số bằng Optuna
- So sánh nhiều mô hình
- Đánh giá bằng các chỉ số:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Tổng hợp kết quả thành bảng so sánh

---

## 2. Folder Streamlit

**Đường dẫn:**
ket_qua/streamlit_order_risk_app

**Mô tả:**

Đây là ứng dụng web xây dựng bằng Streamlit dùng để dự đoán đơn hàng thuộc nhóm:
- Lãi mỏng
- Lãi dày

---

### Cấu trúc thư mục

- `app.py`: Giao diện web chính của ứng dụng
- `preprocessing.py`: Xử lý dữ liệu đầu vào
- `train_model.py`: Huấn luyện và load mô hình
- `requirements.txt`: Danh sách thư viện cần cài đặt
- `sample_input.csv`: Dữ liệu mẫu để test ứng dụng

**Thư mục artifacts:**
- `catboost_order_risk.cbm`: Mô hình CatBoost đã huấn luyện
- `preprocessor.joblib`: Pipeline tiền xử lý dữ liệu
- `model_metrics.json`: Kết quả đánh giá mô hình

---

## 3. Cách chạy ứng dụng Streamlit

### Bước 1: Mở terminal tại thư mục dự án
ket_qua/streamlit_order_risk_app

### Bước 2: Cài đặt thư viện
```bash
pip install -r requirements.txt
### Bước 3: Chạy ứng dụng
python -m streamlit run app.py
### Bước 4: Truy cập ứng dụng 
Sau khi chạy, mở trình duyệt và truy cập 
http://localhost:8501

### 4. Ghi chú 

Đã loại bỏ các file không cần thiết để tối ưu dung lượng:
.venv
__pycache__
file log của Streamlit
Giúp project gọn nhẹ
Dễ nộp bài và triển khai
Tránh file rác khi submit