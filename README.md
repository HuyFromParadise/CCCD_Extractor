# 🪪 Vietnamese CCCD OCR Pipeline (YOLOv8 + VietOCR + FastAPI)

Hệ thống trích xuất thông tin Căn cước công dân (CCCD) Việt Nam tự động với độ chính xác cao. Dự án kết hợp **YOLOv8** cho nhiệm vụ định vị trường thông tin (Bounding Box Detection), **VietOCR** cho nhiệm vụ nhận dạng chữ viết tiếng Việt (Text Recognition), bộ lọc **Regex** để xử lý hậu kỳ (Post-processing) và **FastAPI** để cung cấp RESTful API.

---

## 📌 Các Tính Năng Chính

* **Định vị chính xác**: Sử dụng YOLOv8 phát hiện 8 trường dữ liệu cốt lõi (`id`, `name`, `birth`, `sex`, `nationality`, `place_of_origin`, `place_of_residence`, `expiry`).
* **Nhận dạng tiếng Việt mượt mà**: Tích hợp VietOCR nhận dạng chính xác văn bản có dấu.
* **Tự động làm sạch dữ liệu**: Module `CCCDPostProcessor` sử dụng Regex để chuẩn hóa định dạng Ngày sinh, Số CCCD 12 số, Giới tính và Tên in hoa.
* **Bỏ qua nhiễu**: Tự động lọc bỏ các vùng không chứa chữ như Mã QR, ảnh Chân dung (`portrait`, `qr_code`).
* **Sẵn sàng Production**: Cung cấp REST API bằng FastAPI, tự động sinh tài liệu Swagger UI trực quan.

---

## 📁 Cấu Trúc Dự Án

```text
cccd-ocr-project/
├── app.py                      # REST API Server (FastAPI)
├── main.py                     # Script chạy pipeline kiểm thử trên Terminal
├── label_tester.py             # Script vẽ Bounding Box kiểm tra YOLOv8
├── train_yolo.py               # Script huấn luyện lại mô hình YOLOv8
├── requirements.txt            # Danh sách các thư viện phụ thuộc
├── weights/
│   └── cccd_yolo_best.pt       # Trọng số mô hình YOLOv8
├── src/                        # Thư mục mã nguồn core
│   ├── __init__.py
│   ├── detector.py             # Phát hiện và cắt khung ảnh (YOLOv8)
│   ├── recognizer.py           # Nhận dạng chữ (VietOCR)
│   ├── post_processor.py       # Hậu xử lý dữ liệu (Regex)
│   └── pipeline.py             # Pipeline hợp nhất
└── data/
    ├── input/                  # Chứa ảnh CCCD test đầu vào
    ├── output/                 # Chứa ảnh vẽ debug bounding box
    └── temp/                   # File tạm cho API Server