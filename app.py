# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
from src.pipeline import CCCDPipeline

app = FastAPI(
    title="Vietnamese CCCD OCR API",
    description="API trích xuất thông tin Căn cước công dân sử dụng YOLOv8 & VietOCR",
    version="1.0.0"
)

# Cấu hình CORS để cho phép Frontend (React, Vue, Web UI...) gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo Pipeline OCR (Load model 1 lần duy nhất khi khởi chạy server)
MODEL_WEIGHTS = "weights/cccd_yolo_best.pt"
TEMP_DIR = "data/temp"

os.makedirs(TEMP_DIR, exist_ok=True)
ocr_pipeline = CCCDPipeline(yolo_path=MODEL_WEIGHTS)

# Map tên class sang tên hiển thị chuẩn
FIELD_NAMES = {
    'id': 'Số CCCD',
    'name': 'Họ và tên',
    'birth': 'Ngày sinh',
    'sex': 'Giới tính',
    'nationality': 'Quốc tịch',
    'place_of_origin': 'Quê quán',
    'place_of_residence': 'Nơi thường trú',
    'expiry': 'Có giá trị đến'
}

@app.get("/")
def root():
    return {"status": "success", "message": "CCCD OCR API đang hoạt động bình thường!"}

@app.post("/api/v1/extract-cccd")
async def extract_cccd(file: UploadFile = File(...)):
    # 1. Kiểm tra định dạng file gửi lên
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File tải lên phải là định dạng hình ảnh (jpg, png, jpeg)")

    # 2. Lưu tạm file ảnh vào ổ đĩa với tên ngẫu nhiên (tránh trùng lặp)
    file_extension = file.filename.split(".")[-1]
    temp_filename = f"{uuid.uuid4()}.{file_extension}"
    temp_file_path = os.path.join(TEMP_DIR, temp_filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Trích xuất thông tin bằng Pipeline
        raw_results = ocr_pipeline.process_image(temp_file_path)

        # 4. Định dạng lại JSON đầu ra
        formatted_data = {}
        for key, value in raw_results.items():
            field_name = FIELD_NAMES.get(key, key)
            formatted_data[key] = {
                "label": field_name,
                "value": value
            }

        return {
            "status": 200,
            "success": True,
            "filename": file.filename,
            "data": formatted_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống trong quá trình OCR: {str(e)}")

    finally:
        # 5. Dọn dẹp file tạm sau khi xử lý xong
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            