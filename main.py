# main.py
from src.pipeline import CCCDPipeline

if __name__ == "__main__":
    MODEL_WEIGHTS = "weights/cccd_yolo_best.pt"
    TEST_IMAGE = "data/input/test.JPG"

    ocr_pipeline = CCCDPipeline(yolo_path=MODEL_WEIGHTS)

    # Khai báo tên hiển thị tương ứng với 8 nhãn dữ liệu thực tế
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

    try:
        results = ocr_pipeline.process_image(TEST_IMAGE)
        
        print("\n================ KẾT QUẢ TRÍCH XUẤT CCCD ================")
        for field, value in results.items():
            display_name = FIELD_NAMES.get(field, field.upper())
            print(f"📌 {display_name}: {value}")
        print("=========================================================\n")
        
    except Exception as e:
        print(f"Lỗi trong quá trình xử lý: {e}")