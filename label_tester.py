# label_tester.py
import os
from src.detector import BBoxDetector

if __name__ == "__main__":
    # 1. Đường dẫn mô hình và ảnh đầu vào
    MODEL_WEIGHTS = "weights/cccd_yolo_best.pt"
    INPUT_IMAGE = "data/input/test.JPG"        # Đặt ảnh bạn muốn test ở đây
    OUTPUT_IMAGE = "data/output/labeled_result.png" # Ảnh sau khi khoanh vùng sẽ lưu ở đây

    print("[-] Đang khởi tạo mô hình YOLOv8...")
    detector = BBoxDetector(MODEL_WEIGHTS)

    print(f"[-] Đang xử lý ảnh: {INPUT_IMAGE}")
    try:
        # 2. Gọi hàm khoanh vùng và vẽ nhãn
        saved_path = detector.draw_labels(
            image_path=INPUT_IMAGE, 
            output_path=OUTPUT_IMAGE, 
            conf_threshold=0.35  # Lọc bớt box độ tin cậy thấp < 35%
        )
        
        print("\n==================================================")
        print(f"✅ ĐÃ HOÀN THÀNH! Ảnh kết quả đã được lưu tại:")
        print(f"👉 {os.path.abspath(saved_path)}")
        print("==================================================\n")

    except Exception as e:
        print(f"❌ Lỗi: {e}")