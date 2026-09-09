# src/pipeline.py
from src.detector import BBoxDetector
from src.recognizer import TextRecognizer
from src.post_processor import CCCDPostProcessor

class CCCDPipeline:
    def __init__(self, yolo_path: str):
        print("[-] Đang tải mô hình Detector, VietOCR và PostProcessor...")
        self.detector = BBoxDetector(yolo_path)
        self.recognizer = TextRecognizer()
        self.post_processor = CCCDPostProcessor()
        print("[+] Đã khởi tạo thành công Pipeline!")

    def process_image(self, image_path: str) -> dict:
        # 1. Phát hiện và cắt khung từ YOLOv8
        crops_dict = self.detector.detect_and_crop(image_path)
        raw_results = {}

        # 2. Bỏ qua các nhãn không chứa chữ
        IGNORED_CLASSES = ['portrait', 'qr_code']

        for class_name, items in crops_dict.items():
            if class_name in IGNORED_CLASSES:
                continue

            # Sắp xếp các dòng chữ theo thứ tự từ trên xuống dưới
            items.sort(key=lambda x: x[0])
            
            # 3. Đưa qua VietOCR để đọc văn bản
            texts = [self.recognizer.predict_text(crop) for _, crop in items if crop.size > 0]
            raw_results[class_name] = " ".join(texts)

        # 4. Hậu xử lý dữ liệu qua Regex (Cleaning)
        cleaned_results = self.post_processor.clean_all_fields(raw_results)

        return cleaned_results