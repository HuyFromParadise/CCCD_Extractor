import cv2
import os
from ultralytics import YOLO

class BBoxDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect_and_crop(self, image_path: str) -> dict:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể đọc file ảnh: {image_path}")

        results = self.model.predict(source=img, conf=0.35, verbose=False)
        boxes = results[0].boxes
        crops_by_class = {}

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            class_name = self.model.names[cls_id]

            cropped = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]

            if class_name not in crops_by_class:
                crops_by_class[class_name] = []
            crops_by_class[class_name].append((y1, cropped))

        return crops_by_class

    def draw_labels(self, image_path: str, output_path: str, conf_threshold: float = 0.35):
        """
        Nhận vào đường dẫn ảnh -> Khoanh vùng label + tên class -> Lưu ra file ảnh output
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể đọc file ảnh: {image_path}")

        # Dự đoán bằng YOLOv8
        results = self.model.predict(source=img, conf=conf_threshold, verbose=False)

        # Vẽ bounding box và tên nhãn trực tiếp lên ảnh
        annotated_frame = results[0].plot()

        # Tạo thư mục lưu ảnh nếu chưa tồn tại
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Lưu ảnh kết quả
        cv2.imwrite(output_path, annotated_frame)
        return output_path