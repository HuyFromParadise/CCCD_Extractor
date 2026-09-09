# train_yolo.py
import shutil
import os
from ultralytics import YOLO

if __name__ == "__main__":
    # 1. Khởi tạo model YOLOv8 nano
    model = YOLO("yolov8n.pt")

    # 2. Train mô hình với dataset
    results = model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="cccd_model",
    )

    # 3. Chuyển file weights tốt nhất vào thư mục weights/
    best_weight_src = "runs/detect/cccd_model/weights/best.pt"
    dest_dir = "weights"
    os.makedirs(dest_dir, exist_ok=True)
    
    shutil.copy(best_weight_src, os.path.join(dest_dir, "cccd_yolo_best.pt"))
    print("[+] Hoàn tất Train! File trọng số đã được lưu tại: weights/cccd_yolo_best.pt")