# label_tester.py
import os
from src.detector import BBoxDetector

if __name__ == "__main__":
    MODEL_WEIGHTS = "weights/cccd_yolo_best.pt"
    INPUT_IMAGE = "data/input/test.JPG"      
    OUTPUT_IMAGE = "data/output/labeled_result.png" 
    print("[-] Loading YOLOv8 model...")
    detector = BBoxDetector(MODEL_WEIGHTS)

    print(f"[-] Processing Image: {INPUT_IMAGE}")
    try:
        saved_path = detector.draw_labels(
            image_path=INPUT_IMAGE, 
            output_path=OUTPUT_IMAGE, 
            conf_threshold=0.35  
        )
        
        print("\n==================================================")
        print(f"✅ Successfully! Image result saved in:")
        print(f"👉 {os.path.abspath(saved_path)}")
        print("==================================================\n")

    except Exception as e:
        print(f"❌ Lỗi: {e}")