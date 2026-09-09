import cv2
import torch
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

class TextRecognizer:
    def __init__(self):
        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        config['predictor']['beamsearch'] = False
        self.predictor = Predictor(config)

    def predict_text(self, crop_bgr) -> str:
        if crop_bgr.size == 0:
            return ""
        # Chuyển BGR (OpenCV) -> RGB (PIL Image)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        
        return self.predictor.predict(crop_pil).strip()