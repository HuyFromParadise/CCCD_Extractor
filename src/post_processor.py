# src/post_processor.py
import re

class CCCDPostProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Làm sạch khoảng trắng thừa và ký tự lạ"""
        if not text:
            return ""
        text = re.sub(r'[^\w\s/,\.-]', '', text, flags=re.UNICODE)
        return " ".join(text.split()).strip()

    @staticmethod
    def process_id(text: str) -> str:
        """Số CCCD: Lọc lấy đúng 12 chữ số"""
        digits = re.sub(r'\D', '', text)
        if len(digits) == 12:
            return digits
        return text  # Trả về nguyên bản nếu chưa khớp chuẩn 12 số

    @staticmethod
    def process_date(text: str) -> str:
        """Định dạng ngày tháng: dd/mm/yyyy"""
        # Thay thế các ký tự dễ nhầm lẫn
        text = text.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1')
        match = re.search(r'\d{2}/\d{2}/\d{4}', text)
        if match:
            return match.group(0)
        return text

    @staticmethod
    def process_sex(text: str) -> str:
        """Chuẩn hóa giới tính: Nam / Nữ"""
        text_lower = text.lower()
        if 'nam' in text_lower or 'm' in text_lower:
            return "Nam"
        elif 'nữ' in text_lower or 'nu' in text_lower or 'f' in text_lower:
            return "Nữ"
        return text

    def clean_all_fields(self, raw_results: dict) -> dict:
        """Áp dụng bộ lọc cho toàn bộ dữ liệu OCR"""
        cleaned = {}
        for key, value in raw_results.items():
            val = self.clean_text(value)
            
            if key == 'id':
                val = self.process_id(val)
            elif key in ['birth', 'expiry']:
                val = self.process_date(val)
            elif key == 'sex':
                val = self.process_sex(val)
            elif key == 'name':
                val = val.upper()  # Tên CCCD luôn viết hoa

            cleaned[key] = val
        return cleaned