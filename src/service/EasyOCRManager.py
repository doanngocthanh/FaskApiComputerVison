"""
EasyOCR Manager - Quản lý EasyOCR models local để tránh download lại
Hỗ trợ Vietnamese, English, Chinese Traditional với khả năng pretrain
Tối ưu hóa chỉ sử dụng EasyOCR cho độ ổn định cao
"""

import os
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EasyOCRManager:
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = "temp/models"
        
        self.base_dir = Path(models_dir)
        self.easyocr_dir = self.base_dir / "easyocr"
        
        # Đảm bảo thư mục tồn tại
        self.easyocr_dir.mkdir(parents=True, exist_ok=True)
        
        # OCR engine - chỉ EasyOCR
        self._easyocr_reader = None
        self._easyocr_languages = None
        
        # Language mapping for EasyOCR
        self.language_mapping = {
            "vi": {"easyocr": "vi", "name": "Vietnamese"},
            "en": {"easyocr": "en", "name": "English"},
            "zh": {"easyocr": "ch_tra", "name": "Chinese Traditional"},
            "ch": {"easyocr": "ch_tra", "name": "Chinese Traditional"},
            "ja": {"easyocr": "ja", "name": "Japanese"},
            "ko": {"easyocr": "ko", "name": "Korean"},
            "th": {"easyocr": "th", "name": "Thai"},
            "fr": {"easyocr": "fr", "name": "French"},
            "de": {"easyocr": "de", "name": "German"},
            "es": {"easyocr": "es", "name": "Spanish"},
            "ru": {"easyocr": "ru", "name": "Russian"},
            "ar": {"easyocr": "ar", "name": "Arabic"}
        }
        
        logger.info(f"EasyOCR Manager initialized with models directory: {self.base_dir}")
        
        # Check available engines immediately
        available_engines = self.get_available_engines()
        logger.info(f"Available OCR engines: {available_engines}")
    
    def _initialize_easyocr(self, languages: List[str]) -> Optional[Any]:
        """Initialize EasyOCR với cấu hình đơn giản và ổn định"""
        try:
            import easyocr
            
            # Map languages to EasyOCR format
            easyocr_langs = []
            has_chinese = False
            
            for lang in languages:
                if lang in self.language_mapping:
                    mapped_lang = self.language_mapping[lang]["easyocr"]
                    if mapped_lang in ["ch_tra", "ch_sim"]:
                        has_chinese = True
                        easyocr_langs.append(mapped_lang)
                    else:
                        easyocr_langs.append(mapped_lang)
                else:
                    easyocr_langs.append(lang)
            
            # Chinese Traditional requires English to be included
            if has_chinese and "en" not in easyocr_langs:
                easyocr_langs.append("en")
                logger.info("Đã thêm English vì Chinese Traditional yêu cầu")
            
            if not easyocr_langs:
                easyocr_langs = ["en"]
            
            logger.info(f"Khởi tạo EasyOCR với ngôn ngữ: {easyocr_langs}")
            logger.info(f"Model directory: {self.easyocr_dir}")
            
            # Initialize EasyOCR với cấu hình đơn giản
            reader = easyocr.Reader(
                easyocr_langs, 
                gpu=False,  # Sử dụng CPU để ổn định
                verbose=False,
                model_storage_directory=str(self.easyocr_dir),
                download_enabled=True  # Cho phép download lần đầu
            )
            
            self._easyocr_reader = reader
            logger.info("✅ EasyOCR khởi tạo thành công")
            return reader
            
        except ImportError as e:
            logger.error(f"EasyOCR không có sẵn: {e}")
            return None
        except Exception as e:
            logger.error(f"Lỗi khởi tạo EasyOCR: {e}")
            return None
    
    def preprocess_image_simple(self, image: np.ndarray) -> np.ndarray:
        """Tiền xử lý ảnh đơn giản để cải thiện OCR"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Simple contrast enhancement
            alpha = 1.2  # Contrast control
            beta = 10    # Brightness control
            enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
            
            # Simple denoising
            denoised = cv2.medianBlur(enhanced, 3)
            
            return denoised
            
        except Exception as e:
            logger.warning(f"Preprocessing failed, using original: {e}")
            return image
    
    def extract_text(self, image_array: np.ndarray, languages: List[str] = None, preprocess: bool = True) -> Dict[str, Any]:
        """
        Trích xuất text từ image sử dụng EasyOCR với cấu hình ổn định
        """
        if languages is None:
            languages = ["vi", "en"]
        
        logger.info(f"Bắt đầu OCR với ngôn ngữ: {languages}")
        
        # Use EasyOCR only
        result = self._try_easyocr(image_array, languages, preprocess)
        if result["success"]:
            logger.info("✅ EasyOCR thành công")
            return result
        
        # If EasyOCR failed
        logger.error("❌ EasyOCR thất bại")
        return {
            "success": False,
            "text": "[OCR không khả dụng - Lỗi EasyOCR]",
            "confidence": 0.0,
            "engine": "easyocr",
            "error": result.get("error", "Unknown EasyOCR error")
        }
    
    def _try_easyocr(self, image_array: np.ndarray, languages: List[str], preprocess: bool = True) -> Dict[str, Any]:
        """Thử sử dụng EasyOCR với cấu hình đơn giản"""
        try:
            # Check if we need to reinitialize for different languages
            if (self._easyocr_reader is None or 
                self._easyocr_languages != languages):
                
                logger.info(f"Khởi tạo EasyOCR cho ngôn ngữ: {languages}")
                self._easyocr_reader = self._initialize_easyocr(languages)
                if self._easyocr_reader is not None:
                    self._easyocr_languages = languages.copy()
            
            if self._easyocr_reader is None:
                return {"success": False, "error": "Không thể khởi tạo EasyOCR"}
            
            # Preprocess image if requested
            processed_image = image_array
            if preprocess:
                processed_image = self.preprocess_image_simple(image_array)
            
            # Perform OCR với tham số đơn giản
            logger.info("Đang thực hiện EasyOCR...")
            results = self._easyocr_reader.readtext(
                processed_image,
                detail=1,  # Return bounding boxes
                paragraph=False  # Don't group into paragraphs
            )
            
            if not results:
                return {
                    "success": True,
                    "text": "",
                    "confidence": 0.0,
                    "engine": "easyocr",
                    "details": "Không tìm thấy text",
                    "bboxes": []
                }
            
            # Extract text and confidence
            extracted_texts = []
            confidences = []
            bboxes = []
            
            for result in results:
                if len(result) >= 3:
                    bbox = result[0]
                    text = result[1]
                    confidence = result[2]
                    
                    # Filter by minimum confidence
                    if confidence > 0.5:  # Ngưỡng tin cậy cao hơn
                        extracted_texts.append(text)
                        confidences.append(confidence)
                        
                        # Convert bbox to simple format
                        simple_bbox = [
                            int(min(point[0] for point in bbox)),
                            int(min(point[1] for point in bbox)),
                            int(max(point[0] for point in bbox)),
                            int(max(point[1] for point in bbox))
                        ]
                        bboxes.append(simple_bbox)
            
            combined_text = " ".join(extracted_texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            logger.info(f"EasyOCR kết quả: '{combined_text}' (confidence: {avg_confidence:.2f})")
            
            return {
                "success": True,
                "text": combined_text,
                "confidence": avg_confidence,
                "engine": "easyocr",
                "details": f"Tìm thấy {len(results)} text regions, {len(extracted_texts)} có confidence > 0.5",
                "bboxes": bboxes,
                "texts": extracted_texts,
                "confidences": confidences
            }
            
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_available_engines(self) -> Dict[str, bool]:
        """Kiểm tra các OCR engine có sẵn"""
        engines = {}
        
        # Check EasyOCR only
        try:
            import easyocr
            engines["easyocr"] = True
            logger.info("✅ EasyOCR engine có sẵn")
        except ImportError:
            engines["easyocr"] = False
            logger.warning("❌ EasyOCR engine không có sẵn")
        
        return engines
    
    def get_model_status(self) -> Dict[str, Any]:
        """Lấy thông tin trạng thái model"""
        easyocr_models = list(self.easyocr_dir.glob("*"))
        
        return {
            "easyocr": {
                "directory": str(self.easyocr_dir),
                "models_count": len(easyocr_models),
                "models": [m.name for m in easyocr_models],
                "initialized": self._easyocr_reader is not None,
                "current_languages": self._easyocr_languages or []
            },
            "available_engines": self.get_available_engines(),
            "supported_languages": list(self.language_mapping.keys()),
            "primary_engine": "easyocr"
        }
    
    def preload_models(self, languages: List[str] = None):
        """Preload EasyOCR models để sẵn sàng sử dụng"""
        if languages is None:
            languages = ["vi", "en"]
        
        logger.info(f"Preloading EasyOCR models cho ngôn ngữ: {languages}")
        
        # Preload EasyOCR only
        try:
            result = self._initialize_easyocr(languages)
            if result is not None:
                self._easyocr_languages = languages.copy()
                logger.info("✅ EasyOCR preload thành công")
            else:
                logger.error("❌ EasyOCR preload thất bại")
        except Exception as e:
            logger.error(f"❌ Lỗi preload EasyOCR: {e}")
        
        logger.info("✅ Hoàn thành preload models")


# Global OCR manager instance
easy_ocr_manager = EasyOCRManager()
