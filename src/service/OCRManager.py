"""
OCR Service Manager
Supports multiple OCR engines and languages
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import json

class BaseOCREngine(ABC):
    """Base class for OCR engines"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.supported_languages = []
        self.name = ""
    
    @abstractmethod
    def process_image(self, image_path: str, language: str = "auto") -> Dict[str, Any]:
        """Process image and return OCR results"""
        pass
    
    @abstractmethod
    def process_image_array(self, image: np.ndarray, language: str = "auto") -> Dict[str, Any]:
        """Process numpy image array and return OCR results"""
        pass
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return self.supported_languages
    
    def is_language_supported(self, language: str) -> bool:
        """Check if language is supported"""
        return language in self.supported_languages or language == "auto"

class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR engine implementation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "PaddleOCR"
        self.supported_languages = [
            "ch", "en", "vi", "japan", "korean", "german", "french", 
            "spanish", "portuguese", "russian", "arabic", "hindi", "auto"
        ]
        self._ocr_instance = None
    
    def _get_ocr_instance(self, language: str = "auto"):
        """Get PaddleOCR instance for specific language"""
        try:
            from paddleocr import PaddleOCR
            
            # Map common language codes
            lang_map = {
                "auto": "ch",  # Default to Chinese which works well for multilingual
                "vietnamese": "vi",
                "english": "en",
                "chinese": "ch",
                "japanese": "japan",
                "korean": "korean"
            }
            
            ocr_lang = lang_map.get(language.lower(), language.lower())
            
            if not self.is_language_supported(ocr_lang):
                ocr_lang = "ch"  # Fallback to default
            
            # Create new instance for different language
            return PaddleOCR(
                use_angle_cls=self.config.get("use_angle_cls", True),
                lang=ocr_lang,
                use_gpu=self.config.get("use_gpu", False),
                show_log=self.config.get("show_log", False)
            )
        except ImportError:
            raise ImportError("PaddleOCR not installed. Install with: pip install paddleocr")
    
    def process_image(self, image_path: str, language: str = "auto") -> Dict[str, Any]:
        """Process image file with PaddleOCR"""
        ocr = self._get_ocr_instance(language)
        results = ocr.ocr(image_path, cls=True)
        
        return self._format_paddle_results(results, language)
    
    def process_image_array(self, image: np.ndarray, language: str = "auto") -> Dict[str, Any]:
        """Process numpy image array with PaddleOCR"""
        ocr = self._get_ocr_instance(language)
        results = ocr.ocr(image, cls=True)
        
        return self._format_paddle_results(results, language)
    
    def _format_paddle_results(self, results: List, language: str) -> Dict[str, Any]:
        """Format PaddleOCR results to standard format"""
        texts = []
        confidences = []
        bboxes = []
        
        if results and results[0]:
            for line in results[0]:
                if line:
                    bbox = line[0]
                    text_info = line[1]
                    text = text_info[0]
                    confidence = text_info[1]
                    
                    texts.append(text)
                    confidences.append(float(confidence))
                    bboxes.append(bbox)
        
        return {
            "engine": self.name,
            "language": language,
            "texts": texts,
            "confidences": confidences,
            "bboxes": bboxes,
            "text_count": len(texts)
        }

class ONNXPaddleOCREngine(BaseOCREngine):
    """ONNX PaddleOCR engine implementation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "ONNX-PaddleOCR"
        self.supported_languages = ["ch", "en", "vi", "auto"]
        
        # Import your existing ONNX PaddleOCR
        try:
            from src.service.onnx.PaddleOCR import PaddleOCR
            self._ocr_instance = PaddleOCR()
        except ImportError:
            raise ImportError("ONNX PaddleOCR service not found")
    
    def process_image(self, image_path: str, language: str = "auto") -> Dict[str, Any]:
        """Process image with ONNX PaddleOCR"""
        result = self._ocr_instance.process_image(image_path)
        return self._format_onnx_results(result, language)
    
    def process_image_array(self, image: np.ndarray, language: str = "auto") -> Dict[str, Any]:
        """Process numpy array with ONNX PaddleOCR"""
        # Save temp file for ONNX processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, image)
            tmp_path = tmp.name
        
        try:
            result = self._ocr_instance.process_image(tmp_path)
            return self._format_onnx_results(result, language)
        finally:
            os.unlink(tmp_path)
    
    def _format_onnx_results(self, result: Dict, language: str) -> Dict[str, Any]:
        """Format ONNX PaddleOCR results to standard format"""
        texts = []
        confidences = []
        
        if isinstance(result, dict) and 'texts' in result:
            for item in result['texts']:
                if isinstance(item, list) and len(item) >= 1:
                    texts.append(item[0])
                    if len(item) >= 2:
                        confidences.append(item[1] if isinstance(item[1], (int, float)) else 0.8)
                    else:
                        confidences.append(0.8)
                elif isinstance(item, str):
                    texts.append(item)
                    confidences.append(0.8)
        
        return {
            "engine": self.name,
            "language": language,
            "texts": texts,
            "confidences": confidences,
            "bboxes": result.get('text_regions', []),
            "text_count": len(texts),
            "original_result": result
        }

class EasyOCREngine(BaseOCREngine):
    """EasyOCR engine implementation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "EasyOCR"
        self.supported_languages = [
            "en", "vi", "ch_sim", "ch_tra", "ja", "ko", "th", "ar", "hi", 
            "ru", "de", "fr", "es", "pt", "it", "nl", "auto"
        ]
        self._reader = None
    
    def _get_reader(self, language: str = "auto"):
        """Get EasyOCR reader for specific language"""
        try:
            import easyocr
            
            # Map language codes
            lang_list = []
            if language == "auto":
                lang_list = ["en", "vi"]  # Default multilingual
            elif language == "vietnamese":
                lang_list = ["vi", "en"]
            elif language == "english":
                lang_list = ["en"]
            elif language == "chinese":
                lang_list = ["ch_sim", "en"]
            else:
                lang_list = [language, "en"]  # Add English as fallback
            
            return easyocr.Reader(
                lang_list, 
                gpu=self.config.get("use_gpu", False),
                verbose=self.config.get("verbose", False)
            )
        except ImportError:
            raise ImportError("EasyOCR not installed. Install with: pip install easyocr")
    
    def process_image(self, image_path: str, language: str = "auto") -> Dict[str, Any]:
        """Process image with EasyOCR"""
        reader = self._get_reader(language)
        results = reader.readtext(
            image_path,
            detail=self.config.get("detail", 1),
            paragraph=self.config.get("paragraph", False)
        )
        
        return self._format_easyocr_results(results, language)
    
    def process_image_array(self, image: np.ndarray, language: str = "auto") -> Dict[str, Any]:
        """Process numpy array with EasyOCR"""
        reader = self._get_reader(language)
        results = reader.readtext(
            image,
            detail=self.config.get("detail", 1),
            paragraph=self.config.get("paragraph", False)
        )
        
        return self._format_easyocr_results(results, language)
    
    def _format_easyocr_results(self, results: List, language: str) -> Dict[str, Any]:
        """Format EasyOCR results to standard format"""
        texts = []
        confidences = []
        bboxes = []
        
        for result in results:
            if len(result) >= 3:
                bbox = result[0]
                text = result[1]
                confidence = result[2]
                
                texts.append(text)
                confidences.append(float(confidence))
                bboxes.append(bbox)
        
        return {
            "engine": self.name,
            "language": language,
            "texts": texts,
            "confidences": confidences,
            "bboxes": bboxes,
            "text_count": len(texts)
        }

class TesseractEngine(BaseOCREngine):
    """Tesseract OCR engine implementation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "Tesseract"
        self.supported_languages = [
            "eng", "vie", "chi_sim", "chi_tra", "jpn", "kor", "tha", "ara", 
            "hin", "rus", "deu", "fra", "spa", "por", "ita", "nld", "auto"
        ]
    
    def process_image(self, image_path: str, language: str = "auto") -> Dict[str, Any]:
        """Process image with Tesseract"""
        try:
            import pytesseract
            from PIL import Image
            
            # Map language codes
            lang_map = {
                "auto": "eng+vie",
                "vietnamese": "vie",
                "english": "eng",
                "chinese": "chi_sim",
                "japanese": "jpn",
                "korean": "kor"
            }
            
            tesseract_lang = lang_map.get(language.lower(), "eng")
            
            # Open image
            image = Image.open(image_path)
            
            # Get text with bounding boxes
            data = pytesseract.image_to_data(
                image, 
                lang=tesseract_lang, 
                output_type=pytesseract.Output.DICT
            )
            
            return self._format_tesseract_results(data, language)
            
        except ImportError:
            raise ImportError("Tesseract not installed. Install with: pip install pytesseract")
    
    def process_image_array(self, image: np.ndarray, language: str = "auto") -> Dict[str, Any]:
        """Process numpy array with Tesseract"""
        try:
            import pytesseract
            from PIL import Image
            
            # Convert numpy array to PIL Image
            if len(image.shape) == 3:
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image_pil = Image.fromarray(image)
            
            # Map language codes
            lang_map = {
                "auto": "eng+vie",
                "vietnamese": "vie",
                "english": "eng",
                "chinese": "chi_sim",
                "japanese": "jpn",
                "korean": "kor"
            }
            
            tesseract_lang = lang_map.get(language.lower(), "eng")
            
            # Get text with bounding boxes
            data = pytesseract.image_to_data(
                image_pil, 
                lang=tesseract_lang, 
                output_type=pytesseract.Output.DICT
            )
            
            return self._format_tesseract_results(data, language)
            
        except ImportError:
            raise ImportError("Tesseract not installed. Install with: pip install pytesseract")
    
    def _format_tesseract_results(self, data: Dict, language: str) -> Dict[str, Any]:
        """Format Tesseract results to standard format"""
        texts = []
        confidences = []
        bboxes = []
        
        for i, text in enumerate(data['text']):
            if text.strip():  # Only include non-empty text
                confidence = int(data['conf'][i])
                if confidence > 0:  # Only include text with confidence > 0
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    
                    texts.append(text.strip())
                    confidences.append(confidence / 100.0)  # Normalize to 0-1
                    bboxes.append([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
        
        return {
            "engine": self.name,
            "language": language,
            "texts": texts,
            "confidences": confidences,
            "bboxes": bboxes,
            "text_count": len(texts)
        }

class OCRManager:
    """Main OCR Manager to handle multiple engines"""
    
    def __init__(self):
        self.engines = {}
        self._register_engines()
    
    def _register_engines(self):
        """Register available OCR engines"""
        # Always register ONNX PaddleOCR (your existing implementation)
        try:
            self.engines["onnx-paddle"] = ONNXPaddleOCREngine()
        except Exception as e:
            print(f"Warning: Could not load ONNX PaddleOCR: {e}")
        
        # Try to register PaddleOCR
        try:
            self.engines["paddleocr"] = PaddleOCREngine()
        except Exception as e:
            print(f"Warning: Could not load PaddleOCR: {e}")
        
        # Try to register EasyOCR
        try:
            self.engines["easyocr"] = EasyOCREngine()
        except Exception as e:
            print(f"Warning: Could not load EasyOCR: {e}")
        
        # Try to register Tesseract
        try:
            self.engines["tesseract"] = TesseractEngine()
        except Exception as e:
            print(f"Warning: Could not load Tesseract: {e}")
    
    def get_available_engines(self) -> List[str]:
        """Get list of available OCR engines"""
        return list(self.engines.keys())
    
    def get_engine_info(self, engine_name: str) -> Dict[str, Any]:
        """Get information about specific engine"""
        if engine_name not in self.engines:
            raise ValueError(f"Engine {engine_name} not available")
        
        engine = self.engines[engine_name]
        return {
            "name": engine.name,
            "supported_languages": engine.get_supported_languages(),
            "engine_key": engine_name
        }
    
    def get_all_engines_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available engines"""
        return {
            engine_name: self.get_engine_info(engine_name)
            for engine_name in self.engines.keys()
        }
    
    def process_image(
        self, 
        image_input: Union[str, np.ndarray], 
        engine: str = "auto", 
        language: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process image with specified OCR engine
        
        Args:
            image_input: Image file path or numpy array
            engine: OCR engine to use ("auto", "paddleocr", "easyocr", "tesseract", "onnx-paddle")
            language: Language for OCR ("auto", "vietnamese", "english", "chinese", etc.)
        """
        # Auto-select engine if not specified
        if engine == "auto":
            engine = self._auto_select_engine(language)
        
        if engine not in self.engines:
            raise ValueError(f"Engine {engine} not available. Available: {list(self.engines.keys())}")
        
        ocr_engine = self.engines[engine]
        
        # Check if language is supported
        if not ocr_engine.is_language_supported(language):
            print(f"Warning: Language {language} not supported by {engine}, using auto detection")
            language = "auto"
        
        # Process based on input type
        if isinstance(image_input, str):
            result = ocr_engine.process_image(image_input, language)
        elif isinstance(image_input, np.ndarray):
            result = ocr_engine.process_image_array(image_input, language)
        else:
            raise ValueError("image_input must be file path (str) or numpy array")
        
        # Add metadata
        result["engine_used"] = engine
        result["language_requested"] = language
        
        return result
    
    def _auto_select_engine(self, language: str = "auto") -> str:
        """Auto-select best engine based on language and availability"""
        # Priority order for different languages
        language_engine_priority = {
            "vietnamese": ["paddleocr", "easyocr", "onnx-paddle", "tesseract"],
            "english": ["easyocr", "paddleocr", "tesseract", "onnx-paddle"],
            "chinese": ["paddleocr", "easyocr", "onnx-paddle", "tesseract"],
            "auto": ["paddleocr", "onnx-paddle", "easyocr", "tesseract"]
        }
        
        priority_list = language_engine_priority.get(language, language_engine_priority["auto"])
        
        # Return first available engine from priority list
        for engine_name in priority_list:
            if engine_name in self.engines:
                return engine_name
        
        # Fallback to any available engine
        if self.engines:
            return list(self.engines.keys())[0]
        
        raise RuntimeError("No OCR engines available")
    
    def compare_engines(
        self, 
        image_input: Union[str, np.ndarray], 
        engines: List[str] = None,
        language: str = "auto"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare results from multiple OCR engines
        """
        if engines is None:
            engines = list(self.engines.keys())
        
        results = {}
        
        for engine_name in engines:
            if engine_name in self.engines:
                try:
                    result = self.process_image(image_input, engine_name, language)
                    results[engine_name] = result
                except Exception as e:
                    results[engine_name] = {
                        "error": str(e),
                        "engine": engine_name,
                        "status": "failed"
                    }
        
        return results

# Global OCR manager instance
ocr_manager = OCRManager()
