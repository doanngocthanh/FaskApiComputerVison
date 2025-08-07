import os
import cv2
import numpy as np
import re
from typing import Optional, List, Dict, Any
from src.service.YOLODetector import YOLODetector
from src.service.PaddleOCR import PaddleOCR
from src.service.EasyOCRManager import EasyOCRManager
class  TextExtractor:
    def __init__(self):
        
        self.ocr = PaddleOCR()
    def extract_from_image_en(self, image: np.ndarray) -> str:
        """Extract only text from an image using OCR."""
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array representing an image.")
        
        # Perform OCR on the image
        ocr_result = self.ocr.process_pipeline(image, steps=['detect', 'classify', 'recognize'])
        
        # Extract text from the result
        if isinstance(ocr_result, dict) and 'texts' in ocr_result:
            texts = ocr_result['texts']
            if isinstance(texts, tuple) and len(texts) > 0:
                # Get the first element which contains the text list
                text_list = texts[0] if isinstance(texts[0], list) else []
                return ' '.join(text_list)
            elif isinstance(texts, list):
                return ' '.join(texts)
        
        return ""
    
    def extract_from_image_vi(self, image: np.ndarray) -> str:
        """Extract text from an image using EasyOCR."""
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array representing an image.")
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Initialize EasyOCR manager
        easy_ocr_manager = EasyOCRManager()
        
        # Perform OCR on the image
        ocr_result = easy_ocr_manager.extract_text(image, languages=["vi"], preprocess=True)
          # Extract text from the result
        print(ocr_result)
        if ocr_result["success"] and ocr_result.get("texts"):
            return ' '.join(ocr_result["texts"])
        
        return ""