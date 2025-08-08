import os
import cv2
import numpy as np
import re
from typing import Optional, List, Dict, Any
from src.service.YOLODetector import YOLODetector
from src.service.EasyOCRManager import EasyOCRManager
from src.service.PaddleOCR import PaddleOCR

class TextExtractor:
    def __init__(self):
        # Simple and stable: PaddleOCR ONNX for preprocessing + EasyOCR for text recognition
        self.easy_ocr_manager = EasyOCRManager()
        self.paddle_ocr = PaddleOCR()
        
        print("🔧 TextExtractor initialized with PaddleOCR ONNX (detection) + EasyOCR (recognition)")
        print("📝 Simple and stable OCR solution")
    
    def extract_from_image_en(self, image: np.ndarray) -> str:
        """Extract English text using PaddleOCR ONNX detection + EasyOCR recognition."""
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array representing an image.")
        
        # Method 1: Use PaddleOCR ONNX for detection + EasyOCR for recognition
        try:
            # Step 1: Use PaddleOCR to detect text regions
            text_regions = self.paddle_ocr.detect_text_regions(image)
            
            if text_regions and len(text_regions) > 0:
                print(f"    📍 PaddleOCR detected {len(text_regions)} text regions")
                
                # Step 2: Use PaddleOCR to classify orientation
                cropped_images, angles = self.paddle_ocr.classify_text_orientation(image, text_regions)
                
                # Step 3: Use EasyOCR to recognize text from prepared regions
                extracted_texts = []
                for i, cropped_img in enumerate(cropped_images):
                    try:
                        ocr_result = self.easy_ocr_manager.extract_text(cropped_img, languages=["en"], preprocess=False)
                        if ocr_result["success"] and ocr_result.get("texts"):
                            extracted_texts.extend(ocr_result["texts"])
                    except Exception as e:
                        print(f"    ⚠️ EasyOCR recognition error for region {i}: {str(e)}")
                        continue
                
                if extracted_texts:
                    result_text = ' '.join(extracted_texts).strip()
                    print(f"    ✅ Hybrid result: '{result_text[:50]}...'")
                    return result_text
                    
        except Exception as hybrid_error:
            print(f"    ⚠️ Hybrid approach failed: {str(hybrid_error)}, using direct EasyOCR")
        
        # Method 2: Fallback to direct EasyOCR
        try:
            ocr_result = self.easy_ocr_manager.extract_text(image, languages=["en"])
            
            if ocr_result["success"] and ocr_result.get("texts"):
                return ' '.join(ocr_result["texts"])
                
        except Exception as easy_error:
            print(f"    ❌ EasyOCR English error: {str(easy_error)}")
        
        return ""
    
    def extract_from_image_vi(self, image: np.ndarray) -> str:
        """Extract Vietnamese text using PaddleOCR + EasyOCR."""
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array representing an image.")
        if image is None:
            raise ValueError(f"Could not load image")
        
        # Method 1: Use PaddleOCR ONNX for detection + EasyOCR for recognition
        best_result = ""
        try:
            # Step 1: Use PaddleOCR to detect text regions
            text_regions = self.paddle_ocr.detect_text_regions(image)
            
            if text_regions and len(text_regions) > 0:
                print(f"    📍 PaddleOCR detected {len(text_regions)} text regions")
                
                # Step 2: Use PaddleOCR to classify orientation
                cropped_images, angles = self.paddle_ocr.classify_text_orientation(image, text_regions)
                
                # Step 3: Use EasyOCR to recognize Vietnamese text from prepared regions
                extracted_texts = []
                for i, cropped_img in enumerate(cropped_images):
                    try:
                        ocr_result = self.easy_ocr_manager.extract_text(cropped_img, languages=["vi"], preprocess=False)
                        if ocr_result["success"] and ocr_result.get("texts"):
                            extracted_texts.extend(ocr_result["texts"])
                    except Exception as e:
                        print(f"    ⚠️ EasyOCR recognition error for region {i}: {str(e)}")
                        continue
                
                if extracted_texts:
                    best_result = ' '.join(extracted_texts).strip()
                    print(f"    ✅ Hybrid result: '{best_result[:50]}...'")
                    
        except Exception as hybrid_error:
            print(f"    ⚠️ Hybrid approach failed: {str(hybrid_error)}")
            best_result = ""
        
        # Method 2: Fallback to direct EasyOCR
        easyocr_fallback = ""
        if not best_result:
            try:
                ocr_result = self.easy_ocr_manager.extract_text(image, languages=["vi"])
                
                if ocr_result["success"] and ocr_result.get("texts"):
                    easyocr_fallback = ' '.join(ocr_result["texts"])
                    print(f"    📝 EasyOCR fallback result: '{easyocr_fallback[:50]}...'")
                    
            except Exception as easy_error:
                print(f"    ❌ EasyOCR Vietnamese fallback error: {str(easy_error)}")
        
        # Choose the best result
        results = [r for r in [best_result, easyocr_fallback] if r.strip()]
        
        if not results:
            return ""
        
        # Simple heuristic: prefer longer results with Vietnamese characters
        def has_vietnamese_chars(text: str) -> bool:
            vietnamese_chars = 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'
            vietnamese_chars += vietnamese_chars.upper()
            return any(char in vietnamese_chars for char in text)
        
        scored_results = []
        for result in results:
            score = len(result)
            if has_vietnamese_chars(result):
                score += 50
            scored_results.append((score, result))
        
        best_score, final_result = max(scored_results, key=lambda x: x[0])
        print(f"    🏆 Final Vietnamese result (score: {best_score}): '{final_result[:50]}...'")
        
        return final_result
    
    def extract_from_bbox_en(self, original_image: np.ndarray, bbox: List[int]) -> str:
        """
        Extract English text from bbox on original image
        """
        try:
            x1, y1, x2, y2 = bbox
            h, w = original_image.shape[:2]
            
            # Validate và adjust bbox
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            print(f"    🔍 Extracting English text from bbox [{x1}, {y1}, {x2}, {y2}]")
            
            # Method 1: Use PaddleOCR ONNX detection in bbox area + EasyOCR recognition
            try:
                # Crop với padding
                padding = 10
                padded_x1 = max(0, x1 - padding)
                padded_y1 = max(0, y1 - padding)
                padded_x2 = min(w, x2 + padding)
                padded_y2 = min(h, y2 + padding)
                
                padded_region = original_image[padded_y1:padded_y2, padded_x1:padded_x2]
                
                # Use hybrid approach on padded region
                text_regions = self.paddle_ocr.detect_text_regions(padded_region)
                
                if text_regions and len(text_regions) > 0:
                    cropped_images, angles = self.paddle_ocr.classify_text_orientation(padded_region, text_regions)
                    
                    extracted_texts = []
                    for cropped_img in cropped_images:
                        ocr_result = self.easy_ocr_manager.extract_text(cropped_img, languages=["en"], preprocess=False)
                        if ocr_result["success"] and ocr_result.get("texts"):
                            extracted_texts.extend(ocr_result["texts"])
                    
                    if extracted_texts:
                        result_text = ' '.join(extracted_texts).strip()
                        print(f"    ✅ Hybrid bbox result: '{result_text[:50]}...'")
                        return result_text
                        
            except Exception as hybrid_error:
                print(f"    ⚠️ Hybrid bbox approach failed: {str(hybrid_error)}")
            
            # Method 2: Fallback to direct crop
            cropped_region = original_image[y1:y2, x1:x2]
            if cropped_region.size > 0:
                return self.extract_from_image_en(cropped_region)
                
        except Exception as e:
            print(f"    ❌ Error in bbox English extraction: {str(e)}")
        
        return ""
    
    def extract_from_bbox_vi(self, original_image: np.ndarray, bbox: List[int]) -> str:
        """
        Extract Vietnamese text from bbox on original image
        """
        try:
            x1, y1, x2, y2 = bbox
            h, w = original_image.shape[:2]
            
            # Validate và adjust bbox
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            print(f"    🔍 Extracting Vietnamese text from bbox [{x1}, {y1}, {x2}, {y2}]")
            
            # Method 1: Use PaddleOCR ONNX detection + EasyOCR recognition in bbox (fallback)
            best_result = ""
            try:
                padding = 15
                padded_x1 = max(0, x1 - padding)
                padded_y1 = max(0, y1 - padding)
                padded_x2 = min(w, x2 + padding)
                padded_y2 = min(h, y2 + padding)
                
                padded_region = original_image[padded_y1:padded_y2, padded_x1:padded_x2]
                
                text_regions = self.paddle_ocr.detect_text_regions(padded_region)
                
                if text_regions and len(text_regions) > 0:
                    cropped_images, angles = self.paddle_ocr.classify_text_orientation(padded_region, text_regions)
                    
                    extracted_texts = []
                    for cropped_img in cropped_images:
                        ocr_result = self.easy_ocr_manager.extract_text(cropped_img, languages=["vi"], preprocess=False)
                        if ocr_result["success"] and ocr_result.get("texts"):
                            extracted_texts.extend(ocr_result["texts"])
                    
                    if extracted_texts:
                        best_result = ' '.join(extracted_texts).strip()
                        print(f"    ✅ Hybrid Vietnamese bbox result: '{best_result[:50]}...'")
                        
            except Exception as hybrid_error:
                print(f"    ⚠️ Hybrid Vietnamese bbox approach failed: {str(hybrid_error)}")
                best_result = ""
            
            # Method 2: Fallback to basic crop
            fallback_result = ""
            if not best_result:
                try:
                    cropped_region = original_image[y1:y2, x1:x2]
                    if cropped_region.size > 0:
                        fallback_result = self.extract_from_image_vi(cropped_region)
                        
                except Exception as crop_error:
                    print(f"    ⚠️ Basic crop fallback error: {str(crop_error)}")
            
            # Choose the best result
            results = [r for r in [best_result, fallback_result] if r.strip()]
            
            if not results:
                return ""
            
            # Prefer longer results with Vietnamese characters
            def has_vietnamese_chars(text: str) -> bool:
                vietnamese_chars = 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'
                vietnamese_chars += vietnamese_chars.upper()
                return any(char in vietnamese_chars for char in text)
            
            scored_results = []
            for result in results:
                score = len(result)
                if has_vietnamese_chars(result):
                    score += 50
                scored_results.append((score, result))
            
            best_score, final_result = max(scored_results, key=lambda x: x[0])
            print(f"    🏆 Final Vietnamese bbox result (score: {best_score}): '{final_result[:50]}...'")
            
            return final_result
            
        except Exception as e:
            print(f"    ❌ Error in bbox Vietnamese extraction: {str(e)}")
            return ""
