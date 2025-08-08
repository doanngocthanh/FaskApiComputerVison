import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
import unicodedata
from src.service.YOLODetector import YOLODetector
from src.service.TextExtractor import TextExtractor
from src.service.CardConfigService import card_config_service

class CardDetectionService:
    """Service for Vietnam Citizens Card Detection with advanced text analysis"""
    
    def __init__(self):
        self.text_extractor = TextExtractor()
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for better comparison"""
        # Remove accents/diacritics
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        # Upper case and remove spaces/special chars
        text = ''.join(char for char in text.upper() if char.isalnum())
        return text
    
    def text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity ratio"""
        return SequenceMatcher(None, text1.upper(), text2.upper()).ratio()
    
    def contains_keyword(self, text: str, keyword: str) -> bool:
        """Check if normalized text contains normalized keyword"""
        normalized_text = self.normalize_text(text)
        normalized_keyword = self.normalize_text(keyword)
        return normalized_keyword in normalized_text
    
    def analyze_title_text(self, extracted_text: str, filename: str) -> Optional[str]:
        """Analyze title text to determine card type"""
        print(f"[{filename}] Extracted title text: {extracted_text}")
        
        # Define keywords and thresholds
        cccd_new_keywords = ["CAN CUOC", "CĂN CƯỚC", "CĂN CƯỚC"]
        cccd_old_keywords = ["CONG DAN", "CÔNG DÂN", "CỘNG DÂN", "CUOC CONG DAN", "CĂN CƯỚC CÔNG DÂN"]
        threshold = 0.6  # Lower threshold since we're using contains_keyword
        
        # Normalize extracted text for analysis
        normalized_extracted = self.normalize_text(extracted_text)
        print(f"[{filename}] Normalized title text: {normalized_extracted}")
        
        # Check for CCCD old keywords first (higher priority for old format)
        best_similarity = 0
        detected_type = None
        matched_keyword = None
        
        # Priority 1: Check for OLD CCCD keywords using contains method
        for keyword in cccd_old_keywords:
            if self.contains_keyword(extracted_text, keyword):
                similarity = self.text_similarity(extracted_text, keyword)
                if similarity > best_similarity:
                    best_similarity = similarity
                    detected_type = "cccd_qr_front"
                    matched_keyword = keyword
                    print(f"[{filename}] OLD CCCD keyword found: '{keyword}' in '{extracted_text}' (similarity: {similarity:.2f})")
        
        # Priority 2: Check for NEW CCCD keywords only if no old keyword found
        if not detected_type:
            for keyword in cccd_new_keywords:
                if self.contains_keyword(extracted_text, keyword):
                    similarity = self.text_similarity(extracted_text, keyword)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        detected_type = "cccd_new_front"
                        matched_keyword = keyword
                        print(f"[{filename}] NEW CCCD keyword found: '{keyword}' in '{extracted_text}' (similarity: {similarity:.2f})")
        
        # Fallback: Use similarity matching if no contains match
        if not detected_type and best_similarity < threshold:
            print(f"[{filename}] No direct keyword match, trying similarity matching...")
            for keyword in cccd_old_keywords:
                similarity = self.text_similarity(extracted_text, keyword)
                if similarity > threshold and similarity > best_similarity:
                    best_similarity = similarity
                    detected_type = "cccd_qr_front"
                    matched_keyword = keyword
            
            for keyword in cccd_new_keywords:
                similarity = self.text_similarity(extracted_text, keyword)
                if similarity > threshold and similarity > best_similarity:
                    best_similarity = similarity
                    detected_type = "cccd_new_front"
                    matched_keyword = keyword
        
        if detected_type:
            print(f"[{filename}] Text classification: {detected_type} (matched: '{matched_keyword}', similarity: {best_similarity:.2f})")
            return detected_type
        else:
            print(f"[{filename}] No text classification found - will use confidence-based decision")
            return None
    
    def get_card_info(self, label: str) -> tuple:
        """Get card category and type based on label"""
        if label == "cccd_qr_front":
            card_category = card_config_service.get_card_category_by_id(0)
            card_type = card_config_service.get_card_type_by_id(0)
        elif label == "cccd_qr_back":
            card_category = card_config_service.get_card_category_by_id(0)
            card_type = card_config_service.get_card_type_by_id(1)
        elif label == "cccd_new_front":
            card_category = card_config_service.get_card_category_by_id(5)
            card_type = card_config_service.get_card_type_by_id(0)
        elif label == "cccd_new_back":
            card_category = card_config_service.get_card_category_by_id(5)
            card_type = card_config_service.get_card_type_by_id(1)
        elif label == "gplx_front":
            card_category = card_config_service.get_card_category_by_id(1)
            card_type = card_config_service.get_card_type_by_id(0)
        elif label == "gplx_back":
            card_category = card_config_service.get_card_category_by_id(1)
            card_type = card_config_service.get_card_type_by_id(1)
        else:
            card_category = {"id": -1, "name": "Unknown", "nameEn": "Unknown"}
            card_type = {"id": -1, "name": "Unknown", "nameEn": "Unknown"}
        
        return card_category, card_type
    
    def process_detections(self, detections: List[Dict], image: np.ndarray, filename: str) -> tuple:
        """Process YOLO detections and extract title information"""
        file_detections = []
        title_detected_type = None
        
        for detection in detections:
            detection_keys = list(detection.keys())
            print(f"[{filename}] Detection keys: {detection_keys}")
            
            class_name = detection.get("class_name")
            confidence = detection.get("confidence", 0.0)
            
            if class_name:
                print(f"[{filename}] Detected class name: {class_name}, confidence: {confidence}")
                label = class_name.lower()
                
                # Handle title detection with text analysis
                if label == "title":
                    x1, y1, x2, y2 = detection.get("bbox", [0, 0, 0, 0])
                    cropped_image = image[y1:y2, x1:x2]
                    extracted_text = self.text_extractor.extract_from_image_en(cropped_image)
                    
                    detected_type = self.analyze_title_text(extracted_text, filename)
                    if detected_type:
                        label = detected_type
                        title_detected_type = detected_type
                
                # Get card information
                card_category, card_type = self.get_card_info(label)
                if label not in ["cccd_qr_front", "cccd_qr_back", "cccd_new_front", "cccd_new_back", "gplx_front", "gplx_back"]:
                    label = "unknown"
                    
            else:
                card_category, card_type = self.get_card_info("unknown")
                label = "unknown"
                confidence = 0.0
            
            # Create detection result
            detection_result = {
                "confidence": confidence,
                "detected_label": label,
                "card_category": card_category,
                "card_type": card_type,
                "is_valid_card": label in ["cccd_qr_front", "cccd_qr_back", "cccd_new_front", "cccd_new_back", "gplx_front", "gplx_back"],
                "title_detected_type": title_detected_type
            }
            
            file_detections.append(detection_result)
        
        return file_detections, title_detected_type
    
    def apply_ocr_rules(self, file_detections: List[Dict], has_qr_code: bool, has_portrait: bool, 
                       has_basic_info: bool, filename: str) -> Dict:
        """Apply OCR-based classification rules"""
        print(f"[{filename}] Applying OCR-based classification rules...")
        
        # Rule 1: QR code detected → Prefer old CCCD
        if has_qr_code:
            for det in file_detections:
                if det["detected_label"] in ["cccd_qr_front", "cccd_qr_back"]:
                    print(f"[{filename}] OCR Rule: QR detected → Switched to QR CCCD: {det['detected_label']}")
                    return det
        
        # Rule 2: Portrait + basic info but no QR → Prefer new CCCD or GPLX
        elif has_portrait and has_basic_info and not has_qr_code:
            for det in file_detections:
                if det["detected_label"] in ["cccd_new_front", "cccd_new_back"]:
                    print(f"[{filename}] OCR Rule: Portrait+Basic but no QR → Switched to New CCCD: {det['detected_label']}")
                    return det
            else:
                for det in file_detections:
                    if det["detected_label"] in ["gplx_front", "gplx_back"]:
                        print(f"[{filename}] OCR Rule: Portrait+Basic but no QR → Switched to GPLX: {det['detected_label']}")
                        return det
        
        # Rule 3: Basic info only → Possibly GPLX back
        elif has_basic_info and not has_portrait and not has_qr_code:
            for det in file_detections:
                if det["detected_label"] == "gplx_back":
                    print(f"[{filename}] OCR Rule: Basic info only → Switched to GPLX back: {det['detected_label']}")
                    return det
        
        # Rule 4: QR only → Back side
        elif has_qr_code and not has_portrait:
            for det in file_detections:
                if "back" in det["detected_label"]:
                    print(f"[{filename}] OCR Rule: QR only → Switched to back side: {det['detected_label']}")
                    return det
        
        # Rule 5: Fallback → Prefer old CCCD
        else:
            for det in file_detections:
                if det["detected_label"] in ["cccd_qr_front", "cccd_qr_back"]:
                    print(f"[{filename}] OCR Rule: Fallback → Switched to QR CCCD: {det['detected_label']}")
                    return det
        
        # No rule applied, return highest confidence
        return file_detections[0]
    
    def select_best_detection(self, file_detections: List[Dict], title_detected_type: Optional[str],
                            detected_info_types: set, filename: str) -> Dict:
        """Select the best detection based on title priority and OCR rules"""
        if not file_detections:
            return None
        
        print(f"[{filename}] All detections found:")
        for i, det in enumerate(file_detections):
            print(f"  {i+1}: {det['detected_label']} - confidence: {det['confidence']:.3f}")
        
        # PRIORITY 1: Title detection has absolute priority over confidence (only if detected)
        if title_detected_type and title_detected_type != "unknown":
            print(f"[{filename}] Title detected type: {title_detected_type} - ABSOLUTE PRIORITY over confidence")
            
            # First, try to find exact match with title_detected_type
            title_detection = None
            for det in file_detections:
                if det["detected_label"] == title_detected_type:
                    title_detection = det
                    print(f"[{filename}] TITLE PRIORITY: Found exact match {title_detected_type} with confidence {det['confidence']:.3f}")
                    break
            
            if title_detection:
                best_detection = title_detection
            else:
                # If exact match not found, look for compatible types but still prioritize title logic
                print(f"[{filename}] TITLE PRIORITY: Exact match not found, looking for compatible types")
                if title_detected_type in ["cccd_new_front", "cccd_qr_front"]:
                    for det in file_detections:
                        if det["detected_label"] in ["cccd_new_front", "cccd_qr_front"]:
                            best_detection = det
                            print(f"[{filename}] TITLE PRIORITY (compatible): Selected {det['detected_label']} for title type {title_detected_type}")
                            break
                    else:
                        # If no compatible found, take highest confidence but log the conflict
                        file_detections.sort(key=lambda x: x["confidence"], reverse=True)
                        best_detection = file_detections[0]
                        print(f"[{filename}] WARNING: Title type {title_detected_type} not found, fallback to highest confidence: {best_detection['detected_label']}")
                else:
                    # For other title types, fallback to confidence
                    file_detections.sort(key=lambda x: x["confidence"], reverse=True)
                    best_detection = file_detections[0]
                    print(f"[{filename}] TITLE PRIORITY: Unsupported title type {title_detected_type}, fallback to confidence: {best_detection['detected_label']}")
        
        # PRIORITY 2: No title detection or title_detected_type is unknown - use confidence and OCR rules
        else:
            if not title_detected_type:
                print(f"[{filename}] No title detection found - using confidence-based decision")
            else:
                print(f"[{filename}] Title detection unknown - fallback to confidence-based decision")
                
            file_detections.sort(key=lambda x: x["confidence"], reverse=True)
            best_detection = file_detections[0]
            
            if len(file_detections) > 1:
                first = file_detections[0]
                second = file_detections[1]
                confidence_diff = first["confidence"] - second["confidence"]
                
                print(f"[{filename}] Confidence difference: {confidence_diff:.3f}")
                
                # Apply OCR-based rules if confidence difference is small
                if confidence_diff < 0.1:
                    # Analyze OCR features
                    has_portrait = "portrait" in detected_info_types
                    has_qr_code = "qr_code" in detected_info_types
                    has_basic_info = any(info in detected_info_types for info in ["name", "id", "birth", "sex"])
                    
                    best_detection = self.apply_ocr_rules(file_detections, has_qr_code, has_portrait, has_basic_info, filename)
        
        print(f"[{filename}] Final best detection: {best_detection['detected_label']} with confidence {best_detection['confidence']:.3f}")
        if title_detected_type and title_detected_type != "unknown":
            print(f"[{filename}] Title detection influence: {title_detected_type}")
        else:
            print(f"[{filename}] Decision based on: confidence + OCR rules")
        
        return best_detection

# Create singleton instance
card_detection_service = CardDetectionService()
