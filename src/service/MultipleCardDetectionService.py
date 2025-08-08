import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
import unicodedata
from src.service.YOLODetector import YOLODetector
from src.service.TextExtractor import TextExtractor
from src.service.CardConfigService import card_config_service

class MultipleCardDetectionService:
    """Service for detecting multiple Vietnam Citizens Cards in a single image"""
    
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
    
    def analyze_title_text(self, extracted_text: str, bbox_id: int, filename: str) -> Optional[str]:
        """Analyze title text to determine card type"""
        print(f"[{filename}] Box {bbox_id} - Extracted title text: {extracted_text}")
        
        # Define keywords and thresholds
        cccd_new_keywords = ["CAN CUOC", "CĂN CƯỚC", "CĂN CƯỚC"]
        cccd_old_keywords = ["CONG DAN", "CÔNG DÂN", "CỘNG DÂN", "CUOC CONG DAN", "CĂN CƯỚC CÔNG DÂN"]
        threshold = 0.6
        
        # Normalize extracted text for analysis
        normalized_extracted = self.normalize_text(extracted_text)
        print(f"[{filename}] Box {bbox_id} - Normalized title text: {normalized_extracted}")
        
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
                    print(f"[{filename}] Box {bbox_id} - OLD CCCD keyword found: '{keyword}' in '{extracted_text}' (similarity: {similarity:.2f})")
        
        # Priority 2: Check for NEW CCCD keywords only if no old keyword found
        if not detected_type:
            for keyword in cccd_new_keywords:
                if self.contains_keyword(extracted_text, keyword):
                    similarity = self.text_similarity(extracted_text, keyword)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        detected_type = "cccd_new_front"
                        matched_keyword = keyword
                        print(f"[{filename}] Box {bbox_id} - NEW CCCD keyword found: '{keyword}' in '{extracted_text}' (similarity: {similarity:.2f})")
        
        # Fallback: Use similarity matching if no contains match
        if not detected_type and best_similarity < threshold:
            print(f"[{filename}] Box {bbox_id} - No direct keyword match, trying similarity matching...")
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
            print(f"[{filename}] Box {bbox_id} - Text classification: {detected_type} (matched: '{matched_keyword}', similarity: {best_similarity:.2f})")
            return detected_type
        else:
            print(f"[{filename}] Box {bbox_id} - No text classification found")
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
    
    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def is_contained(self, inner_box: List[int], outer_box: List[int], threshold: float = 0.8) -> bool:
        """Check if inner_box is contained within outer_box"""
        x1_i, y1_i, x2_i, y2_i = inner_box
        x1_o, y1_o, x2_o, y2_o = outer_box
        
        # Calculate area of inner box
        inner_area = (x2_i - x1_i) * (y2_i - y1_i)
        if inner_area <= 0:
            return False
        
        # Calculate intersection
        x1_int = max(x1_i, x1_o)
        y1_int = max(y1_i, y1_o)
        x2_int = min(x2_i, x2_o)
        y2_int = min(y2_i, y2_o)
        
        if x2_int <= x1_int or y2_int <= y1_int:
            return False
        
        intersection = (x2_int - x1_int) * (y2_int - y1_int)
        
        # Check if most of inner box is contained in outer box
        containment_ratio = intersection / inner_area
        return containment_ratio >= threshold
    
    def group_detections_by_location(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[List[Dict]]:
        """Group detections that are in the same location (same card) including title within card"""
        print(f"🔍 Grouping {len(detections)} detections by location...")
        
        try:
            groups = []
            used = set()

            for i, detection in enumerate(detections):
                if i in used:
                    continue
                    
                print(f"  Processing detection {i}: {detection.get('class_name', 'unknown')} (confidence: {detection.get('confidence', 0.0):.3f})")
                
                group = [detection]
                used.add(i)
                bbox1 = detection.get("bbox", [0, 0, 0, 0])
                
                if not bbox1 or len(bbox1) != 4:
                    print(f"    ⚠️ Invalid bbox for detection {i}: {bbox1}")
                    continue
                
                for j, other_detection in enumerate(detections):
                    if j in used:
                        continue
                        
                    bbox2 = other_detection.get("bbox", [0, 0, 0, 0])
                    
                    if not bbox2 or len(bbox2) != 4:
                        print(f"    ⚠️ Invalid bbox for detection {j}: {bbox2}")
                        continue
                    
                    try:
                        # Check IoU for overlapping detections
                        iou = self.calculate_iou(bbox1, bbox2)
                        
                        # Check if one bbox is contained within another (title inside card)
                        is_contained_relationship = False
                        
                        # Case 1: Title inside card bbox
                        if (detection.get('class_name', '').lower() in ['cccd_qr_front', 'cccd_qr_back'] and 
                            other_detection.get('class_name', '').lower() == 'title'):
                            is_contained_relationship = self.is_contained(bbox2, bbox1, threshold=0.7)
                            if is_contained_relationship:
                                print(f"📍 Title bbox contained in {detection.get('class_name', '')} bbox")
                        
                        # Case 2: Card inside title bbox (less common but check anyway)
                        elif (other_detection.get('class_name', '').lower() in ['cccd_qr_front', 'cccd_qr_back'] and 
                              detection.get('class_name', '').lower() == 'title'):
                            is_contained_relationship = self.is_contained(bbox1, bbox2, threshold=0.7)
                            if is_contained_relationship:
                                print(f"📍 {other_detection.get('class_name', '')} bbox contained in title bbox")
                        
                        # Group if high IoU overlap OR containment relationship
                        if iou > iou_threshold or is_contained_relationship:
                            group.append(other_detection)
                            used.add(j)
                            if is_contained_relationship:
                                print(f"🔗 Grouped {detection.get('class_name', '')} + {other_detection.get('class_name', '')} due to containment")
                            elif iou > iou_threshold:
                                print(f"🔗 Grouped {detection.get('class_name', '')} + {other_detection.get('class_name', '')} due to IoU: {iou:.3f}")
                    
                    except Exception as grouping_error:
                        print(f"❌ Error processing detections {i} and {j}: {str(grouping_error)}")
                        print(f"   Detection {i}: {detection}")
                        print(f"   Detection {j}: {other_detection}")
                        continue
                
                groups.append(group)
                print(f"  📦 Created group {len(groups)-1} with {len(group)} detections")
            
            print(f"✅ Grouping completed: {len(groups)} groups created")
            return groups
            
        except Exception as e:
            print(f"❌ Error in group_detections_by_location: {str(e)}")
            print(f"   Input detections: {detections}")
            raise e
    
    def process_detection_group(self, group: List[Dict], image: np.ndarray, group_id: int, 
                              ocr_detections: List[Dict], filename: str) -> Dict:
        """Process a group of detections that belong to the same card"""
        print(f"[{filename}] Processing group {group_id} with {len(group)} detections")
        
        # Find the highest confidence detection in the group
        group.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        primary_detection = group[0]
        
        # Get bounding box for cropping OCR features
        bbox = primary_detection.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        
        # Analyze OCR features within this card region
        card_ocr_types = set()
        for ocr_detection in ocr_detections:
            ocr_bbox = ocr_detection.get("bbox", [0, 0, 0, 0])
            if self.calculate_iou(bbox, ocr_bbox) > 0.1:  # OCR overlaps with card
                class_name = ocr_detection.get("class_name")
                if class_name:
                    card_ocr_types.add(class_name.lower())
        
        print(f"[{filename}] Group {group_id} - OCR features: {card_ocr_types}")
        
        # Process each detection in the group
        processed_detections = []
        title_detected_type = None
        
        # First pass: Extract title information
        for detection in group:
            class_name = detection.get("class_name")
            if class_name and class_name.lower() == "title":
                x1, y1, x2, y2 = detection.get("bbox", [0, 0, 0, 0])
                cropped_image = image[y1:y2, x1:x2]
                extracted_text = self.text_extractor.extract_from_image_en(cropped_image)
                
                detected_type = self.analyze_title_text(extracted_text, group_id, filename)
                if detected_type:
                    title_detected_type = detected_type
        
        # Second pass: Process non-title detections
        for detection in group:
            class_name = detection.get("class_name")
            confidence = detection.get("confidence", 0.0)
            detection_bbox = detection.get("bbox", [0, 0, 0, 0])
            
            if class_name:
                label = class_name.lower()
                
                # Skip title detections - they are only used for enhancement
                if label == "title":
                    continue
                
                # Get card information
                card_category, card_type = self.get_card_info(label)
                if label not in ["cccd_qr_front", "cccd_qr_back", "cccd_new_front", "cccd_new_back", "gplx_front", "gplx_back"]:
                    label = "unknown"
                    
            else:
                card_category, card_type = self.get_card_info("unknown")
                label = "unknown"
                confidence = 0.0
                detection_bbox = [0, 0, 0, 0]
            
            # Create detection result
            detection_result = {
                "confidence": confidence,
                "detected_label": label,
                "bbox": detection_bbox,
                "card_category": card_category,
                "card_type": card_type,
                "is_valid_card": label in ["cccd_qr_front", "cccd_qr_back", "cccd_new_front", "cccd_new_back", "gplx_front", "gplx_back"],
                "title_detected_type": title_detected_type
            }
            
            processed_detections.append(detection_result)
        
        # Handle case where no card detections found, only title
        if not processed_detections and title_detected_type:
            print(f"[{filename}] Group {group_id} - No card detections, creating from title: {title_detected_type}")
            # Find the title detection to use its bbox and confidence
            title_detection = None
            for detection in group:
                if detection.get("class_name", "").lower() == "title":
                    title_detection = detection
                    break
            
            if title_detection:
                card_category, card_type = self.get_card_info(title_detected_type)
                title_result = {
                    "confidence": title_detection.get("confidence", 0.0),
                    "detected_label": title_detected_type,
                    "bbox": title_detection.get("bbox", [0, 0, 0, 0]),
                    "card_category": card_category,
                    "card_type": card_type,
                    "is_valid_card": title_detected_type in ["cccd_qr_front", "cccd_qr_back", "cccd_new_front", "cccd_new_back", "gplx_front", "gplx_back"],
                    "title_detected_type": title_detected_type
                }
                processed_detections.append(title_result)
        
        # Apply selection logic for this group (similar to single card detection)
        if processed_detections:
            # PRIORITY 1: Title detection has absolute priority
            if title_detected_type and title_detected_type != "unknown":
                print(f"[{filename}] Group {group_id} - Title detected type: {title_detected_type} - ABSOLUTE PRIORITY")
                
                # Find exact match with title_detected_type
                best_detection = None
                for det in processed_detections:
                    if det["detected_label"] == title_detected_type:
                        best_detection = det
                        print(f"[{filename}] Group {group_id} - TITLE PRIORITY: Found exact match {title_detected_type}")
                        break
                
                if not best_detection:
                    # Look for compatible types
                    if title_detected_type in ["cccd_new_front", "cccd_qr_front"]:
                        for det in processed_detections:
                            if det["detected_label"] in ["cccd_new_front", "cccd_qr_front"]:
                                best_detection = det
                                print(f"[{filename}] Group {group_id} - TITLE PRIORITY (compatible): Selected {det['detected_label']}")
                                break
                    
                    if not best_detection:
                        # Fallback to highest confidence
                        processed_detections.sort(key=lambda x: x["confidence"], reverse=True)
                        best_detection = processed_detections[0]
                        print(f"[{filename}] Group {group_id} - WARNING: Title type not found, fallback to confidence")
            
            # PRIORITY 2: Use confidence and OCR rules
            else:
                processed_detections.sort(key=lambda x: x["confidence"], reverse=True)
                best_detection = processed_detections[0]
                
                # Apply OCR-based rules if multiple detections with similar confidence
                if len(processed_detections) > 1:
                    first = processed_detections[0]
                    second = processed_detections[1]
                    confidence_diff = first["confidence"] - second["confidence"]
                    
                    if confidence_diff < 0.1:
                        print(f"[{filename}] Group {group_id} - Applying OCR-based rules...")
                        
                        has_portrait = "portrait" in card_ocr_types
                        has_qr_code = "qr_code" in card_ocr_types
                        has_basic_info = any(info in card_ocr_types for info in ["name", "id", "birth", "sex"])
                        
                        # Apply OCR rules
                        if has_qr_code:
                            for det in processed_detections:
                                if det["detected_label"] in ["cccd_qr_front", "cccd_qr_back"]:
                                    best_detection = det
                                    print(f"[{filename}] Group {group_id} - OCR Rule: QR detected → {det['detected_label']}")
                                    break
                        elif has_portrait and has_basic_info and not has_qr_code:
                            for det in processed_detections:
                                if det["detected_label"] in ["cccd_new_front", "cccd_new_back"]:
                                    best_detection = det
                                    print(f"[{filename}] Group {group_id} - OCR Rule: Portrait+Basic no QR → {det['detected_label']}")
                                    break
            
            # Add OCR features and group info to the best detection
            best_detection["ocr_features"] = {
                "has_portrait": "portrait" in card_ocr_types,
                "has_qr_code": "qr_code" in card_ocr_types,
                "has_basic_info": any(info in card_ocr_types for info in ["name", "id", "birth", "sex"]),
                "has_address_info": any(info in card_ocr_types for info in ["place_of_origin", "place_of_residence"]),
                "detected_info_types": list(card_ocr_types)
            }
            best_detection["group_id"] = group_id
            best_detection["group_size"] = len(group)
            
            print(f"[{filename}] Group {group_id} - Final result: {best_detection['detected_label']} (confidence: {best_detection['confidence']:.3f})")
            
            return best_detection
        
        return None
    
    def detect_multiple_cards(self, image: np.ndarray, detections: List[Dict], 
                            ocr_detections: List[Dict], filename: str) -> List[Dict]:
        """Detect multiple cards in a single image"""
        print(f"[{filename}] Processing {len(detections)} detections for multiple card detection")
        
        try:
            if not detections:
                return []
            
            # Debug: Print first detection structure
            if detections:
                print(f"🔍 Sample detection structure: {detections[0]}")
                print(f"🔍 Detection keys: {list(detections[0].keys())}")
            
            # Group detections by location (same physical card)
            detection_groups = self.group_detections_by_location(detections)
            
            print(f"[{filename}] Found {len(detection_groups)} card groups")
            
            # Process each group
            results = []
            for group_id, group in enumerate(detection_groups):
                try:
                    result = self.process_detection_group(group, image, group_id, ocr_detections, filename)
                    if result:
                        results.append(result)
                except Exception as group_error:
                    print(f"❌ Error processing group {group_id}: {str(group_error)}")
                    print(f"   Group data: {group}")
                    continue
            
            # Sort results by confidence for consistent ordering
            results.sort(key=lambda x: x["confidence"], reverse=True)
            
            print(f"[{filename}] Multiple card detection completed: {len(results)} cards found")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in detect_multiple_cards: {str(e)}")
            print(f"   Input data - Detections: {len(detections)}, OCR: {len(ocr_detections)}")
            if detections:
                print(f"   First detection: {detections[0]}")
            raise e

# Create singleton instance
multiple_card_detection_service = MultipleCardDetectionService()
