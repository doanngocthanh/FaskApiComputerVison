from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Path, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import os
import cv2
import numpy as np
from io import BytesIO
import importlib
from src.router.api.__init__ import *  
from src.service.YOLODetector import YOLODetector
from src.service.PaddleOCR import PaddleOCR
from src.service.TextExtractor import TextExtractor
from PIL import Image
from config import PtConfig
from src.service.CardConfigService import card_config_service
from src.service.CardDetectionService import card_detection_service
from difflib import SequenceMatcher
import math

# Response Models
class OCRRegion(BaseModel):
    """OCR region detection result"""
    region_type: str = Field(..., description="Type of OCR region (id, name, birth, etc.)")
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Detection confidence")
    extracted_text: str = Field(..., description="Extracted text content")
    text_direction: Optional[str] = Field(None, description="Text direction (horizontal/vertical)")
    rotation_angle: Optional[float] = Field(None, description="Rotation angle applied")

class CardOCRResult(BaseModel):
    """Complete card OCR result"""
    card_type: str = Field(..., description="Detected card type")
    card_confidence: float = Field(..., description="Card detection confidence")
    total_ocr_regions: int = Field(..., description="Number of OCR regions found")
    ocr_regions: List[OCRRegion] = Field(..., description="List of OCR regions")
    processing_info: Dict[str, Any] = Field(..., description="Processing metadata")

class OCRResponse(BaseModel):
    """API response for OCR"""
    success: bool = Field(True, description="Whether OCR was successful")
    message: str = Field("OCR completed successfully", description="Status message")
    result: Optional[CardOCRResult] = Field(None, description="OCR result if successful")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "OCR completed successfully",
                "result": {
                    "card_type": "cccd_qr_front",
                    "card_confidence": 0.95,
                    "total_ocr_regions": 5,
                    "ocr_regions": [
                        {
                            "region_type": "name",
                            "bbox": [100, 150, 300, 180],
                            "confidence": 0.92,
                            "extracted_text": "NGUYỄN VĂN A",
                            "text_direction": "horizontal",
                            "rotation_angle": 0.0
                        }
                    ],
                    "processing_info": {
                        "image_rotated": False,
                        "paddle_ocr_used": True,
                        "text_extractor_used": True
                    }
                }
            }
        }

# Router setup
router = APIRouter(
    prefix="/api/v1/ocr",
    tags=["VietNam Citizens Card Optical Character Recognition"],
    responses={
        404: {"model": ErrorResponse, "description": "Resource not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)

# Helper functions
def detect_and_correct_image_orientation(image: np.ndarray) -> tuple:
    """
    Detect overall image orientation và rotate về đúng hướng nếu cần
    Returns: (rotation_angle, corrected_image)
    """
    try:
        print("🔄 Detecting overall image orientation...")
        
        # Sử dụng PaddleOCR để detect tất cả text regions trong ảnh
        paddle_ocr = PaddleOCR()
        text_regions = paddle_ocr.detect_text_regions(image)
        
        if not text_regions or len(text_regions) < 3:  # Cần ít nhất 3 regions để judge orientation
            print("⚠️ Not enough text regions for orientation detection")
            return 0.0, image
        
        # Detect orientation cho tất cả regions
        try:
            cropped_images, angles = paddle_ocr.classify_text_orientation(image, text_regions)
            
            if not angles:
                return 0.0, image
            
            print(f"📊 Detected angles: {angles}")
            
            # Analyze angles để quyết định rotation
            # Angles thường là: 0 (normal), 90 (rotated right), 180 (upside down), 270 (rotated left)
            angle_counts = {}
            for angle in angles:
                rounded_angle = round(angle / 90) * 90  # Round to nearest 90 degrees
                if rounded_angle >= 360:
                    rounded_angle -= 360
                elif rounded_angle < 0:
                    rounded_angle += 360
                
                angle_counts[rounded_angle] = angle_counts.get(rounded_angle, 0) + 1
            
            # Tìm angle phổ biến nhất
            most_common_angle = max(angle_counts.items(), key=lambda x: x[1])[0]
            
            print(f"🎯 Most common angle: {most_common_angle}° (appears {angle_counts[most_common_angle]} times)")
            
            # Rotate image if needed
            if most_common_angle != 0:
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                
                # PaddleOCR angles cần convert: 
                # 90 degree text = rotate image -90 to correct it
                rotation_angle = -most_common_angle  
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                
                # Calculate new dimensions
                cos_angle = abs(rotation_matrix[0, 0])
                sin_angle = abs(rotation_matrix[0, 1])
                new_w = int((h * sin_angle) + (w * cos_angle))
                new_h = int((h * cos_angle) + (w * sin_angle))
                
                # Adjust rotation matrix for new center
                rotation_matrix[0, 2] += (new_w / 2) - center[0]
                rotation_matrix[1, 2] += (new_h / 2) - center[1]
                
                corrected_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, 
                                                borderValue=(255, 255, 255))
                
                print(f"✅ Image rotated by {rotation_angle}° (from {image.shape} to {corrected_image.shape})")
                return rotation_angle, corrected_image
            else:
                print("✅ Image orientation is correct")
                return 0.0, image
                
        except Exception as orientation_error:
            print(f"❌ Error in orientation detection: {str(orientation_error)}")
            return 0.0, image
            
    except Exception as e:
        print(f"❌ Error in overall orientation detection: {str(e)}")
        return 0.0, image

def detect_precise_text_regions(image: np.ndarray, yolo_bbox: List[int]) -> List[Dict]:
    """
    Sử dụng PaddleOCR ONNX để phát hiện chính xác vùng text trong bbox YOLO
    Args:
        image: Ảnh gốc
        yolo_bbox: [x1, y1, x2, y2] từ YOLO detection
    Returns:
        List of precise text regions with coordinates
    """
    try:
        # Crop ảnh theo YOLO bbox
        x1, y1, x2, y2 = yolo_bbox
        yolo_region = image[y1:y2, x1:x2]
        
        if yolo_region.size == 0:
            return []
        
        # Sử dụng PaddleOCR để detect text regions chính xác
        paddle_ocr = PaddleOCR()
        text_regions = paddle_ocr.detect_text_regions(yolo_region)
        
        # Convert relative coordinates to absolute coordinates
        precise_regions = []
        for region in text_regions:
            if len(region) >= 4:  # Polygon với ít nhất 4 điểm
                # Convert polygon to bbox and adjust coordinates
                region_array = np.array(region)
                min_x = int(np.min(region_array[:, 0])) + x1
                min_y = int(np.min(region_array[:, 1])) + y1
                max_x = int(np.max(region_array[:, 0])) + x1
                max_y = int(np.max(region_array[:, 1])) + y1
                
                precise_regions.append({
                    "bbox": [min_x, min_y, max_x, max_y],
                    "polygon": [[int(p[0]) + x1, int(p[1]) + y1] for p in region],
                    "area": (max_x - min_x) * (max_y - min_y)
                })
        
        # Sort by area (larger regions first)
        precise_regions.sort(key=lambda x: x["area"], reverse=True)
        
        return precise_regions
        
    except Exception as e:
        print(f"❌ Error in precise text detection: {str(e)}")
        return []

def extract_text_from_precise_region(image: np.ndarray, precise_region: Dict, region_type: str) -> str:
    """
    Extract text từ vùng text được tinh chỉnh chính xác bằng PaddleOCR với auto-rotation
    """
    try:
        bbox = precise_region["bbox"]
        x1, y1, x2, y2 = bbox
        
        # Crop vùng text chính xác
        text_region = image[y1:y2, x1:x2]
        
        if text_region.size == 0:
            return ""
        
        print(f"    🔄 Processing {region_type} region {text_region.shape}")
        
        # Sử dụng PaddleOCR pipeline hoàn chỉnh với orientation detection
        paddle_ocr = PaddleOCR()
        
        try:
            # Step 1: Detect text regions trong crop này
            detected_regions = paddle_ocr.detect_text_regions(text_region)
            
            if not detected_regions:
                print(f"    ⚠️ No text regions detected by PaddleOCR, fallback to TextExtractor")
                # Fallback to TextExtractor
                text_extractor = TextExtractor()
                result = text_extractor.extract_from_image_vi(text_region)
                return result.strip() if result else ""
            
            print(f"    📍 Found {len(detected_regions)} text regions")
            
            # Step 2: Classify orientation và rotate text về đúng hướng
            cropped_images, angles = paddle_ocr.classify_text_orientation(text_region, detected_regions)
            
            print(f"    🔄 Text orientation angles: {angles}")
            
            # Step 3: OCR text từ ảnh đã được xoay
            ocr_results = paddle_ocr.recognize_text(cropped_images)
            
            print(f"    📝 OCR results: {ocr_results}")
            
            # Step 4: Combine tất cả text results
            extracted_texts = []
            for i, result in enumerate(ocr_results):
                try:
                    if isinstance(result, (list, tuple)) and len(result) >= 2:
                        text = result[1]  # Text thường ở index 1
                        confidence = result[2] if len(result) > 2 else 1.0
                        
                        print(f"      Text {i}: '{text}' (confidence: {confidence:.3f})")
                        
                        if confidence > 0.4:  # Giảm threshold confidence
                            extracted_texts.append(str(text))
                    elif isinstance(result, str):
                        extracted_texts.append(result)
                        print(f"      Text {i}: '{result}'")
                except Exception as e:
                    print(f"      ❌ Error processing OCR result {i}: {e}")
                    continue
            
            final_text = " ".join(extracted_texts).strip()
            print(f"    📋 Combined text: '{final_text}'")
            
            # Step 5: Fallback to TextExtractor nếu PaddleOCR không thu được text tốt
            if not final_text or len(final_text.strip()) < 2:
                print(f"    🔄 PaddleOCR result poor, fallback to TextExtractor")
                text_extractor = TextExtractor()
                fallback_result = text_extractor.extract_from_image_vi(text_region)
                
                if fallback_result and len(fallback_result.strip()) > len(final_text.strip()):
                    final_text = fallback_result.strip()
                    print(f"    ✅ TextExtractor better: '{final_text}'")
            
            return final_text
            
        except Exception as paddle_error:
            print(f"    ❌ PaddleOCR error: {str(paddle_error)}, fallback to TextExtractor")
            # Complete fallback to TextExtractor
            text_extractor = TextExtractor()
            return text_extractor.extract_from_image_vi(text_region).strip()
        
    except Exception as e:
        print(f"❌ Error extracting text from precise region: {str(e)}")
        # Final fallback
        try:
            bbox = precise_region["bbox"]
            x1, y1, x2, y2 = bbox
            text_region = image[y1:y2, x1:x2]
            text_extractor = TextExtractor()
            return text_extractor.extract_from_image_vi(text_region).strip()
        except:
            return ""

def detect_overall_text_orientation_and_rotate(image: np.ndarray) -> tuple:
    """
    Detect overall text orientation trong toàn bộ image và rotate nguyên cả hình
    Returns: (rotation_angle, rotated_image, rotation_applied)
    """
    try:
        print("🔄 Detecting overall text orientation for entire image...")
        
        # Sử dụng PaddleOCR để detect tất cả text regions trong toàn ảnh
        paddle_ocr = PaddleOCR()
        text_regions = paddle_ocr.detect_text_regions(image)
        
        if not text_regions or len(text_regions) < 2:  # Cần ít nhất 2 regions để judge orientation
            print("⚠️ Not enough text regions for orientation detection")
            return 0.0, image, False
        
        # Classify orientation cho tất cả text regions
        try:
            cropped_images, angles = paddle_ocr.classify_text_orientation(image, text_regions)
            
            if not angles:
                print("⚠️ No orientation angles detected")
                return 0.0, image, False
            
            print(f"📊 Detected text angles: {angles}")
            
            # Analyze angles để quyết định rotation cho toàn ảnh
            angle_counts = {}
            for angle in angles:
                rounded_angle = round(angle / 90) * 90  # Round to nearest 90 degrees
                if rounded_angle >= 360:
                    rounded_angle -= 360
                elif rounded_angle < 0:
                    rounded_angle += 360
                
                angle_counts[rounded_angle] = angle_counts.get(rounded_angle, 0) + 1
            
            # Tìm angle phổ biến nhất
            most_common_angle = max(angle_counts.items(), key=lambda x: x[1])[0]
            
            print(f"🎯 Most common text angle: {most_common_angle}° (appears {angle_counts[most_common_angle]}/{len(angles)} times)")
            
            # Rotate toàn bộ image nếu cần
            if most_common_angle != 0:
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                
                # Convert PaddleOCR angle to image rotation angle
                rotation_angle = -most_common_angle  # Rotate ngược lại để correct text
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                
                # Calculate new dimensions to prevent cropping
                cos_angle = abs(rotation_matrix[0, 0])
                sin_angle = abs(rotation_matrix[0, 1])
                new_w = int((h * sin_angle) + (w * cos_angle))
                new_h = int((h * cos_angle) + (w * sin_angle))
                
                # Adjust rotation matrix for new center
                rotation_matrix[0, 2] += (new_w / 2) - center[0]
                rotation_matrix[1, 2] += (new_h / 2) - center[1]
                
                rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, 
                                              borderValue=(255, 255, 255))
                
                print(f"✅ Entire image rotated by {rotation_angle}° (from {image.shape} to {rotated_image.shape})")
                return rotation_angle, rotated_image, True
            else:
                print("✅ Text orientation already correct")
                return 0.0, image, False
                
        except Exception as orientation_error:
            print(f"❌ Error in text orientation detection: {str(orientation_error)}")
            return 0.0, image, False
            
    except Exception as e:
        print(f"❌ Error in overall orientation detection: {str(e)}")
        return 0.0, image, False

def extract_text_with_high_quality_bbox(image: np.ndarray, bbox: List[int], region_type: str, lang: str = "vi") -> str:
    """
    Extract text sử dụng bbox trên original image để maintain quality + language-specific OCR
    Args:
        image: Original high-quality image
        bbox: [x1, y1, x2, y2] bounding box
        region_type: Type of region (qr_code, portrait, etc.)
        lang: Language for OCR ("vi" for Vietnamese, "en" for English)
    Returns:
        Best extracted text từ specified language engine
    """
    try:
        x1, y1, x2, y2 = bbox
        
        # Validate bbox
        if x1 >= x2 or y1 >= y2:
            print(f"    ⚠️ Invalid bbox: {bbox}")
            return ""
        
        # Ensure bbox within image bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        print(f"    📝 High-quality extraction for {region_type} from bbox [{x1}, {y1}, {x2}, {y2}] using language: {lang}")
        
        results = {}
        text_extractor = TextExtractor()
        
        # Language-specific OCR processing - Only TextExtractor with EasyOCR
        if lang == "vi":
            print(f"    🇻🇳 Using Vietnamese EasyOCR only...")
            
            # Only TextExtractor Vietnamese with bbox method (EasyOCR based)
            try:
                print(f"    🔄 Extracting Vietnamese text with EasyOCR...")
                vi_text = text_extractor.extract_from_bbox_vi(image, bbox)
                
                if vi_text and len(vi_text.strip()) > 0:
                    results['textextractor_vi'] = vi_text.strip()
                    print(f"    ✅ EasyOCR VI result: '{vi_text[:50]}...'")
                    
            except Exception as vi_error:
                print(f"    ❌ EasyOCR VI error: {str(vi_error)}")
                
        elif lang == "en":
            print(f"    🇺🇸 Using English EasyOCR only...")
            
            # Only TextExtractor English with bbox method (EasyOCR based)
            try:
                print(f"    🔄 Extracting English text with EasyOCR...")
                en_text = text_extractor.extract_from_bbox_en(image, bbox)
                
                if en_text and len(en_text.strip()) > 0:
                    results['textextractor_en'] = en_text.strip()
                    print(f"    ✅ EasyOCR EN result: '{en_text[:50]}...'")
                    
            except Exception as en_error:
                print(f"    ❌ EasyOCR EN error: {str(en_error)}")
        
        else:
            print(f"    ⚠️ Unsupported language '{lang}', fallback to Vietnamese...")
            # Fallback to Vietnamese if unsupported language
            return extract_text_with_high_quality_bbox(image, bbox, region_type, lang="vi")
        
        # Analyze results và chọn best result for specified language
        if not results:
            print(f"    ❌ No text extracted from EasyOCR {lang} engine for {region_type}")
            return ""
        
        print(f"    📊 EasyOCR {lang.upper()} Results for {region_type}:")
        for engine, text in results.items():
            print(f"      {engine}: '{text[:30]}...' (len: {len(text)})")
        
        # Simple selection - since we only have one engine per language now
        best_result = ""
        best_engine = ""
        
        # Language-specific engine selection
        if lang == "vi" and 'textextractor_vi' in results:
            best_result = results['textextractor_vi']
            best_engine = 'textextractor_vi'
        elif lang == "en" and 'textextractor_en' in results:
            best_result = results['textextractor_en']
            best_engine = 'textextractor_en'
        else:
            # Fallback to any available result
            for engine, text in results.items():
                if len(text) > 0:
                    best_result = text
                    best_engine = engine
                    break
        
        if best_result:
            print(f"    🎯 EasyOCR {lang.upper()} result from {best_engine}: '{best_result[:50]}...'")
            return best_result
        else:
            print(f"    ⚠️ No valid {lang} text extracted from {region_type}")
            return ""
        
        # Engine 1: PaddleOCR with original image + bbox (new method)
        try:
            print(f"    � Trying PaddleOCR with original image + bbox...")
            
            # Use high-quality bbox extraction method
            paddle_text = ""
            
            # Crop with padding for PaddleOCR
            padding = 10
            padded_x1 = max(0, x1 - padding)
            padded_y1 = max(0, y1 - padding)
            padded_x2 = min(w, x2 + padding)
            padded_y2 = min(h, y2 + padding)
            
            padded_region = image[padded_y1:padded_y2, padded_x1:padded_x2]
            
            if padded_region.size > 0:
                paddle_ocr = PaddleOCR()
                ocr_result = paddle_ocr.process_pipeline(padded_region, steps=['detect', 'classify', 'recognize'])
                
                if isinstance(ocr_result, dict) and 'texts' in ocr_result:
                    texts = ocr_result['texts']
                    if isinstance(texts, tuple) and len(texts) > 0:
                        text_list = texts[0] if isinstance(texts[0], list) else []
                        paddle_text = ' '.join(text_list)
                    elif isinstance(texts, list):
                        paddle_text = ' '.join(texts)
                
                if paddle_text and len(paddle_text.strip()) > 0:
                    results['paddle_hq'] = paddle_text.strip()
                    print(f"    ✅ PaddleOCR HQ result: '{paddle_text[:50]}...'")
                    
        except Exception as paddle_error:
            print(f"    ❌ PaddleOCR HQ error: {str(paddle_error)}")
        
        # Engine 2: TextExtractor Vietnamese with bbox method (new)
        try:
            print(f"    🔄 Trying TextExtractor VI with original image + bbox...")
            vi_text = text_extractor.extract_from_bbox_vi(image, bbox)
            
            if vi_text and len(vi_text.strip()) > 0:
                results['textextractor_vi_hq'] = vi_text.strip()
                print(f"    ✅ TextExtractor VI HQ result: '{vi_text[:50]}...'")
                
        except Exception as vi_error:
            print(f"    ❌ TextExtractor VI HQ error: {str(vi_error)}")
        
        # Engine 3: TextExtractor English with bbox method (new)
        try:
            print(f"    🔄 Trying TextExtractor EN with original image + bbox...")
            en_text = text_extractor.extract_from_bbox_en(image, bbox)
            
            if en_text and len(en_text.strip()) > 0:
                results['textextractor_en_hq'] = en_text.strip()
                print(f"    ✅ TextExtractor EN HQ result: '{en_text[:50]}...'")
                
        except Exception as en_error:
            print(f"    ❌ TextExtractor EN HQ error: {str(en_error)}")
        
        # Analyze results và chọn best result for specified language
        if not results:
            print(f"    ❌ No text extracted from any {lang} engine for {region_type}")
            return ""
        
        print(f"    📊 {lang.upper()} Results comparison for {region_type}:")
        for engine, text in results.items():
            print(f"      {engine}: '{text[:30]}...' (len: {len(text)})")
        
        # Intelligent selection logic based on language
        best_result = ""
        best_length = 0
        best_engine = ""
        
        # Language-specific engine priorities
        if lang == "vi":
            # Vietnamese text: prefer TextExtractor VI, then PaddleOCR
            engine_priorities = ['textextractor_vi', 'paddle_vi']
        elif lang == "en":
            # English text: prefer TextExtractor EN, then PaddleOCR
            engine_priorities = ['textextractor_en', 'paddle_en']
        else:
            # Fallback: use any available
            engine_priorities = list(results.keys())
        
        # Select best result based on priority và length
        for engine in engine_priorities:
            if engine in results:
                text = results[engine]
                if len(text) > best_length and len(text) > 1:  # Minimum 2 characters
                    best_result = text
                    best_length = len(text)
                    best_engine = engine
        
        # Fallback to any available result
        if not best_result:
            for engine, text in results.items():
                if len(text) > 0:
                    best_result = text
                    best_engine = engine
                    break
        
        if best_result:
            print(f"    🎯 Best {lang.upper()} result from {best_engine}: '{best_result[:50]}...'")
            return best_result
        else:
            print(f"    ⚠️ No valid {lang} text extracted from {region_type}")
            return ""
            
    except Exception as e:
        print(f"    ❌ Error in high-quality bbox extraction: {str(e)}")
        return ""

def process_ocr_with_full_rotation(image: np.ndarray, ocr_detections: List[Dict], lang: str = "vi") -> List[OCRRegion]:
    """
    Process OCR với full image rotation approach và language-specific OCR:
    1. Detect overall text orientation và rotate toàn ảnh
    2. Re-detect YOLO regions trên rotated image 
    3. Extract text với TextExtractor từ corrected image theo ngôn ngữ chỉ định
    Args:
        image: Original image
        ocr_detections: YOLO detection results
        lang: Language for OCR ("vi" for Vietnamese, "en" for English)
    Returns:
        List of OCR regions with language-specific text extraction
    """
    print(f"🔄 Starting full rotation OCR processing with language: {lang.upper()}...")
    
    # Step 1: Detect overall orientation và rotate toàn ảnh nếu cần
    rotation_angle, processed_image, rotation_applied = detect_overall_text_orientation_and_rotate(image)
    
    ocr_regions = []
    current_detections = ocr_detections
    
    # Step 2: Nếu đã rotate image, cần re-detect YOLO regions
    if rotation_applied:
        print("🔍 Re-detecting YOLO regions on rotated image...")
        try:
            # Re-detect với OCR model trên rotated image
            pt_config = PtConfig()
            ocr_model_path = os.path.join(pt_config.get_model_path(), "OCR_QR_CCCD.pt")
            
            if os.path.exists(ocr_model_path):
                ocr_detector = YOLODetector(model_path=ocr_model_path)
                current_detections = ocr_detector.detect(processed_image)
                print(f"📊 Re-detected {len(current_detections)} regions on rotated image")
            else:
                print("⚠️ OCR model not found, using original detections (may be inaccurate)")
                
        except Exception as redetect_error:
            print(f"❌ Error re-detecting regions: {str(redetect_error)}")
            print("⚠️ Using original detections on rotated image")
    
    # Step 3: Process từng detection với TextExtractor
    for detection in current_detections:
        try:
            yolo_bbox = detection.get("bbox", [0, 0, 0, 0])
            class_name = detection.get("class_name", "unknown")
            yolo_confidence = detection.get("confidence", 0.0)
            
            print(f"🔍 Processing {class_name} region with bbox: {yolo_bbox} using {lang.upper()} OCR")
            
            # Extract text với high-quality bbox method và specified language
            extracted_text = extract_text_with_high_quality_bbox(processed_image, yolo_bbox, class_name, lang)
            
            # Tạo OCR region result
            ocr_region = OCRRegion(
                region_type=class_name.lower(),
                bbox=yolo_bbox,
                confidence=float(yolo_confidence),
                extracted_text=extracted_text,
                text_direction="horizontal",
                rotation_angle=rotation_angle if rotation_applied else 0.0
            )
            ocr_regions.append(ocr_region)
            
        except Exception as e:
            print(f"❌ Error processing region {detection}: {str(e)}")
            continue
    
    print(f"✅ Full rotation OCR completed with {lang.upper()}: {len(ocr_regions)} regions processed")
    return ocr_regions

@router.post("/card",
            summary="Precision OCR for Vietnam Citizens Cards",
            description="High-precision OCR using YOLO + PaddleOCR ONNX + TextExtractor workflow",
            response_model=OCRResponse)
async def precision_card_ocr(
    file: UploadFile = File(..., description="Image file containing Vietnam Citizens Card"),
    card_type: Optional[str] = Query(None, description="Specific card type to process (e.g., 'cccd_qr_front')"),
    enable_precise_detection: bool = Query(True, description="Use precision text detection (YOLO → PaddleOCR ONNX)"),
    lang: str = Query("vi", description="OCR language: 'vi' for Vietnamese (default), 'en' for English")
):
    """
    **High-Precision OCR** for Vietnam Citizens Cards using advanced 3-stage workflow:
    
    ## 🔧 **Workflow**:
    1. **YOLO Detection**: Detect text region positions (coarse)
    2. **PaddleOCR ONNX**: Refine text boundaries + orientation detection (precise)  
    3. **TextExtractor**: Extract text from precise regions according to specified language
    
    ## 🌍 **Language Support**:
    - **Vietnamese** (`lang=vi`): Default, optimized for Vietnamese text recognition
    - **English** (`lang=en`): Optimized for English text recognition
    - Single language processing for better accuracy
    
    ## 🎯 **Advantages**:
    - **Language-specific OCR**: Only process with specified language for better accuracy
    - **Reduced noise**: Only process exact text areas
    - **Higher accuracy**: Language-optimized text extraction on clean regions
    - **Orientation handling**: Auto-rotate text to correct direction
    - **Less data loss**: Preserve text boundaries precisely
    
    ## 📋 **Process**:
    - Detect card type (if not provided)
    - For CCCD cards with QR: Use `OCR_QR_CCCD.pt` model
    - YOLO provides rough text locations
    - PaddleOCR ONNX refines text regions and handles rotation
    - TextExtractor performs language-specific OCR on clean regions
    
    ## 🔍 **Supported Cards**:
    - Vietnam Citizens Cards (old format with QR code)
    - Vietnam Citizens Cards (new format) 
    - Driving Licenses (GPLX)
    """
    if not file.filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, png, jpeg).")
    
    temp_files = []
    try:
        # Read and process image
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file or corrupted data.")
        
        print(f"🔍 Processing OCR for image: {file.filename} with language: {lang.upper()}")
        
        # Validate language parameter
        if lang not in ["vi", "en"]:
            raise HTTPException(status_code=400, detail=f"Unsupported language '{lang}'. Supported: 'vi' (Vietnamese), 'en' (English)")
        
        print(f"🌍 OCR Language set to: {lang.upper()} ({'Vietnamese' if lang == 'vi' else 'English'})")
        
        # Step 1: Detect card type if not provided
        detected_card_type = card_type
        card_confidence = 1.0
        
        if not detected_card_type:
            print("🎯 Detecting card type...")
            # Use CardDetectionService to detect card type
            pt_config = PtConfig()
            model_path = os.path.join(pt_config.get_model_path(), "CCCD_FACE_DETECT_2025_NEW_TITLE.pt")
            
            if not os.path.exists(model_path):
                raise HTTPException(status_code=500, detail=f"Card detection model not found at {model_path}")
            
            # Use CardDetectionService to detect card type
            detector = YOLODetector(model_path=model_path)
            card_detections = detector.detect(image)
            
            if not card_detections:
                return OCRResponse(
                    success=False,
                    message="No card detected in image",
                    result=None
                )
            
            # Detect OCR regions for additional verification
            ocr_detector = YOLODetector(model_path=os.path.join(pt_config.get_model_path(), "OCR_QR_CCCD.pt"))
            ocr_detections = ocr_detector.detect(image)
            
            # Extract information types from OCR detections
            detected_info_types = set()
            for ocr_detection in ocr_detections:
                class_name = ocr_detection.get("class_name")
                if class_name:
                    detected_info_types.add(class_name.lower())
            
            # Use CardDetectionService to process detections
            file_detections, title_detected_type = card_detection_service.process_detections(
                card_detections, image, file.filename or "uploaded_file"
            )
            
            # Select best detection using service
            detection_result = card_detection_service.select_best_detection(
                file_detections, title_detected_type, detected_info_types, file.filename or "uploaded_file"
            )
            
            detected_card_type = detection_result["detected_label"]
            card_confidence = detection_result["confidence"]
        
        print(f"🎯 Card type: {detected_card_type} (confidence: {card_confidence:.3f})")
        
        # Step 2: Get OCR regions based on card type
        ocr_detections = []
        
        if "cccd" in detected_card_type.lower() and "qr" in detected_card_type.lower():
            print("🔍 Using OCR_QR_CCCD.pt model for CCCD with QR...")
            pt_config = PtConfig()
            ocr_model_path = os.path.join(pt_config.get_model_path(), "OCR_QR_CCCD.pt")
            
            if not os.path.exists(ocr_model_path):
                raise HTTPException(status_code=500, detail=f"OCR model not found at {ocr_model_path}")
            
            ocr_detector = YOLODetector(model_path=ocr_model_path)
            ocr_detections = ocr_detector.detect(image)
            print(f"📊 Found {len(ocr_detections)} OCR regions")
        
        # Step 3: Process OCR regions with FULL ROTATION approach và language-specific OCR:
        # 1. Detect overall text orientation và rotate toàn ảnh
        # 2. Re-detect YOLO regions trên rotated image  
        # 3. Extract text với TextExtractor từ corrected image theo ngôn ngữ chỉ định
        print(f"📝 Processing OCR with full rotation approach + {lang.upper()} language (detect orientation → rotate entire image → re-detect → language-specific TextExtractor)...")
        ocr_regions = process_ocr_with_full_rotation(image, ocr_detections, lang)
        
        # Create result
        result = CardOCRResult(
            card_type=detected_card_type,
            card_confidence=float(card_confidence),
            total_ocr_regions=len(ocr_regions),
            ocr_regions=ocr_regions,
            processing_info={
                "workflow_used": f"Full Image Rotation → YOLO Re-detection → {lang.upper()} Language OCR",
                "language": lang.upper(),
                "language_description": "Vietnamese" if lang == "vi" else "English",
                "full_image_rotation": True,
                "text_orientation_detection": True,
                "engines_used": [f"PaddleOCR_{lang.upper()}", f"TextExtractor_{lang.upper()}"],
                "single_language_processing": True,
                "ocr_model_used": "OCR_QR_CCCD.pt" if "cccd" in detected_card_type.lower() and "qr" in detected_card_type.lower() else "card_detection_only",
                "image_size": {"width": image.shape[1], "height": image.shape[0]}
            }
        )
        
        print(f"✅ OCR completed with {lang.upper()}: {len(ocr_regions)} regions processed")
        
        return OCRResponse(
            success=True,
            message=f"OCR completed successfully for {detected_card_type} using {lang.upper()} language",
            result=result
        )
    
    except Exception as e:
        print(f"❌ Error in OCR processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")
    
    finally:
        # Clean up temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

@router.post("/card/batch",
            summary="Batch Precision OCR for Multiple Cards",
            description="Process multiple card images with precision OCR workflow")
async def batch_precision_card_ocr(
    files: List[UploadFile] = File(..., description="List of image files"),
    enable_precise_detection: bool = Query(True, description="Use precision text detection for all images"),
    lang: str = Query("vi", description="OCR language for all images: 'vi' for Vietnamese (default), 'en' for English")
):
    """
    Batch processing for multiple card images with precision OCR workflow.
    Uses the same YOLO → PaddleOCR ONNX → TextExtractor workflow for each image 
    with specified language processing.
    
    ## 🌍 **Language Processing**:
    - All images will be processed with the same language setting
    - `lang=vi`: Vietnamese OCR for all images
    - `lang=en`: English OCR for all images
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    
    results = []
    
    for file in files:
        try:
            # Process each file using the precision OCR function với specified language
            result = await precision_card_ocr(
                file=file,
                card_type=None,  # Auto-detect for each image
                enable_precise_detection=enable_precise_detection,
                lang=lang  # Use specified language for all images
            )
            results.append({
                "filename": file.filename,
                "result": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "result": None
            })
    
    return {
        "total_files": len(files),
        "successful": len([r for r in results if "error" not in r]),
        "failed": len([r for r in results if "error" in r]),
        "results": results
    }
