from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Path, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
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
from PIL import Image
from config import PtConfig
from src.service.CardConfigService import card_config_service
from src.service.TextExtractor import TextExtractor
from difflib import SequenceMatcher
router = APIRouter(
    prefix="/api/v1/batch",
    tags=["Batch VietNam Citizens Card Detection"],
    responses={
        404: {"model": ErrorResponse, "description": "Resource not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
@router.post("/card/detect",
            summary="Detect Vietnam Card Type",
            description="Detect and classify Vietnam Citizens Card or Driving License using YOLO + OCR")
async def detect_card(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    
    temp_files = []  # Keep track of temp files for cleanup
    try:
        # Initialize models once for all files
        pt_config = PtConfig()
        model_path = os.path.join(pt_config.get_model_path(), "CCCD_FACE_DETECT_2025_NEW_TITLE.pt")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=500, detail=f"CCCD Old/New detection model not found at {model_path}")
        
        model_path_ocr = os.path.join(pt_config.get_model_path(), "OCR_QR_CCCD.pt")
        if not os.path.exists(model_path_ocr):
            raise HTTPException(status_code=500, detail=f"OCR detection model not found at {model_path_ocr}")
        
        detector = YOLODetector(model_path=model_path)
        ocr_detector = YOLODetector(model_path=model_path_ocr)
        text_extractor = TextExtractor()
        
        # Process each file
        results = []
        
        for file in files:
            if not file.filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "File must be an image (jpg, png, jpeg).",
                    "detections": []
                })
                continue
            
            try:
                # Read file content
                content = await file.read()
                
                # Convert to numpy array for YOLO processing
                nparr = np.frombuffer(content, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": "Invalid image file or corrupted data.",
                        "detections": []
                    })
                    continue
                
                # Detect card regions using the CCCD_OLD_NEW model
                detections = detector.detect(image)

                if not detections:
                    results.append({
                        "filename": file.filename,
                        "success": True,
                        "message": "No card regions detected.",
                        "detections": []
                    })
                    continue
                
                # Detect OCR/QR information for additional verification
                ocr_detections = ocr_detector.detect(image)
                
                # Extract information types from OCR detections
                detected_info_types = set()
                for ocr_detection in ocr_detections:
                    class_name = ocr_detection.get("class_name")
                    if class_name:
                        detected_info_types.add(class_name.lower())
                
                print(f"[{file.filename}] OCR detected info types: {detected_info_types}")
                
                # Analyze OCR features for CCCD classification
                has_portrait = "portrait" in detected_info_types
                has_qr_code = "qr_code" in detected_info_types
                has_basic_info = any(info in detected_info_types for info in ["name", "id", "birth", "sex"])
                has_address_info = any(info in detected_info_types for info in ["place_of_origin", "place_of_residence"])
                
                print(f"[{file.filename}] OCR Analysis - Portrait: {has_portrait}, QR: {has_qr_code}, Basic: {has_basic_info}, Address: {has_address_info}")
                
                # Prepare response with detected regions
                file_detections = []
                
                # Map detection labels to card categories and types
                for detection in detections:
                    detection_keys = list(detection.keys())
                    print(f"[{file.filename}] Detection keys: {detection_keys}")
                    
                    class_name = detection.get("class_name")
                    confidence = detection.get("confidence", 0.0)
                    
                    if class_name:
                        print(f"[{file.filename}] Detected class name: {class_name}, confidence: {confidence}")
                        label = class_name.lower()
                        if label == "title":
                            # Crop title region
                            x1, y1, x2, y2 = detection.get("bbox", [0, 0, 0, 0])
                            cropped_image = image[y1:y2, x1:x2]
                            # Extract text
                            extracted_text = text_extractor.extract_from_image_en(cropped_image)

                            print(f"[{file.filename}] Extracted title text: {extracted_text}")
                            
                            def text_similarity(text1, text2):
                                return SequenceMatcher(None, text1.upper(), text2.upper()).ratio()
                            
                            # Define keywords and thresholds
                            cccd_new_keywords = ["CAN CUOC", "CĂNG CƯỚC", "CĂN CƯỚC"]
                            cccd_old_keywords = ["CONG DAN", "CÔNG DÂN", "CỘNG DÂN","CUOC CONG DAN", "CĂN CƯỚC CÔNG DÂN"]
                            threshold = 0.7
                            
                            # Check for CCCD new keywords
                            best_similarity = 0
                            detected_type = None
                            
                            for keyword in cccd_new_keywords:
                                similarity = max([text_similarity(extracted_text, keyword) for keyword in [keyword]])
                                if similarity > threshold and similarity > best_similarity:
                                    best_similarity = similarity
                                    detected_type = "cccd_new_front"
                            
                            for keyword in cccd_old_keywords:
                                similarity = max([text_similarity(extracted_text, keyword) for keyword in [keyword]])
                                if similarity > threshold and similarity > best_similarity:
                                    best_similarity = similarity
                                    detected_type = "cccd_qr_front"
                            
                            if detected_type:
                                label = detected_type
                                print(f"[{file.filename}] Text classification: {detected_type} (similarity: {best_similarity:.2f})")
                        
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
                            label = "unknown"
                    else:
                        card_category = {"id": -1, "name": "Unknown", "nameEn": "Unknown"}
                        card_type = {"id": -1, "name": "Unknown", "nameEn": "Unknown"}
                        label = "unknown"
                        confidence = 0.0
                    
                    # Create detailed detection result with confidence
                    detection_result = {
                        "confidence": confidence,
                        "detected_label": label,
                        "card_category": card_category,
                        "card_type": card_type,
                        "is_valid_card": label in ["cccd_qr_front", "cccd_qr_back", "cccd_new_front", "cccd_new_back", "gplx_front", "gplx_back"],
                        "ocr_features": {
                            "has_portrait": has_portrait,
                            "has_qr_code": has_qr_code,
                            "has_basic_info": has_basic_info,
                            "has_address_info": has_address_info,
                            "detected_info_types": list(detected_info_types)
                        }
                    }
                    
                    file_detections.append(detection_result)
                
                # Sort detections by confidence and apply selection logic
                if file_detections:
                    file_detections.sort(key=lambda x: x["confidence"], reverse=True)
                    
                    print(f"[{file.filename}] All detections found:")
                    for i, det in enumerate(file_detections):
                        print(f"  {i+1}: {det['detected_label']} - confidence: {det['confidence']:.3f}")
                    
                    # Apply smart detection selection logic
                    best_detection = file_detections[0]
                    
                    if len(file_detections) > 1:
                        first = file_detections[0]
                        second = file_detections[1]
                        confidence_diff = first["confidence"] - second["confidence"]
                        
                        print(f"[{file.filename}] Confidence difference: {confidence_diff:.3f}")
                        
                        # Apply OCR-based rules if confidence difference is small
                        if confidence_diff < 0.1:
                            print(f"[{file.filename}] Applying OCR-based classification rules...")
                            
                            # Rule 1: QR code detected → Prefer old CCCD
                            if has_qr_code:
                                for det in file_detections:
                                    if det["detected_label"] in ["cccd_qr_front", "cccd_qr_back"]:
                                        best_detection = det
                                        print(f"[{file.filename}] OCR Rule: QR detected → Switched to QR CCCD: {det['detected_label']}")
                                        break
                            
                            # Rule 2: Portrait + basic info but no QR → Prefer new CCCD or GPLX
                            elif has_portrait and has_basic_info and not has_qr_code:
                                for det in file_detections:
                                    if det["detected_label"] in ["cccd_new_front", "cccd_new_back"]:
                                        best_detection = det
                                        print(f"[{file.filename}] OCR Rule: Portrait+Basic but no QR → Switched to New CCCD: {det['detected_label']}")
                                        break
                                else:
                                    for det in file_detections:
                                        if det["detected_label"] in ["gplx_front", "gplx_back"]:
                                            best_detection = det
                                            print(f"[{file.filename}] OCR Rule: Portrait+Basic but no QR → Switched to GPLX: {det['detected_label']}")
                                            break
                            
                            # Rule 3: Basic info only → Possibly GPLX back
                            elif has_basic_info and not has_portrait and not has_qr_code:
                                for det in file_detections:
                                    if det["detected_label"] == "gplx_back":
                                        best_detection = det
                                        print(f"[{file.filename}] OCR Rule: Basic info only → Switched to GPLX back: {det['detected_label']}")
                                        break
                            
                            # Rule 4: QR only → Back side
                            elif has_qr_code and not has_portrait:
                                for det in file_detections:
                                    if "back" in det["detected_label"]:
                                        best_detection = det
                                        print(f"[{file.filename}] OCR Rule: QR only → Switched to back side: {det['detected_label']}")
                                        break
                            
                            # Rule 5: Fallback → Prefer old CCCD
                            else:
                                for det in file_detections:
                                    if det["detected_label"] in ["cccd_qr_front", "cccd_qr_back"]:
                                        best_detection = det
                                        print(f"[{file.filename}] OCR Rule: Fallback → Switched to QR CCCD: {det['detected_label']}")
                                        break
                    
                    print(f"[{file.filename}] Final best detection: {best_detection['detected_label']} with confidence {best_detection['confidence']:.3f}")
                    
                    file_detections = [best_detection]
                
                # Add file result
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "detections": file_detections
                })
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e),
                    "detections": []
                })
        
        return {"results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)