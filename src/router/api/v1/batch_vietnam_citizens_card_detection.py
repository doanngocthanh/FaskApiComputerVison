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
from src.service.CardDetectionService import card_detection_service
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
                
                # Use CardDetectionService to process detections
                file_detections, title_detected_type = card_detection_service.process_detections(
                    detections, image, file.filename
                )
                
                # Add OCR features to all detections
                for detection in file_detections:
                    detection["ocr_features"] = {
                        "has_portrait": has_portrait,
                        "has_qr_code": has_qr_code,
                        "has_basic_info": has_basic_info,
                        "has_address_info": has_address_info,
                        "detected_info_types": list(detected_info_types)
                    }
                
                # Select best detection using service
                best_detection = card_detection_service.select_best_detection(
                    file_detections, title_detected_type, detected_info_types, file.filename
                )
                
                if not best_detection:
                    results.append({
                        "filename": file.filename,
                        "success": True,
                        "message": "No valid detections found.",
                        "detections": []
                    })
                    continue
                
                # Add file result
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "detections": [best_detection]
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