from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Path, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid
import os
import cv2
import numpy as np
from io import BytesIO
import importlib
import time
from src.router.api.__init__ import *  
from src.service.YOLODetector import YOLODetector
from src.service.EasyOCRManager import EasyOCRManager
from src.service.LoggingManager import logging_manager
from PIL import Image
from config import PtConfig

# Router setup
router = APIRouter(
    prefix="/api/v1",
    tags=["VietNam Citizens Card Detection"],
    responses={
        404: {"model": ErrorResponse, "description": "Resource not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
card_categories = [{"id":0, "name":"Thẻ Căn Cước Công Dân","nameEn":"Citizens Card"},
                   {"id":1, "name":"Giấy Phép Lái Xe", "nameEn":"Driving License"},
                   {"id":2, "name":"Thẻ Bảo Hiểm Y Tế", "nameEn":"Health Insurance Card"},
                   {"id":3, "name":"Thẻ Ngân Hàng", "nameEn":"Bank Card"},
                   {"id":4, "name":"Thẻ Sinh Viên", "nameEn":"Student Card"},
                   {"id":5, "name":"Thẻ Căn Cước Công Dân Mới", "nameEn":"New Citizens Card"},
                   {"id":6, "name":"Thẻ ", "nameEn":"Old Citizens Card"},
                   ]
card_types =  [{"id":0, "name":"Mặt Trước", "nameEn":"Front"},
               {"id":1, "name":"Mặt Sau", "nameEn":"Back"}]
@router.get("/card/categories")
def get_card_types():
    """
    Endpoint to get the list of card types.
    """
    return card_categories
@router.post("/card/detect")
async def detect_card(request: Request, file: UploadFile = File(...)):
    """
    Endpoint to detect card regions in the uploaded image.
    """
    start_time = time.time()
    
    # Prepare logging data
    log_data = {
        'endpoint': '/api/v1/card/detect',
        'method': 'POST',
        'file_size': 0,
        'request_body': {'filename': file.filename},
        'ip_address': request.client.host if request.client else 'unknown'
    }
    
    if not file.filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        error_msg = "File must be an image (jpg, png, jpeg)."
        log_data.update({
            'success': False,
            'error_message': error_msg,
            'processing_time_ms': int((time.time() - start_time) * 1000)
        })
        logging_manager.log_request(log_data)
        raise HTTPException(status_code=400, detail=error_msg)
    
    temp_files = []  # Keep track of temp files for cleanup
    try:
        # Read file content
        content = await file.read()
        log_data['file_size'] = len(content)
        
        # Convert to numpy array for YOLO processing
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            error_msg = "Invalid image file or corrupted data."
            log_data.update({
                'success': False,
                'error_message': error_msg,
                'processing_time_ms': int((time.time() - start_time) * 1000)
            })
            logging_manager.log_request(log_data)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Initialize YOLO detector
        pt_config = PtConfig()
        model_path = os.path.join(pt_config.get_model_path(), "CCCD_FACE_DETECT.pt")
        if not os.path.exists(model_path):
            error_msg = f"Card detection model not found at {model_path}"
            log_data.update({
                'success': False,
                'error_message': error_msg,
                'processing_time_ms': int((time.time() - start_time) * 1000)
            })
            logging_manager.log_request(log_data)
            raise HTTPException(status_code=500, detail=error_msg)
        
        detector = YOLODetector(model_path=model_path)
        
        model_path_cccd_new= os.path.join(pt_config.get_model_path(), "CCCD_FACE_DETECT_2025_NEW.pt")
        if not os.path.exists(model_path_cccd_new):
            error_msg = f"New card detection model not found at {model_path_cccd_new}"
            log_data.update({
                'success': False,
                'error_message': error_msg,
                'processing_time_ms': int((time.time() - start_time) * 1000)
            })
            logging_manager.log_request(log_data)
            raise HTTPException(status_code=500, detail=error_msg)
        # Use the new model for detection
        detector_cccd_new = YOLODetector(model_path=model_path_cccd_new)
      

        # Detect card regions using numpy array (image)
        detections = detector.detect(image)
        # Detect card regions using the new model as well
        detections_new = detector_cccd_new.detect(image)

        # Combine detections from both models
        all_detections = detections + detections_new

        # Remove duplicates or overlapping detections if needed
        # You can implement logic here to filter out overlapping bounding boxes
        # For now, we'll use all detections
        detections = all_detections
        
        # Prepare response with detected regions
        response = {
            "status": "success",
            "detections": []
        }
        
        if not detections:
            response["message"] = "No card regions detected."
            response["total_detections"] = 0
        else:
            response["total_detections"] = len(detections)
            
            # Map detection labels to card categories and types
            for detection in detections:
                # Extract detection keys for debugging/logging
                detection_keys = list(detection.keys())
                print(f"Detection keys: {detection_keys}")  # Debugging line
                if(detection.get("class_name")):
                    print(f"Detected class name: {detection['class_name']}")  # Debugging line
                    label = detection['class_name'].lower()
                    if label == "cccd_front":
                        card_category = card_categories[0]  # Thẻ Căn Cước Công Dân
                        card_type = card_types[0]  # Mặt Trước
                    elif label == "cccd_backside":
                            card_category = card_categories[0]  # Thẻ Căn Cước Công Dân
                            card_type = card_types[1]  # Mặt Sau
                    elif label == "cccd_new_front":
                        card_category = card_categories[5]
                        card_type = card_types[0]
                    elif label == "cccd_new_back":
                        card_category = card_categories[5]
                        card_type = card_types[1]
                    else:
                            # Unknown or unrecognized label
                            card_category = {"id": -1, "name": "Unknown", "nameEn": "Unknown"}
                            card_type = "unknown"
                
                # Create detailed detection result
                detection_result = {
                    "detected_label": label,
                    "card_category": card_category,
                    "card_type": card_type,
                    "is_valid_card": label in ["cccd_front", "cccd_backside", "cccd_new_front", "cccd_new_back"],
                    "bbox": detection.get('bbox', []),
                    "confidence": detection.get('confidence', 0.0)
                }
                
                response["detections"].append(detection_result)
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        response["processing_time_ms"] = processing_time
        
        # Update log data with success info
        log_data.update({
            'success': True,
            'processing_time_ms': processing_time,
            'total_detections': response["total_detections"],
            'response_body': response
        })
        
        logging_manager.log_request(log_data)
        return response
    
    except HTTPException as e:
        # Log HTTP exceptions
        log_data.update({
            'success': False,
            'error_message': str(e.detail),
            'processing_time_ms': int((time.time() - start_time) * 1000)
        })
        logging_manager.log_request(log_data)
        raise
    except Exception as e:
        error_msg = f"Internal server error: {str(e)}"
        log_data.update({
            'success': False,
            'error_message': error_msg,
            'processing_time_ms': int((time.time() - start_time) * 1000)
        })
        logging_manager.log_request(log_data)
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

