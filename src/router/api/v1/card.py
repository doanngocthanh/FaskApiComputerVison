from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Path
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
from src.router.api.__init__ import *  
from src.service.YOLODetector import YOLODetector
from src.service.PaddleOCR import PaddleOCR
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
                   {"id":5, "name":"Thẻ Căn Cước Công Dân Mới", "nameEn":"New Citizens Card"},]
card_types =  [{"id":0, "name":"Mặt Trước", "nameEn":"Front"},
               {"id":1, "name":"Mặt Sau", "nameEn":"Back"}]
@router.get("/card/categories")
def get_card_types():
    """
    Endpoint to get the list of card types.
    """
    return card_categories
@router.post("/card/detect")
async def detect_card(file: UploadFile = File(...)):
    """
    Endpoint to detect card regions in the uploaded image.
    """
    if not file.filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, png, jpeg).")
    
    temp_files = []  # Keep track of temp files for cleanup
    try:
        # Read file content
        content = await file.read()
        
        # Convert to numpy array for YOLO processing
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file or corrupted data.")
        
        # Initialize YOLO detector
        pt_config = PtConfig()
        model_path = os.path.join(pt_config.get_model_path(), "CCCD_FACE_DETECT.pt")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=500, detail=f"Card detection model not found at {model_path}")
        
        detector = YOLODetector(model_path=model_path)
        
        # Detect card regions using numpy array (image)
        detections = detector.detect(image)
        
        if not detections:
            return {"message": "No card regions detected.", "detections": []}
        
        # Prepare response with detected regions
        response = {
            "detections": []
        }
        
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
                else:
                        # Unknown or unrecognized label
                        card_category = {"id": -1, "name": "Unknown", "nameEn": "Unknown"}
                        card_type = "unknown"
            
            # Create detailed detection result
            detection_result = {
                "detected_label": label,
                "card_category": card_category,
                "card_type": card_type,
                "is_valid_card": label in ["cccd_front", "cccd_backside"]
            }
            
            response["detections"].append(detection_result)
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

