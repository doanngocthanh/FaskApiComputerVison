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
# Response Models for better API documentation
class CardCategory(BaseModel):
    """Card category information"""
    id: int
    name: str
    nameEn: str
    is_active: Optional[bool] = True
    
    class Config:
        schema_extra = {
            "example": {
                "id": 0,
                "name": "Thẻ Căn Cước Công Dân",
                "nameEn": "Citizens Card", 
                "is_active": True
            }
        }

class CardType(BaseModel):
    """Card type information (front/back)"""
    id: int
    name: str
    nameEn: str
    is_active: Optional[bool] = True
    
    class Config:
        schema_extra = {
            "example": {
                "id": 0,
                "name": "Mặt Trước",
                "nameEn": "Front",
                "is_active": True
            }
        }

class OCRFeatures(BaseModel):
    """OCR detected features"""
    has_portrait: bool
    has_qr_code: bool
    has_basic_info: bool
    has_address_info: Optional[bool] = False
    detected_info_types: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "has_portrait": True,
                "has_qr_code": True,
                "has_basic_info": True,
                "has_address_info": True,
                "detected_info_types": ["portrait", "qr_code", "name", "id", "birth"]
            }
        }

class CardDetection(BaseModel):
    """Single card detection result"""
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score (0.0 to 1.0)")
    detected_label: str = Field(..., description="Detected card label (e.g., 'cccd_qr_front', 'gplx_back')")
    card_category: CardCategory
    card_type: CardType
    is_valid_card: bool = Field(..., description="Whether this is a recognized valid card type")
    ocr_features: OCRFeatures
    
    class Config:
        schema_extra = {
            "example": {
                "confidence": 0.95,
                "detected_label": "cccd_qr_front",
                "card_category": {
                    "id": 0,
                    "name": "Thẻ Căn Cước Công Dân",
                    "nameEn": "Citizens Card"
                },
                "card_type": {
                    "id": 0,
                    "name": "Mặt Trước", 
                    "nameEn": "Front"
                },
                "is_valid_card": True,
                "ocr_features": {
                    "has_portrait": True,
                    "has_qr_code": True,
                    "has_basic_info": True,
                    "has_address_info": True,
                    "detected_info_types": ["portrait", "qr_code", "name", "id", "birth"]
                }
            }
        }

class CardDetectionResponse(BaseModel):
    """Card detection API response"""
    detections: List[CardDetection] = Field(..., description="List of detected cards (usually contains best detection)")
    message: Optional[str] = Field(None, description="Status message when no detections found")
    
    class Config:
        schema_extra = {
            "example": {
                "detections": [
                    {
                        "confidence": 0.95,
                        "detected_label": "cccd_qr_front",
                        "card_category": {
                            "id": 0,
                            "name": "Thẻ Căn Cước Công Dân",
                            "nameEn": "Citizens Card"
                        },
                        "card_type": {
                            "id": 0,
                            "name": "Mặt Trước",
                            "nameEn": "Front"
                        },
                        "is_valid_card": True,
                        "ocr_features": {
                            "has_portrait": True,
                            "has_qr_code": True,
                            "has_basic_info": True,
                            "detected_info_types": ["portrait", "qr_code", "name", "id"]
                        }
                    }
                ]
            }
        }

# Router setup
router = APIRouter(
    prefix="/api/v1",
    tags=["VietNam Citizens Card Detection"],
    responses={
        404: {"model": ErrorResponse, "description": "Resource not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
@router.get("/card/categories",
           summary="Get Card Categories",
           description="Retrieve all active card categories from database",
           )
def get_card_types():
    return card_config_service.get_card_categories()

@router.get("/card/types",
           summary="Get Card Types", 
           description="Retrieve all active card types (front/back) from database")
def get_card_types_list():
    return card_config_service.get_card_types()

@router.get("/card/config",
           summary="Get Card Configuration Summary",
           description="Get complete card configuration summary")
def get_card_config():
    return card_config_service.get_config_summary()

@router.post("/card/categories")
def add_card_category(name: str = Form(...), nameEn: str = Form(...)):
    try:
        category_id = card_config_service.add_card_category(name, nameEn)
        return {
            "message": "Card category added successfully",
            "category_id": category_id,
            "category": card_config_service.get_card_category_by_id(category_id)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add card category: {str(e)}")

@router.post("/card/types")
def add_card_type(name: str = Form(...), nameEn: str = Form(...)):
    try:
        type_id = card_config_service.add_card_type(name, nameEn)
        return {
            "message": "Card type added successfully",
            "type_id": type_id,
            "type": card_config_service.get_card_type_by_id(type_id)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add card type: {str(e)}")

@router.put("/card/categories/{category_id}")
def update_card_category(category_id: int, name: str = None, nameEn: str = None, is_active: bool = None):
    """Update existing card category in database."""
    try:
        success = card_config_service.update_card_category(category_id, name, nameEn, is_active)
        if success:
            return {
                "message": "Card category updated successfully",
                "category": card_config_service.get_card_category_by_id(category_id)
            }
        else:
            raise HTTPException(status_code=404, detail="Card category not found or no changes made")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update card category: {str(e)}")

@router.put("/card/types/{type_id}")
def update_card_type(type_id: int, name: str = None, nameEn: str = None, is_active: bool = None):
    """Update existing card type in database."""
    try:
        success = card_config_service.update_card_type(type_id, name, nameEn, is_active)
        if success:
            return {
                "message": "Card type updated successfully",
                "type": card_config_service.get_card_type_by_id(type_id)
            }
        else:
            raise HTTPException(status_code=404, detail="Card type not found or no changes made")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update card type: {str(e)}")

@router.delete("/card/categories/{category_id}")
def delete_card_category(category_id: int):
    """Soft delete card category (set is_active = 0)."""
    try:
        success = card_config_service.delete_card_category(category_id)
        if success:
            return {"message": f"Card category {category_id} deactivated successfully"}
        else:
            raise HTTPException(status_code=404, detail="Card category not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete card category: {str(e)}")

@router.delete("/card/types/{type_id}")
def delete_card_type(type_id: int):
    """Soft delete card type (set is_active = 0)."""
    try:
        success = card_config_service.delete_card_type(type_id)
        if success:
            return {"message": f"Card type {type_id} deactivated successfully"}
        else:
            raise HTTPException(status_code=404, detail="Card type not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete card type: {str(e)}")

@router.post("/card/detect",
            summary="Detect Vietnam Card Type",
            description="Detect and classify Vietnam Citizens Card or Driving License using YOLO + OCR")
async def detect_card(file: UploadFile = File(...)):
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
        model_path = os.path.join(pt_config.get_model_path(), "CCCD_FACE_DETECT_2025_NEW_TITLE.pt")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=500, detail=f"CCCD Old/New detection model not found at {model_path}")
        
        # Initialize OCR model for additional verification
        model_path_ocr = os.path.join(pt_config.get_model_path(), "OCR_QR_CCCD.pt")
        if not os.path.exists(model_path_ocr):
            raise HTTPException(status_code=500, detail=f"OCR detection model not found at {model_path_ocr}")
        
        detector = YOLODetector(model_path=model_path)
        ocr_detector = YOLODetector(model_path=model_path_ocr)

        # Detect card regions using the CCCD_OLD_NEW model
        detections = detector.detect(image)

        if not detections:
            return {"message": "No card regions detected.", "detections": []}
        
        # Detect OCR/QR information for additional verification
        ocr_detections = ocr_detector.detect(image)
        
        # Extract information types from OCR detections
        detected_info_types = set()
        for ocr_detection in ocr_detections:
            class_name = ocr_detection.get("class_name")
            if class_name:
                detected_info_types.add(class_name.lower())
        
        print(f"OCR detected info types: {detected_info_types}")
        
        # Analyze OCR features for CCCD classification
        has_portrait = "portrait" in detected_info_types
        has_qr_code = "qr_code" in detected_info_types
        has_basic_info = any(info in detected_info_types for info in ["name", "id", "birth", "sex"])
        has_address_info = any(info in detected_info_types for info in ["place_of_origin", "place_of_residence"])
        
        print(f"OCR Analysis - Portrait: {has_portrait}, QR: {has_qr_code}, Basic: {has_basic_info}, Address: {has_address_info}")
        
        # Use CardDetectionService to process detections
        file_detections, title_detected_type = card_detection_service.process_detections(
            detections, image, file.filename or "uploaded_file"
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
            file_detections, title_detected_type, detected_info_types, file.filename or "uploaded_file"
        )
        
        if not best_detection:
            return {"message": "No valid detections found.", "detections": []}
        
        # Return only the best detection
        return {
            "detections": [best_detection]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
