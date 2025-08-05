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
           response_description="List of card categories with Vietnamese and English names")
def get_card_types():
    """
    Get the list of card categories from database.
    
    **Output Example:**
    ```json
    [
        {
            "id": 0,
            "name": "Thẻ Căn Cước Công Dân", 
            "nameEn": "Citizens Card",
            "is_active": true
        },
        {
            "id": 1,
            "name": "Giấy Phép Lái Xe",
            "nameEn": "Driving License", 
            "is_active": true
        },
        {
            "id": 5,
            "name": "Thẻ Căn Cước Công Dân Mới",
            "nameEn": "New Citizens Card",
            "is_active": true
        }
    ]
    ```
    
    Returns:
        List of active card categories with bilingual names
    """
    return card_config_service.get_card_categories()

@router.get("/card/types",
           summary="Get Card Types", 
           description="Retrieve all active card types (front/back) from database",
           response_description="List of card types with Vietnamese and English names")
def get_card_types_list():
    """
    Get the list of card types from database.
    
    **Output Example:**
    ```json
    [
        {
            "id": 0,
            "name": "Mặt Trước",
            "nameEn": "Front",
            "is_active": true
        },
        {
            "id": 1, 
            "name": "Mặt Sau",
            "nameEn": "Back",
            "is_active": true
        }
    ]
    ```
    
    Returns:
        List of active card types (front/back orientation)
    """
    return card_config_service.get_card_types()

@router.get("/card/config",
           summary="Get Card Configuration Summary",
           description="Get complete card configuration summary including categories, types, and database info",
           response_description="Complete configuration summary with counts and database path")
def get_card_config():
    """
    Get complete card configuration summary from database.
    
    **Output Example:**
    ```json
    {
        "card_categories_count": 7,
        "card_types_count": 2,
        "card_categories": [...],
        "card_types": [...],
        "database_path": "c:\\WorkSpace\\Rest\\.local\\share\\database\\app.db"
    }
    ```
    
    Returns:
        Configuration summary with counts, data, and system information
    """
    return card_config_service.get_config_summary()

@router.post("/card/categories",
            summary="Add New Card Category",
            description="Add a new card category to the database with Vietnamese and English names",
            response_description="Success response with created category information")
def add_card_category(name: str = Form(..., description="Vietnamese name for the card category", example="Thẻ Học Sinh"),
                     nameEn: str = Form(..., description="English name for the card category", example="Student Card")):
    """
    Add new card category to database.
    
    **Input Example:**
    ```
    name: "Thẻ Học Sinh"
    nameEn: "Student Card"
    ```
    
    **Output Example:**
    ```json
    {
        "message": "Card category added successfully",
        "category_id": 7,
        "category": {
            "id": 7,
            "name": "Thẻ Học Sinh",
            "nameEn": "Student Card",
            "is_active": true
        }
    }
    ```
    
    Returns:
        Success message with new category ID and details
        
    Raises:
        HTTPException: 500 if database operation fails
    """
    """
    Endpoint to add new card category to database.
    """
    try:
        category_id = card_config_service.add_card_category(name, nameEn)
        return {
            "message": "Card category added successfully",
            "category_id": category_id,
            "category": card_config_service.get_card_category_by_id(category_id)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add card category: {str(e)}")

@router.post("/card/types",
            summary="Add New Card Type", 
            description="Add a new card type (orientation) to the database with Vietnamese and English names",
            response_description="Success response with created card type information")
def add_card_type(name: str = Form(..., description="Vietnamese name for the card type", example="Mặt Bên"),
                 nameEn: str = Form(..., description="English name for the card type", example="Side")):
    """
    Add new card type to database.
    
    **Input Example:**
    ```
    name: "Mặt Bên"
    nameEn: "Side"
    ```
    
    **Output Example:**
    ```json
    {
        "message": "Card type added successfully",
        "type_id": 2,
        "type": {
            "id": 2,
            "name": "Mặt Bên",
            "nameEn": "Side",
            "is_active": true
        }
    }
    ```
    
    Returns:
        Success message with new card type ID and details
        
    Raises:
        HTTPException: 500 if database operation fails
    """
    """
    Endpoint to add new card type to database.
    """
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
    """
    Endpoint to update existing card category in database.
    """
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
    """
    Endpoint to update existing card type in database.
    """
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
    """
    Endpoint to soft delete card category (set is_active = 0).
    """
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
    """
    Endpoint to soft delete card type (set is_active = 0).
    """
    try:
        success = card_config_service.delete_card_type(type_id)
        if success:
            return {"message": f"Card type {type_id} deactivated successfully"}
        else:
            raise HTTPException(status_code=404, detail="Card type not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete card type: {str(e)}")

@router.get("/card/demo")
def demo_supported_cards():
    """
    Demo endpoint showing all supported card types and labels.
    """
    return {
        "supported_models": {
            "CCCD_OLD_NEW.pt": {
                "description": "Unified model for Vietnam Citizens Cards and Driving License",
                "supported_labels": [
                    "cccd_qr_front", "cccd_qr_back",  # CCCD Cũ
                    "cccd_new_front", "cccd_new_back",  # CCCD Mới
                    "gplx_front", "gplx_back"  # Giấy Phép Lái Xe
                ]
            },
            "OCR_QR_CCCD.pt": {
                "description": "OCR detection model for additional verification",
                "supported_features": ["portrait", "qr_code", "name", "id", "birth", "sex", "place_of_origin", "place_of_residence"]
            }
        },
        "card_categories": card_config_service.get_card_categories(),
        "card_types": card_config_service.get_card_types(),
        "classification_rules": {
            "rule_1": "QR code detected → CCCD Cũ",
            "rule_2": "Portrait + Basic info (no QR) → CCCD Mới or GPLX",
            "rule_3": "Basic info only (no portrait, no QR) → GPLX Back",
            "rule_4": "QR only (no portrait) → Back side",
            "rule_5": "Fallback → CCCD Cũ"
        }
    }
@router.post("/card/detect",
            response_model=CardDetectionResponse,
            summary="Detect Vietnam Card Type",
            description="Detect and classify Vietnam Citizens Card or Driving License from uploaded image using YOLO + OCR analysis",
            response_description="Card detection results with classification, confidence scores, and OCR features")
async def detect_card(file: UploadFile = File(..., description="Image file containing Vietnam card (JPG, PNG, JPEG format)")):
    """
    Detect and classify Vietnam Citizens Card or Driving License from uploaded image.
    
    **Detection Process:**
    1. YOLO model detects card regions (CCCD_OLD_NEW_GPLX.pt)
    2. OCR model extracts features (OCR_QR_CCCD.pt) 
    3. Smart classification using OCR-based rules
    4. Return best detection with confidence scores
    
    **Supported Card Types:**
    - Thẻ Căn Cước Công Dân Cũ (Old Citizens Card) - với QR code
    - Thẻ Căn Cước Công Dân Mới (New Citizens Card) - không QR code  
    - Giấy Phép Lái Xe (Driving License) - mặt trước và sau
    
    **Output Example (CCCD Cũ - Mặt Trước):**
    ```json
    {
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
                "is_valid_card": true,
                "ocr_features": {
                    "has_portrait": true,
                    "has_qr_code": true,
                    "has_basic_info": true,
                    "has_address_info": true,
                    "detected_info_types": ["portrait", "qr_code", "name", "id", "birth"]
                }
            }
        ]
    }
    ```
    
    **Output Example (GPLX - Mặt Trước):**
    ```json
    {
        "detections": [
            {
                "confidence": 0.89,
                "detected_label": "gplx_front", 
                "card_category": {
                    "id": 1,
                    "name": "Giấy Phép Lái Xe", 
                    "nameEn": "Driving License"
                },
                "card_type": {
                    "id": 0,
                    "name": "Mặt Trước",
                    "nameEn": "Front"
                },
                "is_valid_card": true,
                "ocr_features": {
                    "has_portrait": true,
                    "has_qr_code": false,
                    "has_basic_info": true,
                    "detected_info_types": ["portrait", "name", "id", "birth"]
                }
            }
        ]
    }
    ```
    
    **Output Example (No Detection):**
    ```json
    {
        "message": "No card regions detected.",
        "detections": []
    }
    ```
    
    **Classification Rules:**
    - Rule 1: QR code detected → CCCD Cũ
    - Rule 2: Portrait + Basic info (no QR) → CCCD Mới or GPLX  
    - Rule 3: Basic info only (no portrait, no QR) → GPLX Back
    - Rule 4: QR only (no portrait) → Back side
    - Rule 5: Fallback → CCCD Cũ
    
    Returns:
        Detection results with card classification and OCR feature analysis
        
    Raises:
        HTTPException: 400 if file format is invalid or image is corrupted
        HTTPException: 500 if model loading fails or internal processing error
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
        model_path = os.path.join(pt_config.get_model_path(), "CCCD_OLD_NEW_GPLX.pt")
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
        
        print(f"OCR detected info types: {detected_info_types}")  # Debugging
        
        # Analyze OCR features for CCCD classification
        has_portrait = "portrait" in detected_info_types
        has_qr_code = "qr_code" in detected_info_types
        has_basic_info = any(info in detected_info_types for info in ["name", "id", "birth", "sex"])
        has_address_info = any(info in detected_info_types for info in ["place_of_origin", "place_of_residence"])
        
        print(f"OCR Analysis - Portrait: {has_portrait}, QR: {has_qr_code}, Basic: {has_basic_info}, Address: {has_address_info}")
        
        # Prepare response with detected regions
        response = {
            "detections": []
        }
        
        # Map detection labels to card categories and types
        for detection in detections:
            # Extract detection keys for debugging/logging
            detection_keys = list(detection.keys())
            print(f"Detection keys: {detection_keys}")  # Debugging line
            
            class_name = detection.get("class_name")
            confidence = detection.get("confidence", 0.0)
            
            if class_name:
                print(f"Detected class name: {class_name}, confidence: {confidence}")  # Debugging line
                label = class_name.lower()
                
                if label == "cccd_qr_front":
                    card_category = card_config_service.get_card_category_by_id(0)  # Thẻ Căn Cước Công Dân (Cũ)
                    card_type = card_config_service.get_card_type_by_id(0)  # Mặt Trước
                elif label == "cccd_qr_back":
                    card_category = card_config_service.get_card_category_by_id(0)  # Thẻ Căn Cước Công Dân (Cũ)
                    card_type = card_config_service.get_card_type_by_id(1)  # Mặt Sau
                elif label == "cccd_new_front":
                    card_category = card_config_service.get_card_category_by_id(5)  # Thẻ Căn Cước Công Dân Mới
                    card_type = card_config_service.get_card_type_by_id(0)  # Mặt Trước
                elif label == "cccd_new_back":
                    card_category = card_config_service.get_card_category_by_id(5)  # Thẻ Căn Cước Công Dân Mới
                    card_type = card_config_service.get_card_type_by_id(1)  # Mặt Sau
                elif label == "gplx_front":
                    card_category = card_config_service.get_card_category_by_id(1)  # Giấy Phép Lái Xe
                    card_type = card_config_service.get_card_type_by_id(0)  # Mặt Trước
                elif label == "gplx_back":
                    card_category = card_config_service.get_card_category_by_id(1)  # Giấy Phép Lái Xe
                    card_type = card_config_service.get_card_type_by_id(1)  # Mặt Sau
                else:
                    # Unknown or unrecognized label
                    card_category = {"id": -1, "name": "Unknown", "nameEn": "Unknown"}
                    card_type = {"id": -1, "name": "Unknown", "nameEn": "Unknown"}
                    label = "unknown"
            else:
                # No class name found
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
            
            response["detections"].append(detection_result)
        
        # Sort detections by confidence (highest first) và áp dụng logic chọn detection
        if response["detections"]:
            response["detections"].sort(key=lambda x: x["confidence"], reverse=True)
            
            # Log all detections for debugging
            print(f"All detections found:")
            for i, det in enumerate(response["detections"]):
                print(f"  {i+1}: {det['detected_label']} - confidence: {det['confidence']:.3f}")
            
            # Áp dụng logic chọn detection thông minh
            best_detection = response["detections"][0]
            
            # Nếu có nhiều detections và confidence gần nhau, áp dụng rules với OCR
            if len(response["detections"]) > 1:
                first = response["detections"][0]
                second = response["detections"][1]
                confidence_diff = first["confidence"] - second["confidence"]
                
                print(f"Confidence difference: {confidence_diff:.3f}")
                
                # Nếu chênh lệch confidence nhỏ (< 0.1), áp dụng OCR-based rules
                if confidence_diff < 0.1:
                    print("Applying OCR-based classification rules...")
                    
                    # Rule 1: Nếu có QR code trong OCR → Ưu tiên CCCD cũ
                    if has_qr_code:
                        for det in response["detections"]:
                            if det["detected_label"] in ["cccd_qr_front", "cccd_qr_back"]:
                                best_detection = det
                                print(f"OCR Rule: QR detected → Switched to QR CCCD: {det['detected_label']}")
                                break
                    
                    # Rule 2: Nếu có portrait + basic info nhưng KHÔNG có QR → Ưu tiên CCCD mới hoặc GPLX
                    elif has_portrait and has_basic_info and not has_qr_code:
                        # Ưu tiên CCCD mới trước, sau đó GPLX
                        for det in response["detections"]:
                            if det["detected_label"] in ["cccd_new_front", "cccd_new_back"]:
                                best_detection = det
                                print(f"OCR Rule: Portrait+Basic but no QR → Switched to New CCCD: {det['detected_label']}")
                                break
                        else:
                            # Nếu không có CCCD mới, thử GPLX
                            for det in response["detections"]:
                                if det["detected_label"] in ["gplx_front", "gplx_back"]:
                                    best_detection = det
                                    print(f"OCR Rule: Portrait+Basic but no QR → Switched to GPLX: {det['detected_label']}")
                                    break
                    
                    # Rule 3: Nếu chỉ có basic info mà không có portrait và QR → Có thể là GPLX mặt sau
                    elif has_basic_info and not has_portrait and not has_qr_code:
                        for det in response["detections"]:
                            if det["detected_label"] == "gplx_back":
                                best_detection = det
                                print(f"OCR Rule: Basic info only → Switched to GPLX back: {det['detected_label']}")
                                break
                    
                    # Rule 4: Nếu chỉ có QR mà không có portrait → Mặt sau
                    elif has_qr_code and not has_portrait:
                        for det in response["detections"]:
                            if "back" in det["detected_label"]:
                                best_detection = det
                                print(f"OCR Rule: QR only → Switched to back side: {det['detected_label']}")
                                break
                    
                    # Rule 5: Fallback - ưu tiên CCCD cũ vì model được train cho CCCD cũ
                    else:
                        for det in response["detections"]:
                            if det["detected_label"] in ["cccd_qr_front", "cccd_qr_back"]:
                                best_detection = det
                                print(f"OCR Rule: Fallback → Switched to QR CCCD: {det['detected_label']}")
                                break
            
            print(f"Final best detection: {best_detection['detected_label']} with confidence {best_detection['confidence']:.3f}")
            
            response["detections"] = [best_detection]
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

