from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import uuid
import os
import cv2
import numpy as np
from io import BytesIO
from src.router.api.__init__ import *  
from src.service.YOLODetector import YOLODetector
from src.service.MultipleCardDetectionService import multiple_card_detection_service
from config import PtConfig

# Response Models
class CardCategory(BaseModel):
    """Card category information"""
    id: int
    name: str
    nameEn: str
    is_active: Optional[bool] = True

class CardType(BaseModel):
    """Card type information (front/back)"""
    id: int
    name: str
    nameEn: str
    is_active: Optional[bool] = True

class OCRFeatures(BaseModel):
    """OCR features detected in the card"""
    has_portrait: bool
    has_qr_code: bool
    has_basic_info: bool
    has_address_info: bool
    detected_info_types: List[str]

class MultipleCardDetection(BaseModel):
    """Single card detection result with location"""
    confidence: float = Field(..., description="Detection confidence score (0.0-1.0)")
    detected_label: str = Field(..., description="Detected card type label")
    bbox: List[int] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")
    card_category: CardCategory
    card_type: CardType
    is_valid_card: bool = Field(..., description="Whether this is a recognized valid card type")
    title_detected_type: Optional[str] = Field(None, description="Card type detected from title text analysis")
    ocr_features: OCRFeatures
    group_id: int = Field(..., description="ID of the detection group (same physical card)")
    group_size: int = Field(..., description="Number of detections in the same group")
    
    class Config:
        schema_extra = {
            "example": {
                "confidence": 0.95,
                "detected_label": "cccd_new_front",
                "bbox": [100, 50, 500, 300],
                "card_category": {
                    "id": 5,
                    "name": "Thẻ Căn Cước Công Dân Mới",
                    "nameEn": "New Citizens Card",
                    "is_active": True
                },
                "card_type": {
                    "id": 0,
                    "name": "Mặt Trước",
                    "nameEn": "Front",
                    "is_active": True
                },
                "is_valid_card": True,
                "title_detected_type": "cccd_new_front",
                "ocr_features": {
                    "has_portrait": True,
                    "has_qr_code": False,
                    "has_basic_info": True,
                    "has_address_info": True,
                    "detected_info_types": ["portrait", "name", "id", "birth", "place_of_origin"]
                },
                "group_id": 0,
                "group_size": 2
            }
        }

class MultipleCardDetectionResponse(BaseModel):
    """Response for multiple card detection"""
    message: Optional[str] = None
    total_cards_found: int = Field(..., description="Total number of cards detected")
    detections: List[MultipleCardDetection] = Field(..., description="List of all detected cards with their locations")
    
    class Config:
        schema_extra = {
            "example": {
                "total_cards_found": 2,
                "detections": [
                    {
                        "confidence": 0.95,
                        "detected_label": "cccd_new_front",
                        "bbox": [100, 50, 500, 300],
                        "card_category": {
                            "id": 5,
                            "name": "Thẻ Căn Cước Công Dân Mới",
                            "nameEn": "New Citizens Card"
                        },
                        "card_type": {
                            "id": 0,
                            "name": "Mặt Trước", 
                            "nameEn": "Front"
                        },
                        "is_valid_card": True,
                        "group_id": 0,
                        "group_size": 2
                    },
                    {
                        "confidence": 0.88,
                        "detected_label": "cccd_new_back",
                        "bbox": [600, 50, 1000, 300],
                        "card_category": {
                            "id": 5,
                            "name": "Thẻ Căn Cước Công Dân Mới",
                            "nameEn": "New Citizens Card"
                        },
                        "card_type": {
                            "id": 1,
                            "name": "Mặt Sau",
                            "nameEn": "Back"
                        },
                        "is_valid_card": True,
                        "group_id": 1,
                        "group_size": 1
                    }
                ]
            }
        }

router = APIRouter(
    prefix="/api/v1/multiple",
    tags=["Multiple VietNam Citizens Card Detection"],
    responses={
        404: {"model": ErrorResponse, "description": "Resource not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)

@router.post("/card/detect",
            summary="Detect Multiple Vietnam Cards in Single Image",
            description="Detect and classify multiple Vietnam Citizens Cards or Driving Licenses in a single image using YOLO + OCR",
            response_model=MultipleCardDetectionResponse)
async def detect_multiple_cards(
    file: UploadFile = File(..., description="Image file containing multiple cards"),
    include_bbox: bool = Query(True, description="Include bounding box coordinates in response")
):
    """
    Detect multiple Vietnam Citizens Cards or Driving Licenses in a single uploaded image.
    
    This endpoint can detect and classify multiple cards in one image, including:
    - Vietnam Citizens Cards (old format with QR code)
    - Vietnam Citizens Cards (new format without QR code)  
    - Driving Licenses (GPLX)
    - Both front and back sides of cards
    
    The response includes bounding box coordinates for each detected card, allowing you to
    know the exact location of each card in the image.
    """
    if not file.filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, png, jpeg).")
    
    temp_files = []
    try:
        # Read file content
        content = await file.read()
        
        # Convert to numpy array for YOLO processing
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file or corrupted data.")
        
        # Initialize YOLO detectors
        pt_config = PtConfig()
        model_path = os.path.join(pt_config.get_model_path(), "CCCD_FACE_DETECT_2025_NEW_TITLE.pt")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=500, detail=f"CCCD Old/New detection model not found at {model_path}")
        
        model_path_ocr = os.path.join(pt_config.get_model_path(), "OCR_QR_CCCD.pt")
        if not os.path.exists(model_path_ocr):
            raise HTTPException(status_code=500, detail=f"OCR detection model not found at {model_path_ocr}")
        
        detector = YOLODetector(model_path=model_path)
        ocr_detector = YOLODetector(model_path=model_path_ocr)

        # Detect card regions
        detections = detector.detect(image)
        
        if not detections:
            return MultipleCardDetectionResponse(
                message="No card regions detected.",
                total_cards_found=0,
                detections=[]
            )
        
        # Detect OCR/QR information
        ocr_detections = ocr_detector.detect(image)
        
        print(f"Found {len(detections)} card detections and {len(ocr_detections)} OCR detections")
        
        # Use MultipleCardDetectionService to process all detections
        try:
            print(f"🔍 Starting multiple card detection processing...")
            results = multiple_card_detection_service.detect_multiple_cards(
                image, detections, ocr_detections, file.filename or "uploaded_file"
            )
            print(f"✅ Multiple card detection completed. Found {len(results)} results")
            
        except Exception as detection_error:
            print(f"❌ Error in multiple card detection: {str(detection_error)}")
            print(f"📊 Detection data: {len(detections)} detections, {len(ocr_detections)} OCR detections")
            # Print first few detections for debugging
            for i, det in enumerate(detections[:3]):
                print(f"   Detection {i}: {det}")
            raise HTTPException(status_code=500, detail=f"Detection processing failed: {str(detection_error)}")
        
        # If include_bbox is False, remove bbox from response
        if not include_bbox:
            for result in results:
                if "bbox" in result:
                    del result["bbox"]
        
        return MultipleCardDetectionResponse(
            total_cards_found=len(results),
            detections=results
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

@router.post("/card/detect-batch",
            summary="Detect Multiple Cards in Multiple Images",
            description="Detect multiple cards in multiple uploaded images")
async def detect_multiple_cards_batch(
    files: List[UploadFile] = File(..., description="List of image files"),
    include_bbox: bool = Query(True, description="Include bounding box coordinates in response")
):
    """
    Batch processing: Detect multiple cards in multiple images.
    Each image can contain multiple cards.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    
    temp_files = []
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
                    "total_cards_found": 0,
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
                        "total_cards_found": 0,
                        "detections": []
                    })
                    continue
                
                # Detect card regions
                detections = detector.detect(image)
                
                if not detections:
                    results.append({
                        "filename": file.filename,
                        "success": True,
                        "message": "No card regions detected.",
                        "total_cards_found": 0,
                        "detections": []
                    })
                    continue
                
                # Detect OCR/QR information
                ocr_detections = ocr_detector.detect(image)
                
                # Use MultipleCardDetectionService to process all detections
                card_results = multiple_card_detection_service.detect_multiple_cards(
                    image, detections, ocr_detections, file.filename
                )
                
                # If include_bbox is False, remove bbox from response
                if not include_bbox:
                    for result in card_results:
                        if "bbox" in result:
                            del result["bbox"]
                
                # Add file result
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "total_cards_found": len(card_results),
                    "detections": card_results
                })
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e),
                    "total_cards_found": 0,
                    "detections": []
                })
        
        return {"results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
