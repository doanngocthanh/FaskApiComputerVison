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
    tags=["MRZ Detection"],
    responses={
        404: {"model": ErrorResponse, "description": "Resource not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
@router.post("/mrz/ext")
async def mrz(file: UploadFile = File(...)):
    """
    Endpoint to extract text from MRZ regions in uploaded image.
    """
    # Validate file type
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
        model_path = os.path.join(pt_config.get_model_path(), "MRZ.pt")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=500, detail=f"MRZ model not found at {model_path}")
        
        detector = YOLODetector(model_path=model_path)
        
        # Detect MRZ regions using numpy array (image)
        detections = detector.detect(image)
        
        if not detections:
            return {
                "status": "no_mrz_detected",
                "message": "No MRZ regions detected in the image.",
                "texts": []
            }
        
        # Create temp directory
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Initialize OCR
        ocr = PaddleOCR()
        
        # Process each detected MRZ region and collect all texts
        all_texts = []
       
        for i, detection in enumerate(detections):
            try:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, detection['bbox'])
                
                # Crop the detected region
                cropped_image = image[y1:y2, x1:x2]
                
                if cropped_image.size == 0:
                    continue
                
                # Save cropped image temporarily for OCR processing
                cropped_filename = f"cropped_mrz_{uuid.uuid4()}.jpg"
                cropped_path = os.path.join(temp_dir, cropped_filename)
                cv2.imwrite(cropped_path, cropped_image)
                temp_files.append(cropped_path)
                
                # Process with OCR
                ocr_result = ocr.process_image(cropped_path)
                
                # Extract texts from OCR result
                if isinstance(ocr_result, dict):
                    texts = ocr_result.get('texts', [])
                    all_texts.extend(texts)
                
            except Exception as e:
                # Log error but continue with other detections
                print(f"Error processing detection {i}: {str(e)}")
                continue
        
        # Return only the extracted texts
        return {
            "status": "success",
            "texts": all_texts
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        # Clean up all temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Warning: Could not remove temp file {temp_file}: {str(e)}")
