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
    tags=["VietNam Citizens Card MRZ Extraction"],
    responses={
        404: {"model": ErrorResponse, "description": "Resource not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
@router.post("/mrz/ext")
async def mrz(request: Request, file: UploadFile = File(...)):
    """
    Endpoint to extract text from MRZ regions in uploaded image.
    """
    start_time = time.time()
    
    # Prepare logging data
    log_data = {
        'endpoint': '/api/v1/mrz/ext',
        'method': 'POST',
        'file_size': 0,
        'request_body': {'filename': file.filename},
        'ip_address': request.client.host if request.client else 'unknown'
    }
    
    # Validate file type
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
        model_path = os.path.join(pt_config.get_model_path(), "MRZ.pt")
        if not os.path.exists(model_path):
            error_msg = f"MRZ model not found at {model_path}"
            log_data.update({
                'success': False,
                'error_message': error_msg,
                'processing_time_ms': int((time.time() - start_time) * 1000)
            })
            logging_manager.log_request(log_data)
            raise HTTPException(status_code=500, detail=error_msg)
        
        detector = YOLODetector(model_path=model_path)
        
        # Detect MRZ regions using numpy array (image)
        detections = detector.detect(image)
        
        if not detections:
            response = {
                "status": "no_mrz_detected",
                "message": "No MRZ regions detected in the image.",
                "texts": [],
                "total_detections": 0,
                "total_texts": 0,
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
            
            # Log successful response (no detections is still success)
            log_data.update({
                'success': True,
                'processing_time_ms': response["processing_time_ms"],
                'total_detections': 0,
                'total_texts': 0,
                'response_body': response
            })
            logging_manager.log_request(log_data)
            return response
        
        # Initialize EasyOCR manager
        ocr_manager = EasyOCRManager()
        
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
                
                # Process with EasyOCR directly from numpy array
                extracted_text = ocr_manager.extract_text(cropped_image)
                
                if extracted_text and extracted_text.strip():
                    all_texts.append(extracted_text.strip())
                
            except Exception as e:
                # Log error but continue with other detections
                print(f"Error processing detection {i}: {str(e)}")
                continue
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        # Return extracted texts
        response = {
            "status": "success",
            "texts": all_texts,
            "total_detections": len(detections),
            "total_texts": len(all_texts),
            "processing_time_ms": processing_time
        }
        
        # Update log data with success info
        log_data.update({
            'success': True,
            'processing_time_ms': processing_time,
            'total_detections': len(detections),
            'total_texts': len(all_texts),
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
        # Handle unexpected errors
        error_msg = f"Internal server error: {str(e)}"
        log_data.update({
            'success': False,
            'error_message': error_msg,
            'processing_time_ms': int((time.time() - start_time) * 1000)
        })
        logging_manager.log_request(log_data)
        raise HTTPException(status_code=500, detail=error_msg)
    
    finally:
        # Clean up all temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Warning: Could not remove temp file {temp_file}: {str(e)}")
