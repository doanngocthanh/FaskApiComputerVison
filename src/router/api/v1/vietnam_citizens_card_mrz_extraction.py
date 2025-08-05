from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional, List
import uuid
from src.router.api.__init__ import *  
from src.service.MRZExtractor import MRZExtractor

# Router setup
router = APIRouter(
    prefix="/api/v1",
    tags=["MRZ Detection"],
    responses={
        404: {"model": ErrorResponse, "description": "Resource not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
@router.post("/mrz/ext",
            summary="Extract MRZ from Image", 
            description="Extract Machine Readable Zone (MRZ) text from uploaded Vietnam Citizens Card image using YOLO detection and PaddleOCR",
            response_description="Extracted MRZ text with detected dates and processing information")
async def mrz(file: UploadFile = File(..., description="Image file containing Vietnam Citizens Card (JPG, PNG, JPEG format)")):
    """
    Extract MRZ text from uploaded Vietnam Citizens Card image.
    
    **Process Flow:**
    1. YOLO model detects MRZ regions in the image
    2. PaddleOCR extracts text from detected regions  
    3. Text processing and date extraction (5 different date formats)
    4. Return combined results with confidence scores
    
    **Supported Image Formats:** JPG, PNG, JPEG
    
    **Output Example (Success):**
    ```json
    {
        "status": "success",
        "message": "MRZ extracted successfully",
        "texts": [
            "IDVNM2010029147087201002914<<8",
            "0110130M2610139VNM<<<<<<<<4", 
            "HA<<TRUONG<GIANG<<<<<<<<<<<<"
        ],
        "confidence_scores": [0.95, 0.92, 0.89],
        "processing_time": 1.23,
        "dates_found": ["10/01/30", "26/10/39"],
        "date_extraction": {
            "total_dates_found": 2,
            "date_patterns_used": ["dd/mm/yy"],
            "extraction_method": "regex_pattern_matching"
        }
    }
    ```
    
    **Output Example (No MRZ Found):**
    ```json
    {
        "status": "no_mrz_detected",
        "message": "No MRZ regions detected in the image.",
        "texts": []
    }
    ```
    
    **Output Example (OCR Failed):**
    ```json
    {
        "status": "ocr_failed", 
        "message": "OCR processing failed.",
        "texts": []
    }
    ```
    
    Returns:
        JSON response containing extracted MRZ text, dates, and processing metadata
        
    Raises:
        HTTPException: 400 if file format is invalid
        HTTPException: 500 if internal processing error occurs
    """
    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, png, jpeg).")
    
    try:
        # Read file content
        content = await file.read()
        
        # Initialize MRZ extractor service
        mrz_extractor = MRZExtractor()
        
        # Extract MRZ using service
        result = mrz_extractor.extract_mrz_from_bytes(content)
        
        # Handle different result statuses
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        elif result["status"] == "no_mrz_detected":
            return {
                "status": "no_mrz_detected",
                "message": "No MRZ regions detected in the image.",
                "texts": []
            }
        elif result["status"] == "ocr_failed":
            return {
                "status": "ocr_failed",
                "message": "OCR processing failed.",
                "texts": []
            }
        
        # Return successful result (dates are already included by MRZExtractor service)
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")