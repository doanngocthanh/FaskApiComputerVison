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
@router.post("/mrz/ext")
async def mrz(file: UploadFile = File(...)):
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