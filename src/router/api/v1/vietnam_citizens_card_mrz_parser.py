from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.router.api.__init__ import *
from src.service.MRZParser import MRZParser

# Router setup
router = APIRouter(
    prefix="/api/v1",
    tags=["MRZ Parser"],
    responses={
        404: {"model": ErrorResponse, "description": "Resource not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)

class MRZParseRequest(BaseModel):
    """Request model for MRZ parsing."""
    mrz_string: str
    validate_checksum: Optional[bool] = False
    
    class Config:
        schema_extra = {
            "example": {
                "mrz_string": "IDVNM2010029147087201002914<<80110130M2610139VNM<<<<<<<<4HA<<TRUONG<GIANG<<<<<<<<<<<<<<<<",
                "validate_checksum": False
            }
        }

class MRZParseResponse(BaseModel):
    """Response model for MRZ parsing."""
    status: str
    message: str
    mrz_type: Optional[str] = None
    raw_mrz: Optional[str] = None
    parsed_lines: Optional[dict] = None
    data: dict
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "MRZ parsed successfully",
                "mrz_type": "TD1",
                "raw_mrz": "IDVNM2010029147087201002914<<80110130M2610139VNM<<<<<<<<4HA<<TRUONG<GIANG<<<<<<<<<<<<<<<<",
                "parsed_lines": {
                    "line1": "IDVNM2010029147087201002914<<8",
                    "line2": "0110130M2610139VNM<<<<<<<<4",
                    "line3": "HA<<TRUONG<GIANG<<<<<<<<<<<<"
                },
                "data": {
                    "document_type": "ID",
                    "issuing_country": "VNM",
                    "document_number": "201002914708720",
                    "birth_date": "10/01/30",
                    "expiry_date": "26/10/39",
                    "nationality": "VNM",
                    "sex": "M",
                    "surname": "TRUONG",
                    "given_names": "GIANG",
                    "full_name": "TRUONG GIANG",
                    "dates_found": ["10/01/30", "26/10/39"]
                }
            }
        }

@router.post("/mrz/parse")
async def parse_mrz(request: MRZParseRequest):
    try:
        # Validate input
        if not request.mrz_string or not request.mrz_string.strip():
            raise HTTPException(status_code=400, detail="MRZ string is required and cannot be empty")
        
        # Initialize MRZ parser
        parser = MRZParser()
        
        # Parse MRZ string
        result = parser.parse_mrz_string(request.mrz_string)
        
        # Handle parsing errors
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/mrz/parse/{mrz_string}",
           summary="Parse MRZ via GET", 
           description="Quick MRZ parsing for testing")
async def parse_mrz_get(mrz_string: str):
    try:
        # Validate input
        if not mrz_string or not mrz_string.strip():
            raise HTTPException(status_code=400, detail="MRZ string is required and cannot be empty")
        
        # Initialize MRZ parser
        parser = MRZParser()
        
        # Parse MRZ string
        result = parser.parse_mrz_string(mrz_string)
        
        # Handle parsing errors
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/mrz/demo",
            summary="MRZ Demo",
            description="Demo with sample Vietnam Citizens Card MRZ")
async def parse_mrz_demo():
    try:
        # Sample MRZ from user's request
        sample_mrz = "IDVNM2010029147087201002914<<80110130M2610139VNM<<<<<<<<4HA<<TRUONG<GIANG<<<<<<<<<<<<<<<"
        
        # Initialize MRZ parser
        parser = MRZParser()
        
        # Parse sample MRZ
        result = parser.parse_mrz_string(sample_mrz)
        
        return {
            "demo": True,
            "sample_mrz": sample_mrz,
            "parsed_result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo error: {str(e)}")
