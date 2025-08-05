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

@router.post("/mrz/parse", 
            response_model=MRZParseResponse,
            summary="Parse MRZ String",
            description="Parse Machine Readable Zone (MRZ) string from Vietnam Citizens Card into human-readable information",
            response_description="Parsed MRZ information including personal details, dates, and document information")
async def parse_mrz(request: MRZParseRequest):
    """
    Parse MRZ string into human-readable information.
    
    **Input Example:**
    ```json
    {
        "mrz_string": "IDVNM2010029147087201002914<<80110130M2610139VNM<<<<<<<<4HA<<TRUONG<GIANG<<<<<<<<<<<<<<<<",
        "validate_checksum": false
    }
    ```
    
    **Output Example:**
    ```json
    {
        "status": "success",
        "message": "MRZ parsed successfully",
        "mrz_type": "TD1",
        "data": {
            "document_type": "ID",
            "issuing_country": "VNM", 
            "document_number": "201002914708720",
            "full_name": "TRUONG GIANG",
            "birth_date": "10/01/30",
            "expiry_date": "26/10/39",
            "dates_found": ["10/01/30", "26/10/39"]
        }
    }
    ```
    
    Args:
        request: MRZ parse request containing mrz_string and optional validation settings
        
    Returns:
        Parsed MRZ information including personal details, document info, and extracted dates
        
    Raises:
        HTTPException: 400 if MRZ string is invalid or empty
        HTTPException: 500 if internal parsing error occurs
    """
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
        
        return MRZParseResponse(**result)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/mrz/parse/{mrz_string}",
           summary="Parse MRZ via GET Request", 
           description="Quick MRZ parsing via GET request for testing purposes",
           response_description="Parsed MRZ information in JSON format")
async def parse_mrz_get(mrz_string: str):
    """
    Parse MRZ string via GET request (for quick testing).
    
    **Example URL:**
    ```
    GET /api/v1/mrz/parse/IDVNM2010029147087201002914<<80110130M2610139VNM<<<<<<<<4HA<<TRUONG<GIANG<<<<<<<<<<<<<<<<
    ```
    
    **Output Example:**
    ```json
    {
        "status": "success",
        "message": "MRZ parsed successfully", 
        "data": {
            "document_type": "ID",
            "issuing_country": "VNM",
            "full_name": "TRUONG GIANG",
            "document_number": "201002914708720",
            "dates_found": ["10/01/30", "26/10/39"]
        }
    }
    ```
    
    Args:
        mrz_string: MRZ string to parse (URL encoded)
        
    Returns:
        Parsed MRZ information in JSON format
        
    Raises:
        HTTPException: 400 if MRZ string is invalid or empty
        HTTPException: 500 if internal parsing error occurs
    """
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
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/mrz/demo",
            summary="MRZ Parsing Demo",
            description="Demo endpoint with sample Vietnam Citizens Card MRZ data for testing",
            response_description="Demo response showing parsed sample MRZ data")
async def parse_mrz_demo():
    """
    Demo endpoint with sample MRZ data from Vietnam Citizens Card.
    
    **Sample MRZ Used:**
    ```
    IDVNM2010029147087201002914<<80110130M2610139VNM<<<<<<<<4HA<<TRUONG<GIANG<<<<<<<<<<<<<<<<
    ```
    
    **Output Example:**
    ```json
    {
        "demo": true,
        "sample_mrz": "IDVNM2010029147087201002914<<80110130M2610139VNM<<<<<<<<4HA<<TRUONG<GIANG<<<<<<<<<<<<<<<<",
        "parsed_result": {
            "status": "success",
            "data": {
                "document_type": "ID",
                "issuing_country": "VNM",
                "document_number": "201002914708720", 
                "full_name": "TRUONG GIANG",
                "birth_date": "10/01/30",
                "expiry_date": "26/10/39",
                "sex": "M",
                "nationality": "VNM",
                "dates_found": ["10/01/30", "26/10/39"]
            }
        }
    }
    ```
    
    Returns:
        Demo response with sample MRZ and its parsed information
        
    Raises:
        HTTPException: 500 if demo parsing fails
    """
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
