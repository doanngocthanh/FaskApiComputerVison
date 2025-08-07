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
from PIL import Image
from config import BlogConfig
from pathlib import Path
import re

# Define ErrorResponse model if not imported from __init__
class ErrorResponse(BaseModel):
    detail: str
    error: Optional[str] = None

# Router setup
router = APIRouter(
    prefix="/api/business",
    tags=["Business Config"],
    responses={
        404: {"model": ErrorResponse, "description": "Resource not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
is_business_config= False
@router.get("/health", summary="Get Business Configurations")
async def health_check():
    """
    Health check endpoint to verify if the business configurations are loaded correctly.
    Returns a simple message indicating the service is running.
    """
    return {"business": is_business_config}
