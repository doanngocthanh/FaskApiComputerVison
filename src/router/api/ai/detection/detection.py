from fastapi import APIRouter,UploadFile, File, HTTPException
router = APIRouter()
from config import OnnxConfig, PthConfig, TensorflowConfig, ModelConfig,DBConfig
db_config = DBConfig()
import uuid
import os
from src.service.models_ai import ModelAi
@router.post("/api/ai/detection/")
def detection(File: UploadFile = File(...)):
    """
    Endpoint to handle detection.
    """
    
    return {"message": "Detection endpoint is working"}