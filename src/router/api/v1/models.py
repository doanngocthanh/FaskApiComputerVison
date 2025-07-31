from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Path
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid
import os
import importlib
from src.router.api.__init__ import *  
# Router setup
router = APIRouter(
    prefix="/api/v1/models",
    tags=["Model Management"],
    responses={
        404: {"model": ErrorResponse, "description": "Resource not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)

from config import OnnxConfig, PthConfig, TensorflowConfig, ModelConfig, DBConfig
db_config = DBConfig()

# Create table (move this to a separate migration script in production)
results = db_config.execute_query("""
    CREATE TABLE IF NOT EXISTS models (
        model_id VARCHAR(255) PRIMARY KEY, 
        model_name_id VARCHAR(255) NOT NULL,
        model_name VARCHAR(255) NOT NULL,
        model_type VARCHAR(50) NOT NULL,
        model_path TEXT NOT NULL,
        model_category VARCHAR(50), -- TTS, STT, OCR, YOLO, etc.
        model_size BIGINT,
        model_version VARCHAR(50),
        model_description TEXT,
        model_framework VARCHAR(50), -- pytorch, tensorflow, onnx, etc.
        input_shape VARCHAR(255),
        output_shape VARCHAR(255),
        model_accuracy FLOAT,
        training_dataset VARCHAR(255),
        model_tags TEXT,
        supported_languages VARCHAR(255), -- for TTS/STT models
        sample_rate INTEGER, -- for audio models (TTS/STT)
        max_input_length INTEGER, -- for text/audio processing
        confidence_threshold FLOAT, -- for detection models (YOLO, OCR)
        class_names TEXT, -- JSON array of class names for detection models
        upload_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_public BOOLEAN DEFAULT FALSE,
        uploaded_by VARCHAR(255),
        download_count INTEGER DEFAULT 0,
        model_status VARCHAR(50) DEFAULT 'active',
        package_class_prediction TEXT)
""")

@router.post(
    "/upload_model",
    response_model=ModelUploadResponse,
    status_code=201,
    summary="Upload a new machine learning model",
    description="""
    Upload a machine learning model file to the server. 
    
    **Supported file formats:**
    - `.onnx` - ONNX Runtime models
    - `.pth` - PyTorch models  
    - `.pt` - PyTorch models
    - `.pb` - TensorFlow models
    
    **Process:**
    1. Validates file extension
    2. Generates unique model ID
    3. Saves file to configured directory
    4. Stores metadata in database
    
    **Returns:** Model ID, filename, and success message
    """,
    responses={
        201: {"model": ModelUploadResponse, "description": "Model uploaded successfully"},
        400: {"model": ErrorResponse, "description": "Invalid file type or bad request"},
        500: {"model": ErrorResponse, "description": "Server error during upload"}
    }
)
async def upload_model(
    file: UploadFile = File(
        ..., 
        description="Model file to upload (onnx, pth, pt, pb)",
        media_type="application/octet-stream"
    )
):
    """Upload a model file and store its metadata."""
    # Validate file type based on ModelConfig
    allowed_extensions = ['onnx', 'pth', 'pb', 'pt']
    file_extension = file.filename.split('.')[-1].lower()
    print(f"Uploading file: {file.filename} with extension: {file_extension}")
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )
        
    try:
        # Generate unique model ID
        model_id = str(uuid.uuid4())
        
        # Save the uploaded file to a specific directory
        base_dir = ModelConfig.get_base_path()
        os.makedirs(base_dir, exist_ok=True)
        save_path = os.path.join(base_dir, f"{model_id}_{file.filename}")
        with open(save_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Insert model info into database - FIXED: Changed table name from 'v1/models' to 'models'
        insert_query = """
            INSERT INTO models (model_id, model_name_id, model_name, model_type, model_path, is_public, uploaded_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        db_config.execute_query(insert_query, (
            model_id,
            file.filename.rsplit('.', 1)[0],
            file.filename,
            file_extension,
            save_path,
            False,
            "anonymous"
        ))
        
        return ModelUploadResponse(
            model_id=model_id,
            filename=file.filename,
            message="Model uploaded and saved to database successfully."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List all uploaded models",
    description="""
    Retrieve a list of all models stored in the database.
    
    **Returns:** Complete list of models with their metadata including:
    - Model identifiers and names
    - File paths and types
    - Upload information
    - Public/private status
    """,
    responses={
        200: {"model": ModelListResponse, "description": "List of models retrieved successfully"},
        500: {"model": ErrorResponse, "description": "Database error"}
    }
)
async def list_models():
    """Retrieve all uploaded models from the database."""
    try:
        query = "SELECT * FROM models"
        models = db_config.execute_query(query)
        
        model_list = []
        for model in models:
            model_info = ModelInfo(
                model_id=model[0],
                model_name_id=model[1],
                model_name=model[2],
                model_type=model[3],
                model_path=model[4],
                model_category=model[5],
                model_size=model[6],
                model_version=model[7],
                model_description=model[8],
                model_framework=model[9],
                input_shape=model[10],
                output_shape=model[11],
                model_accuracy=model[12],
                training_dataset=model[13],
                model_tags=model[14],
                supported_languages=model[15],
                sample_rate=model[16],
                max_input_length=model[17],
                confidence_threshold=model[18],
                class_names=model[19],
                upload_at=model[20],
                last_updated=model[21],
                is_public=model[22],
                uploaded_by=model[23],
                download_count=model[24],
                model_status=model[25],
                package_class_prediction=model[26]
            )
            model_list.append(model_info)
        
        return ModelListResponse(models=model_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/directory",
    response_model=ModelDirectoryResponse,
    summary="List model files in directory",
    description="""
    Scan the model storage directory and return all model files found.
    
    **Purpose:** Compare database records with actual files on disk
    **Returns:** List of model files with their file system paths
    """,
    responses={
        200: {"model": ModelDirectoryResponse, "description": "Directory listing successful"},
        404: {"model": ErrorResponse, "description": "Model directory not found"},
        500: {"model": ErrorResponse, "description": "File system error"}
    }
)
async def list_models_directory():
    """List all model files in the storage directory."""
    try:
        base_dir = ModelConfig.get_base_path()
        if not os.path.exists(base_dir):
            raise HTTPException(status_code=404, detail="Model directory does not exist.")
        
        models = []
        for file_name in os.listdir(base_dir):
            if file_name.endswith(('.onnx', '.pth', '.pb', '.pt')):
                models.append(ModelDirectoryInfo(
                    file_name=file_name,
                    file_path=os.path.join(base_dir, file_name)
                ))
        
        return ModelDirectoryResponse(models=models)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/{model_id}",
    response_model=ModelInfo,
    summary="Get detailed model information",
    description="""
    Retrieve complete metadata for a specific model by its ID.
    
    **Returns:** All available information about the model including:
    - Basic info (name, type, path)
    - Technical details (framework, shapes, accuracy)
    - Metadata (tags, description, version)
    - Usage stats (download count, status)
    """,
    responses={
        200: {"model": ModelInfo, "description": "Model details retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Model not found"},
        500: {"model": ErrorResponse, "description": "Database error"}
    }
)
async def get_model_details(
    model_id: str = Path(..., description="Unique identifier of the model")
):
    """Get detailed information about a specific model."""
    try:
        query = "SELECT * FROM models WHERE model_id = ?"
        model = db_config.fetch_one(query, (model_id,))
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return ModelInfo(
            model_id=model[0],
            model_name_id=model[1],  
            model_name=model[2],
            model_type=model[3],
            model_path=model[4],
            model_category=model[5],
            model_size=model[6],
            model_version=model[7],
            model_description=model[8],
            model_framework=model[9],
            input_shape=model[10],
            output_shape=model[11],
            model_accuracy=model[12],
            training_dataset=model[13],
            model_tags=model[14],
            supported_languages=model[15],
            sample_rate=model[16],
            max_input_length=model[17],
            confidence_threshold=model[18],
            class_names=model[19],
            upload_at=model[20],
            last_updated=model[21],
            is_public=model[22],
            uploaded_by=model[23],
            download_count=model[24],
            model_status=model[25],
            package_class_prediction=model[26]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete(
    "/{model_id}",
    response_model=ModelDeleteResponse,
    summary="Delete a model",
    description="""
    Delete a model from the system. Supports both soft and hard deletion.
    
    **Soft Delete (default):**
    - Marks model as 'inactive' in database
    - Keeps file on disk
    - Model becomes inaccessible but recoverable
    
    **Hard Delete (force_delete=true):**
    - Permanently removes database record
    - Deletes file from disk
    - Cannot be recovered
    
    **Use Cases:**
    - Soft delete: Temporarily disable model
    - Hard delete: Free up storage space
    """,
    responses={
        200: {"model": ModelDeleteResponse, "description": "Model deleted successfully"},
        404: {"model": ErrorResponse, "description": "Model not found"},
        500: {"model": ErrorResponse, "description": "Delete operation failed"}
    }
)
async def delete_model(
    model_id: str = Path(..., description="ID of model to delete"),
    force_delete: bool = Query(
        False, 
        description="If true, permanently delete file and database record. If false, mark as inactive."
    )
):
    """Delete a model (soft delete by default, hard delete with force_delete=True)."""
    try:
        # Check if model exists
        model = db_config.fetch_one("SELECT model_id, model_path FROM models WHERE model_id = ?", (model_id,))
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        if force_delete:
            # Hard delete: remove file and database record
            model_path = model[1]
            if os.path.exists(model_path):
                os.remove(model_path)
            
            db_config.execute_query("DELETE FROM models WHERE model_id = ?", (model_id,))
            return ModelDeleteResponse(
                message="Model permanently deleted", 
                model_id=model_id
            )
        else:
            # Soft delete: mark as inactive
            db_config.execute_query("UPDATE models SET model_status = 'inactive' WHERE model_id = ?", (model_id,))
            return ModelDeleteResponse(
                message="Model marked as inactive", 
                model_id=model_id
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/{model_id}/download",
    summary="Download a model file",
    description="""
    Download a model file and increment its download counter.
    
    **Process:**
    1. Validates model exists in database
    2. Checks file exists on disk
    3. Increments download counter
    4. Returns file for download
    
    **Response:** Binary file stream with appropriate headers
    """,
    responses={
        200: {"description": "File download started", "content": {"application/octet-stream": {}}},
        404: {"model": ErrorResponse, "description": "Model or file not found"},
        500: {"model": ErrorResponse, "description": "Download failed"}
    }
)
async def download_model(
    model_id: str = Path(..., description="ID of model to download")
):
    """Download a model file and increment download count."""
    try:
        model = db_config.fetch_one("SELECT model_path, model_name FROM models WHERE model_id = ?", (model_id,))
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_path, model_name = model
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file not found on disk")
        
        # Increment download count
        db_config.execute_query("UPDATE models SET download_count = download_count + 1 WHERE model_id = ?", (model_id,))
        
        return FileResponse(
            path=model_path,
            filename=model_name,
            media_type='application/octet-stream'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))