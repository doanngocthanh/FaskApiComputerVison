from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Path
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid
from src.router.api.__init__ import *  
# Router setup
router = APIRouter(
    prefix="/api/v1/models_setting",
    tags=["Model Management"],
    responses={
        404: {"model": ErrorResponse, "description": "Resource not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)

from config import OnnxConfig, PthConfig, TensorflowConfig, ModelConfig, DBConfig
import json

db_config = DBConfig()

# Create table (move this to a separate migration script in production)
# Create tables for model pipeline processing
results = db_config.execute_query("""
    CREATE TABLE IF NOT EXISTS model_pipelines (
        pipeline_id VARCHAR(36) PRIMARY KEY,
        pipeline_name VARCHAR(100) NOT NULL,
        description TEXT,
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")

results = db_config.execute_query("""
    CREATE TABLE IF NOT EXISTS pipeline_steps (
        step_id VARCHAR(36) PRIMARY KEY,
        pipeline_id VARCHAR(36) NOT NULL,
        model_id VARCHAR(36) NOT NULL,
        step_order INTEGER NOT NULL,
        step_name VARCHAR(100) NOT NULL,
        process_type VARCHAR(50) NOT NULL, -- ocr, detection, qrendcode, etc.
        input_config TEXT, -- JSON config for input processing
        output_config TEXT, -- JSON config for output processing
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (pipeline_id) REFERENCES model_pipelines(pipeline_id),
        FOREIGN KEY (model_id) REFERENCES models(model_id),
        UNIQUE(pipeline_id, step_order)
    );
""")

results = db_config.execute_query("""
    CREATE TABLE IF NOT EXISTS models_setting (
        setting_id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id VARCHAR(36) NOT NULL,
        setting_name VARCHAR(100) NOT NULL,
        setting_value TEXT NOT NULL,
        predict_type VARCHAR(50) NOT NULL, -- ocr, detection, qrendcode, etc.
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (model_id) REFERENCES models(model_id)
    );
""")

# Pydantic models for pipeline management
class PipelineCreate(BaseModel):
    pipeline_name: str = Field(..., max_length=100)
    description: Optional[str] = None
    is_active: bool = True

class PipelineStepCreate(BaseModel):
    model_id: str = Field(..., description="Model ID to use in this step")
    step_order: int = Field(..., ge=1, description="Order of execution (starting from 1)")
    step_name: str = Field(..., max_length=100)
    process_type: str = Field(..., max_length=50, description="Type: ocr, detection, qrendcode, etc.")
    input_config: Optional[dict] = None
    output_config: Optional[dict] = None
    is_active: bool = True

class PipelineWithSteps(BaseModel):
    pipeline_name: str = Field(..., max_length=100)
    description: Optional[str] = None
    steps: List[PipelineStepCreate]

class PipelineResponse(BaseModel):
    pipeline_id: str
    pipeline_name: str
    description: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime

# API endpoints for pipeline management
@router.post("/pipelines", response_model=PipelineResponse)
async def create_pipeline(pipeline: PipelineCreate):
    """Create a new model pipeline"""
    try:
        pipeline_id = str(uuid.uuid4())
        
        query = """
            INSERT INTO model_pipelines (pipeline_id, pipeline_name, description, is_active)
            VALUES (?, ?, ?, ?)
        """
        
        db_config.execute_query(query, (
            pipeline_id,
            pipeline.pipeline_name,
            pipeline.description,
            pipeline.is_active
        ))
        
        # Get created pipeline
        result = db_config.execute_query(
            "SELECT * FROM model_pipelines WHERE pipeline_id = ?", 
            (pipeline_id,)
        )
        
        if result:
            return PipelineResponse(**dict(result[0]))
        else:
            raise HTTPException(status_code=500, detail="Failed to create pipeline")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating pipeline: {str(e)}")

@router.post("/pipelines/{pipeline_id}/steps")
async def add_pipeline_step(
    pipeline_id: str = Path(...),
    step: PipelineStepCreate = None
):
    """Add a step to an existing pipeline"""
    try:
        step_id = str(uuid.uuid4())
        
        # Check if pipeline exists
        pipeline_check = db_config.execute_query(
            "SELECT pipeline_id FROM model_pipelines WHERE pipeline_id = ?",
            (pipeline_id,)
        )
        if not pipeline_check:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        query = """
            INSERT INTO pipeline_steps 
            (step_id, pipeline_id, model_id, step_order, step_name, process_type, input_config, output_config, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        
        db_config.execute_query(query, (
            step_id,
            pipeline_id,
            step.model_id,
            step.step_order,
            step.step_name,
            step.process_type,
            json.dumps(step.input_config) if step.input_config else None,
            json.dumps(step.output_config) if step.output_config else None,
            step.is_active
        ))
        
        return {"step_id": step_id, "message": "Step added successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding step: {str(e)}")

@router.post("/pipelines/complete", response_model=PipelineResponse)
async def create_complete_pipeline(pipeline_data: PipelineWithSteps):
    """Create a pipeline with all its steps in one request"""
    try:
        pipeline_id = str(uuid.uuid4())
        
        # Create pipeline
        pipeline_query = """
            INSERT INTO model_pipelines (pipeline_id, pipeline_name, description, is_active)
            VALUES (?, ?, ?, ?)
        """
        
        db_config.execute_query(pipeline_query, (
            pipeline_id,
            pipeline_data.pipeline_name,
            pipeline_data.description,
            True
        ))
        
        # Add steps
        step_query = """
            INSERT INTO pipeline_steps 
            (step_id, pipeline_id, model_id, step_order, step_name, process_type, input_config, output_config, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        for step in pipeline_data.steps:
            step_id = str(uuid.uuid4())
            db_config.execute_query(step_query, (
                step_id,
                pipeline_id,
                step.model_id,
                step.step_order,
                step.step_name,
                step.process_type,
                json.dumps(step.input_config) if step.input_config else None,
                json.dumps(step.output_config) if step.output_config else None,
                step.is_active
            ))
        
        # Return created pipeline
        result = db_config.execute_query(
            "SELECT * FROM model_pipelines WHERE pipeline_id = ?", 
            (pipeline_id,)
        )
        
        if result:
            return PipelineResponse(**dict(result[0]))
        else:
            raise HTTPException(status_code=500, detail="Failed to create pipeline")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating complete pipeline: {str(e)}")

@router.get("/pipelines")
async def list_pipelines():
    """Get all pipelines"""
    try:
        result = db_config.execute_query("SELECT * FROM model_pipelines ORDER BY created_at DESC")
        return [dict(row) for row in result] if result else []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving pipelines: {str(e)}")

@router.get("/pipelines/{pipeline_id}/steps")
async def get_pipeline_steps(pipeline_id: str = Path(...)):
    """Get all steps for a specific pipeline"""
    try:
        result = db_config.execute_query(
            "SELECT * FROM pipeline_steps WHERE pipeline_id = ? ORDER BY step_order",
            (pipeline_id,)
        )
        return [dict(row) for row in result] if result else []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving pipeline steps: {str(e)}")
