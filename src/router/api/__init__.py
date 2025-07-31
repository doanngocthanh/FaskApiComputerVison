from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# Pydantic Models cho request/response
class ModelUploadResponse(BaseModel):
    model_id: str = Field(..., description="Unique identifier for the uploaded model")
    filename: str = Field(..., description="Original filename of the uploaded model")
    message: str = Field(..., description="Success message")

class ModelInfo(BaseModel):
    model_id: str = Field(..., description="Unique model identifier")
    model_name_id: str = Field(..., description="Model name identifier")
    model_name: str = Field(..., description="Display name of the model")
    model_type: str = Field(..., description="File extension/type (onnx, pth, pb, pt)")
    model_path: str = Field(..., description="Path to the model file on disk")
    model_category: Optional[str] = Field(None, description="Category like TTS, STT, OCR, YOLO")
    model_size: Optional[int] = Field(None, description="Size of model file in bytes")
    model_version: Optional[str] = Field(None, description="Version of the model")
    model_description: Optional[str] = Field(None, description="Detailed description of the model")
    model_framework: Optional[str] = Field(None, description="Framework used (pytorch, tensorflow, onnx)")
    input_shape: Optional[str] = Field(None, description="Expected input tensor shape")
    output_shape: Optional[str] = Field(None, description="Expected output tensor shape")
    model_accuracy: Optional[float] = Field(None, description="Model accuracy score (0.0-1.0)")
    training_dataset: Optional[str] = Field(None, description="Dataset used for training")
    model_tags: Optional[str] = Field(None, description="Tags for categorization")
    supported_languages: Optional[str] = Field(None, description="Supported languages for TTS/STT models")
    sample_rate: Optional[int] = Field(None, description="Sample rate for audio models")
    max_input_length: Optional[int] = Field(None, description="Maximum input length")
    confidence_threshold: Optional[float] = Field(None, description="Confidence threshold for detection models")
    class_names: Optional[str] = Field(None, description="JSON array of class names")
    upload_at: datetime = Field(..., description="Upload timestamp")
    last_updated: datetime = Field(..., description="Last update timestamp")
    is_public: bool = Field(..., description="Whether model is publicly accessible")
    uploaded_by: str = Field(..., description="User who uploaded the model")
    download_count: int = Field(..., description="Number of times model was downloaded")
    model_status: str = Field(..., description="Model status (active, inactive)")
    package_class_prediction: Optional[str] = Field(None, description="Class prediction logic in JSON")

class ModelListResponse(BaseModel):
    models: List[ModelInfo] = Field(..., description="List of all models")

class ModelDirectoryInfo(BaseModel):
    file_name: str = Field(..., description="Name of the model file")
    file_path: str = Field(..., description="Full path to the model file")

class ModelDirectoryResponse(BaseModel):
    models: List[ModelDirectoryInfo] = Field(..., description="List of model files in directory")

class ModelUpdateRequest(BaseModel):
    model_name: Optional[str] = Field(None, description="New model name")
    model_description: Optional[str] = Field(None, description="New model description")
    model_version: Optional[str] = Field(None, description="New model version")
    model_category: Optional[str] = Field(None, description="New model category")
    model_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model accuracy (0.0-1.0)")
    is_public: Optional[bool] = Field(None, description="Whether model should be public")
    model_status: Optional[str] = Field(None, description="New model status")
    model_tags: Optional[str] = Field(None, description="New model tags")

class ModelUpdateResponse(BaseModel):
    message: str = Field(..., description="Success message")
    model_id: str = Field(..., description="ID of updated model")

class ModelDeleteResponse(BaseModel):
    message: str = Field(..., description="Delete operation result message")
    model_id: str = Field(..., description="ID of deleted model")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message details")
