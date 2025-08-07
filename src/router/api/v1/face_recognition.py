"""
Face Recognition API - API endpoints cho nhận diện khuôn mặt
Hỗ trợ:
- Đăng ký khuôn mặt (face registration)
- Xác thực khuôn mặt (face verification) 
- Nhận diện khuôn mặt (face recognition)
- Quản lý users đã đăng ký
- Thống kê và monitoring
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import os
import cv2
import numpy as np
import json
import tempfile
import logging
from pathlib import Path

# Import services
from src.service.FaceRecognitionService import face_recognition_service
from src.router.api.__init__ import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Router setup
router = APIRouter(
    prefix="/api/v1/face",
    tags=["Face Recognition Service"],
    responses={
        404: {"model": ErrorResponse, "description": "Resource not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)

# Pydantic Models
class FaceRegistrationRequest(BaseModel):
    """Face registration request model"""
    user_id: str = Field(..., description="Unique user identifier", min_length=1, max_length=100)
    name: str = Field(..., description="Full name of the user", min_length=1, max_length=200)
    confidence_threshold: Optional[float] = Field(0.7, description="Recognition confidence threshold (0.0-1.0)", ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the user")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_001",
                "name": "Nguyễn Văn A",
                "confidence_threshold": 0.75,
                "metadata": {
                    "department": "IT",
                    "employee_id": "EMP001",
                    "registration_location": "Hà Nội"
                }
            }
        }

class FaceVerificationRequest(BaseModel):
    """Face verification request model"""
    user_id: str = Field(..., description="User ID to verify against", min_length=1, max_length=100)
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_001"
            }
        }

class FaceDetection(BaseModel):
    """Face detection result"""
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Detection confidence", ge=0.0, le=1.0)
    class_name: str = Field(..., description="Detected class name")
    class_id: int = Field(..., description="Class ID")

class FaceMatch(BaseModel):
    """Face match result"""
    user_id: str = Field(..., description="Matched user ID")
    name: str = Field(..., description="User name")
    similarity: float = Field(..., description="Similarity score", ge=0.0, le=1.0)
    confidence_threshold: float = Field(..., description="Required threshold for this user")
    is_match: bool = Field(..., description="Whether similarity exceeds threshold")

class FaceRecognitionResult(BaseModel):
    """Single face recognition result"""
    face_index: int = Field(..., description="Index of detected face")
    face_detection: FaceDetection
    best_match: Optional[FaceMatch] = Field(None, description="Best matching user")
    top_matches: List[FaceMatch] = Field(..., description="Top K matching users")
    matched_user_id: Optional[str] = Field(None, description="Matched user ID if any")

class RegisteredUser(BaseModel):
    """Registered user information"""
    user_id: str
    name: str
    face_image_path: str
    confidence_threshold: float
    created_at: str
    updated_at: str
    is_active: bool

class FaceRegistrationResponse(BaseModel):
    """Face registration response"""
    success: bool
    action: Optional[str] = Field(None, description="Action performed: 'registered' or 'updated'")
    user_id: Optional[str] = None
    name: Optional[str] = None
    face_detection: Optional[FaceDetection] = None
    embedding_size: Optional[int] = None
    face_image_path: Optional[str] = None
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None

class FaceVerificationResponse(BaseModel):
    """Face verification response"""
    success: bool
    verified: bool
    user_id: Optional[str] = None
    name: Optional[str] = None
    similarity: Optional[float] = None
    confidence_threshold: Optional[float] = None
    face_detection: Optional[FaceDetection] = None
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None

class FaceRecognitionResponse(BaseModel):
    """Face recognition response"""
    success: bool
    session_id: str
    faces_detected: int
    faces_processed: Optional[int] = None
    total_registered_users: Optional[int] = None
    results: Optional[List[FaceRecognitionResult]] = None
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None

class SystemStatistics(BaseModel):
    """System statistics"""
    total_registered_users: int
    today_detections: int
    today_verifications: int
    today_success_rate: float
    database_path: str
    models_directory: str

# API Endpoints

@router.post("/register",
            response_model=FaceRegistrationResponse,
            summary="Register Face",
            description="Đăng ký khuôn mặt mới cho người dùng. Ảnh phải chứa đúng 1 khuôn mặt.")
async def register_face(
    file: UploadFile = File(..., description="Image file containing exactly one face"),
    user_id: str = Form(..., description="Unique user identifier"),
    name: str = Form(..., description="Full name of the user"),
    confidence_threshold: float = Form(0.7, description="Recognition confidence threshold"),
    metadata: str = Form("{}", description="Additional metadata as JSON string")
):
    """
    Đăng ký khuôn mặt mới cho người dùng
    
    - **file**: Ảnh chứa khuôn mặt (JPG, PNG)
    - **user_id**: ID duy nhất của người dùng
    - **name**: Tên đầy đủ của người dùng
    - **confidence_threshold**: Ngưỡng confidence cho nhận diện (0.0-1.0)
    - **metadata**: Thông tin bổ sung dạng JSON
    
    **Lưu ý**: 
    - Ảnh phải chứa đúng 1 khuôn mặt
    - Định dạng ảnh: JPG, JPEG, PNG
    - Kích thước tối đa: 10MB
    """
    
    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, jpeg, png)")
    
    # Validate file size (10MB max)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be less than 10MB")
    
    temp_files = []
    
    try:
        # Parse metadata
        try:
            metadata_dict = json.loads(metadata) if metadata != "{}" else {}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON format")
        
        # Validate confidence threshold
        if not 0.0 <= confidence_threshold <= 1.0:
            raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.0 and 1.0")
        
        # Read and process image
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file or corrupted data")
        
        # Register face
        result = face_recognition_service.register_face(
            user_id=user_id,
            name=name,
            image=image,
            confidence_threshold=confidence_threshold,
            metadata=metadata_dict
        )
        
        # Convert face_detection to FaceDetection model if present
        if result.get("face_detection"):
            face_det = result["face_detection"]
            result["face_detection"] = FaceDetection(
                bbox=face_det["bbox"],
                confidence=face_det["confidence"],
                class_name=face_det.get("class_name", "face"),
                class_id=face_det.get("class_id", 0)
            )
        
        return FaceRegistrationResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in register_face: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

@router.post("/verify",
            response_model=FaceVerificationResponse,
            summary="Verify Face",
            description="Xác thực khuôn mặt với một người dùng đã đăng ký cụ thể")
async def verify_face(
    file: UploadFile = File(..., description="Image file containing face to verify"),
    user_id: str = Form(..., description="User ID to verify against")
):
    """
    Xác thực khuôn mặt với người dùng đã đăng ký
    
    - **file**: Ảnh chứa khuôn mặt cần xác thực
    - **user_id**: ID người dùng cần xác thực
    
    **Kết quả**: 
    - Trả về True/False cho việc xác thực thành công
    - Kèm theo điểm tương tự (similarity score)
    - Thông tin khuôn mặt được phát hiện
    """
    
    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, jpeg, png)")
    
    try:
        # Read and process image
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file or corrupted data")
        
        # Verify face
        result = face_recognition_service.verify_face(user_id=user_id, image=image)
        
        # Convert face_detection to FaceDetection model if present
        if result.get("face_detection"):
            face_det = result["face_detection"]
            result["face_detection"] = FaceDetection(
                bbox=face_det["bbox"],
                confidence=face_det["confidence"],
                class_name=face_det.get("class_name", "face"),
                class_id=face_det.get("class_id", 0)
            )
        
        return FaceVerificationResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in verify_face: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/recognize",
            response_model=FaceRecognitionResponse,
            summary="Recognize Faces", 
            description="Nhận diện tất cả khuôn mặt trong ảnh và tìm kiếm trong database users đã đăng ký")
async def recognize_faces(
    file: UploadFile = File(..., description="Image file containing faces to recognize"),
    top_k: int = Form(5, description="Number of top matches to return per face")
):
    """
    Nhận diện khuôn mặt trong ảnh
    
    - **file**: Ảnh chứa các khuôn mặt cần nhận diện
    - **top_k**: Số lượng kết quả tương tự nhất trả về cho mỗi khuôn mặt
    
    **Kết quả**:
    - Phát hiện tất cả khuôn mặt trong ảnh
    - Với mỗi khuôn mặt, tìm kiếm trong database users đã đăng ký
    - Trả về top K matches với điểm tương tự
    - Session ID để tracking
    """
    
    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, jpeg, png)")
    
    # Validate top_k
    if not 1 <= top_k <= 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")
    
    try:
        # Read and process image
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file or corrupted data")
        
        # Recognize faces
        result = face_recognition_service.recognize_face(image=image, top_k=top_k)
        
        # Convert results to Pydantic models
        if result.get("results"):
            converted_results = []
            for res in result["results"]:
                # Convert face_detection
                face_det = res["face_detection"]
                face_detection = FaceDetection(
                    bbox=face_det["bbox"],
                    confidence=face_det["confidence"],
                    class_name=face_det.get("class_name", "face"),
                    class_id=face_det.get("class_id", 0)
                )
                
                # Convert best_match
                best_match = None
                if res.get("best_match"):
                    bm = res["best_match"]
                    best_match = FaceMatch(
                        user_id=bm["user_id"],
                        name=bm["name"],
                        similarity=bm["similarity"],
                        confidence_threshold=bm["confidence_threshold"],
                        is_match=bm["is_match"]
                    )
                
                # Convert top_matches
                top_matches = []
                for tm in res.get("top_matches", []):
                    top_matches.append(FaceMatch(
                        user_id=tm["user_id"],
                        name=tm["name"],
                        similarity=tm["similarity"],
                        confidence_threshold=tm["confidence_threshold"],
                        is_match=tm["is_match"]
                    ))
                
                converted_results.append(FaceRecognitionResult(
                    face_index=res["face_index"],
                    face_detection=face_detection,
                    best_match=best_match,
                    top_matches=top_matches,
                    matched_user_id=res["matched_user_id"]
                ))
            
            result["results"] = converted_results
        
        return FaceRecognitionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in recognize_faces: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/users",
           response_model=List[RegisteredUser],
           summary="Get Registered Users",
           description="Lấy danh sách tất cả người dùng đã đăng ký khuôn mặt")
async def get_registered_users():
    """
    Lấy danh sách tất cả người dùng đã đăng ký khuôn mặt
    
    **Kết quả**:
    - Danh sách users với thông tin cơ bản
    - Thời gian đăng ký và cập nhật
    - Trạng thái active/inactive
    """
    try:
        users = face_recognition_service.get_registered_users()
        
        return [RegisteredUser(
            user_id=user["user_id"],
            name=user["name"],
            face_image_path=user["face_image_path"],
            confidence_threshold=user["confidence_threshold"],
            created_at=user["created_at"],
            updated_at=user["updated_at"],
            is_active=user["is_active"]
        ) for user in users]
        
    except Exception as e:
        logger.error(f"Error in get_registered_users: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.delete("/users/{user_id}",
              summary="Delete User",
              description="Xóa người dùng đã đăng ký (soft delete)")
async def delete_user(user_id: str):
    """
    Xóa người dùng đã đăng ký
    
    - **user_id**: ID người dùng cần xóa
    
    **Lưu ý**: Đây là soft delete, dữ liệu sẽ được đánh dấu inactive
    """
    try:
        result = face_recognition_service.delete_user(user_id)
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/statistics",
           response_model=SystemStatistics,
           summary="Get System Statistics",
           description="Lấy thống kê tổng quan của hệ thống face recognition")
async def get_statistics():
    """
    Lấy thống kê tổng quan của hệ thống
    
    **Bao gồm**:
    - Tổng số users đã đăng ký
    - Số lần detection hôm nay
    - Số lần verification hôm nay
    - Tỷ lệ thành công hôm nay
    - Thông tin database và models
    """
    try:
        stats = face_recognition_service.get_statistics()
        
        if not stats:
            raise HTTPException(status_code=500, detail="Unable to retrieve statistics")
        
        return SystemStatistics(**stats)
        
    except Exception as e:
        logger.error(f"Error in get_statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/users/{user_id}/face-image",
           summary="Get User Face Image",
           description="Lấy ảnh khuôn mặt đã đăng ký của người dùng")
async def get_user_face_image(user_id: str):
    """
    Lấy ảnh khuôn mặt đã đăng ký của người dùng
    
    - **user_id**: ID người dùng
    
    **Kết quả**: File ảnh khuôn mặt đã được crop và lưu trữ
    """
    try:
        users = face_recognition_service.get_registered_users()
        user = next((u for u in users if u["user_id"] == user_id and u["is_active"]), None)
        
        if not user:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        face_image_path = user["face_image_path"]
        
        if not os.path.exists(face_image_path):
            raise HTTPException(status_code=404, detail="Face image file not found")
        
        return FileResponse(
            path=face_image_path,
            media_type="image/jpeg",
            filename=f"{user_id}_face.jpg"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_user_face_image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/detect-faces-only",
            summary="Detect Faces Only",
            description="Chỉ phát hiện khuôn mặt trong ảnh mà không nhận diện")
async def detect_faces_only(
    file: UploadFile = File(..., description="Image file to detect faces in")
):
    """
    Chỉ phát hiện khuôn mặt trong ảnh
    
    - **file**: Ảnh cần phát hiện khuôn mặt
    
    **Kết quả**: 
    - Danh sách các khuôn mặt được phát hiện
    - Bounding box và confidence cho mỗi khuôn mặt
    - Không thực hiện nhận diện (recognition)
    """
    
    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, jpeg, png)")
    
    try:
        # Read and process image
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file or corrupted data")
        
        # Detect faces only
        face_detections = face_recognition_service.detect_faces(image)
        
        # Convert to Pydantic models
        detected_faces = []
        for detection in face_detections:
            detected_faces.append(FaceDetection(
                bbox=detection["bbox"],
                confidence=detection["confidence"],
                class_name=detection.get("class_name", "face"),
                class_id=detection.get("class_id", 0)
            ))
        
        return {
            "success": True,
            "faces_detected": len(detected_faces),
            "detections": detected_faces
        }
        
    except Exception as e:
        logger.error(f"Error in detect_faces_only: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health",
           summary="Health Check",
           description="Kiểm tra tình trạng hoạt động của Face Recognition Service")
async def health_check():
    """
    Kiểm tra tình trạng hoạt động của service
    
    **Kết quả**:
    - Trạng thái service
    - Thông tin models
    - Database connectivity
    """
    try:
        # Test database connection
        stats = face_recognition_service.get_statistics()
        
        # Test face detector
        detector_status = "ok"
        try:
            detector = face_recognition_service.get_face_detector()
            if detector is None:
                detector_status = "not_loaded"
        except Exception:
            detector_status = "error"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected" if stats else "error",
            "face_detector": detector_status,
            "total_registered_users": stats.get("total_registered_users", 0) if stats else 0,
            "service_version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Error in health_check: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "service_version": "1.0.0"
        }
