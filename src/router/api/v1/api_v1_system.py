"""
YOLO Detection + EasyOCR Service API
Service tổng hợp cho detection và OCR đa ngôn ngữ
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
import shutil
import easyocr
from PIL import Image

# Import services
from src.service.YOLODetector import YOLODetector
from src.service.EasyOCRManager import easy_ocr_manager


from src.router.api.__init__ import *
from config import PtConfig

# Router setup
router = APIRouter(
    prefix="/api/v1/system",
    tags=["YOLO Detection + EasyOCR Service"]
)

# Pydantic models
class ModelUploadRequest(BaseModel):
    model_name: str = Field(..., description="Tên model")
    model_description: Optional[str] = Field(None, description="Mô tả model")
    class_names: List[str] = Field(..., description="Danh sách tên class")
    confidence_threshold: Optional[float] = Field(0.5, description="Ngưỡng confidence")

class DetectionResult(BaseModel):
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    class_name: str = Field(..., description="Tên class")
    confidence: float = Field(..., description="Độ tin cậy")
    class_id: int = Field(..., description="ID class")

class OCRResult(BaseModel):
    bbox: List[int] = Field(..., description="Vùng text [x1, y1, x2, y2]")  
    text: str = Field(..., description="Text nhận dạng được")
    confidence: float = Field(..., description="Độ tin cậy text")

class YOLOOCRResponse(BaseModel):
    status: str = Field(..., description="Trạng thái xử lý")
    detection_results: List[DetectionResult] = Field(..., description="Kết quả detection")
    ocr_results: List[OCRResult] = Field(..., description="Kết quả OCR")
    processing_time_ms: int = Field(..., description="Thời gian xử lý (ms)")
    total_detections: int = Field(..., description="Tổng số vùng phát hiện")
    total_texts: int = Field(..., description="Tổng số text nhận dạng")

# Global variables for caching
yolo_models_cache = {}
easyocr_readers_cache = {}

# Supported languages for EasyOCR
SUPPORTED_LANGUAGES = {
    'vi': 'Vietnamese',
    'en': 'English', 
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'th': 'Thai',
    'fr': 'French',
    'de': 'German',
    'es': 'Spanish',
    'ru': 'Russian',
    'ar': 'Arabic'
}

def get_simple_easyocr_reader(languages: List[str]) -> easyocr.Reader:
    """Lấy hoặc tạo EasyOCR reader đơn giản"""
    lang_key = '_'.join(sorted(languages))
    
    if lang_key not in easyocr_readers_cache:
        try:
            print(f"Initializing simple EasyOCR reader for languages: {languages}")
            easyocr_readers_cache[lang_key] = easyocr.Reader(
                languages, 
                gpu=False,  # Sử dụng CPU cho ổn định
                verbose=False
            )
        except Exception as e:
            print(f"Failed to initialize EasyOCR reader: {e}")
            # Return a default reader
            easyocr_readers_cache[lang_key] = easyocr.Reader(['en'], gpu=False, verbose=False)
    
    return easyocr_readers_cache[lang_key]

@router.get("/languages")
async def get_supported_languages():
    return {
        "status": "success",
        "supported_languages": SUPPORTED_LANGUAGES,
        "total": len(SUPPORTED_LANGUAGES)
    }

@router.post("/ocr")
async def ocr_with_model(
    file: UploadFile = File(...),
    model_id: str = Form(None),
    languages: str = Form('["en","vi"]'),
    confidence_threshold: float = Form(0.5)
):
    
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Image must be jpg, jpeg, or png")
    
    temp_files = []
    
    try:
        # Parse languages
        try:
            lang_list = json.loads(languages)
            if not isinstance(lang_list, list):
                raise ValueError("Languages must be a JSON array")
                
            # Validate languages
            invalid_langs = [lang for lang in lang_list if lang not in SUPPORTED_LANGUAGES]
            if invalid_langs:
                raise ValueError(f"Unsupported language(s): {', '.join(invalid_langs)}. Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}")
                
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail=f"Invalid JSON format for languages. Expected format: [\"en\",\"vi\"]. Received: {languages}")
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
            temp_files.append(temp_path)
        
        # Load image
        image = cv2.imread(temp_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        all_ocr_results = []
        total_texts = 0
        detections = []
        
        # Check if model_id is provided for YOLO detection
        if model_id:
            # Load YOLO model and detect objects
            yolo_model = YOLODetector(model_id=model_id)
            detections = yolo_model.detect(temp_path, confidence_threshold=confidence_threshold)
            
            # Process OCR on detected regions
            for detection in detections:
                # Crop detected region
                x1, y1, x2, y2 = detection['bbox']
                cropped = image[int(y1):int(y2), int(x1):int(x2)]
                
                if cropped.size == 0:
                    continue
                    
                # Save cropped image temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as crop_file:
                    cv2.imwrite(crop_file.name, cropped)
                    crop_path = crop_file.name
                    temp_files.append(crop_path)
                
                # OCR on cropped region
                reader = get_simple_easyocr_reader(lang_list)
                ocr_results = reader.readtext(crop_path)
                
                processed_results = []
                for result in ocr_results:
                    try:
                        # EasyOCR returns (bbox, text, confidence)
                        if len(result) == 3:
                            bbox, text, confidence = result
                        elif len(result) == 2:
                            # Some versions might return (text, confidence) 
                            text, confidence = result
                            bbox = None
                        else:
                            print(f"Unexpected OCR result format: {result}")
                            continue
                            
                        if confidence > 0.5:  # OCR confidence threshold
                            if bbox is not None:
                                # Adjust bbox coordinates to original image
                                adjusted_bbox = []
                                for point in bbox:
                                    adjusted_bbox.append([point[0] + x1, point[1] + y1])
                            else:
                                # If no bbox, use detection bbox
                                adjusted_bbox = detection['bbox']
                            
                            processed_results.append({
                                "text": text,
                                "confidence": float(confidence),
                                "bbox": adjusted_bbox
                            })
                            total_texts += 1
                    except Exception as e:
                        print(f"Error processing OCR result {result}: {e}")
                        continue
                
                if processed_results:
                    all_ocr_results.append({
                        "detection_bbox": detection['bbox'],
                        "detection_class": detection['class_name'],
                        "detection_confidence": detection['confidence'],
                        "ocr_results": processed_results
                    })
        else:
            # OCR only without YOLO detection
            reader = get_simple_easyocr_reader(lang_list)
            ocr_results = reader.readtext(temp_path)
            
            for result in ocr_results:
                try:
                    # EasyOCR returns (bbox, text, confidence)
                    if len(result) == 3:
                        bbox, text, confidence = result
                    elif len(result) == 2:
                        # Some versions might return (text, confidence)
                        text, confidence = result
                        bbox = [[0, 0], [100, 0], [100, 20], [0, 20]]  # Default bbox
                    else:
                        print(f"Unexpected OCR result format: {result}")
                        continue
                        
                    if confidence > 0.5:  # OCR confidence threshold
                        all_ocr_results.append({
                            "text": text,
                            "confidence": float(confidence),
                            "bbox": bbox
                        })
                        total_texts += 1
                except Exception as e:
                    print(f"Error processing OCR result {result}: {e}")
                    continue
        
        return {
            "status": "success",
            "results": all_ocr_results,
            "languages_used": lang_list,
            "processing_time_ms": 0,  # Will be calculated by middleware
            "total_detections": len(detections),
            "total_texts": total_texts,
            "model_id": model_id,
            "confidence_threshold": confidence_threshold if model_id else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in OCR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass

@router.post("/models/upload-step1")
async def upload_model_step1(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    model_description: str = Form(""),
    confidence_threshold: float = Form(0.5)
):
    
    if not file.filename.endswith('.pt'):
        raise HTTPException(status_code=400, detail="Model file must be .pt format")
    
    try:
        model_id = str(uuid.uuid4())
        
        # Create model directory
        model_dir = Path("temp/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model file
        model_filename = f"{model_id}.pt"
        model_path = model_dir / model_filename
        
        with open(model_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create temporary model info (without class names yet)
        model_info = {
            "model_id": model_id,
            "model_name": model_name,
            "model_description": model_description,
            "confidence_threshold": confidence_threshold,
            "model_path": str(model_path),
            "upload_time": datetime.now().isoformat(),
            "status": "pending_sample_test"
        }
        
        # Save temporary model info
        info_path = model_dir.parent / "model_info" / f"{model_id}_temp.json"
        info_path.parent.mkdir(exist_ok=True)
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        return {
            "status": "success",
            "model_id": model_id,
            "model_name": model_name,
            "message": "Model uploaded successfully. Now upload a sample image to auto-detect classes.",
            "next_step": f"POST /api/v1/yolo-ocr/models/{model_id}/test-sample"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/models/{model_id}/test-sample")
async def test_sample_and_finalize_model(
    model_id: str,
    sample_image: UploadFile = File(...),
    auto_generate_class_names: bool = Form(True),
    custom_class_names: str = Form("[]")
):
    
    if not sample_image.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Sample image must be jpg, jpeg, or png")
    
    temp_files = []
    
    try:
        # Load temporary model info
        temp_info_path = Path("temp/model_info") / f"{model_id}_temp.json"
        if not temp_info_path.exists():
            raise HTTPException(status_code=404, detail="Model not found or step 1 not completed")
        
        with open(temp_info_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        model_path = model_info['model_path']
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # Save sample image
        sample_filename = f"{model_id}_sample.jpg"
        sample_path = Path("temp/samples") / sample_filename
        sample_path.parent.mkdir(exist_ok=True)
        
        with open(sample_path, "wb") as buffer:
            sample_content = await sample_image.read()
            buffer.write(sample_content)
        
        temp_files.append(str(sample_path))
        
        # Load model and run detection
        detector = YOLODetector(model_path=model_path)
        
        # Load and process sample image
        sample_img = cv2.imread(str(sample_path))
        if sample_img is None:
            raise HTTPException(status_code=400, detail="Invalid sample image")
        
        # Run detection to get class IDs
        detections = detector.detect(sample_img)
        
        # Extract unique class IDs from detections
        detected_class_ids = set()
        sample_results = []
        actual_class_names = {}  # Store actual class names from model
        
        for detection in detections:
            if 'bbox' in detection and 'confidence' in detection and detection['confidence'] >= model_info['confidence_threshold']:
                class_id = detection.get('class_id', 0)
                class_name = detection.get('class_name', f"class_{class_id}")  # Get actual class name from detector
                detected_class_ids.add(class_id)
                actual_class_names[class_id] = class_name  # Store the actual name
                
                sample_results.append({
                    "bbox": detection['bbox'],
                    "class_id": class_id,
                    "class_name": class_name,  # Include actual class name
                    "confidence": detection['confidence']
                })
        
        # Generate class names
        if auto_generate_class_names:
            # Use actual class names from YOLO model detections
            max_class_id = max(detected_class_ids) if detected_class_ids else 0
            class_list = []
            
            for i in range(max_class_id + 1):
                if i in actual_class_names:
                    # Use actual class name from detection
                    class_list.append(actual_class_names[i])
                else:
                    # For unused classes, try to get from YOLO model or use fallback
                    if hasattr(detector.model, 'names') and i in detector.model.names:
                        class_list.append(detector.model.names[i])
                    else:
                        class_list.append(f"unused_class_{i}")
            
            auto_generated = True
            
        else:
            # Use custom class names provided by user
            try:
                class_list = json.loads(custom_class_names)
                if not isinstance(class_list, list):
                    raise ValueError("Custom class names must be a JSON array")
            except:
                raise HTTPException(status_code=400, detail="Invalid JSON format for custom_class_names")
            
            auto_generated = False
        
        # Update sample results with class names (only if not already present)
        for result in sample_results:
            if 'class_name' not in result:
                class_id = result['class_id']
                class_name = class_list[class_id] if class_id < len(class_list) else f"unknown_class_{class_id}"
                result['class_name'] = class_name
        
        # Update model info
        model_info.update({
            "class_names": class_list,
            "sample_image_path": str(sample_path),
            "sample_detection_results": sample_results,
            "detected_class_ids": list(detected_class_ids),
            "auto_generated_class_names": auto_generated,
            "total_classes": len(class_list),
            "detected_classes": len(detected_class_ids),
            "status": "ready"
        })
        
        # Save final model info
        final_info_path = Path("temp/model_info") / f"{model_id}.json"
        with open(final_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        # Remove temporary file
        if temp_info_path.exists():
            temp_info_path.unlink()
        
        # Cache model for immediate use
        yolo_models_cache[model_id] = detector
        
        return {
            "status": "success",
            "model_id": model_id,
            "model_name": model_info['model_name'],
            "class_names": class_list,
            "auto_generated_class_names": auto_generated,
            "detected_class_ids": list(detected_class_ids),
            "total_classes": len(class_list),
            "detected_classes": len(detected_class_ids),
            "sample_detection_results": sample_results,
            "total_detections": len(sample_results),
            "message": f"Model finalized successfully! Found {len(detected_class_ids)} unique classes in sample image.",
            "suggestion": "You can now edit class names using the update-class-names endpoint if needed."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Sample test failed: {str(e)}")

@router.put("/models/{model_id}/update-class-names")
async def update_model_class_names(
    model_id: str,
    class_names: str = Form(...),
    update_sample_results: bool = Form(True)
):
    
    try:
        # Load model info
        info_path = Path("temp/model_info") / f"{model_id}.json"
        if not info_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        with open(info_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        # Parse new class names
        try:
            new_class_list = json.loads(class_names)
            if not isinstance(new_class_list, list):
                raise ValueError("Class names must be a JSON array")
        except:
            raise HTTPException(status_code=400, detail="Invalid JSON format for class_names")
        
        # Update class names
        old_class_names = model_info['class_names']
        model_info['class_names'] = new_class_list
        model_info['auto_generated_class_names'] = False
        model_info['total_classes'] = len(new_class_list)
        
        # Update sample results if requested
        if update_sample_results and 'sample_detection_results' in model_info:
            for result in model_info['sample_detection_results']:
                class_id = result['class_id']
                class_name = new_class_list[class_id] if class_id < len(new_class_list) else f"unknown_class_{class_id}"
                result['class_name'] = class_name
        
        # Save updated model info
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        return {
            "status": "success",
            "model_id": model_id,
            "old_class_names": old_class_names,
            "new_class_names": new_class_list,
            "total_classes": len(new_class_list),
            "sample_results_updated": update_sample_results,
            "message": "Class names updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update class names: {str(e)}")

@router.post("/models/upload")
async def upload_yolo_model_legacy(
    file: UploadFile = File(...),
    sample_image: UploadFile = File(...),
    model_name: str = Form(...),
    model_description: str = Form(""),
    class_names: str = Form(...),
    confidence_threshold: float = Form(0.5)
):
    
    if not file.filename.endswith('.pt'):
        raise HTTPException(status_code=400, detail="Model file must be .pt format")
    
    if not sample_image.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Sample image must be jpg, jpeg, or png")
    
    temp_files = []
    
    try:
        # Parse class names
        try:
            class_list = json.loads(class_names)
            if not isinstance(class_list, list):
                raise ValueError("Class names must be a JSON array")
        except:
            raise HTTPException(status_code=400, detail="Invalid JSON format for class_names")
        
        model_id = str(uuid.uuid4())
        
        # Create model directory
        model_dir = Path("temp/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model file
        model_filename = f"{model_id}.pt"
        model_path = model_dir / model_filename
        
        with open(model_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        temp_files.append(str(model_path))
        
        # Save and process sample image
        sample_filename = f"{model_id}_sample.jpg"
        sample_path = model_dir.parent / "samples" / sample_filename
        sample_path.parent.mkdir(exist_ok=True)
        
        with open(sample_path, "wb") as buffer:
            sample_content = await sample_image.read()
            buffer.write(sample_content)
        
        temp_files.append(str(sample_path))
        
        # Test model with sample image
        detector = YOLODetector(model_path=str(model_path))
        
        # Load and process sample image
        sample_img = cv2.imread(str(sample_path))
        if sample_img is None:
            raise HTTPException(status_code=400, detail="Invalid sample image")
        
        # Run detection
        detections = detector.detect(sample_img)
        
        # Process detection results
        sample_results = []
        for detection in detections:
            if 'bbox' in detection and 'confidence' in detection:
                class_id = detection.get('class_id', 0)
                class_name = class_list[class_id] if class_id < len(class_list) else f"class_{class_id}"
                
                sample_results.append({
                    "bbox": detection['bbox'],
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": detection['confidence']
                })
        
        # Create model info JSON
        model_info = {
            "model_id": model_id,
            "model_name": model_name,
            "model_description": model_description,
            "class_names": class_list,
            "confidence_threshold": confidence_threshold,
            "model_path": str(model_path),
            "sample_image_path": str(sample_path),
            "upload_time": datetime.now().isoformat(),
            "sample_detection_results": sample_results,
            "auto_generated_class_names": False,
            "status": "ready"
        }
        
        # Save model info
        info_path = model_dir.parent / "model_info" / f"{model_id}.json"
        info_path.parent.mkdir(exist_ok=True)
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        # Cache model for immediate use
        yolo_models_cache[model_id] = detector
        
        return {
            "status": "success",
            "model_id": model_id,
            "model_name": model_name,
            "sample_detection_results": sample_results,
            "total_detections": len(sample_results),
            "class_names": class_list,
            "message": f"Model uploaded successfully with {len(sample_results)} detections found (Legacy API)"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/models/list")
async def list_yolo_models():
    try:
        model_info_dir = Path("temp/model_info")
        if not model_info_dir.exists():
            return {"status": "success", "models": [], "total": 0}
        
        models = []
        for info_file in model_info_dir.glob("*.json"):
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                    
                # Check if model file still exists
                if os.path.exists(model_info.get('model_path', '')):
                    models.append({
                        "model_id": model_info['model_id'],
                        "model_name": model_info['model_name'], 
                        "model_description": model_info['model_description'],
                        "class_count": len(model_info['class_names']),
                        "upload_time": model_info['upload_time'],
                        "sample_detections": len(model_info.get('sample_detection_results', []))
                    })
            except:
                continue
        
        return {
            "status": "success",
            "models": models,
            "total": len(models)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@router.post("/models/{model_id}/detect")
async def detect_with_yolo_model(
    model_id: str,
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Image must be jpg, jpeg, or png")
    
    temp_files = []
    
    try:
        # Load model info
        info_path = Path("temp/model_info") / f"{model_id}.json"
        if not info_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        with open(info_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        # Get or load model
        if model_id not in yolo_models_cache:
            model_path = model_info['model_path']
            if not os.path.exists(model_path):
                raise HTTPException(status_code=404, detail="Model file not found")
            yolo_models_cache[model_id] = YOLODetector(model_path=model_path)
        
        detector = yolo_models_cache[model_id]
        
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
            temp_files.append(temp_path)
        
        # Load and process image
        image = cv2.imread(temp_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        start_time = datetime.now()
        
        # Run detection
        detections = detector.detect(image)
        
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Process results
        results = []
        class_names = model_info['class_names']
        
        for detection in detections:
            if detection.get('confidence', 0) >= confidence_threshold:
                class_id = detection.get('class_id', 0)
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                
                results.append(DetectionResult(
                    bbox=detection['bbox'],
                    class_name=class_name,
                    confidence=detection['confidence'],
                    class_id=class_id
                ))
        
        return {
            "status": "success",
            "model_name": model_info['model_name'],
            "detection_results": results,
            "total_detections": len(results),
            "processing_time_ms": processing_time,
            "confidence_threshold": confidence_threshold
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    finally:
        # Clean up
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

@router.post("/models/{model_id}/detect-and-ocr")
async def detect_and_ocr(
    model_id: str,
    file: UploadFile = File(...),
    languages: str = Form('["en","vi"]'),
    confidence_threshold: float = Form(0.5),
    ocr_crop_padding: int = Form(5),
    return_cropped_images: bool = Form(False)
):
    
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Image must be jpg, jpeg, or png")
    
    temp_files = []
    
    try:
        # Parse languages
        try:
            lang_list = json.loads(languages)
            if not isinstance(lang_list, list):
                raise ValueError("Languages must be a JSON array")
            
            # Validate languages
            for lang in lang_list:
                if lang not in SUPPORTED_LANGUAGES:
                    raise ValueError(f"Unsupported language: {lang}")
        except:
            raise HTTPException(status_code=400, detail="Invalid languages format or unsupported language")
        
        # Load model info
        info_path = Path("temp/model_info") / f"{model_id}.json"
        if not info_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        with open(info_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        # Get or load YOLO model
        if model_id not in yolo_models_cache:
            model_path = model_info['model_path']
            if not os.path.exists(model_path):
                raise HTTPException(status_code=404, detail="Model file not found")
            yolo_models_cache[model_id] = YOLODetector(model_path=model_path)
        
        detector = yolo_models_cache[model_id]
        
        # Sử dụng EasyOCRManager đơn giản và ổn định
        # ocr_reader = get_simple_easyocr_reader(lang_list)
        
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
            temp_files.append(temp_path)
        
        # Load and process image
        image = cv2.imread(temp_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        start_time = datetime.now()
        
        # Run YOLO detection
        detections = detector.detect(image)
        
        # Process detection results
        detection_results = []
        ocr_results = []
        class_names = model_info['class_names']
        
        cropped_images_data = []
        
        for i, detection in enumerate(detections):
            if detection.get('confidence', 0) >= confidence_threshold:
                class_id = detection.get('class_id', 0)
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                
                detection_result = DetectionResult(
                    bbox=detection['bbox'],
                    class_name=class_name,
                    confidence=detection['confidence'],
                    class_id=class_id
                )
                detection_results.append(detection_result)
                
                # Crop detected region for OCR
                x1, y1, x2, y2 = detection['bbox']
                
                # Add padding
                h, w = image.shape[:2]
                x1 = max(0, x1 - ocr_crop_padding)
                y1 = max(0, y1 - ocr_crop_padding)
                x2 = min(w, x2 + ocr_crop_padding)
                y2 = min(h, y2 + ocr_crop_padding)
                
                # Crop image
                cropped = image[y1:y2, x1:x2]
                
                if cropped.size > 0:
                    try:
                        # Sử dụng EasyOCRManager để OCR
                        ocr_result = easy_ocr_manager.extract_text(cropped, lang_list, preprocess=True)
                        
                        if ocr_result["success"] and ocr_result.get("bboxes"):
                            # Lấy kết quả từ EasyOCRManager
                            for i, bbox_rel in enumerate(ocr_result["bboxes"]):
                                if i < len(ocr_result["texts"]) and i < len(ocr_result["confidences"]):
                                    text = ocr_result["texts"][i]
                                    confidence = ocr_result["confidences"][i]
                                    
                                    if confidence > 0.5:  # Minimum OCR confidence
                                        # Convert relative bbox to absolute coordinates
                                        abs_bbox = [
                                            int(x1 + bbox_rel[0]),
                                            int(y1 + bbox_rel[1]),
                                            int(x1 + bbox_rel[2]),
                                            int(y1 + bbox_rel[3])
                                        ]
                                        
                                        ocr_results.append(OCRResult(
                                            bbox=abs_bbox,
                                            text=text,
                                            confidence=confidence
                                        ))
                        
                        # Save cropped image if requested
                        if return_cropped_images:
                            import base64
                            _, buffer = cv2.imencode('.jpg', cropped)
                            img_base64 = base64.b64encode(buffer).decode('utf-8')
                            cropped_images_data.append({
                                "detection_index": i,
                                "class_name": class_name,
                                "image_base64": img_base64
                            })
                            
                    except Exception as ocr_error:
                        print(f"OCR failed for detection {i}: {ocr_error}")
                        continue
        
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        response_data = {
            "status": "success",
            "model_name": model_info['model_name'],
            "detection_results": detection_results,
            "ocr_results": ocr_results,
            "processing_time_ms": processing_time,
            "total_detections": len(detection_results),
            "total_texts": len(ocr_results),
            "languages_used": lang_list,
            "confidence_threshold": confidence_threshold
        }
        
        if return_cropped_images:
            response_data["cropped_images"] = cropped_images_data
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection and OCR failed: {str(e)}")
    finally:
        # Clean up
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

@router.post("/ocr/text-only")
async def ocr_text_only(
    file: UploadFile = File(...),
    languages: str = Form('["en","vi"]'),
    return_bboxes: bool = Form(True)
):
    
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Image must be jpg, jpeg, or png")
    
    temp_files = []
    
    try:
        # Parse languages
        try:
            lang_list = json.loads(languages)
            if not isinstance(lang_list, list):
                raise ValueError("Languages must be a JSON array")
            
            # Validate languages
            invalid_langs = [lang for lang in lang_list if lang not in SUPPORTED_LANGUAGES]
            if invalid_langs:
                raise ValueError(f"Unsupported language(s): {', '.join(invalid_langs)}. Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}")
                
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail=f"Invalid JSON format for languages. Expected format: [\"en\",\"vi\"]. Received: {languages}")
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
            temp_files.append(temp_path)
        
        # Load image
        image = cv2.imread(temp_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        start_time = datetime.now()
        
        # Sử dụng EasyOCRManager đơn giản
        ocr_result = easy_ocr_manager.extract_text(image, lang_list, preprocess=True)
        
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Process results
        ocr_results = []
        full_text = []
        
        if ocr_result["success"]:
            if return_bboxes and ocr_result.get("bboxes"):
                # Trả về với bounding boxes
                for i, bbox in enumerate(ocr_result["bboxes"]):
                    if i < len(ocr_result["texts"]) and i < len(ocr_result["confidences"]):
                        text = ocr_result["texts"][i]
                        confidence = ocr_result["confidences"][i]
                        
                        ocr_results.append(OCRResult(
                            bbox=bbox,
                            text=text,
                            confidence=confidence
                        ))
                        full_text.append(text)
            else:
                # Trả về chỉ text
                if ocr_result.get("texts"):
                    for i, text in enumerate(ocr_result["texts"]):
                        if i < len(ocr_result["confidences"]):
                            confidence = ocr_result["confidences"][i]
                            ocr_results.append({
                                "text": text,
                                "confidence": confidence
                            })
                            full_text.append(text)
                else:
                    # Fallback to combined text
                    full_text.append(ocr_result.get("text", ""))
        
        return {
            "status": "success",
            "ocr_results": ocr_results,
            "full_text": "\n".join(full_text),
            "total_texts": len(ocr_results),
            "processing_time_ms": processing_time,
            "languages_used": lang_list
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")
    finally:
        # Clean up
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

@router.delete("/models/{model_id}")
async def delete_yolo_model(model_id: str):
    try:
        # Load model info
        info_path = Path("temp/model_info") / f"{model_id}.json"
        if not info_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        with open(info_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        # Remove from cache
        if model_id in yolo_models_cache:
            del yolo_models_cache[model_id]
        
        # Delete files
        files_to_delete = [
            model_info.get('model_path'),
            model_info.get('sample_image_path'),
            str(info_path)
        ]
        
        deleted_files = []
        for file_path in files_to_delete:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    deleted_files.append(file_path)
                except:
                    pass
        
        return {
            "status": "success",
            "message": f"Model '{model_info['model_name']}' deleted successfully",
            "deleted_files": len(deleted_files)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

@router.get("/models/{model_id}/info")
async def get_model_info(model_id: str):
    try:
        info_path = Path("temp/model_info") / f"{model_id}.json"
        if not info_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        with open(info_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        # Check if model file exists
        model_exists = os.path.exists(model_info.get('model_path', ''))
        sample_exists = os.path.exists(model_info.get('sample_image_path', ''))
        
        return {
            "status": "success",
            "model_info": model_info,
            "model_file_exists": model_exists,
            "sample_image_exists": sample_exists,
            "is_cached": model_id in yolo_models_cache
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.post("/cleanup")
async def cleanup_cache():
    try:
        cleared_yolo = len(yolo_models_cache)
        cleared_ocr = len(easyocr_readers_cache)
        
        yolo_models_cache.clear()
        easyocr_readers_cache.clear()
        
        return {
            "status": "success",
            "message": "Cache cleared successfully",
            "cleared_yolo_models": cleared_yolo,
            "cleared_ocr_readers": cleared_ocr
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
