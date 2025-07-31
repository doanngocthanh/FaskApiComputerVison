import cv2
from ultralytics import YOLO
import numpy as np
from config import PtConfig

class YOLODetector:
    def __init__(self, model_path=PtConfig.get_model, conf_threshold=0.5):
        """
        Initialize YOLO detector
        Args:
            model_path: Path to YOLO model file
            conf_threshold: Confidence threshold for filtering detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
    def detect(self, image_path, get_best_only=False):
        """
        Detect objects in image
        Args:
            image_path: Path to input image
            get_best_only: If True, returns only the highest confidence detection per class
        Returns:
            List of detected objects with bounding boxes and confidence
        """
        # Ensure the image path is valid and load the image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from path: {image_path}")
        else:
            image = image_path
        
        results = self.model(image, conf=self.conf_threshold)
        detected_objects = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    detected_objects.append({
                        'class_name': class_name,
                        'class_id': class_id,
                        'confidence': float(confidence),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
        
        # Filter to get best results
        if get_best_only:
            detected_objects = self._get_best_detections(detected_objects)
        
        # Sort by confidence (highest first)
        detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detected_objects
    
    def detect_from_frame(self, frame, get_best_only=False):
        """
        Detect objects from video frame
        Args:
            frame: OpenCV frame/image array
            get_best_only: If True, returns only the highest confidence detection per class
        Returns:
            List of detected objects
        """
        results = self.model(frame, conf=self.conf_threshold)
        detected_objects = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    detected_objects.append({
                        'class_name': class_name,
                        'class_id': class_id,
                        'confidence': float(confidence),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
        
        # Filter to get best results
        if get_best_only:
            detected_objects = self._get_best_detections(detected_objects)
        
        # Sort by confidence (highest first)
        detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detected_objects
    
    def _get_best_detections(self, detections):
        """
        Get the best detection for each class (highest confidence)
        Args:
            detections: List of all detections
        Returns:
            List of best detections per class
        """
        best_detections = {}
        
        for detection in detections:
            class_id = detection['class_id']
            if class_id not in best_detections or detection['confidence'] > best_detections[class_id]['confidence']:
                best_detections[class_id] = detection
        
        return list(best_detections.values())
    
    def get_top_detection(self, image_path):
        """
        Get only the single best detection (highest confidence)
        Args:
            image_path: Path to input image
        Returns:
            Single best detection or None if no detections
        """
        detections = self.detect(image_path)
        return detections[0] if detections else None