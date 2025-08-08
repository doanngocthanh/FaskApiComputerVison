import os
from argparse import ArgumentParser
from warnings import filterwarnings
import sys

import cv2
import numpy

from src.service.onnx.nets import nn
from src.service.onnx.utils import util
from config import OnnxConfig
filterwarnings("ignore")

class PaddleOCR:
    def __init__(self):
        self.detection = nn.Detection(OnnxConfig().get_model("detection"))
        self.recognition = nn.Recognition(OnnxConfig().get_model("recognition"))
        self.classification = nn.Classification(OnnxConfig().get_model("classification"))
    
    def _resource_path(self, relative_path):
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)
    
    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to Python native types"""
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.bool_):
            return bool(obj)
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        else:
            return obj
    
    def detect_text_regions(self, image_path_or_array):
        """
        Phát hiện vùng chứa text trong ảnh
        Args:
            image_path_or_array: Đường dẫn ảnh hoặc numpy array
        Returns:
            list: Danh sách các điểm tọa độ của vùng text
        """
        if isinstance(image_path_or_array, str):
            frame = cv2.imread(image_path_or_array)
        else:
            frame = image_path_or_array.copy()
        
        if frame is None:
            raise ValueError("Cannot load image")
        
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)  # inplace
        points = self.detection(frame)
        points = util.sort_polygon(list(points))
        
        return self._convert_numpy_types(points)
    
    def classify_text_orientation(self, image_path_or_array, text_regions=None):
        """
        Phân loại hướng của text (xoay text về đúng hướng)
        Args:
            image_path_or_array: Đường dẫn ảnh hoặc numpy array
            text_regions: Các vùng text đã được phát hiện (nếu có)
        Returns:
            tuple: (cropped_images, angles) - ảnh đã cắt và góc xoay
        """
        if isinstance(image_path_or_array, str):
            frame = cv2.imread(image_path_or_array)
        else:
            frame = image_path_or_array.copy()
        
        if frame is None:
            raise ValueError("Cannot load image")
        
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
        
        # Nếu chưa có text regions, phát hiện trước
        if text_regions is None:
            points = self.detection(frame)
            points = util.sort_polygon(list(points))
        else:
            points = text_regions
        
        # Convert points to numpy arrays to avoid list subtraction errors
        import numpy as np
        converted_points = []
        for point_set in points:
            if isinstance(point_set, list):
                converted_points.append(np.array(point_set, dtype=np.float32))
            else:
                converted_points.append(point_set.astype(np.float32))
        points = converted_points
        
        # Cắt ảnh theo vùng text
        cropped_images = [util.crop_image(frame, x) for x in points]
        
        # Phân loại hướng và xoay ảnh
        cropped_images, angles = self.classification(cropped_images)
        
        return cropped_images, self._convert_numpy_types(angles)
    
    def recognize_text(self, cropped_images):
        """
        Nhận diện text từ các ảnh đã được cắt và xoay
        Args:
            cropped_images: Danh sách các ảnh đã được cắt
        Returns:
            list: Kết quả nhận diện text
        """
        results = self.recognition(cropped_images)
        return self._convert_numpy_types(results)
    
    def detect_only(self, image_path_or_array):
        """
        Chỉ phát hiện vùng text (không nhận diện nội dung)
        Args:
            image_path_or_array: Đường dẫn ảnh hoặc numpy array
        Returns:
            dict: Thông tin vùng text
        """
        points = self.detect_text_regions(image_path_or_array)
        
        return {
            "text_regions": points,
            "region_count": len(points),
            "status": "detection_only"
        }
    
    def detect_and_classify(self, image_path_or_array):
        """
        Phát hiện vùng text và phân loại hướng
        Args:
            image_path_or_array: Đường dẫn ảnh hoặc numpy array
        Returns:
            dict: Thông tin vùng text và góc xoay
        """
        if isinstance(image_path_or_array, str):
            frame = cv2.imread(image_path_or_array)
        else:
            frame = image_path_or_array.copy()
        
        points = self.detect_text_regions(frame)
        cropped_images, angles = self.classify_text_orientation(frame, points)
        
        return {
            "text_regions": points,
            "angles": angles,
            "region_count": len(points),
            "status": "detection_and_classification"
        }
    
    def ocr_only(self, cropped_images):
        """
        Chỉ nhận diện text từ các ảnh đã được cắt sẵn
        Args:
            cropped_images: Danh sách các ảnh đã được cắt
        Returns:
            dict: Kết quả nhận diện
        """
        results = self.recognize_text(cropped_images)
        
        return {
            "texts": results,
            "text_count": len(results),
            "status": "ocr_only"
        }
    
    def process_image(self, filepath, save_output=False):
        """
        Xử lý ảnh đầy đủ: phát hiện -> phân loại -> nhận diện
        Args:
            filepath: Đường dẫn ảnh
            save_output: Có lưu kết quả không
        Returns:
            dict: Kết quả đầy đủ
        """
        frame = cv2.imread(filepath) 
        if frame is None:
            raise ValueError("Cannot load image")
        
        image = frame.copy()
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)  # inplace

        # Bước 1: Phát hiện vùng text
        points = self.detection(frame)
        points = util.sort_polygon(list(points))

        # Bước 2: Cắt ảnh và phân loại hướng
        cropped_images = [util.crop_image(frame, x) for x in points]
        cropped_images, angles = self.classification(cropped_images)
        
        # Bước 3: Nhận diện text
        results = self.recognition(cropped_images)

        # Convert numpy types
        results = self._convert_numpy_types(results)
        points = self._convert_numpy_types(points)
        angles = self._convert_numpy_types(angles)
        
        return {
            "texts": results,
            "text_regions": points,
            "angles": angles,
            "text_count": len(results),
            "status": "full_process"
        }
    
    def process_pipeline(self, image_path_or_array, steps=None):
        """
        Pipeline xử lý linh hoạt
        Args:
            image_path_or_array: Đường dẫn ảnh hoặc numpy array
            steps: Danh sách các bước ['detect', 'classify', 'recognize']
        Returns:
            dict: Kết quả theo các bước được chọn
        """
        if steps is None:
            steps = ['detect', 'classify', 'recognize']
        
        if isinstance(image_path_or_array, str):
            frame = cv2.imread(image_path_or_array)
        else:
            frame = image_path_or_array.copy()
        
        if frame is None:
            raise ValueError("Cannot load image")
        
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
        result = {"status": f"pipeline_{'-'.join(steps)}"}
        
        if 'detect' in steps:
            points = self.detection(frame)
            points = util.sort_polygon(list(points))
            result["text_regions"] = self._convert_numpy_types(points)
            result["region_count"] = len(points)
        
        if 'classify' in steps:
            if 'detect' not in steps:
                raise ValueError("Classification requires detection step")
            
            cropped_images = [util.crop_image(frame, x) for x in points]
            cropped_images, angles = self.classification(cropped_images)
            result["angles"] = self._convert_numpy_types(angles)
            result["cropped_images"] = cropped_images  # Keep for recognition
        
        if 'recognize' in steps:
            if 'classify' not in steps:
                if 'detect' not in steps:
                    raise ValueError("Recognition requires detection step")
                cropped_images = [util.crop_image(frame, x) for x in points]
                cropped_images, angles = self.classification(cropped_images)
            
            texts = self.recognition(cropped_images)
            result["texts"] = self._convert_numpy_types(texts)
            result["text_count"] = len(texts)
            
            # Remove cropped_images from result if it exists
            if "cropped_images" in result:
                del result["cropped_images"]
        
        return result

