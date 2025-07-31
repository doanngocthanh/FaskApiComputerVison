from config import OnnxConfig, PthConfig, TensorflowConfig, ModelConfig,DBConfig
from ultralytics import YOLO
import onnxruntime as ort
import os
import cv2
import numpy as np
db_config = DBConfig()
class ModelAi:
    def __init__(self, id):
        self.id = id
        query = "SELECT * FROM models WHERE model_id = ?"
        model = db_config.fetch_one(query, (id,))
        if not model:
            raise ValueError(f"Model with ID {id} not found")
        return {
            "model_id": model[0],
            "model_name_id": model[1],
            "model_name": model[2],
            "model_type": model[3],
            "model_path": model[4],
            "upload_at": model[5],
            "is_public": model[6],
            "uploaded_by": model[7]
        }
    def load_model(self):
            """Load YOLO v8 model in .pt or .onnx format"""
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            file_extension = os.path.splitext(self.model_path)[1].lower()
            
            if file_extension == '.pt':
                # Load PyTorch model
                model = YOLO(self.model_path)
                return model
            elif file_extension == '.onnx':
                # Load ONNX model
                session = ort.InferenceSession(self.model_path)
                return session
            else:
                raise ValueError(f"Unsupported model format: {file_extension}. Only .pt and .onnx are supported")

    def predict(self, image_path):
            """Make prediction using loaded model"""
            model = self.load_model()
            file_extension = os.path.splitext(self.model_path)[1].lower()
            
            if file_extension == '.pt':
                results = model(image_path)
                return results
            elif file_extension == '.onnx':
                # For ONNX, you'll need to preprocess the image and run inference
                
                image = cv2.imread(image_path)
                # Add preprocessing logic here based on your model requirements
                input_tensor = np.expand_dims(image, axis=0).astype(np.float32)
                
                input_name = model.get_inputs()[0].name
                outputs = model.run(None, {input_name: input_tensor})
                return outputs