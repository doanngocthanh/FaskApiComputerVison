import cv2
import numpy as np
from typing import Dict, Any
from src.service.TextExtractor import TextExtractor

if __name__ == "__main__":
    # Example usage
    extractor = TextExtractor()
    image_path = r"C:\Users\dntdo\Downloads\CCCD.v1i.yolov8\train\images\20020438_01JAKR50VENQWY5YAZW8H021BY204850_front_jpg.rf.87e8b9253b7a9396f8225e975e4cc26a.jpg"
    image = cv2.imread(image_path)
    if image is not None:
        result = extractor.extract_from_image_vi(image)
        
        print("Extracted Text:", result)
    else:
        print("Failed to load image.")
 