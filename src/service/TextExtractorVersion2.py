import os
import cv2
import numpy as np
import re
import torch
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from src.service.YOLODetector import YOLODetector
from src.service.EasyOCRManager import EasyOCRManager
from src.service.PaddleOCR import PaddleOCR

# Import VietOCR
try:
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    from PIL import Image
    VIETOCR_AVAILABLE = True
    print("✅ VietOCR imported successfully")
except ImportError as e:
    VIETOCR_AVAILABLE = False
    print(f"⚠️ VietOCR not available: {e}")

class ModelType(Enum):
    """Available VietOCR model types"""
    VGG_SEQ2SEQ = "vgg_seq2seq"
    VGG_TRANSFORMER = "vgg_transformer"
    RESNET_TRANSFORMER = "resnet_transformer"
    TRANSFORMER_OCR = "transformerocr"
    
class OCRStrategy(Enum):
    """OCR processing strategies"""
    PADDLE_EASYOCR = "paddle_easyocr"  # PaddleOCR detection + EasyOCR recognition
    VIETOCR_ONLY = "vietocr_only"      # VietOCR only
    ENSEMBLE_VOTE = "ensemble_vote"    # Vote between multiple methods
    ENSEMBLE_BEST = "ensemble_best"    # Select best result based on confidence
    HYBRID_CASCADE = "hybrid_cascade"  # Try methods in order until success

@dataclass
class OCRResult:
    """OCR result with confidence and metadata"""
    text: str
    confidence: float
    method: str
    processing_time: float
    has_vietnamese_chars: bool = False
    
class VietOCRModelManager:
    """Manages multiple VietOCR models and configurations"""
    
    def __init__(self):
        self.models = {}
        self.configs = {}
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 VietOCR ModelManager initialized on {self.device}")
    
    def create_model_config(self, model_type: ModelType, **kwargs) -> dict:
        """Create optimized configuration for different model types"""
        
        # Base Vietnamese vocabulary
        base_vocab = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'\'()*+,-./:;<=>?@[\\]^_`{|}~ '
        
        # Base configuration
        config = {
            'vocab': base_vocab,
            'device': self.device,
            'quiet': kwargs.get('quiet', False)
        }
        
        # Model-specific configurations
        if model_type == ModelType.VGG_SEQ2SEQ:
            config.update({
                'cnn': {
                    'pretrain': True,
                    'model_name': 'vgg19_bn',
                    'ss': [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
                    'ks': [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
                    'hidden': 256
                },
                'seq_modeling': 'lstm',
                'lstm': {
                    'hidden': 256,
                    'num_layers': 2,
                    'dropout': 0.1
                },
                'predictor': {
                    'beamsearch': kwargs.get('beamsearch', False)
                }
            })
            
        elif model_type == ModelType.VGG_TRANSFORMER:
            config.update({
                'cnn': {
                    'pretrain': True,
                    'model_name': 'vgg19_bn',
                    'ss': [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
                    'ks': [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
                    'hidden': 256
                },
                'seq_modeling': 'transformer',
                'transformer': {
                    'd_model': kwargs.get('d_model', 256),
                    'nhead': kwargs.get('nhead', 8),
                    'num_encoder_layers': kwargs.get('num_encoder_layers', 6),
                    'num_decoder_layers': kwargs.get('num_decoder_layers', 6),
                    'dim_feedforward': kwargs.get('dim_feedforward', 2048),
                    'max_seq_length': kwargs.get('max_seq_length', 1024),
                    'pos_dropout': kwargs.get('pos_dropout', 0.1),
                    'trans_dropout': kwargs.get('trans_dropout', 0.1)
                },
                'predictor': {
                    'beamsearch': kwargs.get('beamsearch', False)
                }
            })
            
        elif model_type == ModelType.RESNET_TRANSFORMER:
            config.update({
                'cnn': {
                    'pretrain': True,
                    'model_name': 'resnet50',
                    'ss': [[2, 2], [2, 2], [2, 1], [2, 1]],
                    'ks': [[2, 2], [2, 2], [2, 1], [2, 1]],
                    'hidden': 512
                },
                'seq_modeling': 'transformer',
                'transformer': {
                    'd_model': kwargs.get('d_model', 512),
                    'nhead': kwargs.get('nhead', 8),
                    'num_encoder_layers': kwargs.get('num_encoder_layers', 6),
                    'num_decoder_layers': kwargs.get('num_decoder_layers', 6),
                    'dim_feedforward': kwargs.get('dim_feedforward', 2048),
                    'max_seq_length': kwargs.get('max_seq_length', 1024),
                    'pos_dropout': kwargs.get('pos_dropout', 0.1),
                    'trans_dropout': kwargs.get('trans_dropout', 0.1)
                },
                'predictor': {
                    'beamsearch': kwargs.get('beamsearch', False)
                }
            })
            
        elif model_type == ModelType.TRANSFORMER_OCR:
            config.update({
                'seq_modeling': 'transformer',
                'transformer': {
                    'd_model': kwargs.get('d_model', 256),
                    'nhead': kwargs.get('nhead', 8),
                    'num_encoder_layers': kwargs.get('num_encoder_layers', 6),
                    'num_decoder_layers': kwargs.get('num_decoder_layers', 6),
                    'dim_feedforward': kwargs.get('dim_feedforward', 2048),
                    'max_seq_length': kwargs.get('max_seq_length', 1024),
                    'pos_dropout': kwargs.get('pos_dropout', 0.1),
                    'trans_dropout': kwargs.get('trans_dropout', 0.1)
                },
                'predictor': {
                    'beamsearch': kwargs.get('beamsearch', False)
                }
            })
        
        # Add common configurations
        config.update({
            'optimizer': {
                'max_lr': kwargs.get('max_lr', 0.0003),
                'pct_start': kwargs.get('pct_start', 0.1)
            },
            'trainer': {
                'batch_size': kwargs.get('batch_size', 32),
                'print_every': 200,
                'valid_every': 4000,
                'iters': 100000
            },
            'dataset': {
                'image_height': kwargs.get('image_height', 32),
                'image_min_width': kwargs.get('image_min_width', 32),
                'image_max_width': kwargs.get('image_max_width', 512)
            },
            'dataloader': {
                'num_workers': kwargs.get('num_workers', 3),
                'pin_memory': True
            },
            'aug': {
                'image_aug': kwargs.get('image_aug', True),
                'masked_language_model': kwargs.get('masked_language_model', True)
            }
        })
        
        return config
    
    def load_model(self, model_type: ModelType, weights_path: str = None, **config_kwargs) -> bool:
        """Load a specific VietOCR model"""
        if not VIETOCR_AVAILABLE:
            print("⚠️ VietOCR not available")
            return False
        
        try:
            # Create config for this model type
            config = self.create_model_config(model_type, **config_kwargs)
            
            # Find weights file
            if weights_path and os.path.exists(weights_path):
                config['weights'] = weights_path
            else:
                # Auto-detect weights
                weights_paths = [
                    f"./weights/{model_type.value}.pth",
                    f"./weights/transformerocr.pth",
                    f".local/share/models/{model_type.value}.pth",
                    rf"\\10.160.99.97\010046\00_AP\to_Thanh\WorkSpace\VietOcr\{model_type.value}.pth"
                ]
                
                weights_found = False
                for path in weights_paths:
                    if os.path.exists(path):
                        config['weights'] = path
                        weights_found = True
                        break
                
                if not weights_found:
                    print(f"⚠️ No weights found for {model_type.value}")
                    return False
            
            # Initialize model
            predictor = Predictor(config)
            
            self.models[model_type] = predictor
            self.configs[model_type] = config
            
            print(f"✅ {model_type.value} model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load {model_type.value}: {e}")
            return False
    
    def predict(self, model_type: ModelType, image: Image.Image) -> OCRResult:
        """Predict text using specific model"""
        if model_type not in self.models:
            return OCRResult("", 0.0, model_type.value, 0.0)
        
        import time
        start_time = time.time()
        
        try:
            result = self.models[model_type].predict(image)
            processing_time = time.time() - start_time
            
            if result and isinstance(result, str):
                confidence = self._calculate_confidence(result)
                has_vietnamese = self._has_vietnamese_chars(result)
                
                return OCRResult(
                    text=result.strip(),
                    confidence=confidence,
                    method=model_type.value,
                    processing_time=processing_time,
                    has_vietnamese_chars=has_vietnamese
                )
            
        except Exception as e:
            print(f"❌ Prediction error with {model_type.value}: {e}")
        
        return OCRResult("", 0.0, model_type.value, time.time() - start_time)
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score based on text characteristics"""
        if not text.strip():
            return 0.0
        
        score = 0.0
        
        # Length bonus (longer texts often more reliable)
        score += min(len(text) * 0.02, 0.3)
        
        # Vietnamese character bonus
        if self._has_vietnamese_chars(text):
            score += 0.2
        
        # Character diversity bonus
        unique_chars = len(set(text.lower()))
        total_chars = len(text)
        if total_chars > 0:
            diversity = unique_chars / total_chars
            score += diversity * 0.2
        
        # Penalty for too many special characters
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if len(text) > 0:
            special_ratio = special_chars / len(text)
            if special_ratio > 0.3:
                score -= 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def _has_vietnamese_chars(self, text: str) -> bool:
        """Check if text contains Vietnamese characters"""
        vietnamese_chars = 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'
        vietnamese_chars += vietnamese_chars.upper()
        return any(char in vietnamese_chars for char in text)

class OptimizedTextExtractor:
    """Enhanced TextExtractor with multiple VietOCR models and intelligent strategy selection"""
    
    def __init__(self, models_to_load: List[ModelType] = None, default_strategy: OCRStrategy = OCRStrategy.HYBRID_CASCADE):
        # Initialize base OCR services
        self.easy_ocr_manager = EasyOCRManager()
        self.paddle_ocr = PaddleOCR()
        
        # Initialize VietOCR model manager
        self.viet_model_manager = VietOCRModelManager()
        
        # Load specified models or default ones
        if models_to_load is None:
            models_to_load = [ModelType.TRANSFORMER_OCR, ModelType.VGG_TRANSFORMER]
        
        self.available_models = []
        for model_type in models_to_load:
            if self.viet_model_manager.load_model(model_type):
                self.available_models.append(model_type)
        
        self.default_strategy = default_strategy
        
        print(f"🚀 OptimizedTextExtractor initialized with {len(self.available_models)} VietOCR models")
        print(f"📋 Available models: {[m.value for m in self.available_models]}")
        print(f"🎯 Default strategy: {default_strategy.value}")
    
    def extract_vietnamese_ensemble(self, image: np.ndarray, strategy: OCRStrategy = None) -> OCRResult:
        """Extract Vietnamese text using ensemble methods"""
        if strategy is None:
            strategy = self.default_strategy
        
        # Convert to PIL Image
        if len(image.shape) == 3 and image.shape[2] == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        
        results = []
        
        if strategy == OCRStrategy.VIETOCR_ONLY:
            # Use only VietOCR models
            for model_type in self.available_models:
                result = self.viet_model_manager.predict(model_type, pil_image)
                if result.text:
                    results.append(result)
        
        elif strategy == OCRStrategy.PADDLE_EASYOCR:
            # Use hybrid PaddleOCR + EasyOCR
            hybrid_result = self._extract_hybrid_paddle_easyocr(image, language="vi")
            if hybrid_result:
                results.append(OCRResult(
                    text=hybrid_result,
                    confidence=0.7,
                    method="paddle_easyocr",
                    processing_time=0.0,
                    has_vietnamese_chars=self.viet_model_manager._has_vietnamese_chars(hybrid_result)
                ))
        
        elif strategy == OCRStrategy.ENSEMBLE_VOTE:
            # Get results from all methods
            all_results = []
            
            # VietOCR results
            for model_type in self.available_models:
                result = self.viet_model_manager.predict(model_type, pil_image)
                if result.text:
                    all_results.append(result)
            
            # Hybrid result
            hybrid_result = self._extract_hybrid_paddle_easyocr(image, language="vi")
            if hybrid_result:
                all_results.append(OCRResult(
                    text=hybrid_result,
                    confidence=0.7,
                    method="paddle_easyocr",
                    processing_time=0.0,
                    has_vietnamese_chars=self.viet_model_manager._has_vietnamese_chars(hybrid_result)
                ))
            
            # Vote for best result
            if all_results:
                results = [self._vote_best_result(all_results)]
        
        elif strategy == OCRStrategy.ENSEMBLE_BEST:
            # Get all results and select highest confidence
            for model_type in self.available_models:
                result = self.viet_model_manager.predict(model_type, pil_image)
                if result.text:
                    results.append(result)
            
            hybrid_result = self._extract_hybrid_paddle_easyocr(image, language="vi")
            if hybrid_result:
                results.append(OCRResult(
                    text=hybrid_result,
                    confidence=0.7,
                    method="paddle_easyocr",
                    processing_time=0.0,
                    has_vietnamese_chars=self.viet_model_manager._has_vietnamese_chars(hybrid_result)
                ))
        
        elif strategy == OCRStrategy.HYBRID_CASCADE:
            # Try methods in order until good result
            
            # Try VietOCR models first (usually best for Vietnamese)
            for model_type in self.available_models:
                result = self.viet_model_manager.predict(model_type, pil_image)
                if result.text and result.confidence > 0.5:
                    results.append(result)
                    break
            
            # If no good VietOCR result, try hybrid method
            if not results:
                hybrid_result = self._extract_hybrid_paddle_easyocr(image, language="vi")
                if hybrid_result:
                    results.append(OCRResult(
                        text=hybrid_result,
                        confidence=0.6,
                        method="paddle_easyocr",
                        processing_time=0.0,
                        has_vietnamese_chars=self.viet_model_manager._has_vietnamese_chars(hybrid_result)
                    ))
            
            # Final fallback to direct EasyOCR
            if not results:
                try:
                    ocr_result = self.easy_ocr_manager.extract_text(image, languages=["vi"])
                    if ocr_result["success"] and ocr_result.get("texts"):
                        fallback_text = ' '.join(ocr_result["texts"])
                        results.append(OCRResult(
                            text=fallback_text,
                            confidence=0.4,
                            method="easyocr_fallback",
                            processing_time=0.0,
                            has_vietnamese_chars=self.viet_model_manager._has_vietnamese_chars(fallback_text)
                        ))
                except Exception as e:
                    print(f"    ⚠️ EasyOCR fallback error: {e}")
        
        # Return best result
        if results:
            best_result = max(results, key=lambda x: x.confidence)
            print(f"    🏆 Best result: '{best_result.text[:50]}...' (confidence: {best_result.confidence:.2f}, method: {best_result.method})")
            return best_result
        
        return OCRResult("", 0.0, "no_result", 0.0)
    
    def _extract_hybrid_paddle_easyocr(self, image: np.ndarray, language: str = "vi") -> str:
        """Extract using PaddleOCR detection + EasyOCR recognition"""
        try:
            text_regions = self.paddle_ocr.detect_text_regions(image)
            
            if text_regions and len(text_regions) > 0:
                cropped_images, angles = self.paddle_ocr.classify_text_orientation(image, text_regions)
                
                extracted_texts = []
                for cropped_img in cropped_images:
                    try:
                        ocr_result = self.easy_ocr_manager.extract_text(
                            cropped_img, 
                            languages=[language], 
                            preprocess=False
                        )
                        if ocr_result["success"] and ocr_result.get("texts"):
                            extracted_texts.extend(ocr_result["texts"])
                    except Exception as e:
                        continue
                
                if extracted_texts:
                    return ' '.join(extracted_texts).strip()
                    
        except Exception as e:
            print(f"    ⚠️ Hybrid extraction error: {e}")
        
        return ""
    
    def _vote_best_result(self, results: List[OCRResult]) -> OCRResult:
        """Vote for best result among multiple OCR results"""
        if not results:
            return OCRResult("", 0.0, "no_result", 0.0)
        
        if len(results) == 1:
            return results[0]
        
        # Simple voting: prefer results with Vietnamese chars and higher confidence
        scored_results = []
        for result in results:
            score = result.confidence
            if result.has_vietnamese_chars:
                score += 0.2
            if len(result.text) > 10:  # Longer texts often more reliable
                score += 0.1
            scored_results.append((score, result))
        
        return max(scored_results, key=lambda x: x[0])[1]
    
    def extract_from_image_vi(self, image: np.ndarray, strategy: OCRStrategy = None) -> str:
        """Enhanced Vietnamese extraction with strategy selection"""
        result = self.extract_vietnamese_ensemble(image, strategy)
        return result.text
    
    def extract_from_image_en(self, image: np.ndarray) -> str:
        """Extract English text (unchanged from original)"""
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array representing an image.")
        
        try:
            text_regions = self.paddle_ocr.detect_text_regions(image)
            
            if text_regions and len(text_regions) > 0:
                print(f"    📍 PaddleOCR detected {len(text_regions)} text regions")
                
                cropped_images, angles = self.paddle_ocr.classify_text_orientation(image, text_regions)
                
                extracted_texts = []
                for i, cropped_img in enumerate(cropped_images):
                    try:
                        ocr_result = self.easy_ocr_manager.extract_text(cropped_img, languages=["en"], preprocess=False)
                        if ocr_result["success"] and ocr_result.get("texts"):
                            extracted_texts.extend(ocr_result["texts"])
                    except Exception as e:
                        print(f"    ⚠️ EasyOCR recognition error for region {i}: {str(e)}")
                        continue
                
                if extracted_texts:
                    result_text = ' '.join(extracted_texts).strip()
                    print(f"    ✅ Hybrid result: '{result_text[:50]}...'")
                    return result_text
                    
        except Exception as hybrid_error:
            print(f"    ⚠️ Hybrid approach failed: {str(hybrid_error)}, using direct EasyOCR")
        
        # Fallback to direct EasyOCR
        try:
            ocr_result = self.easy_ocr_manager.extract_text(image, languages=["en"])
            
            if ocr_result["success"] and ocr_result.get("texts"):
                return ' '.join(ocr_result["texts"])
                
        except Exception as easy_error:
            print(f"    ❌ EasyOCR English error: {str(easy_error)}")
        
        return ""
    
    def extract_from_bbox_vi(self, original_image: np.ndarray, bbox: List[int], strategy: OCRStrategy = None) -> str:
        """Enhanced Vietnamese bbox extraction"""
        try:
            x1, y1, x2, y2 = bbox
            h, w = original_image.shape[:2]
            
            # Validate và adjust bbox
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            print(f"    🔍 Extracting Vietnamese text from bbox [{x1}, {y1}, {x2}, {y2}]")
            
            # Crop with padding
            padding = 15
            padded_x1 = max(0, x1 - padding)
            padded_y1 = max(0, y1 - padding)
            padded_x2 = min(w, x2 + padding)
            padded_y2 = min(h, y2 + padding)
            
            cropped_region = original_image[padded_y1:padded_y2, padded_x1:padded_x2]
            
            if cropped_region.size > 0:
                return self.extract_from_image_vi(cropped_region, strategy)
            
        except Exception as e:
            print(f"    ❌ Error in bbox Vietnamese extraction: {str(e)}")
        
        return ""
    
    def extract_from_bbox_en(self, original_image: np.ndarray, bbox: List[int]) -> str:
        """English bbox extraction (unchanged from original)"""
        try:
            x1, y1, x2, y2 = bbox
            h, w = original_image.shape[:2]
            
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            print(f"    🔍 Extracting English text from bbox [{x1}, {y1}, {x2}, {y2}]")
            
            padding = 10
            padded_x1 = max(0, x1 - padding)
            padded_y1 = max(0, y1 - padding)
            padded_x2 = min(w, x2 + padding)
            padded_y2 = min(h, y2 + padding)
            
            cropped_region = original_image[padded_y1:padded_y2, padded_x1:padded_x2]
            
            if cropped_region.size > 0:
                return self.extract_from_image_en(cropped_region)
                
        except Exception as e:
            print(f"    ❌ Error in bbox English extraction: {str(e)}")
        
        return ""

# Usage examples
if __name__ == "__main__":
    # Initialize with specific models
    extractor = OptimizedTextExtractor(
        models_to_load=[ModelType.TRANSFORMER_OCR, ModelType.VGG_TRANSFORMER],
        default_strategy=OCRStrategy.ENSEMBLE_BEST
    )
    
    # Example usage
    import cv2
    
    # Load image
    image = cv2.imread("test_image.jpg")
    
    # Extract Vietnamese with different strategies
    result1 = extractor.extract_from_image_vi(image, OCRStrategy.VIETOCR_ONLY)
    result2 = extractor.extract_from_image_vi(image, OCRStrategy.ENSEMBLE_VOTE)
    result3 = extractor.extract_from_image_vi(image, OCRStrategy.HYBRID_CASCADE)
    
    print(f"VietOCR only: {result1}")
    print(f"Ensemble vote: {result2}")
    print(f"Hybrid cascade: {result3}")