"""
Face Recognition Service - Quản lý nhận diện khuôn mặt với YOLO và Face Embeddings
Hỗ trợ:
- Face detection với YOLO models
- Face recognition với deep learning embeddings 
- Face registration và verification
- 2D/3D face analysis
- Database management cho face data
"""

import os
import cv2
import numpy as np
import sqlite3
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import base64
from PIL import Image
import pickle

# Import services
from src.service.YOLODetector import YOLODetector
from config import PtConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionService:
    """Service chính cho face recognition"""
    
    def __init__(self, models_dir: str = None, db_path: str = None):
        if models_dir is None:
            models_dir = "temp/models"
        if db_path is None:
            db_path = "temp/face_recognition.db"
            
        self.models_dir = Path(models_dir)
        self.db_path = Path(db_path)
        
        # Đảm bảo thư mục tồn tại
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize face detection models
        self._face_detector = None
        self._face_recognition_model = None
        
        # Initialize database
        self._init_database()
        
        logger.info(f"FaceRecognitionService initialized with models_dir: {self.models_dir}")
        
    def _init_database(self):
        """Khởi tạo database cho face recognition"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bảng registered_faces - lưu thông tin faces đã đăng ký
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS registered_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                face_embedding BLOB NOT NULL,
                face_image_path TEXT,
                confidence_threshold REAL DEFAULT 0.7,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Bảng face_detection_logs - log các lần detection
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_detection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                detected_faces INTEGER DEFAULT 0,
                matched_user_id TEXT,
                confidence REAL,
                detection_time REAL,
                image_path TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Bảng face_verification_logs - log các lần verification
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_verification_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                verification_result BOOLEAN,
                confidence REAL,
                verification_time REAL,
                image_path TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Index cho performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON registered_faces(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_matched_user_id ON face_detection_logs(matched_user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_verification_user_id ON face_verification_logs(user_id)')
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Face recognition database initialized")
    def get_face_detector(self, model_type: str = "yolo"):
        """Lấy face detector model"""
        if self._face_detector is None:
            pt_config = PtConfig()
            
            # Tìm face detection model - ưu tiên FACE_DETECT.pt
            model_candidates = [
                "FACE_DETECT.pt",
                "face_detection.pt",
                "yolov8n-face.pt", 
                "yolov8s-face.pt",
                "face_yolo.pt"
            ]
            
            model_path = None
            for candidate in model_candidates:
                candidate_path = os.path.join(pt_config.get_model_path(), candidate)
                if os.path.exists(candidate_path):
                    model_path = candidate_path
                    logger.info(f"✅ Found face detection model: {candidate}")
                    break
            
            if model_path is None:
                logger.warning("⚠️ Không tìm thấy face detection model, sử dụng YOLO model mặc định")
                # Fallback to default YOLO model
                model_path = pt_config.get_model_path()
            
            self._face_detector = YOLODetector(model_path=model_path, conf_threshold=0.3)
            logger.info(f"✅ Face detector loaded: {model_path}")
            
        return self._face_detector
     
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Phát hiện khuôn mặt trong ảnh
        Args:
            image: numpy array của ảnh
        Returns:
            List các khuôn mặt detected với bbox và confidence
        """
        detector = self.get_face_detector()
        
        try:
            detections = detector.detect(image)
            
            face_detections = []
            for detection in detections:
                # Lọc khuôn mặt với confidence cao
                confidence = detection.get('confidence', 0.0)
                class_name = detection.get('class_name', '').lower()
                
                # Kiểm tra nếu là face class hoặc có confidence cao
                if confidence >= 0.3 and ('face' in class_name or 'person' in class_name or len(class_name) == 0):
                    face_detections.append({
                        'bbox': detection['bbox'],
                        'confidence': confidence,
                        'class_name': detection.get('class_name', 'face'),
                        'class_id': detection.get('class_id', 0)
                    })
            
            # Sort theo confidence (cao nhất trước)
            face_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.info(f"✅ Detected {len(face_detections)} faces")
            return face_detections
            
        except Exception as e:
            logger.error(f"❌ Lỗi detect faces: {e}")
            return []
    
    def extract_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding từ cropped face image
        Args:
            face_image: cropped face image as numpy array
        Returns:
            Face embedding vector hoặc None nếu lỗi
        """
        try:
            # Resize face to standard size
            face_resized = cv2.resize(face_image, (160, 160))
            
            # Normalize pixel values
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            # Simple embedding: flatten và reduce dimension
            # Trong production, nên sử dụng pre-trained face recognition model
            embedding = face_normalized.flatten()
            
            # Reduce dimension using simple PCA-like approach
            # Trong thực tế nên dùng proper face recognition model như FaceNet, ArcFace
            if len(embedding) > 512:
                embedding = embedding[:512]  # Take first 512 features
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Lỗi extract face embedding: {e}")
            return None
    
    def register_face(self, 
                     user_id: str, 
                     name: str, 
                     image: np.ndarray,
                     confidence_threshold: float = 0.7,
                     metadata: Dict = None) -> Dict[str, Any]:
        """
        Đăng ký khuôn mặt mới
        Args:
            user_id: ID người dùng
            name: Tên người dùng  
            image: Ảnh chứa khuôn mặt
            confidence_threshold: Ngưỡng confidence cho recognition
            metadata: Thông tin bổ sung
        Returns:
            Kết quả đăng ký
        """
        start_time = datetime.now()
        
        try:
            # Detect faces trong ảnh
            face_detections = self.detect_faces(image)
            
            if not face_detections:
                return {
                    "success": False,
                    "error": "Không phát hiện được khuôn mặt trong ảnh",
                    "faces_detected": 0
                }
            
            if len(face_detections) > 1:
                return {
                    "success": False,
                    "error": "Phát hiện nhiều hơn 1 khuôn mặt, vui lòng sử dụng ảnh chỉ có 1 khuôn mặt",
                    "faces_detected": len(face_detections)
                }
            
            # Lấy khuôn mặt đầu tiên
            face_detection = face_detections[0]
            x1, y1, x2, y2 = face_detection['bbox']
            
            # Crop khuôn mặt
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return {
                    "success": False,
                    "error": "Không thể crop khuôn mặt từ ảnh"
                }
            
            # Extract face embedding
            face_embedding = self.extract_face_embedding(face_crop)
            
            if face_embedding is None:
                return {
                    "success": False,
                    "error": "Không thể extract face embedding"
                }
            
            # Lưu ảnh khuôn mặt
            face_images_dir = Path("temp/face_images")
            face_images_dir.mkdir(parents=True, exist_ok=True)
            
            face_image_filename = f"{user_id}_{int(datetime.now().timestamp())}.jpg"
            face_image_path = face_images_dir / face_image_filename
            cv2.imwrite(str(face_image_path), face_crop)
            
            # Lưu vào database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Kiểm tra user_id đã tồn tại chưa
            cursor.execute('SELECT id FROM registered_faces WHERE user_id = ? AND is_active = 1', (user_id,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                cursor.execute('''
                    UPDATE registered_faces 
                    SET name = ?, face_embedding = ?, face_image_path = ?,
                        confidence_threshold = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (name, pickle.dumps(face_embedding), str(face_image_path),
                      confidence_threshold, json.dumps(metadata or {}), user_id))
                action = "updated"
            else:
                # Insert new record
                cursor.execute('''
                    INSERT INTO registered_faces 
                    (user_id, name, face_embedding, face_image_path, confidence_threshold, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (user_id, name, pickle.dumps(face_embedding), str(face_image_path),
                      confidence_threshold, json.dumps(metadata or {})))
                action = "registered"
            
            conn.commit()
            conn.close()
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "success": True,
                "action": action,
                "user_id": user_id,
                "name": name,
                "face_detection": face_detection,
                "embedding_size": len(face_embedding),
                "face_image_path": str(face_image_path),
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ Lỗi register face: {e}")
            return {
                "success": False,
                "error": f"Lỗi hệ thống: {str(e)}"
            }
    
    def verify_face(self, user_id: str, image: np.ndarray) -> Dict[str, Any]:
        """
        Xác thực khuôn mặt với user đã đăng ký
        Args:
            user_id: ID người dùng cần xác thực
            image: Ảnh chứa khuôn mặt cần xác thực
        Returns:
            Kết quả xác thực
        """
        start_time = datetime.now()
        
        try:
            # Lấy thông tin user từ database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT name, face_embedding, confidence_threshold, face_image_path
                FROM registered_faces 
                WHERE user_id = ? AND is_active = 1
            ''', (user_id,))
            
            user_data = cursor.fetchone()
            
            if not user_data:
                return {
                    "success": False,
                    "error": f"User ID {user_id} không tồn tại hoặc chưa đăng ký khuôn mặt"
                }
            
            name, registered_embedding_blob, confidence_threshold, face_image_path = user_data
            registered_embedding = pickle.loads(registered_embedding_blob)
            
            # Detect face trong ảnh hiện tại
            face_detections = self.detect_faces(image)
            
            if not face_detections:
                # Log verification
                cursor.execute('''
                    INSERT INTO face_verification_logs 
                    (user_id, verification_result, confidence, verification_time, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, False, 0.0, (datetime.now() - start_time).total_seconds() * 1000,
                      json.dumps({"error": "no_face_detected"})))
                conn.commit()
                conn.close()
                
                return {
                    "success": False,
                    "verified": False,
                    "error": "Không phát hiện được khuôn mặt trong ảnh"
                }
            
            # Lấy face có confidence cao nhất
            best_face = max(face_detections, key=lambda x: x['confidence'])
            x1, y1, x2, y2 = best_face['bbox']
            
            # Crop khuôn mặt
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return {
                    "success": False,
                    "verified": False,
                    "error": "Không thể crop khuôn mặt từ ảnh"
                }
            
            # Extract face embedding
            current_embedding = self.extract_face_embedding(face_crop)
            
            if current_embedding is None:
                return {
                    "success": False,
                    "verified": False,
                    "error": "Không thể extract face embedding"
                }
            
            # Tính similarity
            similarity = np.dot(registered_embedding, current_embedding)
            is_verified = similarity >= confidence_threshold
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Log verification
            cursor.execute('''
                INSERT INTO face_verification_logs 
                (user_id, verification_result, confidence, verification_time, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, is_verified, float(similarity), processing_time,
                  json.dumps({"face_detection": best_face})))
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "verified": is_verified,
                "user_id": user_id,
                "name": name,
                "similarity": float(similarity),
                "confidence_threshold": confidence_threshold,
                "face_detection": best_face,
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ Lỗi verify face: {e}")
            return {
                "success": False,
                "verified": False,
                "error": f"Lỗi hệ thống: {str(e)}"
            }
    
    def recognize_face(self, image: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """
        Nhận diện khuôn mặt trong tất cả users đã đăng ký
        Args:
            image: Ảnh chứa khuôn mặt cần nhận diện
            top_k: Số lượng kết quả top matches trả về
        Returns:
            Kết quả nhận diện với top matches
        """
        start_time = datetime.now()
        session_id = hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        
        try:
            # Detect faces trong ảnh
            face_detections = self.detect_faces(image)
            
            if not face_detections:
                # Log detection
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO face_detection_logs 
                    (session_id, detected_faces, detection_time, metadata)
                    VALUES (?, ?, ?, ?)
                ''', (session_id, 0, (datetime.now() - start_time).total_seconds() * 1000,
                      json.dumps({"error": "no_face_detected"})))
                conn.commit()
                conn.close()
                
                return {
                    "success": False,
                    "session_id": session_id,
                    "error": "Không phát hiện được khuôn mặt trong ảnh",
                    "faces_detected": 0
                }
            
            # Lấy tất cả registered faces
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id, name, face_embedding, confidence_threshold
                FROM registered_faces 
                WHERE is_active = 1
            ''')
            
            registered_users = cursor.fetchall()
            
            if not registered_users:
                return {
                    "success": False,
                    "session_id": session_id,
                    "error": "Không có user nào đã đăng ký khuôn mặt",
                    "faces_detected": len(face_detections)
                }
            
            results = []
            
            # Xử lý từng khuôn mặt detected
            for i, face_detection in enumerate(face_detections):
                x1, y1, x2, y2 = face_detection['bbox']
                
                # Crop khuôn mặt
                face_crop = image[y1:y2, x1:x2]
                
                if face_crop.size == 0:
                    continue
                
                # Extract embedding
                current_embedding = self.extract_face_embedding(face_crop)
                
                if current_embedding is None:
                    continue
                
                # So sánh với tất cả registered users
                matches = []
                
                for user_id, name, embedding_blob, confidence_threshold in registered_users:
                    registered_embedding = pickle.loads(embedding_blob)
                    similarity = np.dot(registered_embedding, current_embedding)
                    
                    matches.append({
                        "user_id": user_id,
                        "name": name,
                        "similarity": float(similarity),
                        "confidence_threshold": confidence_threshold,
                        "is_match": similarity >= confidence_threshold
                    })
                
                # Sort by similarity (cao nhất trước)
                matches.sort(key=lambda x: x['similarity'], reverse=True)
                top_matches = matches[:top_k]
                
                # Lấy best match
                best_match = top_matches[0] if top_matches else None
                matched_user_id = best_match['user_id'] if best_match and best_match['is_match'] else None
                
                results.append({
                    "face_index": i,
                    "face_detection": face_detection,
                    "best_match": best_match,
                    "top_matches": top_matches,
                    "matched_user_id": matched_user_id
                })
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Log detection
            total_matches = sum(1 for r in results if r['matched_user_id'])
            best_overall_match = None
            if results:
                best_result = max(results, key=lambda x: x['best_match']['similarity'] if x['best_match'] else 0)
                best_overall_match = best_result['matched_user_id']
            
            cursor.execute('''
                INSERT INTO face_detection_logs 
                (session_id, detected_faces, matched_user_id, 
                 confidence, detection_time, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, len(face_detections), best_overall_match,
                  results[0]['best_match']['similarity'] if results and results[0]['best_match'] else 0.0,
                  processing_time, json.dumps({"total_matches": total_matches})))
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "session_id": session_id,
                "faces_detected": len(face_detections),
                "faces_processed": len(results),
                "total_registered_users": len(registered_users),
                "results": results,
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ Lỗi recognize face: {e}")
            return {
                "success": False,
                "session_id": session_id,
                "error": f"Lỗi hệ thống: {str(e)}"
            }
    
    def get_registered_users(self) -> List[Dict[str, Any]]:
        """Lấy danh sách tất cả users đã đăng ký"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id, name, face_image_path, confidence_threshold, 
                       created_at, updated_at, is_active
                FROM registered_faces 
                ORDER BY created_at DESC
            ''')
            
            users = []
            for row in cursor.fetchall():
                users.append({
                    "user_id": row[0],
                    "name": row[1],
                    "face_image_path": row[2],
                    "confidence_threshold": row[3],
                    "created_at": row[4],
                    "updated_at": row[5],
                    "is_active": bool(row[6])
                })
            
            conn.close()
            return users
            
        except Exception as e:
            logger.error(f"❌ Lỗi get registered users: {e}")
            return []
    
    def delete_user(self, user_id: str) -> Dict[str, Any]:
        """Xóa user đã đăng ký"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute('SELECT name FROM registered_faces WHERE user_id = ?', (user_id,))
            user = cursor.fetchone()
            
            if not user:
                return {
                    "success": False,
                    "error": f"User ID {user_id} không tồn tại"
                }
            
            # Soft delete - set is_active = 0
            cursor.execute('''
                UPDATE registered_faces 
                SET is_active = 0, updated_at = CURRENT_TIMESTAMP 
                WHERE user_id = ?
            ''', (user_id,))
            
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "user_id": user_id,
                "name": user[0],
                "message": "User đã được xóa thành công"
            }
            
        except Exception as e:
            logger.error(f"❌ Lỗi delete user: {e}")
            return {
                "success": False,
                "error": f"Lỗi hệ thống: {str(e)}"
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Lấy thống kê của hệ thống"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total registered users
            cursor.execute('SELECT COUNT(*) FROM registered_faces WHERE is_active = 1')
            total_users = cursor.fetchone()[0]
            
            # Total detections today
            cursor.execute('''
                SELECT COUNT(*) FROM face_detection_logs 
                WHERE date(created_at) = date('now')
            ''')
            today_detections = cursor.fetchone()[0]
            
            # Total verifications today  
            cursor.execute('''
                SELECT COUNT(*) FROM face_verification_logs 
                WHERE date(created_at) = date('now')
            ''')
            today_verifications = cursor.fetchone()[0]
            
            # Success rate today
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN verification_result = 1 THEN 1 ELSE 0 END) as success
                FROM face_verification_logs 
                WHERE date(created_at) = date('now')
            ''')
            verification_stats = cursor.fetchone()
            success_rate = (verification_stats[1] / verification_stats[0] * 100) if verification_stats[0] > 0 else 0
            
            conn.close()
            
            return {
                "total_registered_users": total_users,
                "today_detections": today_detections,
                "today_verifications": today_verifications,
                "today_success_rate": round(success_rate, 2),
                "database_path": str(self.db_path),
                "models_directory": str(self.models_dir)
            }
            
        except Exception as e:
            logger.error(f"❌ Lỗi get statistics: {e}")
            return {}

# Global instance
face_recognition_service = FaceRecognitionService()
