"""
Advanced Logging Middleware
Middleware ghi log chi tiết các endpoint với khả năng bật/tắt qua admin interface
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import json
import time
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading
import asyncio
from collections import defaultdict

class EndpointLogConfig:
    """Cấu hình log cho từng endpoint"""
    
    def __init__(self):
        self.config_file = "config/logging_config.json"
        self.log_dir = "logs"
        self.default_config = {
            "global_enabled": True,
            "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR
            "max_response_length": 1000,  # Giới hạn độ dài response log
            "retention_days": 30,
            "endpoints": {}  # Cấu hình riêng cho từng endpoint
        }
        self.config = self.load_config()
        self.ensure_log_directory()
    
    def ensure_log_directory(self):
        """Tạo thư mục logs nếu chưa có"""
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path("config").mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> Dict:
        """Load cấu hình từ file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Merge với default config
                    for key, value in self.default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            return self.default_config.copy()
        except Exception as e:
            logging.error(f"Failed to load logging config: {e}")
            return self.default_config.copy()
    
    def save_config(self):
        """Lưu cấu hình vào file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to save logging config: {e}")
    
    def is_endpoint_enabled(self, path: str, method: str) -> bool:
        """Kiểm tra endpoint có được log không"""
        if not self.config.get("global_enabled", True):
            return False
        
        endpoint_key = f"{method}:{path}"
        endpoint_config = self.config["endpoints"].get(endpoint_key, {})
        
        return endpoint_config.get("enabled", True)  # Mặc định là enabled
    
    def get_endpoint_log_level(self, path: str, method: str) -> str:
        """Lấy log level cho endpoint"""
        endpoint_key = f"{method}:{path}"
        endpoint_config = self.config["endpoints"].get(endpoint_key, {})
        
        return endpoint_config.get("log_level", self.config.get("log_level", "INFO"))
    
    def should_log_request_body(self, path: str, method: str) -> bool:
        """Có nên log request body không"""
        endpoint_key = f"{method}:{path}"
        endpoint_config = self.config["endpoints"].get(endpoint_key, {})
        
        return endpoint_config.get("log_request_body", method in ["POST", "PUT", "PATCH"])
    
    def should_log_response_body(self, path: str, method: str) -> bool:
        """Có nên log response body không"""
        endpoint_key = f"{method}:{path}"
        endpoint_config = self.config["endpoints"].get(endpoint_key, {})
        
        return endpoint_config.get("log_response_body", True)
    
    def update_endpoint_config(self, path: str, method: str, config: Dict):
        """Cập nhật cấu hình cho endpoint"""
        endpoint_key = f"{method}:{path}"
        if endpoint_key not in self.config["endpoints"]:
            self.config["endpoints"][endpoint_key] = {}
        
        self.config["endpoints"][endpoint_key].update(config)
        self.save_config()

class EndpointLogger:
    """Logger chuyên dụng cho endpoints"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.loggers = {}
        self.setup_loggers()
    
    def setup_loggers(self):
        """Setup các logger cho từng loại log"""
        # Logger cho request/response
        self.request_logger = self._create_logger(
            "endpoint_requests", 
            os.path.join(self.log_dir, "requests.log")
        )
        
        # Logger cho errors
        self.error_logger = self._create_logger(
            "endpoint_errors",
            os.path.join(self.log_dir, "errors.log")
        )
        
        # Logger cho performance
        self.performance_logger = self._create_logger(
            "endpoint_performance",
            os.path.join(self.log_dir, "performance.log")
        )
    
    def _create_logger(self, name: str, log_file: str):
        """Tạo logger với file handler"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        
        # Tránh duplicate handlers
        if logger.handlers:
            return logger
        
        # File handler
        handler = logging.FileHandler(log_file, encoding='utf-8')
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def log_request(self, log_data: Dict, level: str = "INFO"):
        """Log request data"""
        message = self._format_log_message(log_data)
        
        if level == "DEBUG":
            self.request_logger.debug(message)
        elif level == "WARNING":
            self.request_logger.warning(message)
        elif level == "ERROR":
            self.request_logger.error(message)
        else:
            self.request_logger.info(message)
    
    def log_error(self, log_data: Dict):
        """Log error data"""
        message = self._format_log_message(log_data)
        self.error_logger.error(message)
    
    def log_performance(self, log_data: Dict):
        """Log performance data"""
        message = self._format_log_message(log_data)
        self.performance_logger.info(message)
    
    def _format_log_message(self, log_data: Dict) -> str:
        """Format log message as JSON string"""
        try:
            return json.dumps(log_data, ensure_ascii=False, separators=(',', ':'))
        except Exception:
            return str(log_data)

class AdvancedLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware log chi tiết các endpoint calls"""
    
    def __init__(self, app):
        super().__init__(app)
        self.config = EndpointLogConfig()
        self.logger = EndpointLogger()
        
        # Stats tracking
        self.stats = defaultdict(lambda: {
            "total_requests": 0,
            "error_count": 0,
            "avg_response_time": 0,
            "last_called": None
        })
        
        # Cache để tối ưu performance
        self.endpoint_cache = {}
        
        # Background cleanup task
        self.start_cleanup_task()
    
    def start_cleanup_task(self):
        """Khởi động task cleanup logs định kỳ"""
        def cleanup_worker():
            while True:
                try:
                    self.cleanup_old_logs()
                    time.sleep(24 * 60 * 60)  # Chạy mỗi ngày
                except Exception as e:
                    logging.error(f"Log cleanup error: {e}")
        
        thread = threading.Thread(target=cleanup_worker, daemon=True)
        thread.start()
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = self._generate_request_id()
        
        # Extract request info
        client_ip = self._get_client_ip(request)
        path = request.url.path
        method = request.method
        
        # Check if logging is enabled for this endpoint
        if not self.config.is_endpoint_enabled(path, method):
            return await call_next(request)
        
        # Prepare log data
        log_data = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "client_ip": client_ip,
            "method": method,
            "path": path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "user_agent": request.headers.get("user-agent", ""),
        }
        
        # Log request body if configured
        if self.config.should_log_request_body(path, method):
            try:
                body = await request.body()
                if body:
                    # Limit body size for logging
                    max_length = self.config.config.get("max_response_length", 1000)
                    body_str = body.decode('utf-8')[:max_length]
                    log_data["request_body"] = body_str
                    if len(body) > max_length:
                        log_data["request_body_truncated"] = True
                
                # Re-create request with body for next middleware
                request = Request(request.scope, receive=self._create_receive(body))
            except Exception as e:
                log_data["request_body_error"] = str(e)
        
        # Process request
        try:
            response = await call_next(request)
            end_time = time.time()
            response_time = end_time - start_time
            
            # Update log data with response info
            log_data.update({
                "status_code": response.status_code,
                "response_time_ms": round(response_time * 1000, 2),
                "response_headers": dict(response.headers)
            })
            
            # Log response body if configured and not too large
            if self.config.should_log_response_body(path, method):
                response_body = await self._extract_response_body(response)
                if response_body:
                    max_length = self.config.config.get("max_response_length", 1000)
                    if len(response_body) <= max_length:
                        log_data["response_body"] = response_body
                    else:
                        log_data["response_body"] = response_body[:max_length]
                        log_data["response_body_truncated"] = True
            
            # Determine log level based on status code
            if response.status_code >= 500:
                level = "ERROR"
                self.logger.log_error(log_data)
            elif response.status_code >= 400:
                level = "WARNING"
            else:
                level = self.config.get_endpoint_log_level(path, method)
            
            # Log the request
            self.logger.log_request(log_data, level)
            
            # Update stats
            self._update_stats(path, method, response_time, response.status_code >= 400)
            
            # Log performance if slow
            if response_time > 1.0:  # Slow request threshold
                perf_data = {
                    "request_id": request_id,
                    "path": path,
                    "method": method,
                    "response_time_ms": round(response_time * 1000, 2),
                    "status_code": response.status_code,
                    "client_ip": client_ip
                }
                self.logger.log_performance(perf_data)
            
            return response
            
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            # Log error
            error_data = log_data.copy()
            error_data.update({
                "error": str(e),
                "error_type": type(e).__name__,
                "response_time_ms": round(response_time * 1000, 2)
            })
            
            self.logger.log_error(error_data)
            self._update_stats(path, method, response_time, True)
            
            raise
    
    def _generate_request_id(self) -> str:
        """Tạo unique request ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _get_client_ip(self, request: Request) -> str:
        """Lấy IP thực của client"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _create_receive(self, body: bytes):
        """Tạo receive callable với body đã đọc"""
        async def receive():
            return {"type": "http.request", "body": body}
        return receive
    
    async def _extract_response_body(self, response: Response) -> Optional[str]:
        """Extract response body cho logging"""
        try:
            # Chỉ log text responses
            content_type = response.headers.get("content-type", "")
            if not any(ct in content_type.lower() for ct in ["json", "text", "xml"]):
                return None
            
            if hasattr(response, 'body'):
                body = response.body
                if isinstance(body, bytes):
                    return body.decode('utf-8', errors='ignore')
                return str(body)
        except Exception:
            pass
        
        return None
    
    def _update_stats(self, path: str, method: str, response_time: float, is_error: bool):
        """Cập nhật thống kê endpoint"""
        key = f"{method}:{path}"
        stats = self.stats[key]
        
        stats["total_requests"] += 1
        if is_error:
            stats["error_count"] += 1
        
        # Update average response time
        current_avg = stats["avg_response_time"]
        total_requests = stats["total_requests"]
        stats["avg_response_time"] = (current_avg * (total_requests - 1) + response_time) / total_requests
        stats["last_called"] = datetime.now().isoformat()
    
    def cleanup_old_logs(self):
        """Cleanup logs cũ theo retention policy"""
        try:
            retention_days = self.config.config.get("retention_days", 30)
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            log_files = [
                "requests.log",
                "errors.log", 
                "performance.log"
            ]
            
            for log_file in log_files:
                log_path = os.path.join(self.logger.log_dir, log_file)
                if os.path.exists(log_path):
                    self._rotate_log_file(log_path, cutoff_date)
            
            logging.info(f"Log cleanup completed. Removed logs older than {retention_days} days")
            
        except Exception as e:
            logging.error(f"Log cleanup failed: {e}")
    
    def _rotate_log_file(self, log_path: str, cutoff_date: datetime):
        """Rotate log file, keeping only recent entries"""
        try:
            # Read all lines
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Filter lines by date
            filtered_lines = []
            for line in lines:
                try:
                    # Extract timestamp from log line
                    timestamp_str = line.split(' | ')[0]
                    log_date = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    
                    if log_date >= cutoff_date:
                        filtered_lines.append(line)
                except:
                    # Keep line if can't parse date
                    filtered_lines.append(line)
            
            # Write back filtered lines
            with open(log_path, 'w', encoding='utf-8') as f:
                f.writelines(filtered_lines)
                
        except Exception as e:
            logging.error(f"Failed to rotate log file {log_path}: {e}")
    
    def get_stats(self) -> Dict:
        """Lấy thống kê endpoint"""
        return dict(self.stats)
    
    def get_config(self) -> Dict:
        """Lấy cấu hình hiện tại"""
        return self.config.config
    
    def update_config(self, new_config: Dict):
        """Cập nhật cấu hình"""
        self.config.config.update(new_config)
        self.config.save_config()

# Global instance
logging_middleware = AdvancedLoggingMiddleware(None)
