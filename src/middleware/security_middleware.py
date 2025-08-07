"""
Security Middleware
Middleware bảo mật với CORS, IP whitelist, Rate limiting
Tích hợp với Rate Limiting API
"""

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
import logging
import time
import json
import os
from typing import Dict, List, Optional
from collections import defaultdict, deque
import ipaddress

class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        
        # Rate limiting config files (từ rate limiting API)
        self.rate_limit_config_file = 'data/rate_limit_config.json'
        self.ip_lists_file = 'data/ip_lists.json'
        self.blocked_requests_file = 'data/blocked_requests.json'
        
        # Default IP Whitelist - cho phép local và các IP tin cậy
        self.default_allowed_ips = [
            "127.0.0.1",
            "::1",
            "localhost",
            "192.168.0.0/16",  # Local network
            "10.0.0.0/8",      # Private network
            "172.16.0.0/12"    # Private network
        ]
        
        # Default Rate limiting - số request tối đa mỗi IP trong time window
        self.default_rate_limit = {
            "enabled": True,
            "requests_per_minute": 60,
            "burst_limit": 100,
            "window_size": 60,  # seconds
            "block_duration": 15,  # minutes
            "strategy": "sliding-window"
        }
        
        # Load cấu hình từ rate limiting API
        self.rate_limit_config = self._load_rate_limit_config()
        self.allowed_ips = self._load_ip_whitelist()
        self.blacklist_ips = self._load_ip_blacklist()
        self.endpoint_limits = self._load_endpoint_limits()
        
        # Lưu trữ request history
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.blocked_ips: Dict[str, float] = {}  # ip -> blocked_until_timestamp
        
        # Blocked paths - chặn truy cập các đường dẫn nguy hiểm
        self.blocked_paths = [
            "/.env",
            "/config.py",
            "/requirements.txt",
            "/.git",
            "/src/",
            "/database/",
            "/logs/",
            "/__pycache__",
            ".pyc",
            ".py",
            "/admin/database",
            "/admin/logs",
            "/admin/config"
        ]
        
        # Sensitive endpoints cần rate limit nghiêm ngặt hơn
        self.sensitive_endpoints = [
            "/api/admin/auth/login",
            "/api/admin/auth/change-password",
            "/api/admin/settings"
        ]
        
        # Cache thời gian để reload config
        self.last_config_reload = time.time()
        self.config_reload_interval = 300  # 5 minutes

    def _load_json_file(self, file_path: str, default_data: dict = None):
        """Load data từ JSON file"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load {file_path}: {e}")
        return default_data or {}

    def _load_rate_limit_config(self):
        """Load global rate limit configuration"""
        config = self._load_json_file(self.rate_limit_config_file, {})
        return config.get('global', self.default_rate_limit)

    def _load_ip_whitelist(self):
        """Load IP whitelist từ rate limiting API"""
        ip_lists = self._load_json_file(self.ip_lists_file, {})
        whitelist = ip_lists.get('whitelist', [])
        
        # Combine với default IPs
        combined_ips = self.default_allowed_ips.copy()
        for ip_entry in whitelist:
            if isinstance(ip_entry, dict):
                combined_ips.append(ip_entry.get('ip', ''))
            else:
                combined_ips.append(str(ip_entry))
        
        return [ip for ip in combined_ips if ip]

    def _load_ip_blacklist(self):
        """Load IP blacklist từ rate limiting API"""
        ip_lists = self._load_json_file(self.ip_lists_file, {})
        blacklist = ip_lists.get('blacklist', [])
        
        blacklist_ips = []
        for ip_entry in blacklist:
            if isinstance(ip_entry, dict):
                blacklist_ips.append(ip_entry.get('ip', ''))
            else:
                blacklist_ips.append(str(ip_entry))
        
        return [ip for ip in blacklist_ips if ip]

    def _load_endpoint_limits(self):
        """Load endpoint-specific rate limits"""
        config = self._load_json_file(self.rate_limit_config_file, {})
        endpoints = config.get('endpoints', [])
        
        # Convert to dict for easier lookup
        endpoint_dict = {}
        for ep in endpoints:
            if isinstance(ep, dict) and ep.get('enabled', True):
                key = f"{ep.get('method', '*')}:{ep.get('path', '')}"
                endpoint_dict[key] = {
                    'requests_per_minute': ep.get('requests_per_minute', 60),
                    'burst_limit': ep.get('burst_limit', 100)
                }
        
        return endpoint_dict

    def _reload_config_if_needed(self):
        """Reload configuration nếu cần thiết"""
        current_time = time.time()
        if current_time - self.last_config_reload > self.config_reload_interval:
            try:
                self.rate_limit_config = self._load_rate_limit_config()
                self.allowed_ips = self._load_ip_whitelist()
                self.blacklist_ips = self._load_ip_blacklist()
                self.endpoint_limits = self._load_endpoint_limits()
                self.last_config_reload = current_time
                logging.info("Rate limiting configuration reloaded")
            except Exception as e:
                logging.error(f"Failed to reload rate limiting config: {e}")

    def _is_ip_blacklisted(self, ip: str) -> bool:
        """Kiểm tra IP có trong blacklist không"""
        try:
            client_ip = ipaddress.ip_address(ip)
            
            for blacklisted in self.blacklist_ips:
                if "/" in blacklisted:
                    # CIDR network
                    if client_ip in ipaddress.ip_network(blacklisted, strict=False):
                        return True
                else:
                    # Single IP
                    if str(client_ip) == blacklisted:
                        return True
            
            return False
        except ValueError:
            return False

    def _log_blocked_request(self, ip: str, endpoint: str, reason: str):
        """Log blocked request vào file"""
        try:
            blocked_requests = self._load_json_file(self.blocked_requests_file, [])
            
            blocked_entry = {
                "id": len(blocked_requests) + 1,
                "ip": ip,
                "endpoint": endpoint,
                "reason": reason,
                "blocked_at": time.time(),
                "blocked_until": time.time() + (self.rate_limit_config.get('block_duration', 15) * 60),
                "request_count": len(self.request_history.get(ip, [])),
                "user_agent": "N/A"
            }
            
            blocked_requests.append(blocked_entry)
            
            # Save to file
            with open(self.blocked_requests_file, 'w', encoding='utf-8') as f:
                json.dump(blocked_requests, f, ensure_ascii=False, indent=2, default=str)
                
        except Exception as e:
            logging.error(f"Failed to log blocked request: {e}")

    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        request_time = time.time()
        
        # Reload config nếu cần
        self._reload_config_if_needed()
        
        # Skip rate limiting cho admin rate limit API để tránh deadlock
        if request.url.path.startswith("/api/admin/rate-limit"):
            response = await call_next(request)
            self._add_security_headers(response)
            return response
        
        # 1. Check if rate limiting is enabled
        if not self.rate_limit_config.get('enabled', True):
            response = await call_next(request)
            self._add_security_headers(response)
            return response
        
        # 2. Check IP blacklist
        if self._is_ip_blacklisted(client_ip):
            logging.warning(f"Blocked request from blacklisted IP: {client_ip}")
            self._log_blocked_request(client_ip, request.url.path, "IP blacklisted")
            return self._security_response("Access denied: IP blacklisted")
        
        # 3. Check if IP is temporarily blocked
        if client_ip in self.blocked_ips:
            if time.time() < self.blocked_ips[client_ip]:
                logging.warning(f"Blocked request from temporarily blocked IP: {client_ip}")
                return self._security_response("Access denied: IP temporarily blocked")
            else:
                # Unblock IP
                del self.blocked_ips[client_ip]
        
        # 4. Check IP whitelist
        if not self._is_ip_allowed(client_ip):
            logging.warning(f"Blocked request from unauthorized IP: {client_ip}")
            self._log_blocked_request(client_ip, request.url.path, "IP not whitelisted")
            return self._security_response("Access denied: IP not authorized")
        
        # 5. Check blocked paths
        if self._is_path_blocked(request.url.path):
            logging.warning(f"Blocked access to sensitive path: {request.url.path} from {client_ip}")
            self._log_blocked_request(client_ip, request.url.path, "Blocked path access")
            return self._security_response("Access denied: Path not allowed")
        
        # 6. Rate limiting
        if not self._check_rate_limit(client_ip, request.url.path, request.method, request_time):
            logging.warning(f"Rate limit exceeded for IP: {client_ip}")
            # Block IP temporarily
            block_duration = self.rate_limit_config.get('block_duration', 15) * 60
            self.blocked_ips[client_ip] = time.time() + block_duration
            self._log_blocked_request(client_ip, request.url.path, "Rate limit exceeded")
            return self._security_response("Rate limit exceeded")
        
        # 7. Log request
        self._log_request(client_ip, request.url.path, request.method)
        
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(response)
        
        return response

    def _add_security_headers(self, response):
        """Thêm security headers"""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    def _get_client_ip(self, request: Request) -> str:
        """Lấy IP thực của client"""
        # Check X-Forwarded-For header (for proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"

    def _is_ip_allowed(self, ip: str) -> bool:
        """Kiểm tra IP có được phép truy cập không"""
        if ip == "unknown":
            return False
            
        try:
            client_ip = ipaddress.ip_address(ip)
            
            for allowed in self.allowed_ips:
                if "/" in allowed:
                    # CIDR network
                    if client_ip in ipaddress.ip_network(allowed, strict=False):
                        return True
                else:
                    # Single IP
                    if str(client_ip) == allowed or allowed == "localhost":
                        return True
            
            return False
        except ValueError:
            # Invalid IP format
            return False

    def _is_path_blocked(self, path: str) -> bool:
        """Kiểm tra đường dẫn có bị chặn không"""
        path_lower = path.lower()
        
        for blocked in self.blocked_paths:
            if blocked in path_lower:
                return True
        
        return False

    def _check_rate_limit(self, ip: str, path: str, method: str, current_time: float) -> bool:
        """Kiểm tra rate limit với cấu hình từ API"""
        history = self.request_history[ip]
        
        # Remove old requests outside time window
        window_size = self.rate_limit_config.get("window_size", 60)
        cutoff_time = current_time - window_size
        while history and history[0] < cutoff_time:
            history.popleft()
        
        # Check current rate
        requests_in_window = len(history)
        
        # Get endpoint-specific limit
        endpoint_key = f"{method}:{path}"
        endpoint_limit = self.endpoint_limits.get(endpoint_key)
        
        # Fallback to wildcard method
        if not endpoint_limit:
            endpoint_key = f"*:{path}"
            endpoint_limit = self.endpoint_limits.get(endpoint_key)
        
        # Determine rate limit
        if endpoint_limit:
            max_requests = endpoint_limit['requests_per_minute']
            burst_limit = endpoint_limit.get('burst_limit', max_requests)
        elif any(sensitive in path for sensitive in self.sensitive_endpoints):
            max_requests = 10  # 10 requests per minute for sensitive endpoints
            burst_limit = 15
        else:
            max_requests = self.rate_limit_config.get("requests_per_minute", 60)
            burst_limit = self.rate_limit_config.get("burst_limit", 100)
        
        # Check burst limit first
        if requests_in_window >= burst_limit:
            return False
        
        # Check regular rate limit
        if requests_in_window >= max_requests:
            return False
        
        # Add current request
        history.append(current_time)
        return True

    def _log_request(self, ip: str, path: str, method: str):
        """Log request để theo dõi"""
        logging.info(f"Request: {method} {path} from {ip}")

    def _security_response(self, message: str):
        """Trả về response bảo mật"""
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": "Access Denied",
                "message": message
            }
        )

def setup_cors(app):
    """Setup CORS middleware"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:3000",  # For development frontend
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Requested-With",
            "Accept",
            "Origin",
            "Access-Control-Request-Method",
            "Access-Control-Request-Headers",
        ],
    )

class IPWhitelistConfig:
    """Cấu hình IP whitelist có thể thay đổi"""
    
    def __init__(self):
        self.whitelist_file = "config/allowed_ips.txt"
        self.default_ips = [
            "127.0.0.1",
            "::1", 
            "192.168.0.0/16"
        ]
    
    def load_whitelist(self) -> List[str]:
        """Load IP whitelist từ file"""
        try:
            with open(self.whitelist_file, 'r') as f:
                ips = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                return ips if ips else self.default_ips
        except FileNotFoundError:
            return self.default_ips
    
    def save_whitelist(self, ips: List[str]):
        """Lưu IP whitelist vào file"""
        import os
        os.makedirs(os.path.dirname(self.whitelist_file), exist_ok=True)
        
        with open(self.whitelist_file, 'w') as f:
            f.write("# IP Whitelist - One IP/CIDR per line\n")
            f.write("# Lines starting with # are comments\n\n")
            for ip in ips:
                f.write(f"{ip}\n")
