"""
Security Management API
API quản lý cấu hình bảo mật
"""

from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
from datetime import datetime, timedelta

from src.middleware.auth_middleware import get_current_user
from src.middleware.security_middleware import IPWhitelistConfig

router = APIRouter(prefix="/api/admin/security", tags=["security"])

class IPWhitelistRequest(BaseModel):
    ips: List[str]

class SecurityStatsResponse(BaseModel):
    total_requests: int
    blocked_requests: int
    rate_limited_requests: int
    top_ips: List[Dict[str, Any]]
    recent_blocks: List[Dict[str, Any]]

@router.get("/ip-whitelist")
async def get_ip_whitelist(current_user: dict = Depends(get_current_user)):
    """Lấy danh sách IP được phép truy cập"""
    config = IPWhitelistConfig()
    whitelist = config.load_whitelist()
    
    return {
        "success": True,
        "whitelist": whitelist,
        "count": len(whitelist)
    }

@router.post("/ip-whitelist")
async def update_ip_whitelist(
    request: IPWhitelistRequest,
    current_user: dict = Depends(get_current_user)
):
    """Cập nhật danh sách IP được phép truy cập"""
    try:
        config = IPWhitelistConfig()
        config.save_whitelist(request.ips)
        
        logging.info(f"IP whitelist updated by {current_user.get('username')}")
        
        return {
            "success": True,
            "message": "IP whitelist updated successfully",
            "count": len(request.ips)
        }
    except Exception as e:
        logging.error(f"Failed to update IP whitelist: {e}")
        raise HTTPException(status_code=500, detail="Failed to update IP whitelist")

@router.get("/stats")
async def get_security_stats(current_user: dict = Depends(get_current_user)):
    """Lấy thống kê bảo mật"""
    # Đây là mock data - trong thực tế sẽ lấy từ database hoặc log files
    stats = {
        "total_requests": 12450,
        "blocked_requests": 45,
        "rate_limited_requests": 23,
        "top_ips": [
            {"ip": "127.0.0.1", "requests": 8932, "blocked": 0},
            {"ip": "192.168.1.100", "requests": 2341, "blocked": 12},
            {"ip": "10.0.0.55", "requests": 1177, "blocked": 33}
        ],
        "recent_blocks": [
            {
                "ip": "203.162.4.101", 
                "reason": "Rate limit exceeded",
                "timestamp": "2025-01-07T10:30:45Z",
                "path": "/api/admin/auth/login"
            },
            {
                "ip": "45.77.161.78",
                "reason": "IP not authorized", 
                "timestamp": "2025-01-07T10:25:12Z",
                "path": "/admin/dashboard"
            }
        ]
    }
    
    return {
        "success": True,
        "stats": stats
    }

@router.get("/rate-limits")
async def get_rate_limits(current_user: dict = Depends(get_current_user)):
    """Lấy cấu hình rate limiting"""
    return {
        "success": True,
        "rate_limits": {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "sensitive_endpoints_per_minute": 10,
            "window_size_seconds": 60
        }
    }

@router.post("/rate-limits")
async def update_rate_limits(
    limits: Dict[str, int],
    current_user: dict = Depends(get_current_user)
):
    """Cập nhật cấu hình rate limiting"""
    # Validate input
    required_fields = ["requests_per_minute", "requests_per_hour", "sensitive_endpoints_per_minute"]
    for field in required_fields:
        if field not in limits or limits[field] <= 0:
            raise HTTPException(status_code=400, detail=f"Invalid {field}")
    
    # Lưu cấu hình (mock - trong thực tế sẽ lưu vào database hoặc config file)
    logging.info(f"Rate limits updated by {current_user.get('username')}: {limits}")
    
    return {
        "success": True,
        "message": "Rate limits updated successfully",
        "new_limits": limits
    }

@router.get("/blocked-paths")
async def get_blocked_paths(current_user: dict = Depends(get_current_user)):
    """Lấy danh sách đường dẫn bị chặn"""
    blocked_paths = [
        "/.env",
        "/config.py", 
        "/requirements.txt",
        "/.git",
        "/src/",
        "/database/",
        "/logs/",
        "/__pycache__",
        ".pyc",
        ".py"
    ]
    
    return {
        "success": True,
        "blocked_paths": blocked_paths,
        "count": len(blocked_paths)
    }

@router.post("/block-ip")
async def block_ip_temporarily(
    ip: str,
    duration_minutes: int = 60,
    current_user: dict = Depends(get_current_user)
):
    """Chặn IP tạm thời"""
    if duration_minutes <= 0 or duration_minutes > 1440:  # Max 24 hours
        raise HTTPException(status_code=400, detail="Duration must be between 1-1440 minutes")
    
    # Thêm IP vào blacklist tạm thời (mock implementation)
    logging.warning(f"IP {ip} blocked temporarily for {duration_minutes} minutes by {current_user.get('username')}")
    
    return {
        "success": True,
        "message": f"IP {ip} blocked for {duration_minutes} minutes",
        "expires_at": (datetime.now() + timedelta(minutes=duration_minutes)).isoformat()
    }

@router.delete("/unblock-ip/{ip}")
async def unblock_ip(
    ip: str,
    current_user: dict = Depends(get_current_user)
):
    """Bỏ chặn IP"""
    # Xóa IP khỏi blacklist (mock implementation)
    logging.info(f"IP {ip} unblocked by {current_user.get('username')}")
    
    return {
        "success": True,
        "message": f"IP {ip} has been unblocked"
    }

@router.get("/cors-settings")
async def get_cors_settings(current_user: dict = Depends(get_current_user)):
    """Lấy cấu hình CORS"""
    return {
        "success": True,
        "cors_settings": {
            "allowed_origins": [
                "http://localhost:8000",
                "http://127.0.0.1:8000", 
                "http://localhost:3000"
            ],
            "allowed_methods": ["GET", "POST", "PUT", "DELETE"],
            "allowed_headers": [
                "Authorization",
                "Content-Type",
                "X-Requested-With",
                "Accept",
                "Origin"
            ],
            "allow_credentials": True
        }
    }
