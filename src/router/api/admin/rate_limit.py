"""
Rate Limiting Management API
API quản lý rate limiting cho hệ thống
"""

from fastapi import APIRouter, Request, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
import ipaddress

from src.middleware.auth_middleware import get_current_user

# Data storage files
RATE_LIMIT_CONFIG_FILE = 'data/rate_limit_config.json'
BLOCKED_REQUESTS_FILE = 'data/blocked_requests.json'
IP_LISTS_FILE = 'data/ip_lists.json'

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

router = APIRouter(prefix="/api/admin/rate-limit", tags=["rate-limiting"])

# Pydantic models
class RateLimitConfig(BaseModel):
    enabled: bool = True
    requests_per_minute: int = 60
    burst_limit: int = 100
    window_size: int = 60  # seconds
    block_duration: int = 15  # minutes
    strategy: str = "sliding-window"  # fixed-window, sliding-window, token-bucket

class EndpointRateLimit(BaseModel):
    id: Optional[int] = None
    path: str
    method: str = "*"
    enabled: bool = True
    requests_per_minute: int = 60
    burst_limit: int = 100
    description: Optional[str] = None

class IPWhitelistEntry(BaseModel):
    id: Optional[int] = None
    ip: str
    description: Optional[str] = None
    added_at: Optional[datetime] = None

class UpdateStats(BaseModel):
    status: str
    timestamp: datetime

# Utility functions for JSON storage
def load_json_data(file_path: str, default_data=None):
    """Load data from JSON file"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return default_data or {}
    except Exception:
        return default_data or {}

def save_json_data(file_path: str, data):
    """Save data to JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        return True
    except Exception:
        return False

def get_current_config():
    """Get current rate limit configuration"""
    return load_json_data(RATE_LIMIT_CONFIG_FILE, {
        "global": {
            "enabled": True,
            "requests_per_minute": 60,
            "burst_limit": 100,
            "window_size": 60,
            "block_duration": 15,
            "strategy": "sliding-window"
        },
        "endpoints": [
            {
                "id": 1,
                "path": "/api/v1/vietnam-citizens-card-detection",
                "method": "POST",
                "enabled": True,
                "requests_per_minute": 30,
                "burst_limit": 50,
                "description": "Card detection endpoint"
            },
            {
                "id": 2,
                "path": "/api/v1/vietnam-citizens-card-mrz-extraction",
                "method": "POST",
                "enabled": True,
                "requests_per_minute": 20,
                "burst_limit": 40,
                "description": "MRZ extraction endpoint"
            }
        ],
        "ip_whitelist": [],
        "ip_blacklist": []
    })

def get_blocked_requests():
    """Get blocked requests data"""
    blocked_data = load_json_data(BLOCKED_REQUESTS_FILE, [])
    
    # Generate sample blocked requests if empty
    if not blocked_data:
        now = datetime.now()
        blocked_data = [
            {
                "id": 1,
                "ip": "192.168.1.100",
                "endpoint": "/api/v1/vietnam-citizens-card-detection",
                "reason": "Rate limit exceeded",
                "blocked_at": (now - timedelta(minutes=5)).isoformat(),
                "blocked_until": (now + timedelta(minutes=10)).isoformat(),
                "request_count": 150,
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            {
                "id": 2,
                "ip": "203.113.45.22",
                "endpoint": "/api/v1/vietnam-citizens-card-mrz-extraction",
                "reason": "Suspicious activity",
                "blocked_at": (now - timedelta(minutes=15)).isoformat(),
                "blocked_until": (now + timedelta(minutes=5)).isoformat(),
                "request_count": 200,
                "user_agent": "python-requests/2.28.1"
            }
        ]
        save_json_data(BLOCKED_REQUESTS_FILE, blocked_data)
    
    return blocked_data

def get_ip_lists():
    """Get IP whitelist and blacklist"""
    return load_json_data(IP_LISTS_FILE, {
        "whitelist": [
            {
                "id": 1,
                "ip": "192.168.1.0/24",
                "description": "Internal network",
                "added_at": datetime.now().isoformat()
            },
            {
                "id": 2,
                "ip": "203.113.45.0/24",
                "description": "Partner network",
                "added_at": (datetime.now() - timedelta(days=1)).isoformat()
            }
        ],
        "blacklist": [
            {
                "id": 1,
                "ip": "123.45.67.89",
                "description": "Malicious IP",
                "added_at": datetime.now().isoformat()
            }
        ]
    })

# Initialize data files on module load
def init_data_files():
    """Initialize JSON data files with default values"""
    if not os.path.exists(RATE_LIMIT_CONFIG_FILE):
        save_json_data(RATE_LIMIT_CONFIG_FILE, get_current_config())
    if not os.path.exists(BLOCKED_REQUESTS_FILE):
        save_json_data(BLOCKED_REQUESTS_FILE, [])
    if not os.path.exists(IP_LISTS_FILE):
        save_json_data(IP_LISTS_FILE, {"whitelist": [], "blacklist": []})

init_data_files()

# === Configuration Endpoints ===

@router.get("/config")
async def get_rate_limit_config(current_user: dict = Depends(get_current_user)):
    """Lấy cấu hình rate limiting toàn cục"""
    try:
        config = get_current_config()
        return {
            "success": True,
            "config": config["global"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")

@router.post("/config")
async def update_rate_limit_config(
    config: RateLimitConfig,
    current_user: dict = Depends(get_current_user)
):
    """Cập nhật cấu hình rate limiting toàn cục"""
    try:
        current_config = get_current_config()
        current_config["global"] = config.dict()
        save_json_data(RATE_LIMIT_CONFIG_FILE, current_config)
        
        return {
            "success": True,
            "message": "Rate limit configuration updated successfully",
            "config": config.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")

# === Endpoint Management ===

@router.get("/endpoints")
async def get_endpoint_limits(current_user: dict = Depends(get_current_user)):
    """Lấy danh sách rate limit cho từng endpoint"""
    try:
        config = get_current_config()
        return {
            "success": True,
            "endpoints": config.get("endpoints", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get endpoints: {str(e)}")

@router.post("/endpoints")
async def add_endpoint_limit(
    endpoint: EndpointRateLimit,
    current_user: dict = Depends(get_current_user)
):
    """Thêm rate limit cho endpoint mới"""
    try:
        config = get_current_config()
        endpoints = config.get("endpoints", [])
        
        # Generate new ID
        new_id = max([ep.get("id", 0) for ep in endpoints], default=0) + 1
        endpoint_data = endpoint.dict()
        endpoint_data["id"] = new_id
        endpoint_data["created_at"] = datetime.now().isoformat()
        
        endpoints.append(endpoint_data)
        config["endpoints"] = endpoints
        save_json_data(RATE_LIMIT_CONFIG_FILE, config)
        
        return {
            "success": True,
            "message": "Endpoint rate limit added successfully",
            "endpoint": endpoint_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add endpoint: {str(e)}")

@router.put("/endpoints/{endpoint_id}")
async def update_endpoint_limit(
    endpoint_id: int,
    endpoint: EndpointRateLimit,
    current_user: dict = Depends(get_current_user)
):
    """Cập nhật rate limit cho endpoint"""
    try:
        config = get_current_config()
        endpoints = config.get("endpoints", [])
        
        for i, ep in enumerate(endpoints):
            if ep.get("id") == endpoint_id:
                endpoint_data = endpoint.dict()
                endpoint_data["id"] = endpoint_id
                endpoint_data["updated_at"] = datetime.now().isoformat()
                endpoints[i] = endpoint_data
                break
        else:
            raise HTTPException(status_code=404, detail="Endpoint not found")
        
        config["endpoints"] = endpoints
        save_json_data(RATE_LIMIT_CONFIG_FILE, config)
        
        return {
            "success": True,
            "message": "Endpoint rate limit updated successfully",
            "endpoint": endpoint_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update endpoint: {str(e)}")

@router.delete("/endpoints/{endpoint_id}")
async def delete_endpoint_limit(
    endpoint_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Xóa rate limit cho endpoint"""
    try:
        config = get_current_config()
        endpoints = config.get("endpoints", [])
        
        initial_length = len(endpoints)
        endpoints = [ep for ep in endpoints if ep.get("id") != endpoint_id]
        
        if len(endpoints) == initial_length:
            raise HTTPException(status_code=404, detail="Endpoint not found")
        
        config["endpoints"] = endpoints
        save_json_data(RATE_LIMIT_CONFIG_FILE, config)
        
        return {
            "success": True,
            "message": "Endpoint rate limit deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete endpoint: {str(e)}")

# === IP Management ===

@router.get("/ip-lists")
async def get_ip_lists_endpoint(current_user: dict = Depends(get_current_user)):
    """Lấy danh sách IP whitelist và blacklist"""
    try:
        ip_lists = get_ip_lists()
        return {
            "success": True,
            "whitelist": ip_lists.get("whitelist", []),
            "blacklist": ip_lists.get("blacklist", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get IP lists: {str(e)}")

@router.post("/ip-lists/whitelist")
async def add_ip_to_whitelist(
    ip_entry: IPWhitelistEntry,
    current_user: dict = Depends(get_current_user)
):
    """Thêm IP vào whitelist"""
    try:
        # Validate IP address
        try:
            ipaddress.ip_network(ip_entry.ip, strict=False)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid IP address or CIDR")
        
        ip_lists = get_ip_lists()
        whitelist = ip_lists.get("whitelist", [])
        
        # Generate new ID
        new_id = max([ip.get("id", 0) for ip in whitelist], default=0) + 1
        ip_data = ip_entry.dict()
        ip_data["id"] = new_id
        ip_data["added_at"] = datetime.now().isoformat()
        
        whitelist.append(ip_data)
        ip_lists["whitelist"] = whitelist
        save_json_data(IP_LISTS_FILE, ip_lists)
        
        return {
            "success": True,
            "message": "IP added to whitelist successfully",
            "ip": ip_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add IP to whitelist: {str(e)}")

@router.post("/ip-lists/blacklist")
async def add_ip_to_blacklist(
    ip_entry: IPWhitelistEntry,
    current_user: dict = Depends(get_current_user)
):
    """Thêm IP vào blacklist"""
    try:
        # Validate IP address
        try:
            ipaddress.ip_network(ip_entry.ip, strict=False)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid IP address or CIDR")
        
        ip_lists = get_ip_lists()
        blacklist = ip_lists.get("blacklist", [])
        
        # Generate new ID
        new_id = max([ip.get("id", 0) for ip in blacklist], default=0) + 1
        ip_data = ip_entry.dict()
        ip_data["id"] = new_id
        ip_data["added_at"] = datetime.now().isoformat()
        
        blacklist.append(ip_data)
        ip_lists["blacklist"] = blacklist
        save_json_data(IP_LISTS_FILE, ip_lists)
        
        return {
            "success": True,
            "message": "IP added to blacklist successfully",
            "ip": ip_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add IP to blacklist: {str(e)}")

@router.delete("/ip-lists/whitelist/{ip_id}")
async def remove_ip_from_whitelist(
    ip_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Xóa IP khỏi whitelist"""
    try:
        ip_lists = get_ip_lists()
        whitelist = ip_lists.get("whitelist", [])
        
        initial_length = len(whitelist)
        whitelist = [ip for ip in whitelist if ip.get("id") != ip_id]
        
        if len(whitelist) == initial_length:
            raise HTTPException(status_code=404, detail="IP not found in whitelist")
        
        ip_lists["whitelist"] = whitelist
        save_json_data(IP_LISTS_FILE, ip_lists)
        
        return {
            "success": True,
            "message": "IP removed from whitelist successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove IP from whitelist: {str(e)}")

@router.delete("/ip-lists/blacklist/{ip_id}")
async def remove_ip_from_blacklist(
    ip_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Xóa IP khỏi blacklist"""
    try:
        ip_lists = get_ip_lists()
        blacklist = ip_lists.get("blacklist", [])
        
        initial_length = len(blacklist)
        blacklist = [ip for ip in blacklist if ip.get("id") != ip_id]
        
        if len(blacklist) == initial_length:
            raise HTTPException(status_code=404, detail="IP not found in blacklist")
        
        ip_lists["blacklist"] = blacklist
        save_json_data(IP_LISTS_FILE, ip_lists)
        
        return {
            "success": True,
            "message": "IP removed from blacklist successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove IP from blacklist: {str(e)}")

# === Blocked IPs Management ===

@router.get("/blocked-ips")
async def get_blocked_ips(current_user: dict = Depends(get_current_user)):
    """Lấy danh sách IP bị block"""
    try:
        blocked_requests = get_blocked_requests()
        now = datetime.now()
        
        # Filter active blocks
        active_blocks = []
        for req in blocked_requests:
            try:
                blocked_until = datetime.fromisoformat(req.get("blocked_until", ""))
                if blocked_until > now:
                    active_blocks.append(req)
            except:
                continue
        
        return {
            "success": True,
            "blocked_ips": active_blocks,
            "total": len(active_blocks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get blocked IPs: {str(e)}")

@router.delete("/blocked-ips/{ip}")
async def unblock_ip(
    ip: str,
    current_user: dict = Depends(get_current_user)
):
    """Bỏ block IP"""
    try:
        blocked_requests = get_blocked_requests()
        
        # Remove blocks for this IP
        initial_length = len(blocked_requests)
        blocked_requests = [req for req in blocked_requests if req.get("ip") != ip]
        
        if len(blocked_requests) == initial_length:
            raise HTTPException(status_code=404, detail="IP not found in blocked list")
        
        save_json_data(BLOCKED_REQUESTS_FILE, blocked_requests)
        
        return {
            "success": True,
            "message": f"IP {ip} has been unblocked successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unblock IP: {str(e)}")

# === Statistics and Monitoring ===

@router.get("/stats")
async def get_rate_limit_stats(current_user: dict = Depends(get_current_user)):
    """Lấy thống kê rate limiting"""
    try:
        config = get_current_config()
        blocked_requests = get_blocked_requests()
        ip_lists = get_ip_lists()
        
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Calculate stats
        total_requests_today = 2847  # Sample data
        blocked_requests_today = len([req for req in blocked_requests 
                                    if datetime.fromisoformat(req.get("blocked_at", "")) >= today_start])
        
        active_blocks = len([req for req in blocked_requests 
                           if datetime.fromisoformat(req.get("blocked_until", "")) > now])
        
        return {
            "success": True,
            "stats": {
                "total_requests_today": total_requests_today,
                "blocked_requests_today": blocked_requests_today,
                "active_blocked_ips": active_blocks,
                "total_endpoints": len(config.get("endpoints", [])),
                "whitelist_ips": len(ip_lists.get("whitelist", [])),
                "blacklist_ips": len(ip_lists.get("blacklist", []))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.get("/monitoring")
async def get_monitoring_data(current_user: dict = Depends(get_current_user)):
    """Lấy dữ liệu monitoring real-time"""
    try:
        # Generate sample monitoring data
        now = datetime.now()
        monitoring_data = []
        
        for i in range(24):
            hour_time = now - timedelta(hours=i)
            monitoring_data.append({
                "timestamp": hour_time.isoformat(),
                "total_requests": 120 - i * 2,
                "blocked_requests": max(0, 15 - i),
                "avg_response_time": 150 + i * 5,
                "active_ips": 45 + i
            })
        
        return {
            "success": True,
            "monitoring_data": list(reversed(monitoring_data))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring data: {str(e)}")

@router.get("/real-time")
async def get_real_time_data(current_user: dict = Depends(get_current_user)):
    """Lấy dữ liệu real-time cho dashboard"""
    try:
        now = datetime.now()
        
        # Generate real-time sample data
        real_time_data = {
            "current_requests": 45,
            "requests_per_second": 12.5,
            "blocked_ips_count": 3,
            "top_ips": [
                {"ip": "192.168.1.100", "requests": 150, "status": "blocked"},
                {"ip": "203.113.45.22", "requests": 89, "status": "warning"},
                {"ip": "10.0.0.50", "requests": 67, "status": "normal"},
                {"ip": "172.16.0.25", "requests": 45, "status": "normal"}
            ],
            "endpoint_stats": [
                {
                    "endpoint": "/api/v1/vietnam-citizens-card-detection",
                    "requests": 245,
                    "blocked": 12,
                    "avg_response_time": 250
                },
                {
                    "endpoint": "/api/v1/vietnam-citizens-card-mrz-extraction", 
                    "requests": 189,
                    "blocked": 8,
                    "avg_response_time": 180
                }
            ],
            "last_updated": now.isoformat()
        }
        
        return {
            "success": True,
            "data": real_time_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get real-time data: {str(e)}")
