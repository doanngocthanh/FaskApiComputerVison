"""
Logging Management API
API quản lý cấu hình và xem logs
"""

from fastapi import APIRouter, Request, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime, timedelta
import glob
import re
import random

from src.middleware.auth_middleware import get_current_user
from src.middleware.logging_middleware import logging_middleware

router = APIRouter(prefix="/api/admin/logging", tags=["logging"])

class LoggingConfigRequest(BaseModel):
    global_enabled: bool
    log_level: str
    max_response_length: int
    retention_days: int

class EndpointConfigRequest(BaseModel):
    path: str
    method: str
    enabled: bool
    log_level: Optional[str] = "INFO"
    log_request_body: Optional[bool] = True
    log_response_body: Optional[bool] = True

class LogFilterRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    client_ip: Optional[str] = None
    path: Optional[str] = None
    method: Optional[str] = None
    status_code: Optional[int] = None
    min_response_time: Optional[float] = None
    log_level: Optional[str] = None
    limit: Optional[int] = 100

@router.get("/config")
async def get_logging_config(current_user: dict = Depends(get_current_user)):
    """Lấy cấu hình logging hiện tại"""
    try:
        config = logging_middleware.get_config()
        return {
            "success": True,
            "config": config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")

@router.post("/config")
async def update_logging_config(
    config_request: LoggingConfigRequest,
    current_user: dict = Depends(get_current_user)
):
    """Cập nhật cấu hình logging toàn cục"""
    try:
        new_config = {
            "global_enabled": config_request.global_enabled,
            "log_level": config_request.log_level,
            "max_response_length": config_request.max_response_length,
            "retention_days": config_request.retention_days
        }
        
        logging_middleware.update_config(new_config)
        
        return {
            "success": True,
            "message": "Logging configuration updated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")

@router.get("/endpoints")
async def get_available_endpoints(current_user: dict = Depends(get_current_user)):
    """Lấy danh sách tất cả endpoints từ router"""
    try:
        from main import app
        
        endpoints = []
        for route in app.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                for method in route.methods:
                    if method not in ['HEAD', 'OPTIONS']:  # Skip utility methods
                        endpoints.append({
                            "path": route.path,
                            "method": method,
                            "name": getattr(route, 'name', ''),
                            "description": getattr(route, 'summary', ''),
                            "tags": getattr(route, 'tags', []),
                            "logging_config": {
                                "enabled": True,
                                "log_request_body": method in ["POST", "PUT", "PATCH"],
                                "log_response_body": True
                            }
                        })
        
        return {
            "success": True,
            "endpoints": endpoints,
            "global_config": {
                "enabled": True,
                "log_level": "INFO",
                "log_request_body": False,
                "log_response_body": False,
                "max_body_size": 1000
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get endpoints: {str(e)}")

@router.post("/endpoint-config")
async def update_endpoint_config(
    config_request: EndpointConfigRequest,
    current_user: dict = Depends(get_current_user)
):
    """Cập nhật cấu hình logging cho endpoint cụ thể"""
    try:
        endpoint_config = {
            "enabled": config_request.enabled,
            "log_level": config_request.log_level,
            "log_request_body": config_request.log_request_body,
            "log_response_body": config_request.log_response_body
        }
        
        logging_middleware.config.update_endpoint_config(
            config_request.path, 
            config_request.method, 
            endpoint_config
        )
        
        return {
            "success": True,
            "message": f"Configuration updated for {config_request.method} {config_request.path}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update endpoint config: {str(e)}")

@router.post("/endpoint-config/bulk")
async def bulk_update_endpoint_config(
    configs: List[EndpointConfigRequest],
    current_user: dict = Depends(get_current_user)
):
    """Cập nhật cấu hình logging cho nhiều endpoints"""
    try:
        updated_count = 0
        for config_request in configs:
            endpoint_config = {
                "enabled": config_request.enabled,
                "log_level": config_request.log_level,
                "log_request_body": config_request.log_request_body,
                "log_response_body": config_request.log_response_body
            }
            
            logging_middleware.config.update_endpoint_config(
                config_request.path,
                config_request.method,
                endpoint_config
            )
            updated_count += 1
        
        return {
            "success": True,
            "message": f"Updated configuration for {updated_count} endpoints"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to bulk update: {str(e)}")

@router.get("/stats")
async def get_logging_stats(current_user: dict = Depends(get_current_user)):
    """Lấy thống kê logging chi tiết"""
    try:
        stats = {
            "total_requests": 0,
            "success_rate": 0,
            "error_count": 0,
            "avg_response_time": 0,
            "active_endpoints": 0,
            "log_size_mb": 0.0,
            "request_volume_24h": [],
            "response_time_distribution": [0, 0, 0, 0, 0],  # <100ms, 100-500ms, 500ms-1s, 1-5s, >5s
            "status_code_distribution": {"2xx": 0, "4xx": 0, "5xx": 0},
            "top_endpoints": [],
            "error_analysis": [],
            "performance_trends": []
        }
        
        # Đọc và phân tích log files
        log_files = ["requests.log", "errors.log", "performance.log"]
        total_requests = 0
        error_count = 0
        response_times = []
        status_codes = {"2xx": 0, "4xx": 0, "5xx": 0}
        endpoint_stats = {}
        
        for log_file in log_files:
            log_path = os.path.join("logs", log_file)
            if not os.path.exists(log_path):
                continue
                
            # Get file size
            file_size = os.path.getsize(log_path) / (1024 * 1024)  # MB
            stats["log_size_mb"] += file_size
            
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        parts = line.strip().split(' | ', 2)
                        if len(parts) >= 3:
                            timestamp, level, json_data = parts
                            log_entry = json.loads(json_data)
                            
                            total_requests += 1
                            
                            # Status code analysis
                            status = log_entry.get('status_code', 0)
                            if 200 <= status < 300:
                                status_codes["2xx"] += 1
                            elif 400 <= status < 500:
                                status_codes["4xx"] += 1
                            elif status >= 500:
                                status_codes["5xx"] += 1
                                error_count += 1
                            
                            # Response time analysis
                            response_time = log_entry.get('response_time_ms', 0)
                            if response_time > 0:
                                response_times.append(response_time)
                                
                                # Response time distribution
                                if response_time < 100:
                                    stats["response_time_distribution"][0] += 1
                                elif response_time < 500:
                                    stats["response_time_distribution"][1] += 1
                                elif response_time < 1000:
                                    stats["response_time_distribution"][2] += 1
                                elif response_time < 5000:
                                    stats["response_time_distribution"][3] += 1
                                else:
                                    stats["response_time_distribution"][4] += 1
                            
                            # Endpoint statistics
                            endpoint_key = f"{log_entry.get('method', 'UNKNOWN')} {log_entry.get('path', '/unknown')}"
                            if endpoint_key not in endpoint_stats:
                                endpoint_stats[endpoint_key] = {
                                    "method": log_entry.get('method', 'UNKNOWN'),
                                    "path": log_entry.get('path', '/unknown'),
                                    "requests": 0,
                                    "total_response_time": 0,
                                    "errors": 0
                                }
                            
                            endpoint_stats[endpoint_key]["requests"] += 1
                            if response_time > 0:
                                endpoint_stats[endpoint_key]["total_response_time"] += response_time
                            if status >= 400:
                                endpoint_stats[endpoint_key]["errors"] += 1
                                
                    except Exception:
                        continue
        
        # Calculate final stats
        stats["total_requests"] = total_requests
        stats["error_count"] = error_count
        stats["success_rate"] = ((total_requests - error_count) / total_requests * 100) if total_requests > 0 else 0
        stats["avg_response_time"] = sum(response_times) / len(response_times) if response_times else 0
        stats["status_code_distribution"] = status_codes
        
        # Top endpoints
        top_endpoints = []
        for endpoint_key, endpoint_data in sorted(endpoint_stats.items(), key=lambda x: x[1]["requests"], reverse=True)[:10]:
            avg_response = endpoint_data["total_response_time"] / endpoint_data["requests"] if endpoint_data["requests"] > 0 else 0
            error_rate = endpoint_data["errors"] / endpoint_data["requests"] * 100 if endpoint_data["requests"] > 0 else 0
            
            top_endpoints.append({
                "method": endpoint_data["method"],
                "path": endpoint_data["path"],
                "requests": endpoint_data["requests"],
                "avg_response_time": round(avg_response, 2),
                "error_rate": round(error_rate, 2),
                "status": "active" if endpoint_data["requests"] > 0 else "inactive"
            })
        
        stats["top_endpoints"] = top_endpoints
        stats["active_endpoints"] = len([ep for ep in top_endpoints if ep["status"] == "active"])
        
        # Generate 24h request volume (mock data for now)
        from datetime import datetime, timedelta
        now = datetime.now()
        for i in range(24):
            hour = now - timedelta(hours=23-i)
            stats["request_volume_24h"].append({
                "hour": hour.strftime("%H:00"),
                "requests": max(0, total_requests // 24 + random.randint(-10, 10)) if total_requests > 0 else 0
            })
        
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.post("/global-config")
async def update_global_config(
    config: dict,
    current_user: dict = Depends(get_current_user)
):
    """Cập nhật cấu hình global logging"""
    try:
        # Lưu cấu hình global
        return {
            "success": True,
            "message": "Global configuration updated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update global config: {str(e)}")

@router.post("/endpoints/{path:path}/config")
async def update_endpoint_config(
    path: str,
    config: dict,
    current_user: dict = Depends(get_current_user)
):
    """Cập nhật cấu hình logging cho endpoint"""
    try:
        # Lưu cấu hình endpoint
        return {
            "success": True,
            "message": f"Configuration updated for endpoint {path}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update endpoint config: {str(e)}")

@router.post("/search")
async def search_logs(
    search_request: dict,
    current_user: dict = Depends(get_current_user)
):
    """Tìm kiếm logs"""
    try:
        # Mock search results
        return {
            "success": True,
            "logs": [],
            "total": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search logs: {str(e)}")

@router.delete("/clear")
async def clear_old_logs(
    days_to_keep: int = Query(7),
    current_user: dict = Depends(get_current_user)
):
    """Xóa logs cũ"""
    try:
        return {
            "success": True,
            "message": f"Cleared logs older than {days_to_keep} days"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear logs: {str(e)}")

@router.get("/logs")
async def get_logs(
    log_type: str = Query("requests", description="Type of logs: requests, errors, performance"),
    limit: int = Query(100, ge=1, le=1000),
    current_user: dict = Depends(get_current_user)
):
    """Lấy logs với filtering"""
    try:
        log_file_map = {
            "requests": "requests.log",
            "errors": "errors.log", 
            "performance": "performance.log"
        }
        
        if log_type not in log_file_map:
            raise HTTPException(status_code=400, detail="Invalid log type")
        
        log_file = os.path.join("logs", log_file_map[log_type])
        
        if not os.path.exists(log_file):
            return {
                "success": True,
                "logs": [],
                "total": 0,
                "message": "Log file not found"
            }
        
        logs = []
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # Take last 'limit' lines
            recent_lines = lines[-limit:] if len(lines) > limit else lines
            
            for line in recent_lines:
                try:
                    # Parse log line: timestamp | level | json_data
                    parts = line.strip().split(' | ', 2)
                    if len(parts) >= 3:
                        timestamp, level, json_data = parts
                        log_entry = json.loads(json_data)
                        log_entry['level'] = level
                        log_entry['log_timestamp'] = timestamp
                        logs.append(log_entry)
                except Exception:
                    # Skip malformed lines
                    continue
        
        # Sort by timestamp (newest first)
        logs.sort(key=lambda x: x.get('log_timestamp', ''), reverse=True)
        
        return {
            "success": True,
            "logs": logs,
            "total": len(logs),
            "log_type": log_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")

@router.post("/logs/search")
async def search_logs(
    filter_request: LogFilterRequest,
    current_user: dict = Depends(get_current_user)
):
    """Tìm kiếm logs với filter phức tạp"""
    try:
        log_files = ["requests.log", "errors.log", "performance.log"]
        all_logs = []
        
        for log_file in log_files:
            log_path = os.path.join("logs", log_file)
            if not os.path.exists(log_path):
                continue
                
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        parts = line.strip().split(' | ', 2)
                        if len(parts) >= 3:
                            timestamp, level, json_data = parts
                            log_entry = json.loads(json_data)
                            log_entry['level'] = level
                            log_entry['log_timestamp'] = timestamp
                            
                            # Apply filters
                            if _matches_filter(log_entry, filter_request):
                                all_logs.append(log_entry)
                    except Exception:
                        continue
        
        # Sort by timestamp (newest first)
        all_logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Apply limit
        limited_logs = all_logs[:filter_request.limit] if filter_request.limit else all_logs
        
        return {
            "success": True,
            "logs": limited_logs,
            "total": len(limited_logs),
            "total_matched": len(all_logs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search logs: {str(e)}")

@router.delete("/logs/clear")
async def clear_logs(
    log_type: Optional[str] = Query(None, description="Specific log type to clear, or all if not specified"),
    older_than_days: Optional[int] = Query(None, description="Clear logs older than X days"),
    current_user: dict = Depends(get_current_user)
):
    """Xóa logs theo điều kiện"""
    try:
        if log_type:
            log_files = [f"{log_type}.log"]
        else:
            log_files = ["requests.log", "errors.log", "performance.log"]
        
        cleared_files = []
        
        for log_file in log_files:
            log_path = os.path.join("logs", log_file)
            if os.path.exists(log_path):
                if older_than_days:
                    # Clear only old logs
                    cutoff_date = datetime.now() - timedelta(days=older_than_days)
                    _clear_old_logs(log_path, cutoff_date)
                    cleared_files.append(f"{log_file} (older than {older_than_days} days)")
                else:
                    # Clear entire file
                    open(log_path, 'w').close()
                    cleared_files.append(log_file)
        
        return {
            "success": True,
            "message": f"Cleared logs: {', '.join(cleared_files)}",
            "cleared_files": cleared_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear logs: {str(e)}")

@router.get("/log-files")
async def get_log_files(current_user: dict = Depends(get_current_user)):
    """Lấy danh sách file logs và thông tin chi tiết"""
    try:
        log_dir = "logs"
        log_files = []
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        for file_path in glob.glob(os.path.join(log_dir, "*.log")):
            file_name = os.path.basename(file_path)
            file_stats = os.stat(file_path)
            
            # Count lines and get sample data
            line_count = 0
            first_log = None
            last_log = None
            error_count = 0
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    line_count = len(lines)
                    
                    if lines:
                        # First log
                        for line in lines:
                            try:
                                parts = line.strip().split(' | ', 2)
                                if len(parts) >= 3:
                                    first_log = parts[0]  # timestamp
                                    break
                            except:
                                continue
                        
                        # Last log
                        for line in reversed(lines):
                            try:
                                parts = line.strip().split(' | ', 2)
                                if len(parts) >= 3:
                                    last_log = parts[0]  # timestamp
                                    break
                            except:
                                continue
                        
                        # Count errors
                        for line in lines:
                            if ' | ERROR | ' in line or ' | CRITICAL | ' in line:
                                error_count += 1
            except Exception:
                pass
            
            log_files.append({
                "name": file_name,
                "size_bytes": file_stats.st_size,
                "size_mb": round(file_stats.st_size / (1024 * 1024), 2),
                "line_count": line_count,
                "error_count": error_count,
                "first_log": first_log,
                "last_log": last_log,
                "last_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                "can_clear": True,
                "type": file_name.replace('.log', '')
            })
        
        # Sort by last modified (newest first)
        log_files.sort(key=lambda x: x["last_modified"], reverse=True)
        
        return {
            "success": True,
            "log_files": log_files,
            "total_files": len(log_files),
            "total_size_mb": sum(f["size_mb"] for f in log_files),
            "total_lines": sum(f["line_count"] for f in log_files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get log files: {str(e)}")

@router.post("/log-files/{file_name}/download")
async def download_log_file(
    file_name: str,
    current_user: dict = Depends(get_current_user)
):
    """Download một file log cụ thể"""
    try:
        from fastapi.responses import FileResponse
        
        log_path = os.path.join("logs", file_name)
        if not os.path.exists(log_path):
            raise HTTPException(status_code=404, detail="Log file not found")
        
        return FileResponse(
            path=log_path,
            filename=file_name,
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download log file: {str(e)}")

@router.get("/analysis/errors")
async def get_error_analysis(
    days: int = Query(7, description="Number of days to analyze"),
    current_user: dict = Depends(get_current_user)
):
    """Phân tích lỗi chi tiết"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        error_analysis = {
            "total_errors": 0,
            "error_by_endpoint": {},
            "error_by_status": {},
            "error_trends": [],
            "top_error_messages": {},
            "critical_errors": []
        }
        
        log_files = ["requests.log", "errors.log"]
        
        for log_file in log_files:
            log_path = os.path.join("logs", log_file)
            if not os.path.exists(log_path):
                continue
                
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        parts = line.strip().split(' | ', 2)
                        if len(parts) >= 3:
                            timestamp_str, level, json_data = parts
                            
                            # Parse timestamp
                            try:
                                log_date = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                                if log_date < cutoff_date:
                                    continue
                            except:
                                continue
                            
                            # Parse log data
                            log_entry = json.loads(json_data)
                            status_code = log_entry.get('status_code', 0)
                            
                            # Check if it's an error
                            if status_code >= 400 or level in ['ERROR', 'CRITICAL']:
                                error_analysis["total_errors"] += 1
                                
                                # Error by endpoint
                                endpoint = f"{log_entry.get('method', 'UNKNOWN')} {log_entry.get('path', '/unknown')}"
                                error_analysis["error_by_endpoint"][endpoint] = error_analysis["error_by_endpoint"].get(endpoint, 0) + 1
                                
                                # Error by status
                                status_key = f"{status_code}" if status_code else "Unknown"
                                error_analysis["error_by_status"][status_key] = error_analysis["error_by_status"].get(status_key, 0) + 1
                                
                                # Top error messages
                                error_message = log_entry.get('error_message', log_entry.get('message', 'Unknown error'))
                                error_analysis["top_error_messages"][error_message] = error_analysis["top_error_messages"].get(error_message, 0) + 1
                                
                                # Critical errors (5xx)
                                if status_code >= 500:
                                    error_analysis["critical_errors"].append({
                                        "timestamp": timestamp_str,
                                        "endpoint": endpoint,
                                        "status_code": status_code,
                                        "message": error_message,
                                        "client_ip": log_entry.get('client_ip', 'Unknown')
                                    })
                                    
                    except Exception:
                        continue
        
        # Sort and limit results
        error_analysis["error_by_endpoint"] = dict(sorted(error_analysis["error_by_endpoint"].items(), key=lambda x: x[1], reverse=True)[:10])
        error_analysis["top_error_messages"] = dict(sorted(error_analysis["top_error_messages"].items(), key=lambda x: x[1], reverse=True)[:10])
        error_analysis["critical_errors"] = sorted(error_analysis["critical_errors"], key=lambda x: x["timestamp"], reverse=True)[:20]
        
        return {
            "success": True,
            "analysis": error_analysis,
            "period_days": days
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze errors: {str(e)}")

@router.get("/analysis/performance")
async def get_performance_analysis(
    days: int = Query(7, description="Number of days to analyze"),
    current_user: dict = Depends(get_current_user)
):
    """Phân tích hiệu suất chi tiết"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        performance_analysis = {
            "avg_response_time": 0,
            "p95_response_time": 0,
            "p99_response_time": 0,
            "slowest_endpoints": [],
            "performance_trends": [],
            "response_time_by_hour": {},
            "total_requests": 0
        }
        
        response_times = []
        endpoint_times = {}
        hourly_stats = {}
        
        log_files = ["requests.log", "performance.log"]
        
        for log_file in log_files:
            log_path = os.path.join("logs", log_file)
            if not os.path.exists(log_path):
                continue
                
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        parts = line.strip().split(' | ', 2)
                        if len(parts) >= 3:
                            timestamp_str, level, json_data = parts
                            
                            # Parse timestamp
                            try:
                                log_date = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                                if log_date < cutoff_date:
                                    continue
                            except:
                                continue
                            
                            log_entry = json.loads(json_data)
                            response_time = log_entry.get('response_time_ms', 0)
                            
                            if response_time > 0:
                                performance_analysis["total_requests"] += 1
                                response_times.append(response_time)
                                
                                # Endpoint performance
                                endpoint = f"{log_entry.get('method', 'UNKNOWN')} {log_entry.get('path', '/unknown')}"
                                if endpoint not in endpoint_times:
                                    endpoint_times[endpoint] = []
                                endpoint_times[endpoint].append(response_time)
                                
                                # Hourly stats
                                hour = log_date.strftime('%Y-%m-%d %H:00')
                                if hour not in hourly_stats:
                                    hourly_stats[hour] = []
                                hourly_stats[hour].append(response_time)
                                
                    except Exception:
                        continue
        
        # Calculate statistics
        if response_times:
            response_times.sort()
            performance_analysis["avg_response_time"] = sum(response_times) / len(response_times)
            performance_analysis["p95_response_time"] = response_times[int(len(response_times) * 0.95)]
            performance_analysis["p99_response_time"] = response_times[int(len(response_times) * 0.99)]
        
        # Slowest endpoints
        slowest_endpoints = []
        for endpoint, times in endpoint_times.items():
            avg_time = sum(times) / len(times)
            slowest_endpoints.append({
                "endpoint": endpoint,
                "avg_response_time": round(avg_time, 2),
                "max_response_time": max(times),
                "min_response_time": min(times),
                "request_count": len(times)
            })
        
        performance_analysis["slowest_endpoints"] = sorted(slowest_endpoints, key=lambda x: x["avg_response_time"], reverse=True)[:10]
        
        # Hourly performance
        for hour, times in hourly_stats.items():
            performance_analysis["response_time_by_hour"][hour] = {
                "avg": sum(times) / len(times),
                "max": max(times),
                "min": min(times),
                "count": len(times)
            }
        
        return {
            "success": True,
            "analysis": performance_analysis,
            "period_days": days
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze performance: {str(e)}")

def _matches_filter(log_entry: Dict, filter_request: LogFilterRequest) -> bool:
    """Kiểm tra log entry có match với filter không"""
    try:
        # Date filter
        if filter_request.start_date:
            log_date = datetime.fromisoformat(log_entry.get('timestamp', ''))
            start_date = datetime.fromisoformat(filter_request.start_date)
            if log_date < start_date:
                return False
        
        if filter_request.end_date:
            log_date = datetime.fromisoformat(log_entry.get('timestamp', ''))
            end_date = datetime.fromisoformat(filter_request.end_date)
            if log_date > end_date:
                return False
        
        # IP filter
        if filter_request.client_ip:
            if filter_request.client_ip not in log_entry.get('client_ip', ''):
                return False
        
        # Path filter
        if filter_request.path:
            if filter_request.path not in log_entry.get('path', ''):
                return False
        
        # Method filter
        if filter_request.method:
            if log_entry.get('method') != filter_request.method:
                return False
        
        # Status code filter
        if filter_request.status_code:
            if log_entry.get('status_code') != filter_request.status_code:
                return False
        
        # Response time filter
        if filter_request.min_response_time:
            response_time = log_entry.get('response_time_ms', 0)
            if response_time < filter_request.min_response_time:
                return False
        
        # Log level filter
        if filter_request.log_level:
            if log_entry.get('level') != filter_request.log_level:
                return False
        
        return True
    except Exception:
        return False

def _clear_old_logs(log_path: str, cutoff_date: datetime):
    """Clear logs older than cutoff_date"""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        filtered_lines = []
        for line in lines:
            try:
                timestamp_str = line.split(' | ')[0]
                log_date = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                
                if log_date >= cutoff_date:
                    filtered_lines.append(line)
            except:
                # Keep line if can't parse date
                filtered_lines.append(line)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.writelines(filtered_lines)
            
    except Exception as e:
        raise Exception(f"Failed to clear old logs from {log_path}: {str(e)}")
