"""
Monitoring API - REST endpoints cho monitoring và statistics
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import json
from pathlib import Path

# Import logging manager (will create simple version if schedule not available)
try:
    from src.service.LoggingManager import logging_manager
except ImportError:
    # Fallback simple logging manager
    class SimpleLoggingManager:
        def get_statistics(self, period='today'):
            return {"message": "Logging manager not available"}
        
        def get_performance_metrics(self, metric_type=None, hours=24):
            return []
        
        def cleanup_old_logs(self, keep_days=30):
            return {"message": "Cleanup not available"}
    
    logging_manager = SimpleLoggingManager()

# Router setup
router = APIRouter(
    prefix="/api/v1/yolo-ocr/monitoring",
    tags=["Monitoring & Logging"],
    responses={
        404: {"description": "Resource not found"},
        500: {"description": "Internal server error"}
    }
)

# Pydantic models
class StatisticsResponse(BaseModel):
    period: str
    start_time: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    average_processing_time_ms: float
    top_endpoints: Dict[str, int]
    language_usage: Dict[str, int]
    error_types: Dict[str, int]

class PerformanceMetric(BaseModel):
    timestamp: str
    metric_type: Optional[str] = None
    value: float
    metadata: Dict[str, Any]

class CleanupResponse(BaseModel):
    deleted_requests: int
    deleted_metrics: int
    cutoff_date: str
    message: str

@router.get("/")
async def monitoring_dashboard():
    """Trang dashboard monitoring"""
    
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OCR Monitoring Dashboard</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                overflow: hidden;
            }
            .header { 
                background: linear-gradient(45deg, #2c3e50, #3498db);
                color: white; 
                padding: 30px;
                text-align: center;
            }
            .header h1 { font-size: 2.5rem; margin-bottom: 10px; }
            .header p { opacity: 0.9; font-size: 1.1rem; }
            
            .dashboard { padding: 30px; }
            .stats-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .stat-card {
                background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                padding: 25px;
                border-radius: 12px;
                border-left: 5px solid #007bff;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }
            .stat-card:hover { transform: translateY(-5px); }
            .stat-card h3 { color: #495057; margin-bottom: 10px; }
            .stat-value { font-size: 2rem; font-weight: bold; color: #007bff; }
            .stat-label { color: #6c757d; font-size: 0.9rem; }
            
            .controls { 
                display: flex; 
                gap: 15px; 
                margin-bottom: 30px;
                flex-wrap: wrap;
            }
            .btn {
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 1rem;
                font-weight: 500;
                transition: all 0.3s ease;
                text-decoration: none;
                display: inline-block;
            }
            .btn-primary { 
                background: linear-gradient(45deg, #007bff, #0056b3);
                color: white; 
            }
            .btn-primary:hover { 
                background: linear-gradient(45deg, #0056b3, #004085);
                transform: translateY(-2px);
            }
            .btn-success { 
                background: linear-gradient(45deg, #28a745, #1e7e34);
                color: white; 
            }
            .btn-warning { 
                background: linear-gradient(45deg, #ffc107, #e0a800);
                color: #212529; 
            }
            .btn-danger { 
                background: linear-gradient(45deg, #dc3545, #bd2130);
                color: white; 
            }
            
            .section {
                background: #f8f9fa;
                border-radius: 12px;
                padding: 25px;
                margin-bottom: 25px;
                border: 1px solid #dee2e6;
            }
            .section h2 {
                color: #495057;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #007bff;
            }
            
            .logs-container {
                background: #212529;
                color: #fff;
                padding: 20px;
                border-radius: 8px;
                font-family: 'Courier New', monospace;
                max-height: 400px;
                overflow-y: auto;
                margin-top: 15px;
            }
            
            .loading {
                text-align: center;
                padding: 40px;
                color: #6c757d;
            }
            
            .error {
                background: #f8d7da;
                color: #721c24;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #f5c6cb;
                margin: 15px 0;
            }
            
            .success {
                background: #d4edda;
                color: #155724;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #c3e6cb;
                margin: 15px 0;
            }
            
            @media (max-width: 768px) {
                .controls { flex-direction: column; }
                .stats-grid { grid-template-columns: 1fr; }
                .header h1 { font-size: 2rem; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 OCR Monitoring Dashboard</h1>
                <p>Real-time monitoring và thống kê cho OCR Service</p>
            </div>
            
            <div class="dashboard">
                <!-- Statistics Cards -->
                <div class="stats-grid" id="statsGrid">
                    <div class="loading">Đang tải thống kê...</div>
                </div>
                
                <!-- Controls -->
                <div class="controls">
                    <button class="btn btn-primary" onclick="loadStats('today')">📅 Hôm nay</button>
                    <button class="btn btn-primary" onclick="loadStats('week')">📆 Tuần này</button>
                    <button class="btn btn-primary" onclick="loadStats('month')">🗓️ Tháng này</button>
                    <button class="btn btn-success" onclick="loadPerformance()">⚡ Performance</button>
                    <button class="btn btn-warning" onclick="cleanupLogs(7)">🧹 Cleanup 7 ngày</button>
                    <button class="btn btn-danger" onclick="cleanupLogs(30)">🗑️ Cleanup 30 ngày</button>
                </div>
                
                <!-- Real-time Statistics -->
                <div class="section">
                    <h2>📈 Thống kê Real-time</h2>
                    <div id="realtimeStats">
                        <div class="loading">Đang tải...</div>
                    </div>
                </div>
                
                <!-- Performance Metrics -->
                <div class="section">
                    <h2>⚡ Performance Metrics</h2>
                    <div id="performanceMetrics">
                        <div class="loading">Đang tải...</div>
                    </div>
                </div>
                
                <!-- System Logs -->
                <div class="section">
                    <h2>📋 System Status</h2>
                    <div id="systemStatus">
                        <div class="success">✅ OCR Service đang hoạt động bình thường</div>
                        <div>🕐 Last updated: <span id="lastUpdate"></span></div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let currentPeriod = 'today';
            
            // Load statistics
            async function loadStats(period = 'today') {
                currentPeriod = period;
                try {
                    const response = await fetch(`/api/v1/yolo-ocr/monitoring/statistics?period=${period}`);
                    const data = await response.json();
                    
                    if (response.ok) {
                        displayStats(data);
                    } else {
                        showError('Không thể tải thống kê: ' + data.detail);
                    }
                } catch (error) {
                    showError('Lỗi kết nối: ' + error.message);
                }
            }
            
            // Display statistics
            function displayStats(stats) {
                const grid = document.getElementById('statsGrid');
                grid.innerHTML = `
                    <div class="stat-card">
                        <h3>📊 Tổng Requests</h3>
                        <div class="stat-value">${stats.total_requests || 0}</div>
                        <div class="stat-label">Trong ${stats.period}</div>
                    </div>
                    <div class="stat-card">
                        <h3>✅ Thành công</h3>
                        <div class="stat-value">${stats.successful_requests || 0}</div>
                        <div class="stat-label">${stats.success_rate || 0}% success rate</div>
                    </div>
                    <div class="stat-card">
                        <h3>❌ Thất bại</h3>
                        <div class="stat-value">${stats.failed_requests || 0}</div>
                        <div class="stat-label">Requests bị lỗi</div>
                    </div>
                    <div class="stat-card">
                        <h3>⚡ Thời gian xử lý</h3>
                        <div class="stat-value">${Math.round(stats.average_processing_time_ms || 0)}ms</div>
                        <div class="stat-label">Trung bình</div>
                    </div>
                `;
                
                // Update real-time stats
                const realtimeDiv = document.getElementById('realtimeStats');
                realtimeDiv.innerHTML = `
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                        <div>
                            <h4>🔥 Top Endpoints</h4>
                            <div class="logs-container" style="max-height: 200px;">
                                ${Object.entries(stats.top_endpoints || {}).map(([endpoint, count]) => 
                                    `<div>${endpoint}: ${count} requests</div>`
                                ).join('') || '<div>Không có dữ liệu</div>'}
                            </div>
                        </div>
                        <div>
                            <h4>🌍 Language Usage</h4>
                            <div class="logs-container" style="max-height: 200px;">
                                ${Object.entries(stats.language_usage || {}).map(([lang, count]) => 
                                    `<div>${lang}: ${count} requests</div>`
                                ).join('') || '<div>Không có dữ liệu</div>'}
                            </div>
                        </div>
                    </div>
                `;
            }
            
            // Load performance metrics
            async function loadPerformance() {
                try {
                    const response = await fetch('/api/v1/yolo-ocr/monitoring/performance');
                    const data = await response.json();
                    
                    if (response.ok) {
                        displayPerformance(data);
                    } else {
                        showError('Không thể tải performance metrics');
                    }
                } catch (error) {
                    showError('Lỗi kết nối: ' + error.message);
                }
            }
            
            // Display performance metrics
            function displayPerformance(metrics) {
                const div = document.getElementById('performanceMetrics');
                div.innerHTML = `
                    <div class="logs-container">
                        ${metrics.map(metric => 
                            `<div>[${new Date(metric.timestamp).toLocaleString()}] ${metric.metric_type || 'Unknown'}: ${metric.value}</div>`
                        ).join('') || '<div>Không có performance metrics</div>'}
                    </div>
                `;
            }
            
            // Cleanup logs
            async function cleanupLogs(days) {
                if (!confirm(`Bạn có chắc muốn xóa logs cũ hơn ${days} ngày?`)) return;
                
                try {
                    const response = await fetch(`/api/v1/monitoring/cleanup?keep_days=${days}`, {
                        method: 'POST'
                    });
                    const data = await response.json();
                    
                    if (response.ok) {
                        showSuccess(`✅ Đã dọn dẹp ${data.deleted_requests} requests và ${data.deleted_metrics} metrics`);
                        loadStats(currentPeriod); // Reload stats
                    } else {
                        showError('Không thể dọn dẹp logs: ' + data.detail);
                    }
                } catch (error) {
                    showError('Lỗi kết nối: ' + error.message);
                }
            }
            
            // Show error message
            function showError(message) {
                const systemStatus = document.getElementById('systemStatus');
                systemStatus.innerHTML = `<div class="error">❌ ${message}</div>`;
            }
            
            // Show success message
            function showSuccess(message) {
                const systemStatus = document.getElementById('systemStatus');
                systemStatus.innerHTML = `<div class="success">${message}</div>`;
                setTimeout(() => {
                    systemStatus.innerHTML = `
                        <div class="success">✅ OCR Service đang hoạt động bình thường</div>
                        <div>🕐 Last updated: <span id="lastUpdate">${new Date().toLocaleString()}</span></div>
                    `;
                }, 3000);
            }
            
            // Update timestamp
            function updateTimestamp() {
                const lastUpdate = document.getElementById('lastUpdate');
                if (lastUpdate) {
                    lastUpdate.textContent = new Date().toLocaleString();
                }
            }
            
            // Auto refresh every 30 seconds
            setInterval(() => {
                loadStats(currentPeriod);
                updateTimestamp();
            }, 30000);
            
            // Initial load
            loadStats('today');
            updateTimestamp();
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@router.get("/statistics", response_model=Dict[str, Any])
async def get_statistics(
    period: str = Query('today', description="Period: today, week, month")
):
    """Lấy thống kê requests theo period"""
    try:
        if period not in ['today', 'week', 'month']:
            raise HTTPException(status_code=400, detail="Invalid period. Use: today, week, month")
        
        stats = logging_manager.get_statistics(period)
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.get("/performance")
async def get_performance_metrics(
    metric_type: Optional[str] = Query(None, description="Metric type to filter"),
    hours: int = Query(24, description="Hours to look back")
):
    """Lấy performance metrics"""
    try:
        metrics = logging_manager.get_performance_metrics(metric_type, hours)
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@router.post("/cleanup", response_model=Dict[str, Any])
async def cleanup_old_logs(
    keep_days: int = Query(30, description="Number of days to keep")
):
    """Dọn dẹp logs cũ"""
    try:
        if keep_days < 1:
            raise HTTPException(status_code=400, detail="keep_days must be at least 1")
        
        result = logging_manager.cleanup_old_logs(keep_days)
        result["message"] = f"Successfully cleaned up logs older than {keep_days} days"
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup logs: {str(e)}")

@router.get("/logs")
async def get_detailed_logs(
    page: int = Query(1, ge=1, description="Page number (starting from 1)"),
    limit: int = Query(20, ge=1, le=100, description="Number of logs per page"),
    endpoint: Optional[str] = Query(None, description="Filter by endpoint"),
    success: Optional[bool] = Query(None, description="Filter by success status"),
):
    """Lấy detailed logs với phân trang và filter"""
    try:
        offset = (page - 1) * limit
        
        # Get logs
        logs = logging_manager.get_detailed_logs(
            limit=limit, 
            offset=offset, 
            endpoint=endpoint, 
            success=success
        )
        
        # Get total count
        total_count = logging_manager.get_log_count(endpoint=endpoint, success=success)
        total_pages = (total_count + limit - 1) // limit
        
        return {
            "logs": logs,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "total_count": total_count,
                "per_page": limit,
                "has_next": page < total_pages,
                "has_prev": page > 1
            },
            "filters": {
                "endpoint": endpoint,
                "success": success
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")

@router.get("/logs/{log_id}")
async def get_log_detail(log_id: str):
    """Lấy chi tiết của một log theo ID"""
    try:
        logs = logging_manager.get_detailed_logs(limit=1, offset=0)
        
        # Tìm log theo ID
        for log in logs:
            if log.get('id') == log_id:
                return log
        
        raise HTTPException(status_code=404, detail="Log not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get log detail: {str(e)}")

@router.get("/request/{request_id}")
async def get_request_detail(request_id: str):
    """Lấy chi tiết request theo request_id UUID"""
    try:
        request_detail = logging_manager.get_request_by_id(request_id)
        
        if not request_detail:
            raise HTTPException(status_code=404, detail=f"Request with ID {request_id} not found")
        
        return {
            "request_id": request_id,
            "request_detail": request_detail,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get request detail: {str(e)}")

@router.get("/requests/search")
async def search_requests(
    endpoint: Optional[str] = Query(None, description="Filter by endpoint"),
    success: Optional[bool] = Query(None, description="Filter by success status"),
    limit: int = Query(50, description="Number of results to return"),
    offset: int = Query(0, description="Number of results to skip")
):
    """Search requests with filters"""
    try:
        requests = logging_manager.get_detailed_logs(
            limit=limit, 
            offset=offset, 
            endpoint_filter=endpoint,
            success_filter=success
        )
        
        total_count = logging_manager.get_log_count(endpoint=endpoint, success=success)
        
        return {
            "requests": requests,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search requests: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "OCR Monitoring",
        "version": "1.0.0"
    }
