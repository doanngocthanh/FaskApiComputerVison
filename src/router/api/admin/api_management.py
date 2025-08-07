"""
API Management Router
Quản lý hiển thị/ẩn các API endpoints
"""

from fastapi import APIRouter, HTTPException, Form, Query, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime

from src.service.ApiManagementService import api_management
from src.middleware.auth_middleware import get_current_user

# Router setup
router = APIRouter(
    prefix="/api/admin",
    tags=["API Management"]
)

# Pydantic models
class EndpointUpdate(BaseModel):
    id: int = Field(..., description="ID của endpoint")
    is_visible: bool = Field(..., description="Trạng thái hiển thị")
    is_public: Optional[bool] = Field(None, description="Trạng thái public")

class BulkUpdateRequest(BaseModel):
    updates: List[EndpointUpdate] = Field(..., description="Danh sách cập nhật")

class SettingUpdate(BaseModel):
    key: str = Field(..., description="Key của setting")
    value: str = Field(..., description="Giá trị setting")

class EndpointSearchResponse(BaseModel):
    id: int
    endpoint_path: str
    endpoint_method: str
    router_name: str
    endpoint_name: str
    description: str
    is_visible: bool
    is_public: bool
    tags: List[str]

@router.get("/management")
async def api_management_page(user: dict = Depends(get_current_user)):
    """Redirect to main admin management page"""
    return RedirectResponse(url="/admin/management")

@router.get("/endpoints")
async def get_all_endpoints(user: dict = Depends(get_current_user)):
    """Lấy tất cả endpoints - Requires Authentication"""
    try:
        endpoints = api_management.get_all_endpoints()
        return {
            "status": "success",
            "endpoints": endpoints,
            "total": len(endpoints)
        }
    except Exception as e:
        logging.error(f"Failed to get endpoints: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/endpoints/visible")
async def get_visible_endpoints(user: dict = Depends(get_current_user)):
    """Lấy các endpoints hiển thị - Requires Authentication"""
    try:
        endpoints = api_management.get_visible_endpoints()
        return {
            "status": "success",
            "endpoints": endpoints,
            "total": len(endpoints)
        }
    except Exception as e:
        logging.error(f"Failed to get visible endpoints: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync")
async def sync_endpoints(user: dict = Depends(get_current_user)):
    """Đồng bộ endpoints từ code với real-time refresh - Requires Authentication"""
    try:
        stats = api_management.sync_endpoints()
        # Refresh Swagger sau khi sync
        api_management.invalidate_swagger_cache()
        return {
            "status": "success",
            "message": "Endpoints synced successfully and Swagger refreshed",
            "stats": stats
        }
    except Exception as e:
        logging.error(f"Failed to sync endpoints: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/refresh-swagger")
async def refresh_swagger(user: dict = Depends(get_current_user)):
    """Force refresh Swagger documentation - Requires Authentication"""
    try:
        success = api_management.invalidate_swagger_cache()
        return {
            "status": "success" if success else "warning",
            "message": "Swagger cache invalidated" if success else "Could not invalidate cache"
        }
    except Exception as e:
        logging.error(f"Failed to refresh swagger: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/endpoints/{endpoint_id}/visibility")
async def update_endpoint_visibility(
    endpoint_id: int,
    is_visible: bool = Form(...),
    is_public: bool = Form(None),
    user: dict = Depends(get_current_user)
):
    """Cập nhật trạng thái hiển thị endpoint với real-time refresh - Requires Authentication"""
    try:
        success = api_management.update_endpoint_visibility_with_refresh(endpoint_id, is_visible, is_public)
        if success:
            return {
                "status": "success", 
                "message": "Endpoint visibility updated and Swagger refreshed"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update endpoint visibility")
    except Exception as e:
        logging.error(f"Failed to update endpoint visibility: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/endpoints/bulk-update")
async def bulk_update_endpoints(request: BulkUpdateRequest, user: dict = Depends(get_current_user)):
    """Cập nhật hàng loạt endpoints với real-time refresh - Requires Authentication"""
    try:
        logging.info(f"Bulk update request: {len(request.updates)} updates")
        
        # Validate updates
        valid_updates = []
        for update in request.updates:
            try:
                # Convert Pydantic model to dict
                update_dict = {
                    'id': update.id,
                    'is_visible': update.is_visible
                }
                if update.is_public is not None:
                    update_dict['is_public'] = update.is_public
                    
                valid_updates.append(update_dict)
                logging.info(f"Valid update: endpoint {update.id} -> visible={update.is_visible}, public={update.is_public}")
            except Exception as e:
                logging.error(f"Invalid update for endpoint {update.id}: {e}")
                continue
        
        if not valid_updates:
            raise HTTPException(status_code=400, detail="No valid updates provided")
        
        stats = api_management.bulk_update_visibility_with_refresh(valid_updates)
        
        logging.info(f"Bulk update completed: {stats}")
        
        return {
            "status": "success",
            "message": f"Updated {stats['updated']} endpoints successfully and refreshed Swagger",
            "stats": stats,
            "processed": len(valid_updates),
            "total_requested": len(request.updates)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to bulk update endpoints: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/settings")
async def get_all_settings(user: dict = Depends(get_current_user)):
    """Lấy tất cả settings - Requires Authentication"""
    try:
        settings = api_management.get_all_settings()
        return {
            "status": "success",
            "settings": settings
        }
    except Exception as e:
        logging.error(f"Failed to get settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/settings")
async def update_setting(request: SettingUpdate):
    """Cập nhật setting với real-time refresh"""
    try:
        success = api_management.update_setting_with_refresh(request.key, request.value)
        if success:
            return {
                "status": "success", 
                "message": "Setting updated successfully and Swagger refreshed"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update setting")
    except Exception as e:
        logging.error(f"Failed to update setting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/swagger-config")
async def get_swagger_config():
    """Lấy cấu hình Swagger hiện tại"""
    try:
        config = api_management.get_swagger_config()
        return {
            "status": "success",
            "config": config
        }
    except Exception as e:
        logging.error(f"Failed to get swagger config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search")
async def search_endpoints(
    q: str = Query(..., description="Search term"),
    router_name: Optional[str] = Query(None, description="Router name filter"),
    method: Optional[str] = Query(None, description="HTTP method filter")
):
    """Tìm kiếm endpoints"""
    try:
        endpoints = api_management.search_endpoints(q, router_name, method)
        return {
            "status": "success",
            "endpoints": endpoints,
            "total": len(endpoints),
            "search_term": q
        }
    except Exception as e:
        logging.error(f"Failed to search endpoints: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/endpoints/{endpoint_id}")
async def delete_endpoint(endpoint_id: int):
    """Xóa endpoint"""
    try:
        success = api_management.delete_endpoint(endpoint_id)
        if success:
            return {"status": "success", "message": "Endpoint deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete endpoint")
    except Exception as e:
        logging.error(f"Failed to delete endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug/pending-changes")
async def debug_pending_changes():
    """Debug endpoint để kiểm tra các thay đổi pending"""
    try:
        all_endpoints = api_management.get_all_endpoints()
        return {
            "status": "success",
            "total_endpoints": len(all_endpoints),
            "endpoints_sample": all_endpoints[:5] if all_endpoints else [],
            "message": "Debug information retrieved"
        }
    except Exception as e:
        logging.error(f"Debug failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
