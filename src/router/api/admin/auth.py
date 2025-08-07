"""
Authentication Router
Quản lý login/logout cho admin
"""

from fastapi import APIRouter, HTTPException, Request, Depends, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from typing import Optional
import logging

from src.service.AuthService import auth_service
from src.middleware.auth_middleware import get_current_user

# Router setup
router = APIRouter(
    prefix="/api/admin/auth",
    tags=["Authentication"]
)

# Pydantic models
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: dict

class PasswordChangeRequest(BaseModel):
    old_password: str
    new_password: str

@router.get("/login", response_class=HTMLResponse)
async def login_page():
    """Redirect to main admin login page"""
    return RedirectResponse(url="/admin/login")

@router.post("/login")
async def login(request: Request, login_data: LoginRequest):
    """API login"""
    try:
        # Get client info
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("User-Agent")
        
        # Authenticate user
        user_info = auth_service.authenticate_user(
            login_data.username, 
            login_data.password,
            ip_address,
            user_agent
        )
        
        if not user_info:
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password"
            )
        
        logging.info(f"Admin login successful: {login_data.username} from {ip_address}")
        
        return {
            "access_token": user_info["access_token"],
            "token_type": user_info["token_type"],
            "user": {
                "id": user_info["id"],
                "username": user_info["username"],
                "email": user_info["email"],
                "full_name": user_info["full_name"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/logout")
async def logout(request: Request, user: dict = Depends(get_current_user)):
    """API logout"""
    try:
        # Extract token để xóa session
        auth_header = request.headers.get("Authorization")
        token = None
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
        else:
            token = request.cookies.get("admin_token")
        
        if token:
            auth_service.logout(token)
        
        logging.info(f"Admin logout: {user['username']}")
        
        response = JSONResponse(content={"message": "Logged out successfully"})
        response.delete_cookie("admin_token", path="/")
        return response
        
    except Exception as e:
        logging.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/check")
async def check_auth(request: Request):
    """Check authentication status - Enhanced version"""
    try:
        # Extract token from request
        auth_header = request.headers.get("Authorization")
        token = None
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
        else:
            token = request.cookies.get("admin_token")
        
        if not token:
            raise HTTPException(
                status_code=401,
                detail="No authentication token provided"
            )
        
        # Verify token
        user_info = auth_service.verify_token(token)
        if not user_info:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        
        # Get additional user info
        full_user_info = auth_service.db_config.fetch_one("""
            SELECT id, username, email, full_name, created_at, last_login
            FROM admin_users WHERE id = ?
        """, (user_info["user_id"],))
        
        if not full_user_info:
            raise HTTPException(
                status_code=401,
                detail="User not found"
            )
        
        return {
            "authenticated": True,
            "user": {
                "id": full_user_info[0],
                "username": full_user_info[1],
                "email": full_user_info[2],
                "full_name": full_user_info[3],
                "created_at": full_user_info[4],
                "last_login": full_user_info[5]
            },
            "token_valid": True,
            "session_info": {
                "ip_address": request.client.host if request.client else None,
                "user_agent": request.headers.get("User-Agent"),
                "token_expires": user_info.get("exp")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Auth check error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Authentication check failed"
        )

@router.get("/profile")
async def get_profile(user: dict = Depends(get_current_user)):
    """Lấy thông tin profile"""
    try:
        # Get full user info từ database
        user_info = auth_service.db_config.fetch_one("""
            SELECT id, username, email, full_name, created_at, last_login
            FROM admin_users WHERE id = ?
        """, (user["user_id"],))
        
        if not user_info:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get active sessions
        sessions = auth_service.get_active_sessions(user["user_id"])
        
        return {
            "id": user_info[0],
            "username": user_info[1],
            "email": user_info[2],
            "full_name": user_info[3],
            "created_at": user_info[4],
            "last_login": user_info[5],
            "active_sessions": len(sessions),
            "sessions": sessions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Profile error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/change-password")
async def change_password(
    request: PasswordChangeRequest,
    user: dict = Depends(get_current_user)
):
    """Đổi password"""
    try:
        success = auth_service.change_password(
            user["user_id"],
            request.old_password,
            request.new_password
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Invalid old password")
        
        logging.info(f"Password changed for user: {user['username']}")
        
        return {"message": "Password changed successfully. Please login again."}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Password change error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/sessions")
async def get_sessions(user: dict = Depends(get_current_user)):
    """Lấy danh sách sessions của user"""
    try:
        sessions = auth_service.get_active_sessions(user["user_id"])
        return {
            "sessions": sessions,
            "total": len(sessions)
        }
    except Exception as e:
        logging.error(f"Get sessions error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Cleanup task - có thể chạy định kỳ
@router.post("/cleanup")
async def cleanup_sessions(user: dict = Depends(get_current_user)):
    """Cleanup expired sessions (admin only)"""
    try:
        auth_service.cleanup_expired_sessions()
        return {"message": "Expired sessions cleaned up"}
    except Exception as e:
        logging.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
