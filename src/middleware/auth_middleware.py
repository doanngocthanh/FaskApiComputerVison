"""
Auth Middleware
Middleware xác thực cho admin routes
"""

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
from typing import Optional

from src.service.AuthService import auth_service

class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.protected_paths = [
            "/api/admin",  # Tất cả admin API routes
            "/admin"       # Tất cả admin template routes
        ]
        self.excluded_paths = [
            "/api/admin/auth/login",  # Login API endpoint
            "/api/admin/auth/check",  # Auth check endpoint  
            "/admin/login"            # Login page template
        ]

    async def dispatch(self, request: Request, call_next):
        # Store current request path for use in _unauthorized_response
        self.current_request_path = request.url.path
        
        # Check if this is a protected path
        if self._is_protected_path(request.url.path):
            # Debug logging
            logging.info(f"Protected path accessed: {request.url.path}")
            logging.info(f"Request cookies: {request.cookies}")
            logging.info(f"Request headers: {dict(request.headers)}")
            
            # Get token from Authorization header or cookie
            token = self._extract_token(request)
            logging.info(f"Extracted token: {'***' if token else 'None'}")
            
            if not token:
                logging.warning(f"No token found for path: {request.url.path}")
                return self._unauthorized_response("Missing authentication token")
            
            # Verify token
            user_info = auth_service.verify_token(token)
            if not user_info:
                logging.warning(f"Invalid token for path: {request.url.path}")
                return self._unauthorized_response("Invalid or expired token")
            
            # Add user info to request state
            request.state.user = user_info
            logging.info(f"User authenticated: {user_info.get('username', 'unknown')}")
            
        response = await call_next(request)
        return response

    def _is_protected_path(self, path: str) -> bool:
        """Kiểm tra path có cần authentication không"""
        # Bỏ qua excluded paths
        for excluded in self.excluded_paths:
            if path.startswith(excluded):
                return False
        
        # Kiểm tra protected paths
        for protected in self.protected_paths:
            if path.startswith(protected):
                return True
        
        return False

    def _extract_token(self, request: Request) -> Optional[str]:
        """Extract token từ Authorization header hoặc cookie"""
        # Try Authorization header first
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        
        # Try cookie
        token = request.cookies.get("admin_token")
        if token:
            return token
        
        return None

    def _unauthorized_response(self, message: str):
        """Trả về response unauthorized"""
        # For template routes, redirect to login page
        if self.current_request_path and self.current_request_path.startswith("/admin") and not self.current_request_path.startswith("/api/admin"):
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url="/admin/login", status_code=302)
        
        # For API routes, return JSON response
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": "Unauthorized",
                "message": message,
                "redirect": "/admin/login"
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

def get_current_user(request: Request) -> dict:
    """Helper function để lấy current user từ request"""
    if hasattr(request.state, 'user'):
        return request.state.user
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated"
    )
