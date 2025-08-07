from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from src.service.ConfigService import ConfigService
from src.service.LoggingManager import LoggingManager
import os

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Initialize services
config_service = ConfigService()
logging_manager = LoggingManager()

@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Admin dashboard page"""
    return templates.TemplateResponse("admin/dashboard.html", {
        "request": request,
        "page_title": "Dashboard"
    })

@router.get("/admin/management", response_class=HTMLResponse)
async def admin_management(request: Request):
    """API management page"""
    return templates.TemplateResponse("admin/management.html", {
        "request": request,
        "page_title": "API Management"
    })

@router.get("/admin/settings", response_class=HTMLResponse)
async def admin_settings(request: Request):
    """Settings page"""
    return templates.TemplateResponse("admin/settings.html", {
        "request": request,
        "page_title": "Settings"
    })

@router.get("/admin/profile", response_class=HTMLResponse)
async def admin_profile(request: Request):
    """Profile page"""
    return templates.TemplateResponse("admin/profile.html", {
        "request": request,
        "page_title": "Profile"
    })

@router.get("/admin/security", response_class=HTMLResponse)
async def admin_security(request: Request):
    """Security management page"""
    return templates.TemplateResponse("admin/security.html", {
        "request": request,
        "page_title": "Security Management"
    })

@router.get("/admin/logging", response_class=HTMLResponse)
async def admin_logging(request: Request):
    """Logging management page"""
    return templates.TemplateResponse("admin/logging.html", {
        "request": request,
        "page_title": "Logging Management"
    })

@router.get("/admin/login", response_class=HTMLResponse)
async def admin_login(request: Request):
    """Login page"""
    return templates.TemplateResponse("admin/login.html", {
        "request": request,
        "page_title": "Admin Login"
    })
