from config import RouterConfig, MiddlewareConfig
from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from src.service.SwaggerConfigService import swagger_config_service
from src.middleware.auth_middleware import AuthMiddleware, get_current_user
from src.middleware.security_middleware import SecurityMiddleware, setup_cors
from src.middleware.logging_middleware import AdvancedLoggingMiddleware
import uvicorn
import os



# Create FastAPI app with metadata
app = FastAPI(
    title="AI Processing API",
    description="AI Processing API with OCR, object detection capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure templates
templates = Jinja2Templates(directory="templates")

# Configure Swagger documentation
swagger_config_service.configure_database_driven_swagger(app)

# Setup CORS
setup_cors(app)

# Add logging middleware (should be first to log all requests)
app.add_middleware(AdvancedLoggingMiddleware)

# Add security middleware 
#app.add_middleware(SecurityMiddleware)

# Add auth middleware (should be last in auth chain)
app.add_middleware(AuthMiddleware)

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint to serve the new frontend interface
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Main landing page"""
    return templates.TemplateResponse("index.html", {"request": request})

# ==================== ADMIN ROUTES ====================

@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    """Admin login page"""
    return templates.TemplateResponse("admin/login.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, user: dict = Depends(get_current_user)):
    """Admin dashboard"""
    return templates.TemplateResponse("admin/dashboard.html", {"request": request, "user": user})

@app.get("/admin/management", response_class=HTMLResponse)
async def admin_management(request: Request, user: dict = Depends(get_current_user)):
    """API Management page"""
    return templates.TemplateResponse("admin/management.html", {"request": request, "user": user})

@app.get("/admin/settings", response_class=HTMLResponse)
async def admin_settings(request: Request, user: dict = Depends(get_current_user)):
    """Admin settings page"""
    return templates.TemplateResponse("admin/settings.html", {"request": request, "user": user})

@app.get("/admin/profile", response_class=HTMLResponse)
async def admin_profile(request: Request, user: dict = Depends(get_current_user)):
    """Admin profile page"""
    return templates.TemplateResponse("admin/profile.html", {"request": request, "user": user})

@app.get("/admin/security", response_class=HTMLResponse)
async def admin_security(request: Request, user: dict = Depends(get_current_user)):
    """Security management page"""
    return templates.TemplateResponse("admin/security.html", {"request": request, "user": user})

@app.get("/admin/logging", response_class=HTMLResponse)
async def admin_logging(request: Request, user: dict = Depends(get_current_user)):
    """Logging management page"""
    return templates.TemplateResponse("admin/logging.html", {"request": request, "user": user})

@app.get("/admin/logging-enhanced", response_class=HTMLResponse)
async def admin_logging_enhanced(request: Request, user: dict = Depends(get_current_user)):
    """Enhanced logging management page"""
    return templates.TemplateResponse("admin/logging_enhanced.html", {"request": request, "user": user})

@app.get("/admin/rate-limit", response_class=HTMLResponse)
async def admin_rate_limit(request: Request, user: dict = Depends(get_current_user)):
    """Rate limiting management page"""
    return templates.TemplateResponse("admin/rate_limit.html", {"request": request, "user": user})

# ==================== END ADMIN ROUTES ====================

# Include additional routers from config
RouterConfig().include_routers(app, RouterConfig().api_dir, "src.router.api")
# Add middleware
MiddlewareConfig.add_cors_middleware(app)

if __name__ == "__main__":
    # Khởi tạo API Management Service và sync endpoints khi start app
    try:
        from src.service.ApiManagementService import api_management
        print("🔄 Syncing API endpoints...")
        stats = api_management.sync_endpoints()
        print(f"✅ Sync completed: {stats['new_endpoints']} new, {stats['updated_endpoints']} updated")
    except Exception as e:
        print(f"⚠️ Failed to sync endpoints: {e}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)