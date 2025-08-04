from config import RouterConfig, MiddlewareConfig
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os

# Import logging components
try:
    from src.service.LoggingManager import logging_manager
    from src.middleware.logging_middleware import OCRLoggingMiddleware
    LOGGING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Logging components not available: {e}")
    LOGGING_AVAILABLE = False

# Create FastAPI app with metadata
app = FastAPI(
    title="AI Processing API with Monitoring",
    description="Comprehensive AI processing API with OCR, object detection, monitoring and logging capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add logging middleware if available
if LOGGING_AVAILABLE:
    app.add_middleware(OCRLoggingMiddleware, logging_manager=logging_manager)
    print("✅ Logging middleware added successfully")

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint to serve the web interface
@app.get("/")
async def read_root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "AI Processing API with Monitoring is running", "docs": "/docs", "monitoring": "/api/v1/monitoring"}

# Include additional routers from config
RouterConfig().include_routers(app, RouterConfig().api_dir, "src.router.api")

# Add middleware
MiddlewareConfig.add_cors_middleware(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)