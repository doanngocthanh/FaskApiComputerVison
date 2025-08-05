from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

# Router setup
router = APIRouter(
    prefix="",
    tags=["Frontend"],
    responses={
        404: {"description": "Resource not found"},
        500: {"description": "Internal server error"}
    }
)

# Get static directory path
static_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "static")

@router.get("/", response_class=HTMLResponse)
async def home():
    """
    Serve main frontend application
    """
    frontend_index = os.path.join(static_dir, "frontend", "index.html")
    return FileResponse(frontend_index)

@router.get("/app", response_class=HTMLResponse) 
async def app():
    """
    Alternative route to frontend application
    """
    frontend_index = os.path.join(static_dir, "frontend", "index.html")
    return FileResponse(frontend_index)

@router.get("/frontend", response_class=HTMLResponse)
async def frontend():
    """
    Alternative route to frontend application
    """
    frontend_index = os.path.join(static_dir, "frontend", "index.html")
    return FileResponse(frontend_index)
