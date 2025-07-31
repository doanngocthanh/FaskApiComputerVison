from fastapi import APIRouter

router = APIRouter()
from src.service.PaddleOCR import *
@router.get("/")
async def hello_world():
    return {"message": "Hello, World!"}