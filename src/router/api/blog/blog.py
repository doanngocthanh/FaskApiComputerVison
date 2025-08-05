from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Path, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid
import os
import cv2
import numpy as np
from io import BytesIO
import importlib
import time
from src.router.api.__init__ import *  
from src.service.YOLODetector import YOLODetector
from src.service.EasyOCRManager import EasyOCRManager
from PIL import Image
from config import BlogConfig
import markdown
from markdown.extensions import codehilite, tables, toc
from pathlib import Path
import re

# Define ErrorResponse model if not imported from __init__
class ErrorResponse(BaseModel):
    detail: str
    error: Optional[str] = None

# Router setup
router = APIRouter(
    prefix="/api/v1/blog",
    tags=["Blogs Documentation"],
    responses={
        404: {"model": ErrorResponse, "description": "Resource not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)

# Base models for blog responses
class BlogPost(BaseModel):
    id: str
    title: str
    content: str
    created_at: datetime
    file_path: str

class BlogListResponse(BaseModel):
    blogs: List[BlogPost]
    total: int


def fix_image_paths(html_content: str) -> str:
    """Fix relative image paths to work with static file serving"""
    # Replace src="filename.jpg" with src="/static/content/filename.jpg"
    html_content = re.sub(
        r'src="(?!http|/static)([^"]+\.(jpg|jpeg|png|gif|webp|svg))"',
        r'src="/static/content/\1"',
        html_content,
        flags=re.IGNORECASE
    )
    
    # Also handle any remaining relative paths in markdown image syntax
    html_content = re.sub(
        r'<img([^>]*) src="(?!http|/static)([^"]+)"',
        r'<img\1 src="/static/content/\2"',
        html_content,
        flags=re.IGNORECASE
    )
    
    return html_content


# Get all blog posts
@router.get("/", response_model=BlogListResponse)
async def get_all_blogs():
    """Get all blog posts from markdown files"""
    try:
        content_dir = Path(BlogConfig().get_blog_path())
        if not content_dir.exists():
            content_dir.mkdir(parents=True, exist_ok=True)
        
        blogs = []
        for md_file in content_dir.glob("*.md"):
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Convert markdown to HTML with extensions
            html_content = markdown.markdown(
                content,
                extensions=[
                    'markdown.extensions.fenced_code',
                    'markdown.extensions.tables',
                    'markdown.extensions.toc',
                    'markdown.extensions.codehilite',
                    'markdown.extensions.extra'
                ]
            )
            
            # Fix relative image paths to absolute paths
            html_content = fix_image_paths(html_content)
            
            # Extract title from first h1 or use filename
            title = md_file.stem
            if content.startswith('#'):
                title = content.split('\n')[0].replace('#', '').strip()
            
            blog = BlogPost(
                id=md_file.stem,
                title=title,
                content=html_content,
                created_at=datetime.fromtimestamp(md_file.stat().st_mtime),
                file_path=str(md_file)
            )
            blogs.append(blog)
        
        # Sort by creation time (newest first)
        blogs.sort(key=lambda x: x.created_at, reverse=True)
        
        return BlogListResponse(blogs=blogs, total=len(blogs))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to load blog posts")
        # Get single blog post
@router.get("/{blog_id}", response_model=BlogPost)
async def get_blog_by_id(blog_id: str = Path(description="Blog ID")):
            """Get a specific blog post by ID"""
            try:
                content_dir = Path(BlogConfig().get_blog_path())
                md_file = content_dir / f"{blog_id}.md"
                
                if not md_file.exists():
                    raise HTTPException(status_code=404, detail="Blog post not found")
                
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Convert markdown to HTML with extensions
                html_content = markdown.markdown(
                    content,
                    extensions=[
                        'markdown.extensions.fenced_code',
                        'markdown.extensions.tables',
                        'markdown.extensions.toc',
                        'markdown.extensions.codehilite',
                        'markdown.extensions.extra'
                    ]
                )
                
                # Fix relative image paths to absolute paths
                html_content = fix_image_paths(html_content)
                
                # Extract title
                title = blog_id
                if content.startswith('#'):
                    title = content.split('\n')[0].replace('#', '').strip()
                
                return BlogPost(
                    id=blog_id,
                    title=title,
                    content=html_content,
                    created_at=datetime.fromtimestamp(md_file.stat().st_mtime),
                    file_path=str(md_file)
                )
            
            except HTTPException:
                raise
            except Exception as e:
                
                raise HTTPException(status_code=500, detail="Failed to load blog post")
 