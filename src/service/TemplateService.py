"""
Template Service for rendering HTML pages with shared navigation
Manages template rendering and navigation inclusion
"""

import os
from typing import Dict, Optional

class TemplateService:
    def __init__(self, static_dir: str = "static"):
        self.static_dir = static_dir
        self.pages_dir = os.path.join(static_dir, "pages")
        self.components_dir = os.path.join(static_dir, "components")
        
        # Cache for templates
        self._template_cache = {}
        self._navigation_cache = None
    
    def get_navigation_html(self) -> str:
        """Get navigation HTML from components"""
        if self._navigation_cache is None:
            nav_file = os.path.join(self.components_dir, "navigation.html")
            try:
                with open(nav_file, 'r', encoding='utf-8') as f:
                    self._navigation_cache = f.read()
            except FileNotFoundError:
                self._navigation_cache = "<!-- Navigation not found -->"
        
        return self._navigation_cache
    
    def render_page(self, 
                   page_title: str = "AI Vision",
                   page_icon: str = "🔍",
                   page_description: str = "AI Vision System",
                   content_file: str = "",
                   page_styles: str = "",
                   page_scripts: str = "",
                   active_page: str = "") -> str:
        """
        Render a page using the layout template system
        
        Args:
            page_title: Title of the page
            page_icon: Icon for the page
            page_description: Description of the page
            content_file: HTML file containing page content (optional)
            page_styles: Additional CSS styles for the page
            page_scripts: Additional JavaScript for the page
            active_page: Active navigation item
            
        Returns:
            Rendered HTML string
        """
        
        # Use layout template
        template_file = os.path.join(self.pages_dir, "layout.html")
        
        if "layout.html" not in self._template_cache:
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    self._template_cache["layout.html"] = f.read()
            except FileNotFoundError:
                return f"<html><body><h1>Layout template not found</h1></body></html>"
        
        template_content = self._template_cache["layout.html"]
        
        # Load page content if specified
        page_content = ""
        if content_file:
            content_path = os.path.join(self.pages_dir, content_file)
            try:
                with open(content_path, 'r', encoding='utf-8') as f:
                    page_content = f.read()
                    # Extract content between <body> tags if it's a full HTML file
                    if '<body>' in page_content and '</body>' in page_content:
                        start = page_content.find('<body>') + 6
                        end = page_content.find('</body>')
                        page_content = page_content[start:end].strip()
                        # Remove navigation placeholder if exists
                        page_content = page_content.replace('<!-- Fixed Navigation Menu -->', '')
                        page_content = page_content.replace('<div class="container">', '')
                        page_content = page_content.replace('</div>\n</body>', '')
            except FileNotFoundError:
                page_content = f"<p>Content file {content_file} not found</p>"
        
        # Replace template variables
        replacements = {
            '{{PAGE_TITLE}}': page_title,
            '{{PAGE_ICON}}': page_icon,
            '{{PAGE_DESCRIPTION}}': page_description,
            '{{PAGE_CONTENT}}': page_content,
            '{{PAGE_STYLES}}': page_styles,
            '{{PAGE_SCRIPTS}}': page_scripts,
            '{{ACTIVE_PAGE}}': active_page
        }
        
        rendered_content = template_content
        for placeholder, value in replacements.items():
            rendered_content = rendered_content.replace(placeholder, value)
        
        return rendered_content
    
    def render_analytics_page(self) -> str:
        """Render analytics page"""
        content = """
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon">📊</div>
                <div class="stat-content">
                    <h3>Total Requests</h3>
                    <p class="stat-number">1,234</p>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🎯</div>
                <div class="stat-content">
                    <h3>Success Rate</h3>
                    <p class="stat-number">98.5%</p>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">⚡</div>
                <div class="stat-content">
                    <h3>Avg Response Time</h3>
                    <p class="stat-number">150ms</p>
                </div>
            </div>
        </div>
        
        <style>
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .stat-card {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            gap: 15px;
            border-left: 4px solid #3498db;
        }
        
        .stat-icon {
            font-size: 2rem;
        }
        
        .stat-content h3 {
            margin: 0 0 5px 0;
            color: #2c3e50;
            font-size: 1rem;
        }
        
        .stat-number {
            font-size: 1.8rem;
            font-weight: bold;
            color: #3498db;
            margin: 0;
        }
        </style>
        """
        
        return self.render_page(
            page_title="Analytics Dashboard",
            page_icon="📊",
            page_description="System analytics and statistics",
            page_content=content,
            active_page="analytics"
        )
    
    def render_models_page(self) -> str:
        """Render models page"""
        content = """
        <div class="models-grid">
            <div class="model-card">
                <div class="model-icon">🤖</div>
                <div class="model-info">
                    <h3>YOLO Detection</h3>
                    <p>Object detection and classification</p>
                    <span class="model-status active">Active</span>
                </div>
            </div>
            <div class="model-card">
                <div class="model-icon">🔤</div>
                <div class="model-info">
                    <h3>EasyOCR</h3>
                    <p>Text recognition and extraction</p>
                    <span class="model-status active">Active</span>
                </div>
            </div>
            <div class="model-card">
                <div class="model-icon">📄</div>
                <div class="model-info">
                    <h3>MRZ Parser</h3>
                    <p>Machine readable zone processing</p>
                    <span class="model-status active">Active</span>
                </div>
            </div>
        </div>
        
        <style>
        .models-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .model-card {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            gap: 20px;
            border: 1px solid #e9ecef;
            transition: transform 0.2s ease;
        }
        
        .model-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .model-icon {
            font-size: 2.5rem;
        }
        
        .model-info h3 {
            margin: 0 0 8px 0;
            color: #2c3e50;
        }
        
        .model-info p {
            margin: 0 0 10px 0;
            color: #7f8c8d;
        }
        
        .model-status {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        
        .model-status.active {
            background: #d4edda;
            color: #155724;
        }
        </style>
        """
        
        return self.render_page(
            page_title="AI Models",
            page_icon="🤖",
            page_description="Manage and monitor AI models",
            page_content=content,
            active_page="models"
        )
    
    def clear_cache(self):
        """Clear template cache"""
        self._template_cache.clear()
        self._navigation_cache = None

# Global template service instance
template_service = TemplateService()
