"""
Migration script to convert existing HTML files to use shared navigation
"""
import os
import re
from typing import Dict, List

class HTMLMigrationTool:
    def __init__(self, pages_dir: str = "static/pages"):
        self.pages_dir = pages_dir
        self.backup_dir = os.path.join(pages_dir, "backup")
        
        # Ensure backup directory exists
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def extract_page_info(self, html_content: str) -> Dict[str, str]:
        """Extract page information from HTML content"""
        info = {
            'title': 'AI Vision',
            'icon': '🔍',
            'description': 'AI Vision System',
            'content': '',
            'active_page': ''
        }
        
        # Extract title
        title_match = re.search(r'<title>([^<]+) - AI Vision</title>', html_content)
        if title_match:
            info['title'] = title_match.group(1)
            
        # Extract main content (everything between container divs, excluding nav)
        content_pattern = r'<div class="container".*?>(.*?)</div>\s*(?:<script|</body>)'
        content_match = re.search(content_pattern, html_content, re.DOTALL)
        if content_match:
            content = content_match.group(1).strip()
            
            # Remove page header if exists
            content = re.sub(r'<div class="page-header">.*?</div>', '', content, flags=re.DOTALL)
            
            # Remove alert info if it's the default one
            content = re.sub(r'<div class="alert alert-info">.*?Under Construction.*?</div>', '', content, flags=re.DOTALL)
            
            # Remove default buttons if they exist
            content = re.sub(r'<div style="text-align: center; margin-top: 40px;">.*?</div>', '', content, flags=re.DOTALL)
            
            info['content'] = content.strip()
        
        return info
    
    def generate_template_usage(self, filename: str, page_info: Dict[str, str]) -> str:
        """Generate template service usage for a page"""
        
        page_name = filename.replace('.html', '')
        
        # Map filename to page info
        page_mapping = {
            'analytics': {'icon': '📊', 'desc': 'System analytics and statistics'},
            'models': {'icon': '🤖', 'desc': 'Manage and monitor AI models'},
            'monitoring': {'icon': '📺', 'desc': 'Real-time system monitoring'},
            'ocr': {'icon': '🔤', 'desc': 'Optical Character Recognition'},
            'detection': {'icon': '🎯', 'desc': 'Object detection and classification'},
            'upload': {'icon': '📤', 'desc': 'Upload and process images'},
            'proxy-management': {'icon': '🔄', 'desc': 'API proxy management and configuration'},
            'blog': {'icon': '📝', 'desc': 'Latest news and updates'}
        }
        
        if page_name in page_mapping:
            icon = page_mapping[page_name]['icon']
            desc = page_mapping[page_name]['desc']
        else:
            icon = '🔍'
            desc = 'AI Vision System'
        
        # Generate route code
        route_code = f'''@router.get("/{page_name}", response_class=HTMLResponse)
async def {page_name}_page():
    """{page_info['title']} page"""'''
        
        if page_info['content'].strip():
            route_code += f'''
    content = """
    {page_info['content']}
    """
    
    return template_service.render_page(
        page_title="{page_info['title']}",
        page_icon="{icon}",
        page_description="{desc}",
        page_content=content,
        active_page="{page_name}"
    )'''
        else:
            route_code += f'''
    return template_service.render_page(
        page_title="{page_info['title']}",
        page_icon="{icon}",
        page_description="{desc}",
        active_page="{page_name}"
    )'''
        
        return route_code
    
    def migrate_file(self, filename: str) -> str:
        """Migrate a single HTML file"""
        file_path = os.path.join(self.pages_dir, filename)
        backup_path = os.path.join(self.backup_dir, filename)
        
        if not os.path.exists(file_path):
            return f"File {filename} not found"
        
        # Read original file
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Create backup
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Extract page information
        page_info = self.extract_page_info(original_content)
        
        # Generate route code
        route_code = self.generate_template_usage(filename, page_info)
        
        return route_code
    
    def migrate_all_files(self) -> Dict[str, str]:
        """Migrate all HTML files in the pages directory"""
        results = {}
        
        # Get all HTML files
        html_files = [f for f in os.listdir(self.pages_dir) 
                     if f.endswith('.html') and f != 'template.html']
        
        for filename in html_files:
            try:
                route_code = self.migrate_file(filename)
                results[filename] = route_code
            except Exception as e:
                results[filename] = f"Error: {str(e)}"
        
        return results
    
    def generate_complete_router(self, migration_results: Dict[str, str]) -> str:
        """Generate complete router file with all migrated routes"""
        
        router_header = '''"""
Static Pages Router - Migrated from individual HTML files
Handles rendering of static pages with shared navigation
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from src.service.TemplateService import template_service

router = APIRouter()

'''
        
        router_content = router_header
        
        for filename, route_code in migration_results.items():
            if not route_code.startswith("Error"):
                router_content += route_code + "\n\n"
        
        return router_content

# Usage example
if __name__ == "__main__":
    migrator = HTMLMigrationTool("c:/WorkSpace/Rest/static/pages")
    results = migrator.migrate_all_files()
    
    print("Migration Results:")
    print("=" * 50)
    
    for filename, result in results.items():
        print(f"\n{filename}:")
        print("-" * 30)
        print(result)
    
    # Generate complete router
    complete_router = migrator.generate_complete_router(results)
    
    print("\n" + "=" * 50)
    print("COMPLETE ROUTER FILE:")
    print("=" * 50)
    print(complete_router)
