"""
Auto-inject Global Navigation Script
Automatically adds the global navigation loader to all HTML files
"""
import os
import re
from pathlib import Path

class HTMLNavigationInjector:
    def __init__(self, pages_dir: str = "static/pages"):
        self.pages_dir = pages_dir
        self.backup_dir = os.path.join(pages_dir, "backup_original")
        
        # Ensure backup directory exists
        os.makedirs(self.backup_dir, exist_ok=True)
        
        self.injection_script = '''
    <!-- Global Navigation Loader -->
    <script src="/static/components/global-nav-loader.js"></script>'''
    
    def should_inject(self, content: str) -> bool:
        """Check if injection is needed"""
        return 'global-nav-loader.js' not in content
    
    def inject_navigation_loader(self, html_content: str) -> str:
        """Inject navigation loader script into HTML"""
        
        # Find the best place to inject (before closing body tag or after existing scripts)
        injection_patterns = [
            (r'(</body>)', rf'{self.injection_script}\n\1'),
            (r'(</script>\s*</body>)', rf'\1{self.injection_script}\n</body>'),
            (r'(</head>)', rf'{self.injection_script}\n\1'),
        ]
        
        modified_content = html_content
        injected = False
        
        for pattern, replacement in injection_patterns:
            if re.search(pattern, modified_content, re.IGNORECASE):
                modified_content = re.sub(pattern, replacement, modified_content, flags=re.IGNORECASE)
                injected = True
                break
        
        if not injected:
            # Fallback: append before closing HTML tag
            modified_content = re.sub(
                r'(</html>)', 
                rf'{self.injection_script}\n\1', 
                modified_content, 
                flags=re.IGNORECASE
            )
        
        return modified_content
    
    def process_file(self, filename: str) -> dict:
        """Process a single HTML file"""
        file_path = os.path.join(self.pages_dir, filename)
        backup_path = os.path.join(self.backup_dir, filename)
        
        if not os.path.exists(file_path) or not filename.endswith('.html'):
            return {"status": "skipped", "reason": "Not an HTML file or doesn't exist"}
        
        try:
            # Read original file
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Check if injection needed
            if not self.should_inject(original_content):
                return {"status": "skipped", "reason": "Already has global navigation loader"}
            
            # Create backup
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            
            # Inject navigation loader
            modified_content = self.inject_navigation_loader(original_content)
            
            # Write modified file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            return {
                "status": "success", 
                "message": f"Injected global navigation loader into {filename}",
                "backup_created": backup_path
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def process_all_files(self) -> dict:
        """Process all HTML files in the directory"""
        results = {}
        
        if not os.path.exists(self.pages_dir):
            return {"error": f"Directory {self.pages_dir} does not exist"}
        
        html_files = [f for f in os.listdir(self.pages_dir) if f.endswith('.html')]
        
        for filename in html_files:
            results[filename] = self.process_file(filename)
        
        return results
    
    def create_master_template(self):
        """Create a master template that all pages can use"""
        template_content = '''<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{PAGE_TITLE}} - AI Vision</title>
    
    <!-- Navigation CSS -->
    <link rel="stylesheet" href="/static/components/navigation.css">
    
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 80px 20px 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow: hidden;
            padding: 30px;
        }

        .page-header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid #f0f0f0;
        }

        .page-header h1 {
            color: #2c3e50;
            font-size: 2.5rem;
            margin-bottom: 10px;
        }

        .page-header p {
            color: #7f8c8d;
            font-size: 1.1rem;
        }

        .btn {
            display: inline-block;
            padding: 12px 24px;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            transition: background-color 0.3s ease;
            border: none;
            cursor: pointer;
            font-size: 1rem;
        }

        .btn:hover {
            background: #2980b9;
        }

        /* Page-specific styles will be injected here */
        {{PAGE_STYLES}}
    </style>
</head>
<body>
    <!-- Navigation will be loaded by global-nav-loader.js -->
    <div id="navigation-container"></div>

    <div class="container">
        <div class="page-header">
            <h1>{{PAGE_ICON}} {{PAGE_TITLE}}</h1>
            <p>{{PAGE_DESCRIPTION}}</p>
        </div>

        <!-- Page content -->
        <div class="page-content">
            {{PAGE_CONTENT}}
        </div>
    </div>

    <!-- Global Navigation Loader -->
    <script src="/static/components/global-nav-loader.js"></script>
    
    <!-- Page-specific scripts -->
    {{PAGE_SCRIPTS}}
    
    <script>
        // Set active page for navigation
        if (window.globalNavLoader) {
            window.globalNavLoader.setActive('{{ACTIVE_PAGE}}');
        }
    </script>
</body>
</html>'''
        
        template_path = os.path.join(self.pages_dir, "master-template.html")
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        return template_path

# Usage
if __name__ == "__main__":
    injector = HTMLNavigationInjector("c:/WorkSpace/Rest/static/pages")
    
    print("🚀 Starting HTML Navigation Injection Process")
    print("=" * 60)
    
    # Process all files
    results = injector.process_all_files()
    
    # Display results
    for filename, result in results.items():
        status = result["status"]
        if status == "success":
            print(f"✅ {filename}: {result['message']}")
        elif status == "skipped":
            print(f"⏭️  {filename}: {result['reason']}")
        else:
            print(f"❌ {filename}: {result['message']}")
    
    # Create master template
    template_path = injector.create_master_template()
    print(f"\n📄 Master template created: {template_path}")
    
    print(f"\n🎯 Summary:")
    successful = sum(1 for r in results.values() if r["status"] == "success")
    skipped = sum(1 for r in results.values() if r["status"] == "skipped")
    errors = sum(1 for r in results.values() if r["status"] == "error")
    
    print(f"   ✅ Successfully processed: {successful}")
    print(f"   ⏭️  Skipped: {skipped}")
    print(f"   ❌ Errors: {errors}")
    print(f"   📁 Backups stored in: {injector.backup_dir}")
