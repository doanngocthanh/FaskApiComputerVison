"""
Remove Old Navigation Script
Automatically removes old navigation elements from HTML files
"""
import os
import re
from pathlib import Path

class NavigationCleaner:
    def __init__(self, pages_dir: str = "static/pages"):
        self.pages_dir = pages_dir
        self.backup_dir = os.path.join(pages_dir, "backup_before_nav_removal")
        
        # Ensure backup directory exists
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def remove_old_navigation(self, html_content: str) -> str:
        """Remove old navigation elements from HTML content"""
        
        # Pattern to match navigation section from opening nav to closing nav
        nav_patterns = [
            # Match complete nav element with class containing "nav"
            r'<nav\s+class="[^"]*nav[^"]*">.*?</nav>',
            # Match nav with fixed-nav class specifically
            r'<nav\s+class="fixed-nav">.*?</nav>',
            # Match any nav element
            r'<nav[^>]*>.*?</nav>',
            # Match comment + nav structure
            r'<!--\s*Fixed Navigation Menu\s*-->.*?</nav>',
        ]
        
        modified_content = html_content
        removed_count = 0
        
        for pattern in nav_patterns:
            matches = re.findall(pattern, modified_content, re.DOTALL | re.IGNORECASE)
            if matches:
                modified_content = re.sub(pattern, '', modified_content, flags=re.DOTALL | re.IGNORECASE)
                removed_count += len(matches)
                break  # Only remove once to avoid over-processing
        
        # Also remove standalone navigation CSS if it exists
        css_patterns = [
            r'/\*\s*Fixed Navigation Menu\s*\*/.*?/\*[^}]*\*/',
            r'\.fixed-nav\s*{[^}]*}',
            r'\.nav-container\s*{[^}]*}',
            r'\.nav-brand\s*{[^}]*}',
            r'\.nav-menu\s*{[^}]*}',
            r'\.nav-item[^}]*{[^}]*}',
            r'\.nav-toggle[^}]*{[^}]*}',
        ]
        
        # Remove navigation-specific CSS (but keep other styles)
        for css_pattern in css_patterns:
            modified_content = re.sub(css_pattern, '', modified_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove old navigation JavaScript
        js_patterns = [
            r"document\.getElementById\('navToggle'\)\.addEventListener.*?}\);",
            r"document\.querySelector\('\.nav-toggle'\)\.addEventListener.*?}\);",
        ]
        
        for js_pattern in js_patterns:
            modified_content = re.sub(js_pattern, '', modified_content, flags=re.DOTALL | re.IGNORECASE)
        
        return modified_content, removed_count
    
    def clean_file(self, filename: str) -> dict:
        """Clean navigation from a single HTML file"""
        file_path = os.path.join(self.pages_dir, filename)
        backup_path = os.path.join(self.backup_dir, filename)
        
        if not os.path.exists(file_path) or not filename.endswith('.html'):
            return {"status": "skipped", "reason": "Not an HTML file or doesn't exist"}
        
        try:
            # Read original file
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Check if old navigation exists
            has_old_nav = bool(re.search(r'<nav[^>]*class="[^"]*nav[^"]*"', original_content, re.IGNORECASE))
            
            if not has_old_nav:
                return {"status": "skipped", "reason": "No old navigation found"}
            
            # Create backup
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            
            # Remove old navigation
            cleaned_content, removed_count = self.remove_old_navigation(original_content)
            
            # Write cleaned file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            return {
                "status": "success",
                "message": f"Removed {removed_count} navigation elements from {filename}",
                "backup_created": backup_path,
                "removed_count": removed_count
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def clean_all_files(self) -> dict:
        """Clean navigation from all HTML files"""
        results = {}
        
        if not os.path.exists(self.pages_dir):
            return {"error": f"Directory {self.pages_dir} does not exist"}
        
        html_files = [f for f in os.listdir(self.pages_dir) 
                     if f.endswith('.html') and not f.startswith('master-template')]
        
        for filename in html_files:
            results[filename] = self.clean_file(filename)
        
        return results
    
    def add_navigation_css_link(self, html_content: str) -> str:
        """Add link to shared navigation CSS if not present"""
        css_link = '<link rel="stylesheet" href="/static/components/navigation.css">'
        
        if 'navigation.css' in html_content:
            return html_content
        
        # Add CSS link in head section
        head_pattern = r'(<head[^>]*>)'
        if re.search(head_pattern, html_content, re.IGNORECASE):
            html_content = re.sub(
                head_pattern, 
                rf'\1\n    {css_link}', 
                html_content, 
                flags=re.IGNORECASE
            )
        
        return html_content
    
    def complete_cleanup(self) -> dict:
        """Complete cleanup with CSS link addition"""
        results = self.clean_all_files()
        
        # Add CSS links to cleaned files
        for filename, result in results.items():
            if result.get("status") == "success":
                file_path = os.path.join(self.pages_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    updated_content = self.add_navigation_css_link(content)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    
                    result["css_added"] = True
                except Exception as e:
                    result["css_error"] = str(e)
        
        return results

# Usage
if __name__ == "__main__":
    cleaner = NavigationCleaner("c:/WorkSpace/Rest/static/pages")
    
    print("🧹 Starting Navigation Cleanup Process")
    print("=" * 60)
    
    # Complete cleanup
    results = cleaner.complete_cleanup()
    
    # Display results
    for filename, result in results.items():
        status = result["status"]
        if status == "success":
            removed = result.get("removed_count", 0)
            css_added = "✅ CSS linked" if result.get("css_added") else "❌ CSS link failed"
            print(f"✅ {filename}: Removed {removed} nav elements | {css_added}")
        elif status == "skipped":
            print(f"⏭️  {filename}: {result['reason']}")
        else:
            print(f"❌ {filename}: {result['message']}")
    
    print(f"\n🎯 Summary:")
    successful = sum(1 for r in results.values() if r["status"] == "success")
    skipped = sum(1 for r in results.values() if r["status"] == "skipped")
    errors = sum(1 for r in results.values() if r["status"] == "error")
    total_removed = sum(r.get("removed_count", 0) for r in results.values() if r["status"] == "success")
    
    print(f"   ✅ Successfully cleaned: {successful}")
    print(f"   ⏭️  Skipped: {skipped}")
    print(f"   ❌ Errors: {errors}")
    print(f"   🗑️  Total nav elements removed: {total_removed}")
    print(f"   📁 Backups stored in: {cleaner.backup_dir}")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Test HTML files to ensure navigation loads correctly")
    print(f"   2. Check that /static/components/navigation.css is accessible")
    print(f"   3. Verify /static/components/global-nav-loader.js works properly")
