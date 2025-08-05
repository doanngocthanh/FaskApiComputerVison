import os
import re

def check_navigation_in_files():
    """Check navigation setup in all HTML files"""
    pages_dir = "static/pages"
    issues = []
    fixed_files = []
    
    for filename in os.listdir(pages_dir):
        if filename.endswith('.html') and filename != 'nav-test.html':
            filepath = os.path.join(pages_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for CSS link
            has_css = 'navigation.css' in content
            
            # Check for global-nav-loader.js
            has_loader = 'global-nav-loader.js' in content
            
            # Check for navigation placeholder
            has_placeholder = '<!-- Fixed Navigation Menu -->' in content
            
            # Check for empty CSS blocks (problematic)
            empty_css_blocks = len(re.findall(r'/\* Fixed Navigation Menu \*/\s*\n(\s*\n){5,}', content))
            
            print(f"\n📄 {filename}:")
            print(f"  📋 CSS Link: {'✅' if has_css else '❌'}")
            print(f"  ⚙️ JS Loader: {'✅' if has_loader else '❌'}")  
            print(f"  🏷️ Placeholder: {'✅' if has_placeholder else '❌'}")
            print(f"  🚫 Empty CSS: {'⚠️ ' + str(empty_css_blocks) if empty_css_blocks > 0 else '✅'}")
            
            if not has_css or not has_loader or not has_placeholder or empty_css_blocks > 0:
                issues.append(filename)
            else:
                fixed_files.append(filename)
    
    print(f"\n📊 SUMMARY:")
    print(f"✅ Working files: {len(fixed_files)}")
    print(f"❌ Files with issues: {len(issues)}")
    
    if issues:
        print(f"\n🔧 Files needing fixes:")
        for issue in issues:
            print(f"  - {issue}")
    
    if fixed_files:
        print(f"\n✅ Files working correctly:")
        for fixed in fixed_files:
            print(f"  - {fixed}")

if __name__ == "__main__":
    check_navigation_in_files()
