# 🎨 Layout System - AI Vision Platform

## 📋 Tổng quan
Hệ thống layout mới đã được tạo để đồng bộ navigation và tái sử dụng giao diện trên tất cả các trang.

## 🗂️ Cấu trúc Files

### 📂 Router & Service
- `src/router/api/pages/static_pages.py` - Router chính xử lý tất cả các trang
- `src/service/TemplateService.py` - Service render template với layout chung
- `static/pages/layout.html` - Template layout chung cho tất cả trang

### 📂 Components
- `static/components/navigation.html` - Navigation component tái sử dụng
- `static/components/navigation.css` - CSS styles cho navigation
- `static/components/global-nav-loader.js` - JavaScript auto-load navigation

### 📂 Content Files
- `static/pages/home-content.html` - Nội dung trang chủ
- `static/pages/analytics.html` - Trang analytics (sử dụng layout chung)
- `static/pages/models.html` - Trang quản lý models
- `static/pages/detection.html` - Trang detection & OCR
- `static/pages/ocr.html` - Trang OCR processing
- `static/pages/upload.html` - Trang upload models
- `static/pages/monitoring.html` - Trang monitoring
- `static/pages/blog.html` - Trang blog

## 🚀 URL Routes

### 📍 Trang chính
- `http://localhost:8000/` → Home page với layout chung
- `http://localhost:8000/analytics` → Analytics Dashboard  
- `http://localhost:8000/models` → Models Management
- `http://localhost:8000/detection` → Detection & OCR
- `http://localhost:8000/ocr` → OCR Processing
- `http://localhost:8000/upload` → Upload Models
- `http://localhost:8000/monitoring` → System Monitoring
- `http://localhost:8000/blog` → Blog & Updates
- `http://localhost:8000/proxy-management` → Proxy Management

### 📍 Legacy Support (tương thích ngược)
- `http://localhost:8000/static/pages/[page].html` → Truy cập trực tiếp file HTML

## ⚙️ Template Variables

Layout template hỗ trợ các biến:
- `{{PAGE_TITLE}}` - Tiêu đề trang
- `{{PAGE_ICON}}` - Icon của trang  
- `{{PAGE_DESCRIPTION}}` - Mô tả trang
- `{{PAGE_CONTENT}}` - Nội dung chính
- `{{PAGE_STYLES}}` - CSS tùy chỉnh cho trang
- `{{PAGE_SCRIPTS}}` - JavaScript tùy chỉnh cho trang
- `{{ACTIVE_PAGE}}` - Trang hiện tại (để highlight navigation)

## 🎯 Ưu điểm

### ✅ Đồng bộ Navigation
- Navigation được load tự động trên tất cả trang
- Highlight trang hiện tại
- Responsive design cho mobile

### ✅ Tái sử dụng Layout
- Một template chung cho tất cả trang
- Không cần copy/paste CSS và HTML cơ bản
- Dễ maintain và update

### ✅ SEO & Performance  
- Clean URLs (không cần .html extension)
- Fast loading với template caching
- Proper HTML structure

### ✅ Developer Experience
- Chỉ cần tạo content HTML, không cần full page
- Template service tự động xử lý layout
- Easy to add new pages

## 📝 Cách thêm trang mới

1. **Thêm vào PAGES dict trong static_pages.py:**
```python
"new-page": {
    "title": "New Page Title",
    "icon": "🆕", 
    "description": "Description of new page",
    "template": "new-page.html"
}
```

2. **Tạo content file tại static/pages/new-page.html:**
```html
<div class="page-content">
    <h2>Your page content here</h2>
    <p>Only need to write the main content, not full HTML structure</p>
</div>
```

3. **Access tại:** `http://localhost:8000/new-page`

## 🔧 Customization

### CSS tùy chỉnh cho trang
```python
template_service.render_page(
    page_title="Custom Page",
    content_file="custom.html", 
    page_styles="""
    .custom-style {
        background: #f0f0f0;
    }
    """,
    active_page="custom"
)
```

### JavaScript tùy chỉnh cho trang
```python
template_service.render_page(
    page_title="Interactive Page",
    content_file="interactive.html",
    page_scripts="""
    <script>
        console.log('Page loaded');
    </script>
    """,
    active_page="interactive"  
)
```

## 🎉 Kết luận
Hệ thống layout mới giúp:
- Đồng bộ navigation trên tất cả trang ✅
- Tái sử dụng code và giảm duplicate ✅
- Dễ maintain và extend ✅
- URL clean và SEO friendly ✅
- Developer experience tốt hơn ✅
