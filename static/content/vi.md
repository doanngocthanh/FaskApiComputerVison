# ChangeLog 🚀

## Version 1.4.0 - Database Configuration Management System 📊
*Ngày: 5 Tháng 8, 2025*

### ✨ Tính năng mới
- **Hệ thống quản lý cấu hình thẻ qua Database**: 
  - Tạo service `CardConfigService` để quản lý card categories và card types trong SQLite database
  - Thay thế hardcoded arrays bằng dynamic data từ database
  - Hỗ trợ CRUD operations cho card categories và card types
  - Soft delete với trường `is_active` thay vì xóa hoàn toàn

### 🔧 API Endpoints mới
- `GET /api/v1/card/categories` - Lấy danh sách loại thẻ từ database
- `GET /api/v1/card/types` - Lấy danh sách kiểu thẻ (mặt trước/sau) từ database
- `GET /api/v1/card/config` - Lấy tổng quan cấu hình thẻ từ database
- `POST /api/v1/card/categories` - Thêm loại thẻ mới
- `POST /api/v1/card/types` - Thêm kiểu thẻ mới
- `PUT /api/v1/card/categories/{id}` - Cập nhật loại thẻ
- `PUT /api/v1/card/types/{id}` - Cập nhật kiểu thẻ
- `DELETE /api/v1/card/categories/{id}` - Vô hiệu hóa loại thẻ
- `DELETE /api/v1/card/types/{id}` - Vô hiệu hóa kiểu thẻ

### 📊 Database Schema
```sql
-- Bảng card_categories
CREATE TABLE card_categories (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    nameEn TEXT NOT NULL,
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Bảng card_types  
CREATE TABLE card_types (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    nameEn TEXT NOT NULL,
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 🏗️ Cấu trúc dữ liệu mặc định
**Card Categories:**
- ID 0: Thẻ Căn Cước Công Dân (Citizens Card)
- ID 1: Giấy Phép Lái Xe (Driving License)
- ID 2: Thẻ Bảo Hiểm Y Tế (Health Insurance Card)
- ID 3: Thẻ Ngân Hàng (Bank Card)
- ID 4: Thẻ Sinh Viên (Student Card)
- ID 5: Thẻ Căn Cước Công Dân Mới (New Citizens Card)
- ID 6: Thẻ Căn Cước Công Dân Cũ (Old Citizens Card)

**Card Types:**
- ID 0: Mặt Trước (Front)
- ID 1: Mặt Sau (Back)

### 🔄 Cải tiến hệ thống
- **Auto-initialization**: Database tự động tạo bảng và dữ liệu mặc định khi khởi động
- **Singleton Pattern**: CardConfigService sử dụng singleton để đảm bảo consistency
- **Error Handling**: Comprehensive error handling cho tất cả database operations
- **Flexible Updates**: Hỗ trợ partial updates với optional parameters

### 🛠️ Technical Improvements
- Loại bỏ hardcoded arrays từ `vietnam_citizens_card_detection.py`
- Tích hợp `DBConfig` class có sẵn trong `config.py`
- Cải thiện separation of concerns với dedicated service layer
- Hỗ trợ timestamps cho audit trail

---

## Version 1.3.0 - Navigation System Unification 🧭
*Ngày: 5 Tháng 8, 2025*

### ✅ Giải quyết xung đột Navigation
- **Thống nhất hệ thống navigation**: Loại bỏ multiple conflicting navigation systems
- **Global Navigation Loader**: Sử dụng `global-nav-loader.js` cho tất cả pages
- **Consistent URL Structure**: Tất cả navigation links sử dụng `/static/pages/` paths
- **Template Router Disabled**: Tạm thời disable `static_pages.py` để tránh conflicts

### 🎨 UI/UX Improvements  
- Loại bỏ ~80 lines hardcoded navigation CSS từ `index.html`
- Consistent menu appearance across all pages
- Better user experience với unified navigation behavior

---

## Version 1.2.0 - Enhanced Card Detection & API Proxy 🎯
*Ngày: Trước đó*

### 🔍 Card Detection Enhancements
- **GPLX Support**: Thêm hỗ trợ Giấy Phép Lái Xe (Driving License)
- **Smart Classification**: Intelligent rules sử dụng OCR features
- **Multi-model Integration**: CCCD_OLD_NEW.pt + OCR_QR_CCCD.pt

### 🌐 API Proxy System
- Dynamic routing với SQLite configuration
- Proxy management interface
- Support for external API redirection

### 📅 Date Extraction
- **Enhanced MRZ Processing**: 5 regex patterns cho date extraction
- Support multiple date formats (dd/mm/yyyy, dd/mm/yy, etc.)
- Comprehensive text analysis

---



