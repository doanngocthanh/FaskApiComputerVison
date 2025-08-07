"""
API Management Service
Quản lý hiển thị/ẩn các API endpoints trong Swagger documentation
"""

import json
import logging
from typing import List, Dict, Any, Optional
from config import DBConfig
from datetime import datetime

class ApiManagementService:
    def __init__(self):
        self.db_config = DBConfig()
        self.init_database()
        
    def init_database(self):
        """Khởi tạo bảng api_endpoints nếu chưa tồn tại"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS api_endpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            endpoint_path TEXT NOT NULL,
            endpoint_method TEXT NOT NULL,
            router_name TEXT NOT NULL,
            endpoint_name TEXT NOT NULL,
            description TEXT,
            is_visible BOOLEAN DEFAULT 1,
            is_public BOOLEAN DEFAULT 1,
            tags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(endpoint_path, endpoint_method)
        )
        """
        
        create_settings_table = """
        CREATE TABLE IF NOT EXISTS api_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setting_key TEXT NOT NULL UNIQUE,
            setting_value TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        try:
            # Kiểm tra xem bảng có tồn tại và có cấu trúc cũ không
            existing_table = self.db_config.fetch_one(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='api_endpoints'"
            )
            
            if existing_table and 'UNIQUE(endpoint_path, endpoint_method)' not in existing_table[0]:
                print("🔄 Updating database schema for composite unique constraint...")
                # Backup dữ liệu cũ
                backup_data = self.db_config.fetch_all("""
                    SELECT endpoint_path, endpoint_method, router_name, endpoint_name, 
                           description, is_visible, is_public, tags
                    FROM api_endpoints
                """)
                
                # Xóa bảng cũ
                self.db_config.execute_query("DROP TABLE IF EXISTS api_endpoints")
                
                # Tạo bảng mới với constraint đúng
                self.db_config.execute_query(create_table_query)
                
                # Khôi phục dữ liệu (chỉ giữ lại unique combinations)
                unique_endpoints = {}
                for row in backup_data:
                    key = (row[0], row[1])  # (endpoint_path, endpoint_method)
                    if key not in unique_endpoints:
                        unique_endpoints[key] = row
                
                for endpoint_data in unique_endpoints.values():
                    try:
                        self.db_config.execute_query("""
                            INSERT INTO api_endpoints 
                            (endpoint_path, endpoint_method, router_name, endpoint_name, 
                             description, is_visible, is_public, tags)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, endpoint_data)
                    except:
                        pass  # Skip duplicates
                
                print("✅ Database schema updated successfully")
            else:
                # Tạo bảng mới nếu chưa tồn tại
                self.db_config.execute_query(create_table_query)
            
            self.db_config.execute_query(create_settings_table)
            
            # Thêm các setting mặc định
            self.init_default_settings()
            
            logging.info("✅ API Management database initialized successfully")
        except Exception as e:
            logging.error(f"❌ Failed to initialize API Management database: {e}")
            print(f"❌ Database initialization error: {e}")
            raise
            
    def init_default_settings(self):
        """Khởi tạo các setting mặc định"""
        default_settings = [
            ("swagger_title", "FastAPI Computer Vision System", "Tiêu đề Swagger documentation"),
            ("swagger_description", "API for Computer Vision and OCR Services", "Mô tả Swagger documentation"),
            ("swagger_version", "1.0.0", "Phiên bản API"),
            ("show_error_responses", "false", "Hiển thị error response schemas"),
            ("show_crud_endpoints", "false", "Hiển thị CRUD endpoints"),
            ("show_internal_endpoints", "false", "Hiển thị internal endpoints"),
            ("documentation_mode", "clean", "Chế độ documentation: minimal, clean, full")
        ]
        
        for key, value, desc in default_settings:
            existing = self.db_config.fetch_one(
                "SELECT id FROM api_settings WHERE setting_key = ?", (key,)
            )
            if not existing:
                self.db_config.execute_query(
                    "INSERT INTO api_settings (setting_key, setting_value, description) VALUES (?, ?, ?)",
                    (key, value, desc)
                )
    
    def discover_endpoints(self) -> List[Dict[str, Any]]:
        """Tự động phát hiện tất cả endpoints trong hệ thống"""
        import os
        import importlib
        import inspect
        import sys
        from fastapi import APIRouter
        from fastapi.routing import APIRoute
        
        endpoints = []
        
        # Lấy đường dẫn gốc của project
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        api_dir = os.path.join(project_root, "src", "router", "api")
        
        print(f"🔍 Scanning API directory: {api_dir}")
        
        def scan_directory(directory, module_prefix=""):
            if not os.path.exists(directory):
                print(f"⚠️ Directory not found: {directory}")
                return
                
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                
                if os.path.isdir(item_path) and not item.startswith("__"):
                    # Quét thư mục con
                    new_prefix = f"{module_prefix}.{item}" if module_prefix else item
                    scan_directory(item_path, new_prefix)
                    
                elif item.endswith(".py") and not item.startswith("__"):
                    module_name = item[:-3]
                    try:
                        # Xây dựng module path
                        if module_prefix:
                            full_module_path = f"src.router.api.{module_prefix}.{module_name}"
                        else:
                            full_module_path = f"src.router.api.{module_name}"
                        
                        print(f"📂 Attempting to import: {full_module_path}")
                        
                        # Import module
                        module = importlib.import_module(full_module_path)
                        
                        if hasattr(module, 'router') and isinstance(module.router, APIRouter):
                            router = module.router
                            router_name = f"{module_prefix}.{module_name}" if module_prefix else module_name
                            
                            print(f"✅ Found router: {router_name}")
                            
                            # Lấy thông tin từ router
                            for route in router.routes:
                                if isinstance(route, APIRoute) and hasattr(route, 'methods'):
                                    for method in route.methods:
                                        if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                                            # Lấy thông tin chi tiết từ route
                                            endpoint_function = route.endpoint
                                            function_name = getattr(endpoint_function, '__name__', 'unknown')
                                            
                                            # Lấy summary và description từ decorator hoặc docstring
                                            summary = getattr(route, 'summary', '')
                                            description = getattr(route, 'description', '')
                                            
                                            if not summary and hasattr(endpoint_function, '__doc__'):
                                                doc = endpoint_function.__doc__
                                                if doc:
                                                    lines = doc.strip().split('\n')
                                                    summary = lines[0].strip() if lines else ''
                                            
                                            endpoint_info = {
                                                'endpoint_path': route.path,
                                                'endpoint_method': method.upper(),
                                                'router_name': router_name,
                                                'endpoint_name': route.name or function_name,
                                                'description': summary or description or '',
                                                'tags': ','.join(router.tags) if hasattr(router, 'tags') and router.tags else router_name,
                                                'is_visible': True,
                                                'is_public': True,
                                                'function_name': function_name
                                            }
                                            endpoints.append(endpoint_info)
                                            print(f"  📍 {method.upper()} {route.path} -> {function_name}")
                            
                    except ImportError as e:
                        print(f"⚠️ Import error for {full_module_path}: {e}")
                        continue
                    except Exception as e:
                        print(f"⚠️ Error scanning {full_module_path}: {e}")
                        continue
        
        try:
            scan_directory(api_dir)
            print(f"✅ Discovered {len(endpoints)} endpoints")
        except Exception as e:
            print(f"❌ Failed to discover endpoints: {e}")
            logging.error(f"❌ Failed to discover endpoints: {e}")
            
        return endpoints
    
    def sync_endpoints(self) -> Dict[str, Any]:
        """Đồng bộ endpoints từ code vào database"""
        discovered_endpoints = self.discover_endpoints()
        
        stats = {
            'total_discovered': len(discovered_endpoints),
            'new_endpoints': 0,
            'updated_endpoints': 0,
            'existing_endpoints': 0,
            'errors': 0
        }
        
        for endpoint in discovered_endpoints:
            try:
                existing = self.db_config.fetch_one(
                    "SELECT id, is_visible, is_public FROM api_endpoints WHERE endpoint_path = ? AND endpoint_method = ?",
                    (endpoint['endpoint_path'], endpoint['endpoint_method'])
                )
                
                if existing:
                    # Cập nhật thông tin endpoint (giữ nguyên is_visible và is_public)
                    self.db_config.execute_query("""
                        UPDATE api_endpoints SET 
                            router_name = ?, endpoint_name = ?, description = ?, 
                            tags = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE endpoint_path = ? AND endpoint_method = ?
                    """, (
                        endpoint['router_name'], endpoint['endpoint_name'], 
                        endpoint['description'], endpoint['tags'],
                        endpoint['endpoint_path'], endpoint['endpoint_method']
                    ))
                    stats['updated_endpoints'] += 1
                    print(f"🔄 Updated: {endpoint['endpoint_method']} {endpoint['endpoint_path']}")
                else:
                    # Thêm endpoint mới
                    self.db_config.execute_query("""
                        INSERT INTO api_endpoints 
                        (endpoint_path, endpoint_method, router_name, endpoint_name, description, tags, is_visible, is_public)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        endpoint['endpoint_path'], endpoint['endpoint_method'],
                        endpoint['router_name'], endpoint['endpoint_name'],
                        endpoint['description'], endpoint['tags'],
                        endpoint['is_visible'], endpoint['is_public']
                    ))
                    stats['new_endpoints'] += 1
                    print(f"✅ Added: {endpoint['endpoint_method']} {endpoint['endpoint_path']}")
                    
            except Exception as e:
                stats['errors'] += 1
                print(f"❌ Error syncing {endpoint['endpoint_method']} {endpoint['endpoint_path']}: {e}")
                logging.error(f"Error syncing endpoint {endpoint}: {e}")
                continue
        
        print(f"📊 Sync Summary: {stats['new_endpoints']} new, {stats['updated_endpoints']} updated, {stats['errors']} errors")
        return stats
    
    def get_all_endpoints(self) -> List[Dict[str, Any]]:
        """Lấy tất cả endpoints từ database"""
        results = self.db_config.fetch_all("""
            SELECT id, endpoint_path, endpoint_method, router_name, endpoint_name, 
                   description, is_visible, is_public, tags, created_at, updated_at
            FROM api_endpoints 
            ORDER BY router_name, endpoint_path, endpoint_method
        """)
        
        endpoints = []
        for row in results:
            endpoints.append({
                'id': row[0],
                'endpoint_path': row[1],
                'endpoint_method': row[2],
                'router_name': row[3],
                'endpoint_name': row[4],
                'description': row[5],
                'is_visible': bool(row[6]),
                'is_public': bool(row[7]),
                'tags': row[8].split(',') if row[8] else [],
                'created_at': row[9],
                'updated_at': row[10]
            })
            
        return endpoints
    
    def get_visible_endpoints(self) -> List[Dict[str, Any]]:
        """Lấy các endpoints được set hiển thị"""
        results = self.db_config.fetch_all("""
            SELECT endpoint_path, endpoint_method, router_name, endpoint_name, description, tags
            FROM api_endpoints 
            WHERE is_visible = 1 AND is_public = 1
            ORDER BY router_name, endpoint_path
        """)
        
        endpoints = []
        for row in results:
            endpoints.append({
                'endpoint_path': row[0],
                'endpoint_method': row[1],
                'router_name': row[2],
                'endpoint_name': row[3],
                'description': row[4],
                'tags': row[5].split(',') if row[5] else []
            })
            
        return endpoints
    
    def update_endpoint_visibility(self, endpoint_id: int, is_visible: bool, is_public: bool = None) -> bool:
        """Cập nhật trạng thái hiển thị của endpoint"""
        try:
            if is_public is not None:
                self.db_config.execute_query("""
                    UPDATE api_endpoints 
                    SET is_visible = ?, is_public = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (is_visible, is_public, endpoint_id))
            else:
                self.db_config.execute_query("""
                    UPDATE api_endpoints 
                    SET is_visible = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (is_visible, endpoint_id))
            return True
        except Exception as e:
            logging.error(f"Failed to update endpoint visibility: {e}")
            return False
    
    def bulk_update_visibility(self, updates: List[Dict[str, Any]]) -> Dict[str, int]:
        """Cập nhật hàng loạt trạng thái hiển thị"""
        stats = {'updated': 0, 'failed': 0}
        
        print(f"🔄 Processing {len(updates)} bulk updates...")
        
        for update in updates:
            try:
                endpoint_id = update.get('id')
                is_visible = update.get('is_visible')
                is_public = update.get('is_public')
                
                if endpoint_id is None or is_visible is None:
                    print(f"❌ Invalid update data: {update}")
                    stats['failed'] += 1
                    continue
                
                success = self.update_endpoint_visibility(endpoint_id, is_visible, is_public)
                if success:
                    stats['updated'] += 1
                    print(f"✅ Updated endpoint {endpoint_id}: visible={is_visible}, public={is_public}")
                else:
                    stats['failed'] += 1
                    print(f"❌ Failed to update endpoint {endpoint_id}")
                    
            except Exception as e:
                stats['failed'] += 1
                print(f"❌ Error processing update {update}: {e}")
                logging.error(f"Error in bulk update: {e}")
                
        print(f"📊 Bulk update completed: {stats['updated']} updated, {stats['failed']} failed")
        return stats
    
    def get_setting(self, key: str) -> Optional[str]:
        """Lấy giá trị setting"""
        result = self.db_config.fetch_one(
            "SELECT setting_value FROM api_settings WHERE setting_key = ?", (key,)
        )
        return result[0] if result else None
    
    def update_setting(self, key: str, value: str) -> bool:
        """Cập nhật setting"""
        try:
            existing = self.db_config.fetch_one(
                "SELECT id FROM api_settings WHERE setting_key = ?", (key,)
            )
            
            if existing:
                self.db_config.execute_query("""
                    UPDATE api_settings 
                    SET setting_value = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE setting_key = ?
                """, (value, key))
            else:
                self.db_config.execute_query("""
                    INSERT INTO api_settings (setting_key, setting_value)
                    VALUES (?, ?)
                """, (key, value))
            return True
        except Exception as e:
            logging.error(f"Failed to update setting {key}: {e}")
            return False
    
    def get_all_settings(self) -> Dict[str, str]:
        """Lấy tất cả settings"""
        results = self.db_config.fetch_all(
            "SELECT setting_key, setting_value FROM api_settings"
        )
        return {row[0]: row[1] for row in results}
    
    def get_swagger_config(self) -> Dict[str, Any]:
        """Lấy cấu hình Swagger từ database"""
        settings = self.get_all_settings()
        visible_endpoints = self.get_visible_endpoints()
        
        return {
            'title': settings.get('swagger_title', 'FastAPI Application'),
            'description': settings.get('swagger_description', 'API Documentation'),
            'version': settings.get('swagger_version', '1.0.0'),
            'show_error_responses': settings.get('show_error_responses', 'false').lower() == 'true',
            'show_crud_endpoints': settings.get('show_crud_endpoints', 'false').lower() == 'true',
            'show_internal_endpoints': settings.get('show_internal_endpoints', 'false').lower() == 'true',
            'documentation_mode': settings.get('documentation_mode', 'clean'),
            'visible_endpoints': visible_endpoints,
            'total_visible_endpoints': len(visible_endpoints)
        }
    
    def invalidate_swagger_cache(self):
        """Vô hiệu hóa cache Swagger để force reload từ database"""
        try:
            from src.service.SwaggerConfigService import swagger_config_service
            if hasattr(swagger_config_service, 'app') and swagger_config_service.app:
                # Reset OpenAPI schema để force regeneration
                swagger_config_service.app.openapi_schema = None
                print("🔄 Swagger cache invalidated - will reload from database")
                return True
        except Exception as e:
            print(f"⚠️ Failed to invalidate swagger cache: {e}")
            return False
        return False
    
    def update_endpoint_visibility_with_refresh(self, endpoint_id: int, is_visible: bool, is_public: bool = None) -> bool:
        """Cập nhật trạng thái hiển thị của endpoint và refresh Swagger"""
        success = self.update_endpoint_visibility(endpoint_id, is_visible, is_public)
        if success:
            self.invalidate_swagger_cache()
        return success
    
    def bulk_update_visibility_with_refresh(self, updates: List[Dict[str, Any]]) -> Dict[str, int]:
        """Cập nhật hàng loạt trạng thái hiển thị và refresh Swagger"""
        stats = self.bulk_update_visibility(updates)
        if stats['updated'] > 0:
            self.invalidate_swagger_cache()
        return stats
    
    def update_setting_with_refresh(self, key: str, value: str) -> bool:
        """Cập nhật setting và refresh Swagger"""
        success = self.update_setting(key, value)
        if success:
            self.invalidate_swagger_cache()
        return success
    
    def delete_endpoint(self, endpoint_id: int) -> bool:
        """Xóa endpoint khỏi database"""
        try:
            self.db_config.execute_query(
                "DELETE FROM api_endpoints WHERE id = ?", (endpoint_id,)
            )
            return True
        except Exception as e:
            logging.error(f"Failed to delete endpoint: {e}")
            return False
    
    def search_endpoints(self, search_term: str, router_name: str = None, method: str = None) -> List[Dict[str, Any]]:
        """Tìm kiếm endpoints"""
        query = """
            SELECT id, endpoint_path, endpoint_method, router_name, endpoint_name, 
                   description, is_visible, is_public, tags
            FROM api_endpoints 
            WHERE (endpoint_path LIKE ? OR endpoint_name LIKE ? OR description LIKE ?)
        """
        params = [f"%{search_term}%", f"%{search_term}%", f"%{search_term}%"]
        
        if router_name:
            query += " AND router_name = ?"
            params.append(router_name)
            
        if method:
            query += " AND endpoint_method = ?"
            params.append(method.upper())
            
        query += " ORDER BY router_name, endpoint_path"
        
        results = self.db_config.fetch_all(query, params)
        
        endpoints = []
        for row in results:
            endpoints.append({
                'id': row[0],
                'endpoint_path': row[1],
                'endpoint_method': row[2],
                'router_name': row[3],
                'endpoint_name': row[4],
                'description': row[5],
                'is_visible': bool(row[6]),
                'is_public': bool(row[7]),
                'tags': row[8].split(',') if row[8] else []
            })
            
        return endpoints

# Global instance
api_management = ApiManagementService()
