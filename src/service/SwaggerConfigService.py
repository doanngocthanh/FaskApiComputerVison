"""
Swagger Configuration Service
Service để quản lý cấu hình Swagger documentation
"""

from typing import Dict, Any, Optional, List
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


class SwaggerConfigService:
    """Service để cấu hình Swagger documentation"""
    
    def __init__(self):
        self.excluded_tags = []
        self.excluded_paths = []
        self.custom_responses = {}
        self.app = None  # Store reference to FastAPI app
        
    def set_app(self, app: FastAPI):
        """Lưu reference đến FastAPI app"""
        self.app = app
        
    def configure_minimal_swagger(self, app: FastAPI) -> None:
        """
        Cấu hình Swagger tối giản - loại bỏ error responses không cần thiết
        """
        def custom_openapi():
            if app.openapi_schema:
                return app.openapi_schema
                
            openapi_schema = get_openapi(
                title="AI Processing API",
                version="1.0.0",
                description="AI Processing API with minimal documentation",
                routes=app.routes,
            )
            
            # Loại bỏ HTTPValidationError schema
            if "components" in openapi_schema:
                if "schemas" in openapi_schema["components"]:
                    # Loại bỏ các schema error không cần thiết
                    schemas_to_remove = [
                        "HTTPValidationError",
                        "ValidationError"
                    ]
                    
                    for schema_name in schemas_to_remove:
                        if schema_name in openapi_schema["components"]["schemas"]:
                            del openapi_schema["components"]["schemas"][schema_name]
                    
                    # Loại bỏ các ErrorResponse schemas
                    error_schemas = [k for k in openapi_schema["components"]["schemas"].keys() 
                                   if "ErrorResponse" in k]
                    for schema_name in error_schemas:
                        del openapi_schema["components"]["schemas"][schema_name]
            
            # Cấu hình paths - chỉ giữ lại các endpoint cần thiết
            if "paths" in openapi_schema:
                filtered_paths = {}
                
                for path, methods in openapi_schema["paths"].items():
                    # Chỉ giữ lại các endpoint không bị loại trừ
                    if not any(excluded in path for excluded in self.excluded_paths):
                        filtered_methods = {}
                        
                        for method, details in methods.items():
                            # Loại bỏ error responses
                            if "responses" in details:
                                # Chỉ giữ lại 200 response
                                filtered_responses = {}
                                if "200" in details["responses"]:
                                    filtered_responses["200"] = details["responses"]["200"]
                                    # Loại bỏ schema trong 200 response
                                    if "content" in filtered_responses["200"]:
                                        for content_type in filtered_responses["200"]["content"]:
                                            if "schema" in filtered_responses["200"]["content"][content_type]:
                                                # Thay thế schema phức tạp bằng schema đơn giản
                                                filtered_responses["200"]["content"][content_type]["schema"] = {
                                                    "type": "object",
                                                    "description": "Success response"
                                                }
                                
                                details["responses"] = filtered_responses
                            
                            # Loại bỏ các tags bị loại trừ
                            if "tags" in details:
                                if not any(tag in self.excluded_tags for tag in details["tags"]):
                                    filtered_methods[method] = details
                            else:
                                filtered_methods[method] = details
                        
                        if filtered_methods:
                            filtered_paths[path] = filtered_methods
                
                openapi_schema["paths"] = filtered_paths
            
            app.openapi_schema = openapi_schema
            return app.openapi_schema
        
        app.openapi = custom_openapi
    
    def configure_clean_swagger(self, app: FastAPI) -> None:
        """
        Cấu hình Swagger hoàn toàn sạch - chỉ hiển thị endpoints cần thiết
        """
        def clean_openapi():
            if app.openapi_schema:
                return app.openapi_schema
            
            # Tạo schema tùy chỉnh hoàn toàn
            openapi_schema = {
                "openapi": "3.1.0",
                "info": {
                    "title": "AI Processing API",
                    "description": "AI Processing API with clean documentation",
                    "version": "1.0.0"
                },
                "paths": {},
                "components": {
                    "schemas": {}
                }
            }
            
            # Chỉ thêm các endpoints được phép
            allowed_paths = [
                "/",
                "/api/v1/card/categories",
                "/api/v1/card/types", 
                "/api/v1/card/config",
                "/api/v1/card/demo",
                "/api/v1/card/detect",
                "/api/v1/mrz/parse/{mrz_string}",
                "/api/v1/mrz/demo",
                "/api/v1/system/languages",
                "/api/v1/system/models/list",
                "/api/v1/system/models/{model_id}/detect",
                "/api/v1/system/models/{model_id}/detect-and-ocr",
                "/api/v1/system/ocr/text-only",
                "/api/v1/system/models/{model_id}/info"
            ]
            
            # Lấy openapi schema gốc
            original_schema = get_openapi(
                title="AI Processing API",
                version="1.0.0", 
                description="AI Processing API",
                routes=app.routes,
            )
            
            # Chỉ copy các paths được phép
            if "paths" in original_schema:
                for path in allowed_paths:
                    if path in original_schema["paths"]:
                        # Copy path nhưng chỉ giữ response 200
                        path_data = original_schema["paths"][path].copy()
                        
                        for method in path_data:
                            if "responses" in path_data[method]:
                                # Chỉ giữ response 200
                                new_responses = {}
                                if "200" in path_data[method]["responses"]:
                                    new_responses["200"] = {
                                        "description": "Successful Response",
                                        "content": {
                                            "application/json": {
                                                "schema": {
                                                    "type": "object",
                                                    "description": "Success response"
                                                }
                                            }
                                        }
                                    }
                                path_data[method]["responses"] = new_responses
                        
                        openapi_schema["paths"][path] = path_data
            
            app.openapi_schema = openapi_schema
            return app.openapi_schema
        
        app.openapi = clean_openapi
    
    def exclude_tags(self, tags: List[str]) -> None:
        """Loại trừ các tags khỏi Swagger documentation"""
        self.excluded_tags.extend(tags)
    
    def exclude_paths(self, paths: List[str]) -> None:
        """Loại trừ các paths khỏi Swagger documentation"""
        self.excluded_paths.extend(paths)
    
    def configure_database_driven_swagger(self, app: FastAPI) -> None:
        """
        Cấu hình Swagger dựa trên database settings
        """
        self.app = app  # Store app reference
        
        def database_openapi():
            # Always reload from database - no caching
            app.openapi_schema = None
            
            # Import API Management Service để lấy config từ database
            try:
                from src.service.ApiManagementService import api_management
                config = api_management.get_swagger_config()
                print(f"🔧 Loading Swagger config from database: {config['total_visible_endpoints']} visible endpoints")
            except Exception as e:
                print(f"Failed to load API management config: {e}")
                # Fallback to default config
                config = {
                    'title': 'FastAPI Computer Vision System',
                    'description': 'API for Computer Vision and OCR Services',
                    'version': '1.0.0',
                    'visible_endpoints': [],
                    'show_error_responses': False,
                    'documentation_mode': 'clean'
                }
            
            openapi_schema = get_openapi(
                title=config.get('title', 'FastAPI Application'),
                version=config.get('version', '1.0.0'),
                description=config.get('description', 'API Documentation'),
                routes=app.routes,
            )
            
            # Lọc endpoints dựa trên database settings
            visible_endpoints = config.get('visible_endpoints', [])
            
            # Nếu có cấu hình endpoints trong database, chỉ hiển thị những endpoints được chọn
            if visible_endpoints:
                visible_paths = set()
                for endpoint in visible_endpoints:
                    visible_paths.add(endpoint['endpoint_path'])
                
                filtered_paths = {}
                for path_name, path_item in openapi_schema.get("paths", {}).items():
                    if path_name in visible_paths:
                        filtered_paths[path_name] = path_item
                        
                openapi_schema["paths"] = filtered_paths
                print(f"📄 Filtered to {len(filtered_paths)} visible paths in Swagger")
            
            # Xử lý response schemas dựa trên settings
            if not config.get('show_error_responses', False):
                for path_name, path_item in openapi_schema.get("paths", {}).items():
                    for method_name, operation in path_item.items():
                        if isinstance(operation, dict) and "responses" in operation:
                            responses = operation["responses"]
                            
                            # Chỉ giữ lại success responses
                            cleaned_responses = {}
                            for status_code in ["200", "201", "202", "204"]:
                                if status_code in responses:
                                    cleaned_responses[status_code] = responses[status_code]
                            
                            if cleaned_responses:
                                operation["responses"] = cleaned_responses
                            else:
                                # Nếu không có success response, tạo default
                                operation["responses"] = {
                                    "200": {
                                        "description": "Successful Response",
                                        "content": {
                                            "application/json": {
                                                "schema": {"type": "object"}
                                            }
                                        }
                                    }
                                }
            
            # Làm sạch components schemas
            if "components" in openapi_schema and "schemas" in openapi_schema["components"]:
                schemas = openapi_schema["components"]["schemas"]
                
                # Loại bỏ error schemas nếu setting không cho phép
                if not config.get('show_error_responses', False):
                    schemas_to_remove = []
                    for schema_name in schemas.keys():
                        if any(error_term in schema_name.lower() for error_term in 
                              ['error', 'exception', 'validation', 'http']):
                            schemas_to_remove.append(schema_name)
                    
                    for schema_name in schemas_to_remove:
                        schemas.pop(schema_name, None)
            
            return openapi_schema
            
        app.openapi = database_openapi

    def configure_production_swagger(self, app: FastAPI) -> None:
        """
        Cấu hình Swagger cho production - chỉ hiển thị các API cần thiết
        """
        # Loại trừ các tags CRUD
        self.exclude_tags([
            "YOLO Detection + EasyOCR Service",  # Có thể loại bỏ hoàn toàn
        ])
        
        # Loại trừ các paths không cần thiết
        self.exclude_paths([
            "/api/v1/system/models/upload",
            "/api/v1/system/models/{model_id}/test-sample",
            "/api/v1/system/models/{model_id}/update-class-names",
            "/api/v1/card/categories",
            "/api/v1/card/types",
        ])
        
        self.configure_clean_swagger(app)
    
    def configure_development_swagger(self, app: FastAPI) -> None:
        """
        Cấu hình Swagger cho development - hiển thị tất cả nhưng clean
        """
        self.configure_clean_swagger(app)
    
    def disable_swagger(self, app: FastAPI) -> None:
        """
        Tắt hoàn toàn Swagger documentation
        """
        app.docs_url = None
        app.redoc_url = None
        app.openapi_url = None


# Singleton instance
swagger_config_service = SwaggerConfigService()
