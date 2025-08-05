"""
OCR Request Logging Middleware
Tự động log tất cả OCR requests và responses với UUID tracking
"""

import time
import json
import uuid
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import logging

# Setup logging
logger = logging.getLogger(__name__)

class OCRLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, logging_manager=None):
        super().__init__(app)
        self.logging_manager = logging_manager
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Skip logging for non-OCR endpoints
        if not self._should_log_request(request):
            return await call_next(request)
        
        start_time = time.time()
        
        # Prepare request data
        request_data = await self._prepare_request_data(request, request_id)
        
        # Process request
        response = None
        error_message = ""
        success = True
        response_body_data = None
        
        try:
            response = await call_next(request)
            
            # Capture response body for logging (only for JSON responses)
            if (hasattr(response, 'media_type') and response.media_type == 'application/json') or \
               (hasattr(response, 'headers') and response.headers.get('content-type', '').startswith('application/json')):
                
                # For JSON responses, try to capture the body
                if hasattr(response, 'body'):
                    try:
                        # Get body as bytes and decode
                        if isinstance(response.body, bytes):
                            body_str = response.body.decode('utf-8')
                        else:
                            body_str = str(response.body)
                        
                        # Parse JSON
                        response_body_data = json.loads(body_str)
                    except Exception as e:
                        logger.debug(f"Could not capture response body: {e}")
            
            # Check if response indicates error
            if hasattr(response, 'status_code') and response.status_code >= 400:
                success = False
                if response_body_data:
                    error_message = response_body_data.get('detail', f'HTTP {response.status_code}')
                else:
                    error_message = f'HTTP {response.status_code}'
            
        except Exception as e:
            success = False
            error_message = str(e)
            response = JSONResponse(
                status_code=500,
                content={"detail": f"Internal server error: {str(e)}"}
            )
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        # Prepare response data with captured body
        response_data = self._prepare_response_data(response, processing_time, response_body_data)
        
        # Combine request and response data
        log_data = {
            **request_data,
            **response_data,
            'success': success,
            'error_message': error_message,
            'processing_time_ms': processing_time,
            'request_id': request_id,
            'timestamp': time.time()
        }
        
        # Log the request
        try:
            if self.logging_manager:
                self.logging_manager.log_request(log_data)
                
                # Log performance metric
                if success:
                    self.logging_manager.log_performance(
                        'processing_time',
                        processing_time,
                        {
                            'endpoint': request_data['endpoint'],
                            'method': request_data['method'],
                            'languages': request_data.get('languages', [])
                        }
                    )
                else:
                    # Log error
                    self.logging_manager.log_error({
                        'endpoint': request_data['endpoint'],
                        'method': request_data['method'],
                        'error': error_message,
                        'processing_time_ms': processing_time,
                        'ip_address': request_data.get('ip_address', ''),
                        'user_agent': request_data.get('user_agent', '')
                    })
        except Exception as log_error:
            logger.error(f"Failed to log request: {log_error}")
        
        return response
    
    def _should_log_request(self, request: Request) -> bool:
        """Xác định có nên log request này không"""
        path = request.url.path
        
        # Log các OCR endpoints
        ocr_endpoints = [
            '/api/v1/yolo-ocr/',
            '/api/v1/ocr/',
        ]
        
        # Skip monitoring endpoints để tránh recursive logging
        skip_endpoints = [
            '/api/v1/monitoring/',
            '/docs',
            '/redoc',
            '/openapi.json',
            '/favicon.ico'
        ]
        
        # Skip nếu là monitoring endpoint
        for skip in skip_endpoints:
            if path.startswith(skip):
                return False
        
        # ENABLE LOGGING for all OCR endpoints - only log response, not request body
        # This avoids consuming multipart/form-data and causing 422 errors
        # if request.headers.get('content-type', '').startswith('multipart/form-data'):
        #     return False
        
        # Log nếu là OCR endpoint
        for ocr in ocr_endpoints:
            if path.startswith(ocr):
                return True
        
        return False
    
    async def _prepare_request_data(self, request: Request, request_id: str) -> dict:
        """Chuẩn bị dữ liệu request để log"""
        data = {
            'request_id': request_id,
            'endpoint': request.url.path,
            'method': request.method,
            'ip_address': self._get_client_ip(request),
            'user_agent': request.headers.get('user-agent', ''),
            'file_size': 0,
            'languages': [],
            'model_id': '',
            'total_detections': 0,
            'total_texts': 0,
            'confidence_threshold': 0.0,
            'request_body': {}
        }
        
        # Capture request parameters từ query và form
        request_params = {}
        
        # Query parameters
        if request.query_params:
            request_params['query_params'] = dict(request.query_params)
        
        # Path parameters
        if hasattr(request, 'path_params') and request.path_params:
            request_params['path_params'] = dict(request.path_params)
            # Extract model_id từ path
            if 'model_id' in request.path_params:
                data['model_id'] = request.path_params['model_id']
        
        # Extract basic request info without reading multipart body
        # This prevents consuming the request stream and causing 422 errors
        
        # Query parameters
        if request.query_params:
            request_params['query_params'] = dict(request.query_params)
        
        # Path parameters
        if hasattr(request, 'path_params') and request.path_params:
            request_params['path_params'] = dict(request.path_params)
            # Extract model_id từ path
            if 'model_id' in request.path_params:
                data['model_id'] = request.path_params['model_id']
        
        # Extract model_id from URL path if present
        path_parts = request.url.path.split('/')
        if 'models' in path_parts:
            try:
                model_index = path_parts.index('models')
                if model_index + 1 < len(path_parts):
                    data['model_id'] = path_parts[model_index + 1]
            except:
                pass
        
        # Note: We skip reading form data to avoid consuming multipart stream
        if request.headers.get('content-type', '').startswith('multipart/form-data'):
            request_params['note'] = 'multipart/form-data - request body not logged to prevent 422 errors'
            data['file_size'] = 0  # Will be extracted from response if available
        data['request_body'] = request_params
        return data
    
    def _prepare_response_data(self, response: Response, processing_time: int, response_body_data: dict = None) -> dict:
        """Chuẩn bị dữ liệu response để log"""
        data = {
            'total_detections': 0,
            'total_texts': 0,
            'response_body': {},
            'ocr_results': [],
            'detection_results': []
        }
        
        # Use captured response body data if available
        if response_body_data:
            try:
                # Store full response (excluding large binary data)
                filtered_response = {}
                for key, value in response_body_data.items():
                    if key in ['cropped_images']:  # Skip large binary data
                        filtered_response[key] = f"[{len(value)} items]" if isinstance(value, list) else "[binary data]"
                    else:
                        filtered_response[key] = value
                
                data['response_body'] = filtered_response
                
                # Extract detection and OCR counts
                data['total_detections'] = response_body_data.get('total_detections', 0)
                data['total_texts'] = response_body_data.get('total_texts', 0)
                
                # Extract results arrays (limited for logging)
                if 'detection_results' in response_body_data:
                    data['detection_results'] = response_body_data['detection_results'][:5]  # Limit to first 5
                    data['total_detections'] = len(response_body_data['detection_results'])
                
                if 'ocr_results' in response_body_data:
                    data['ocr_results'] = response_body_data['ocr_results'][:5]  # Limit to first 5
                    data['total_texts'] = len(response_body_data['ocr_results'])
                
                # Extract additional useful info
                if 'languages_used' in response_body_data:
                    data['languages_used'] = response_body_data['languages_used']
                
                if 'model_name' in response_body_data:
                    data['model_name'] = response_body_data['model_name']
                
                if 'confidence_threshold' in response_body_data:
                    data['confidence_threshold'] = response_body_data['confidence_threshold']
                
                return data
            except Exception as e:
                print(f"Error processing response body data: {e}")
                # Continue with fallback processing
        
        try:
            # Try to extract data from response body if it's JSON
            if hasattr(response, 'body') and response.status_code == 200:
                try:
                    # For StreamingResponse or regular Response, get body differently
                    if hasattr(response, 'body'):
                        if callable(response.body):
                            body_str = str(response.body)
                        else:
                            body_str = response.body.decode('utf-8') if isinstance(response.body, bytes) else str(response.body)
                    else:
                        # Try to get body from response content
                        body_str = ""
                    
                    # Try to parse as JSON
                    if body_str and body_str.strip():
                        response_data = json.loads(body_str)
                        
                        # Store full response (excluding large binary data)
                        filtered_response = {}
                        for key, value in response_data.items():
                            if key in ['cropped_images']:  # Skip large binary data
                                filtered_response[key] = f"[{len(value)} items]" if isinstance(value, list) else "[binary data]"
                            else:
                                filtered_response[key] = value
                        
                        data['response_body'] = filtered_response
                        
                        # Extract detection and OCR counts
                        data['total_detections'] = response_data.get('total_detections', 0)
                        data['total_texts'] = response_data.get('total_texts', 0)
                        
                        # Extract results arrays (limited for logging)
                        if 'detection_results' in response_data:
                            data['detection_results'] = response_data['detection_results'][:5]  # Limit to first 5
                            data['total_detections'] = len(response_data['detection_results'])
                        
                        if 'ocr_results' in response_data:
                            data['ocr_results'] = response_data['ocr_results'][:5]  # Limit to first 5
                            data['total_texts'] = len(response_data['ocr_results'])
                        
                        # Extract additional useful info
                        if 'languages_used' in response_data:
                            data['languages_used'] = response_data['languages_used']
                        
                        if 'model_name' in response_data:
                            data['model_name'] = response_data['model_name']
                        
                        if 'confidence_threshold' in response_data:
                            data['confidence_threshold'] = response_data['confidence_threshold']
                    else:
                        # Empty body or couldn't get body content
                        data['response_body'] = {
                            'status_code': response.status_code,
                            'content_type': response.headers.get('content-type', ''),
                            'note': 'Empty response body or could not read body'
                        }
                    
                except json.JSONDecodeError as json_error:
                    logger.debug(f"Could not parse response as JSON: {json_error}")
                    # If JSON parsing fails, store what we can
                    data['response_body'] = {
                        'status_code': response.status_code,
                        'content_type': response.headers.get('content-type', ''),
                        'parse_error': f'JSON decode error: {str(json_error)}',
                        'body_preview': body_str[:200] if 'body_str' in locals() else 'Could not read body'
                    }
                except Exception as parse_error:
                    logger.debug(f"Could not parse response body: {parse_error}")
                    # If JSON parsing fails, store status info
                    data['response_body'] = {
                        'status_code': response.status_code,
                        'content_type': response.headers.get('content-type', ''),
                        'parse_error': str(parse_error)
                    }
            else:
                # For non-200 responses or non-JSON responses
                data['response_body'] = {
                    'status_code': response.status_code,
                    'content_type': response.headers.get('content-type', ''),
                    'headers': dict(response.headers)
                }
        
        except Exception as e:
            logger.warning(f"Failed to extract response data: {e}")
            data['response_body'] = {'error': f"Failed to parse response: {str(e)}"}
        
        return data
    
    def _get_client_ip(self, request: Request) -> str:
        """Lấy IP address của client"""
        # Check for forwarded headers first
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        if hasattr(request, 'client') and request.client:
            return request.client.host
        
        return 'unknown'
