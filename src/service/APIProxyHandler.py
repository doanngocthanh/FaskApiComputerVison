"""
API Proxy Handler - Handles dynamic routing and request forwarding
"""

import httpx
import time
import uuid
from typing import Dict, Any, Optional
from fastapi import Request, Response
from src.service.ProxyConfigService import ProxyConfigService


class APIProxyHandler:
    """Handler for API proxy operations"""
    
    def __init__(self):
        self.config_service = ProxyConfigService()
        self.http_client = httpx.AsyncClient()
    
    async def route_request(self, 
                          input_setting: str, 
                          request: Request) -> Dict[str, Any]:
        """Route request through proxy configuration"""
        
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Get proxy configuration
            config = self.config_service.get_proxy_config(input_setting)
            
            if not config:
                return {
                    "error": "Proxy configuration not found",
                    "input_setting": input_setting,
                    "status_code": 404
                }
            
            # Build target URL
            target_url = self._build_target_url(config, request)
            
            # Prepare headers
            headers = self._prepare_headers(config, request)
            
            # Prepare parameters
            params = self._prepare_params(config, request)
            
            # Make the request
            method = config.get('method', 'GET').upper()
            
            if method == 'GET':
                response = await self.http_client.get(
                    target_url,
                    params=params,
                    headers=headers,
                    timeout=30.0
                )
            elif method == 'POST':
                # Handle POST data
                content_type = request.headers.get('content-type', '')
                if 'application/json' in content_type:
                    json_data = await request.json()
                    response = await self.http_client.post(
                        target_url,
                        json=json_data,
                        params=params,
                        headers=headers,
                        timeout=30.0
                    )
                elif 'multipart/form-data' in content_type:
                    form_data = await request.form()
                    response = await self.http_client.post(
                        target_url,
                        data=form_data,
                        params=params,
                        headers=headers,
                        timeout=30.0
                    )
                else:
                    body = await request.body()
                    response = await self.http_client.post(
                        target_url,
                        content=body,
                        params=params,
                        headers=headers,
                        timeout=30.0
                    )
            else:
                return {
                    "error": f"HTTP method {method} not supported",
                    "status_code": 405
                }
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Log usage
            self.config_service.log_proxy_usage(
                config_id=config['config_id'],
                request_id=request_id,
                source_uri=str(request.url),
                target_uri=target_url,
                method=method,
                status_code=response.status_code,
                response_time=response_time
            )
            
            # Return response
            try:
                response_data = response.json()
            except:
                response_data = response.text
            
            return {
                "status_code": response.status_code,
                "data": response_data,
                "headers": dict(response.headers),
                "response_time": round(response_time, 3),
                "request_id": request_id,
                "target_url": target_url,
                "proxy_config": {
                    "input_setting": config['input_setting'],
                    "config_id": config['config_id'],
                    "description": config.get('description')
                }
            }
            
        except httpx.TimeoutException:
            response_time = time.time() - start_time
            error_msg = "Request timeout"
            
            if 'config' in locals():
                self.config_service.log_proxy_usage(
                    config_id=config['config_id'],
                    request_id=request_id,
                    source_uri=str(request.url),
                    target_uri=target_url if 'target_url' in locals() else '',
                    method=method if 'method' in locals() else 'GET',
                    status_code=408,
                    response_time=response_time,
                    error_message=error_msg
                )
            
            return {
                "error": error_msg,
                "status_code": 408,
                "response_time": round(response_time, 3),
                "request_id": request_id
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = str(e)
            
            if 'config' in locals():
                self.config_service.log_proxy_usage(
                    config_id=config['config_id'],
                    request_id=request_id,
                    source_uri=str(request.url),
                    target_uri=target_url if 'target_url' in locals() else '',
                    method=method if 'method' in locals() else 'GET',
                    status_code=500,
                    response_time=response_time,
                    error_message=error_msg
                )
            
            return {
                "error": error_msg,
                "status_code": 500,
                "response_time": round(response_time, 3),
                "request_id": request_id
            }
    
    def _build_target_url(self, config: Dict, request: Request) -> str:
        """Build target URL from configuration"""
        base_url = config['target_uri']
        
        # Handle base URL that might need localhost resolution
        if base_url.startswith('/'):
            # Relative URL - use current host
            host = request.headers.get('host', 'localhost:8000')
            scheme = 'https' if request.url.scheme == 'https' else 'http'
            base_url = f"{scheme}://{host}{base_url}"
        
        return base_url
    
    def _prepare_headers(self, config: Dict, request: Request) -> Dict[str, str]:
        """Prepare headers for target request"""
        headers = {}
        
        # Copy relevant headers from original request
        for header_name, header_value in request.headers.items():
            if header_name.lower() not in ['host', 'content-length']:
                headers[header_name] = header_value
        
        # Apply header mapping from config
        header_mapping = config.get('header_mapping', {})
        for source_header, target_header in header_mapping.items():
            if source_header in request.headers:
                headers[target_header] = request.headers[source_header]
        
        return headers
    
    def _prepare_params(self, config: Dict, request: Request) -> Dict[str, Any]:
        """Prepare query parameters for target request"""
        params = {}
        
        # Start with default parameters from config
        default_params = config.get('default_params', {})
        params.update(default_params)
        
        # Add query parameters from original request
        for param_name, param_value in request.query_params.items():
            params[param_name] = param_value
        
        # Apply query parameter mapping from config
        query_mapping = config.get('query_mapping', {})
        mapped_params = {}
        
        for source_param, target_param in query_mapping.items():
            if source_param in params:
                mapped_params[target_param] = params[source_param]
                # Remove original parameter if it's being mapped
                if source_param != target_param:
                    params.pop(source_param, None)
        
        params.update(mapped_params)
        
        return params
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
