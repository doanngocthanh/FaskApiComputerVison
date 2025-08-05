"""
API Proxy Service - Manages proxy configurations and routing
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class ProxyConfigService:
    """Service for managing API proxy configurations"""
    
    def __init__(self, db_path: str = "temp/logs/proxy_configs.db"):
        """Initialize proxy config service with database"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Proxy configurations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS proxy_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_id TEXT UNIQUE NOT NULL,
                    input_setting TEXT NOT NULL,
                    target_uri TEXT NOT NULL,
                    method TEXT DEFAULT 'GET',
                    description TEXT,
                    default_params TEXT,  -- JSON string
                    query_mapping TEXT,   -- JSON string for parameter mapping
                    header_mapping TEXT,  -- JSON string for header mapping
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Proxy usage logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS proxy_usage_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_id TEXT NOT NULL,
                    request_id TEXT,
                    source_uri TEXT,
                    target_uri TEXT,
                    method TEXT,
                    status_code INTEGER,
                    response_time REAL,
                    error_message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (config_id) REFERENCES proxy_configs (config_id)
                )
            """)
            
            conn.commit()
    
    def create_proxy_config(self, 
                          input_setting: str,
                          target_uri: str,
                          method: str = "GET",
                          description: str = None,
                          default_params: Dict = None,
                          query_mapping: Dict = None,
                          header_mapping: Dict = None) -> str:
        """Create a new proxy configuration"""
        
        config_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO proxy_configs 
                (config_id, input_setting, target_uri, method, description, 
                 default_params, query_mapping, header_mapping)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                config_id,
                input_setting,
                target_uri,
                method,
                description,
                json.dumps(default_params or {}),
                json.dumps(query_mapping or {}),
                json.dumps(header_mapping or {})
            ))
            
            conn.commit()
        
        return config_id
    
    def get_proxy_config(self, input_setting: str) -> Optional[Dict]:
        """Get proxy configuration by input_setting"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.row_factory = sqlite3.Row
            
            cursor.execute("""
                SELECT * FROM proxy_configs 
                WHERE input_setting = ? AND is_active = 1
                ORDER BY updated_at DESC
                LIMIT 1
            """, (input_setting,))
            
            row = cursor.fetchone()
            
            if row:
                config = dict(row)
                # Parse JSON fields
                config['default_params'] = json.loads(config['default_params'] or '{}')
                config['query_mapping'] = json.loads(config['query_mapping'] or '{}')
                config['header_mapping'] = json.loads(config['header_mapping'] or '{}')
                return config
            
            return None
    
    def list_proxy_configs(self, active_only: bool = True) -> List[Dict]:
        """List all proxy configurations"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.row_factory = sqlite3.Row
            
            query = "SELECT * FROM proxy_configs"
            params = []
            
            if active_only:
                query += " WHERE is_active = 1"
            
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            configs = []
            for row in rows:
                config = dict(row)
                # Parse JSON fields
                config['default_params'] = json.loads(config['default_params'] or '{}')
                config['query_mapping'] = json.loads(config['query_mapping'] or '{}')
                config['header_mapping'] = json.loads(config['header_mapping'] or '{}')
                configs.append(config)
            
            return configs
    
    def update_proxy_config(self, config_id: str, **kwargs) -> bool:
        """Update proxy configuration"""
        
        allowed_fields = [
            'input_setting', 'target_uri', 'method', 'description',
            'default_params', 'query_mapping', 'header_mapping', 'is_active'
        ]
        
        update_fields = []
        update_values = []
        
        for field, value in kwargs.items():
            if field in allowed_fields:
                if field in ['default_params', 'query_mapping', 'header_mapping']:
                    value = json.dumps(value)
                update_fields.append(f"{field} = ?")
                update_values.append(value)
        
        if not update_fields:
            return False
        
        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        update_values.append(config_id)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = f"UPDATE proxy_configs SET {', '.join(update_fields)} WHERE config_id = ?"
            cursor.execute(query, update_values)
            
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_proxy_config(self, config_id: str) -> bool:
        """Delete (deactivate) proxy configuration"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE proxy_configs 
                SET is_active = 0, updated_at = CURRENT_TIMESTAMP
                WHERE config_id = ?
            """, (config_id,))
            
            conn.commit()
            return cursor.rowcount > 0
    
    def log_proxy_usage(self, 
                       config_id: str,
                       request_id: str,
                       source_uri: str,
                       target_uri: str,
                       method: str,
                       status_code: int,
                       response_time: float,
                       error_message: str = None):
        """Log proxy usage"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO proxy_usage_logs
                (config_id, request_id, source_uri, target_uri, method, 
                 status_code, response_time, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                config_id, request_id, source_uri, target_uri,
                method, status_code, response_time, error_message
            ))
            
            conn.commit()
    
    def get_usage_stats(self, config_id: str = None, days: int = 7) -> Dict:
        """Get proxy usage statistics"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            base_query = """
                SELECT 
                    COUNT(*) as total_requests,
                    AVG(response_time) as avg_response_time,
                    COUNT(CASE WHEN status_code >= 200 AND status_code < 300 THEN 1 END) as success_requests,
                    COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_requests
                FROM proxy_usage_logs 
                WHERE timestamp >= datetime('now', '-{} days')
            """.format(days)
            
            params = []
            
            if config_id:
                base_query += " AND config_id = ?"
                params.append(config_id)
            
            cursor.execute(base_query, params)
            stats = cursor.fetchone()
            
            return {
                'total_requests': stats[0] or 0,
                'avg_response_time': round(stats[1] or 0, 3),
                'success_requests': stats[2] or 0,
                'error_requests': stats[3] or 0,
                'success_rate': round((stats[2] or 0) / max(stats[0] or 1, 1) * 100, 2)
            }
