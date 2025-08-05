
import os
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid
import time
from collections import defaultdict
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoggingManager:
    def __init__(self, log_dir: str = None):
        if log_dir is None:
            log_dir = "temp/logs"
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Database for structured logging
        self.db_path = self.log_dir / "ocr_requests.db"
        self.init_database()
        
        # JSON log files
        self.requests_log = self.log_dir / "requests.json"
        self.errors_log = self.log_dir / "errors.json"
        self.performance_log = self.log_dir / "performance.json"
        
        # Statistics cache
        self.stats_cache = {}
        self.cache_lock = threading.Lock()
        
        # Last cleanup time
        self.last_cleanup = datetime.now()
        
        logger.info(f"Logging Manager initialized with directory: {self.log_dir}")
    
    def init_database(self):
        """Khởi tạo SQLite database cho structured logging"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Check if database exists and get current schema
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ocr_requests'")
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                # Check current columns
                cursor.execute("PRAGMA table_info(ocr_requests)")
                current_columns = [row[1] for row in cursor.fetchall()]
                
                # Expected columns with request_id
                expected_columns = [
                    'id', 'request_id', 'timestamp', 'endpoint', 'method', 'languages', 'model_id',
                    'processing_time_ms', 'success', 'error_message', 'file_size',
                    'total_detections', 'total_texts', 'ip_address', 'user_agent',
                    'request_body', 'response_body', 'confidence_threshold',
                    'ocr_results', 'detection_results'
                ]
                
                # Add missing columns
                for column in expected_columns:
                    if column not in current_columns:
                        if column in ['request_body', 'response_body', 'ocr_results', 'detection_results', 'request_id']:
                            cursor.execute(f'ALTER TABLE ocr_requests ADD COLUMN {column} TEXT')
                        elif column == 'confidence_threshold':
                            cursor.execute(f'ALTER TABLE ocr_requests ADD COLUMN {column} REAL')
                        else:
                            cursor.execute(f'ALTER TABLE ocr_requests ADD COLUMN {column} TEXT')
            else:
                # Create new table with all columns
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ocr_requests (
                        id TEXT PRIMARY KEY,
                        request_id TEXT UNIQUE,
                        timestamp TEXT,
                        endpoint TEXT,
                        method TEXT,
                        languages TEXT,
                        model_id TEXT,
                        processing_time_ms REAL,
                        success BOOLEAN,
                        error_message TEXT,
                        file_size INTEGER,
                        total_detections INTEGER,
                        total_texts INTEGER,
                        ip_address TEXT,
                        user_agent TEXT,
                        request_body TEXT,
                        response_body TEXT,
                        confidence_threshold REAL,
                        ocr_results TEXT,
                        detection_results TEXT
                    )
                ''')
                
                logger.info("✅ Created ocr_requests table with all columns")
            
            # Tạo bảng performance metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    metric_type TEXT,
                    metric_value REAL,
                    metadata TEXT
                )
            ''')
            
            # Tạo indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON ocr_requests(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_endpoint ON ocr_requests(endpoint)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_success ON ocr_requests(success)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON performance_metrics(timestamp)')
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Database initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
    
    def log_request(self, request_data: Dict[str, Any]):
        """Log một OCR request với request_id tracking"""
        try:
            # Use existing request_id or generate new one
            request_id = request_data.get('request_id', str(uuid.uuid4()))
            timestamp = datetime.now().isoformat()
            
            # Auto cleanup check (every 24 hours)
            if (datetime.now() - self.last_cleanup).days >= 1:
                self._auto_cleanup()
                self.last_cleanup = datetime.now()
            
            # Log to database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get current table schema
            cursor.execute("PRAGMA table_info(ocr_requests)")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Prepare data for insertion
            values = []
            placeholders = []
            
            # Base data mapping
            data_mapping = {
                'id': str(uuid.uuid4()),  # Auto-generated primary key
                'request_id': request_id,  # User-trackable ID
                'timestamp': timestamp,
                'endpoint': request_data.get('endpoint', ''),
                'method': request_data.get('method', 'POST'),
                'languages': json.dumps(request_data.get('languages', [])),
                'model_id': request_data.get('model_id', ''),
                'processing_time_ms': request_data.get('processing_time_ms', 0),
                'success': request_data.get('success', False),
                'error_message': request_data.get('error_message', ''),
                'file_size': request_data.get('file_size', 0),
                'total_detections': request_data.get('total_detections', 0),
                'total_texts': request_data.get('total_texts', 0),
                'ip_address': request_data.get('ip_address', ''),
                'user_agent': request_data.get('user_agent', ''),
                'request_body': json.dumps(request_data.get('request_body', {})),
                'response_body': json.dumps(request_data.get('response_body', {})),
                'confidence_threshold': request_data.get('confidence_threshold', 0.0),
                'ocr_results': json.dumps(request_data.get('ocr_results', [])),
                'detection_results': json.dumps(request_data.get('detection_results', []))
            }
            
            # Build INSERT query based on available columns
            for column in columns:
                if column in data_mapping:
                    values.append(data_mapping[column])
                    placeholders.append('?')
            
            if values:
                column_names = ', '.join([col for col in columns if col in data_mapping])
                placeholders_str = ', '.join(placeholders)
                
                query = f'INSERT INTO ocr_requests ({column_names}) VALUES ({placeholders_str})'
                cursor.execute(query, values)
            
            conn.commit()
            conn.close()
            
            # Log to JSON file (for backup)
            self._append_to_json_log(self.requests_log, {
                'id': data_mapping['id'],
                'request_id': request_id,
                'timestamp': timestamp,
                **request_data
            })
            
            # Update statistics cache
            self._update_stats_cache(request_data)
            
            logger.info(f"📝 Logged request {request_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to log request: {e}")
    
    def log_error(self, error_data: Dict[str, Any]):
        """Log một error"""
        try:
            error_entry = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                **error_data
            }
            
            self._append_to_json_log(self.errors_log, error_entry)
            logger.error(f"🚨 Logged error: {error_data.get('message', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to log error: {e}")
    
    def log_performance(self, metric_type: str, value: float, metadata: Dict = None):
        """Log performance metric"""
        try:
            metric_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Log to database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics VALUES (?, ?, ?, ?, ?)
            ''', (
                metric_id,
                timestamp,
                metric_type,
                value,
                json.dumps(metadata or {})
            ))
            
            conn.commit()
            conn.close()
            
            # Log to JSON file
            self._append_to_json_log(self.performance_log, {
                'id': metric_id,
                'timestamp': timestamp.isoformat(),
                'metric_type': metric_type,
                'value': value,
                'metadata': metadata or {}
            })
            
        except Exception as e:
            logger.error(f"❌ Failed to log performance: {e}")
    
    def get_statistics(self, period: str = 'today') -> Dict[str, Any]:
        """Lấy thống kê requests theo period (today, week, month)"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Xác định thời gian
            now = datetime.now()
            if period == 'today':
                start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif period == 'week':
                start_time = now - timedelta(days=7)
            elif period == 'month':
                start_time = now - timedelta(days=30)
            else:
                start_time = now - timedelta(days=1)
            
            # Total requests
            cursor.execute('''
                SELECT COUNT(*) FROM ocr_requests 
                WHERE timestamp >= ?
            ''', (start_time,))
            total_requests = cursor.fetchone()[0]
            
            # Success rate
            cursor.execute('''
                SELECT COUNT(*) FROM ocr_requests 
                WHERE timestamp >= ? AND success = 1
            ''', (start_time,))
            successful_requests = cursor.fetchone()[0]
            
            success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
            
            # Average processing time
            cursor.execute('''
                SELECT AVG(processing_time_ms) FROM ocr_requests 
                WHERE timestamp >= ? AND success = 1
            ''', (start_time,))
            avg_processing_time = cursor.fetchone()[0] or 0
            
            # Top endpoints
            cursor.execute('''
                SELECT endpoint, COUNT(*) as count FROM ocr_requests 
                WHERE timestamp >= ?
                GROUP BY endpoint 
                ORDER BY count DESC 
                LIMIT 5
            ''', (start_time,))
            top_endpoints = dict(cursor.fetchall())
            
            # Language usage
            cursor.execute('''
                SELECT languages, COUNT(*) as count FROM ocr_requests 
                WHERE timestamp >= ? AND languages != ''
                GROUP BY languages 
                ORDER BY count DESC 
                LIMIT 5
            ''', (start_time,))
            language_usage = dict(cursor.fetchall())
            
            # Error types
            cursor.execute('''
                SELECT error_message, COUNT(*) as count FROM ocr_requests 
                WHERE timestamp >= ? AND success = 0 AND error_message != ''
                GROUP BY error_message 
                ORDER BY count DESC 
                LIMIT 5
            ''', (start_time,))
            error_types = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'period': period,
                'start_time': start_time.isoformat(),
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': total_requests - successful_requests,
                'success_rate': round(success_rate, 2),
                'average_processing_time_ms': round(avg_processing_time, 2),
                'top_endpoints': top_endpoints,
                'language_usage': language_usage,
                'error_types': error_types
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get statistics: {e}")
            return {
                'period': period,
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'success_rate': 0,
                'average_processing_time_ms': 0,
                'top_endpoints': {},
                'language_usage': {},
                'error_types': {}
            }
    
    def get_performance_metrics(self, metric_type: str = None, hours: int = 24) -> List[Dict]:
        """Lấy performance metrics"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            start_time = datetime.now() - timedelta(hours=hours)
            
            if metric_type:
                cursor.execute('''
                    SELECT timestamp, metric_value, metadata FROM performance_metrics 
                    WHERE timestamp >= ? AND metric_type = ?
                    ORDER BY timestamp DESC
                ''', (start_time, metric_type))
            else:
                cursor.execute('''
                    SELECT timestamp, metric_type, metric_value, metadata FROM performance_metrics 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (start_time,))
            
            results = []
            for row in cursor.fetchall():
                if metric_type:
                    results.append({
                        'timestamp': row[0],
                        'value': row[1],
                        'metadata': json.loads(row[2])
                    })
                else:
                    results.append({
                        'timestamp': row[0],
                        'metric_type': row[1],
                        'value': row[2],
                        'metadata': json.loads(row[3])
                    })
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get performance metrics: {e}")
            return []
    
    def cleanup_old_logs(self, keep_days: int = 30):
        """Dọn dẹp logs cũ hơn keep_days ngày"""
        try:
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            
            # Cleanup database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Delete old requests
            cursor.execute('DELETE FROM ocr_requests WHERE timestamp < ?', (cutoff_date,))
            deleted_requests = cursor.rowcount
            
            # Delete old performance metrics
            cursor.execute('DELETE FROM performance_metrics WHERE timestamp < ?', (cutoff_date,))
            deleted_metrics = cursor.rowcount
            
            # Vacuum database to reclaim space
            cursor.execute('VACUUM')
            
            conn.commit()
            conn.close()
            
            # Cleanup JSON log files (archive old logs)
            self._archive_old_json_logs(cutoff_date)
            
            logger.info(f"🧹 Cleanup completed: {deleted_requests} requests, {deleted_metrics} metrics deleted")
            
            return {
                'deleted_requests': deleted_requests,
                'deleted_metrics': deleted_metrics,
                'cutoff_date': cutoff_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup logs: {e}")
            return {'deleted_requests': 0, 'deleted_metrics': 0, 'cutoff_date': datetime.now().isoformat()}
    
    def _auto_cleanup(self):
        """Tự động dọn dẹp logs hàng ngày"""
        try:
            # Weekly cleanup - giữ logs 7 ngày
            now = datetime.now()
            if now.weekday() == 6:  # Sunday
                logger.info("🗑️ Running weekly cleanup...")
                self.cleanup_old_logs(keep_days=7)
            
            # Monthly cleanup - giữ logs 30 ngày (chỉ chạy ngày đầu tháng)
            if now.day == 1:
                logger.info("🗑️ Running monthly cleanup...")
                self.cleanup_old_logs(keep_days=30)
                
        except Exception as e:
            logger.error(f"❌ Auto cleanup failed: {e}")
    
    def _append_to_json_log(self, log_file: Path, data: Dict):
        """Append data to JSON log file"""
        try:
            # Read existing data
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Append new data
            logs.append(data)
            
            # Keep only last 1000 entries to prevent file from growing too large
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            # Write back
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Failed to append to JSON log: {e}")
    
    def _update_stats_cache(self, request_data: Dict):
        """Update statistics cache"""
        try:
            with self.cache_lock:
                if 'daily_stats' not in self.stats_cache:
                    self.stats_cache['daily_stats'] = defaultdict(int)
                
                today = datetime.now().strftime('%Y-%m-%d')
                self.stats_cache['daily_stats'][f'{today}_total'] += 1
                
                if request_data.get('success', False):
                    self.stats_cache['daily_stats'][f'{today}_success'] += 1
                else:
                    self.stats_cache['daily_stats'][f'{today}_error'] += 1
                    
        except Exception as e:
            logger.error(f"❌ Failed to update stats cache: {e}")
    
    def _archive_old_json_logs(self, cutoff_date: datetime):
        """Archive old JSON logs"""
        try:
            archive_dir = self.log_dir / "archive"
            archive_dir.mkdir(exist_ok=True)
            
            for log_file in [self.requests_log, self.errors_log, self.performance_log]:
                if log_file.exists():
                    # Create archive filename with timestamp
                    archive_name = f"{log_file.stem}_{cutoff_date.strftime('%Y%m%d')}.json"
                    archive_path = archive_dir / archive_name
                    
                    # Copy file to archive instead of moving to keep current logs
                    import shutil
                    shutil.copy2(log_file, archive_path)
                    logger.info(f"📦 Archived {log_file.name} to {archive_name}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to archive logs: {e}")

    def get_detailed_logs(self, limit: int = 50, offset: int = 0, 
                         endpoint_filter: str = None, success_filter: bool = None) -> List[Dict]:
        """Lấy detailed logs với request/response"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Build query
            where_conditions = []
            params = []
            
            if endpoint_filter:
                where_conditions.append("endpoint LIKE ?")
                params.append(f"%{endpoint_filter}%")
            
            if success_filter is not None:
                where_conditions.append("success = ?")
                params.append(success_filter)
            
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)
            
            query = f'''
                SELECT id, request_id, timestamp, endpoint, method, languages, model_id, 
                       processing_time_ms, success, error_message, file_size,
                       total_detections, total_texts, ip_address, user_agent,
                       request_body, response_body, confidence_threshold,
                       ocr_results, detection_results
                FROM ocr_requests 
                {where_clause}
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            '''
            
            params.extend([limit, offset])
            cursor.execute(query, params)
            
            columns = [
                'id', 'request_id', 'timestamp', 'endpoint', 'method', 'languages', 'model_id',
                'processing_time_ms', 'success', 'error_message', 'file_size',
                'total_detections', 'total_texts', 'ip_address', 'user_agent',
                'request_body', 'response_body', 'confidence_threshold',
                'ocr_results', 'detection_results'
            ]
            
            results = []
            for row in cursor.fetchall():
                record = dict(zip(columns, row))
                
                # Parse JSON fields
                try:
                    record['languages'] = json.loads(record['languages'] or '[]')
                    record['request_body'] = json.loads(record['request_body'] or '{}')
                    record['response_body'] = json.loads(record['response_body'] or '{}')
                    record['ocr_results'] = json.loads(record['ocr_results'] or '[]')
                    record['detection_results'] = json.loads(record['detection_results'] or '[]')
                except json.JSONDecodeError:
                    pass
                
                # Format timestamp
                record['formatted_time'] = datetime.fromisoformat(record['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                
                results.append(record)
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get detailed logs: {e}")
            return []

    def get_log_count(self, endpoint: str = None, success: bool = None) -> int:
        """Đếm tổng số logs"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            where_conditions = []
            params = []
            
            if endpoint:
                where_conditions.append("endpoint LIKE ?")
                params.append(f"%{endpoint}%")
            
            if success is not None:
                where_conditions.append("success = ?")
                params.append(success)
            
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)
            
            query = f"SELECT COUNT(*) FROM ocr_requests {where_clause}"
            cursor.execute(query, params)
            
            count = cursor.fetchone()[0]
            conn.close()
            return count
            
        except Exception as e:
            logger.error(f"❌ Failed to get log count: {e}")
            return 0

    def get_request_by_id(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Lấy chi tiết request theo request_id"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM ocr_requests 
                WHERE request_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (request_id,))
            
            row = cursor.fetchone()
            if not row:
                conn.close()
                return None
            
            # Get column names
            cursor.execute("PRAGMA table_info(ocr_requests)")
            column_names = [row[1] for row in cursor.fetchall()]
            
            # Create dictionary from row data
            result = dict(zip(column_names, row))
            
            # Parse JSON fields
            json_fields = ['languages', 'request_body', 'response_body', 'ocr_results', 'detection_results']
            for field in json_fields:
                if field in result and result[field]:
                    try:
                        result[field] = json.loads(result[field])
                    except (json.JSONDecodeError, TypeError):
                        result[field] = []
            
            conn.close()
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to get request by ID {request_id}: {e}")
            return None

# Global logging manager instance
logging_manager = LoggingManager()
