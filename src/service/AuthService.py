"""
Authentication Service
Quản lý xác thực cho admin panel
"""

import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
from config import DBConfig

class AuthService:
    def __init__(self):
        self.db_config = DBConfig()
        self.secret_key = self._get_or_create_secret_key()
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 60 * 24  # 24 hours
        self.init_database()
        self.init_default_admin()

    def _get_or_create_secret_key(self) -> str:
        """Lấy hoặc tạo secret key"""
        try:
            result = self.db_config.fetch_one(
                "SELECT config_value FROM app_settings WHERE config_key = 'jwt_secret_key'"
            )
            if result:
                return result[0]
            else:
                # Tạo secret key mới
                secret_key = secrets.token_urlsafe(32)
                self.db_config.execute_query(
                    "INSERT OR REPLACE INTO app_settings (config_key, config_value) VALUES (?, ?)",
                    ('jwt_secret_key', secret_key)
                )
                return secret_key
        except:
            # Fallback nếu chưa có bảng settings
            return secrets.token_urlsafe(32)

    def init_database(self):
        """Khởi tạo bảng users nếu chưa có"""
        try:
            self.db_config.execute_query("""
                CREATE TABLE IF NOT EXISTS admin_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT,
                    full_name TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            """)
            
            # Tạo bảng sessions
            self.db_config.execute_query("""
                CREATE TABLE IF NOT EXISTS admin_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    token_hash TEXT NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    FOREIGN KEY (user_id) REFERENCES admin_users (id)
                )
            """)
            
            logging.info("✅ Auth database tables initialized")
            
        except Exception as e:
            logging.error(f"❌ Failed to init auth database: {e}")

    def init_default_admin(self):
        """Tạo admin mặc định nếu chưa có"""
        try:
            # Kiểm tra xem đã có admin nào chưa
            existing_admin = self.db_config.fetch_one(
                "SELECT id FROM admin_users WHERE username = 'admin'"
            )
            
            if not existing_admin:
                # Tạo admin mặc định với password = 'admin123'
                password_hash = self._hash_password('admin123')
                self.db_config.execute_query("""
                    INSERT INTO admin_users (username, password_hash, email, full_name, is_active)
                    VALUES (?, ?, ?, ?, ?)
                """, ('admin', password_hash, 'admin@example.com', 'System Administrator', 1))
                
                logging.info("✅ Default admin user created (username: admin, password: admin123)")
                
        except Exception as e:
            logging.error(f"❌ Failed to create default admin: {e}")

    def _hash_password(self, password: str) -> str:
        """Hash password với salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}${password_hash.hex()}"

    def _verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password"""
        try:
            salt, hash_hex = hashed_password.split('$')
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return hash_hex == password_hash.hex()
        except:
            return False

    def authenticate_user(self, username: str, password: str, ip_address: str = None, user_agent: str = None) -> Optional[Dict[str, Any]]:
        """Xác thực user"""
        try:
            user = self.db_config.fetch_one("""
                SELECT id, username, password_hash, email, full_name, is_active
                FROM admin_users 
                WHERE username = ? AND is_active = 1
            """, (username,))
            
            if not user:
                return None
                
            user_id, username, password_hash, email, full_name, is_active = user
            
            if not self._verify_password(password, password_hash):
                return None
            
            # Update last login
            self.db_config.execute_query(
                "UPDATE admin_users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                (user_id,)
            )
            
            # Tạo access token
            token = self.create_access_token(user_id, username)
            
            # Lưu session
            self._save_session(user_id, token, ip_address, user_agent)
            
            return {
                "id": user_id,
                "username": username,
                "email": email,
                "full_name": full_name,
                "access_token": token,
                "token_type": "bearer"
            }
            
        except Exception as e:
            logging.error(f"Authentication error: {e}")
            return None

    def create_access_token(self, user_id: int, username: str) -> str:
        """Tạo JWT access token"""
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        payload = {
            "user_id": user_id,
            "username": username,
            "exp": expire,
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Kiểm tra session có còn valid không
            session = self.db_config.fetch_one("""
                SELECT s.id, s.user_id, u.username, u.is_active
                FROM admin_sessions s
                JOIN admin_users u ON s.user_id = u.id
                WHERE s.token_hash = ? AND s.expires_at > CURRENT_TIMESTAMP AND u.is_active = 1
            """, (self._hash_token(token),))
            
            if not session:
                return None
                
            return {
                "user_id": payload["user_id"],
                "username": payload["username"]
            }
            
        except jwt.ExpiredSignatureError:
            logging.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logging.warning("Invalid token")
            return None
        except Exception as e:
            logging.error(f"Token verification error: {e}")
            return None

    def _save_session(self, user_id: int, token: str, ip_address: str = None, user_agent: str = None):
        """Lưu session vào database"""
        try:
            token_hash = self._hash_token(token)
            expires_at = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
            
            self.db_config.execute_query("""
                INSERT INTO admin_sessions (user_id, token_hash, expires_at, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, token_hash, expires_at.isoformat(), ip_address, user_agent))
            
        except Exception as e:
            logging.error(f"Failed to save session: {e}")

    def _hash_token(self, token: str) -> str:
        """Hash token để lưu trong database"""
        return hashlib.sha256(token.encode()).hexdigest()

    def logout(self, token: str) -> bool:
        """Logout user - xóa session"""
        try:
            token_hash = self._hash_token(token)
            self.db_config.execute_query(
                "DELETE FROM admin_sessions WHERE token_hash = ?",
                (token_hash,)
            )
            return True
        except Exception as e:
            logging.error(f"Logout error: {e}")
            return False

    def cleanup_expired_sessions(self):
        """Xóa các session đã hết hạn"""
        try:
            self.db_config.execute_query(
                "DELETE FROM admin_sessions WHERE expires_at < CURRENT_TIMESTAMP"
            )
        except Exception as e:
            logging.error(f"Session cleanup error: {e}")

    def get_active_sessions(self, user_id: int) -> list:
        """Lấy các session đang active của user"""
        try:
            sessions = self.db_config.fetch_all("""
                SELECT id, created_at, ip_address, user_agent, expires_at
                FROM admin_sessions 
                WHERE user_id = ? AND expires_at > CURRENT_TIMESTAMP
                ORDER BY created_at DESC
            """, (user_id,))
            
            return [
                {
                    "id": session[0],
                    "created_at": session[1],
                    "ip_address": session[2],
                    "user_agent": session[3],
                    "expires_at": session[4]
                }
                for session in sessions
            ]
        except Exception as e:
            logging.error(f"Failed to get active sessions: {e}")
            return []

    def change_password(self, user_id: int, old_password: str, new_password: str) -> bool:
        """Đổi password"""
        try:
            # Verify old password
            user = self.db_config.fetch_one(
                "SELECT password_hash FROM admin_users WHERE id = ?",
                (user_id,)
            )
            
            if not user or not self._verify_password(old_password, user[0]):
                return False
            
            # Update password
            new_password_hash = self._hash_password(new_password)
            self.db_config.execute_query(
                "UPDATE admin_users SET password_hash = ? WHERE id = ?",
                (new_password_hash, user_id)
            )
            
            # Xóa tất cả session cũ để force re-login
            self.db_config.execute_query(
                "DELETE FROM admin_sessions WHERE user_id = ?",
                (user_id,)
            )
            
            return True
            
        except Exception as e:
            logging.error(f"Password change error: {e}")
            return False

# Singleton instance
auth_service = AuthService()
