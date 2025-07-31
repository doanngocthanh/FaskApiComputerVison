import os
import importlib
from fastapi.middleware.cors import CORSMiddleware
import sqlite3

class RouterConfig:
    def __init__(self):
        self.api_dir = os.path.join(os.path.dirname(__file__), "src/router/api")
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    def include_routers(self, app, current_dir, base_module_path):
        for entry in os.listdir(current_dir):
            entry_path = os.path.join(current_dir, entry)
            if os.path.isdir(entry_path):
                self.include_routers(app, entry_path, f"{base_module_path}.{entry}")
            elif entry.endswith(".py"):
                module_name = entry[:-3]
                module_path = f"{base_module_path}.{module_name}"
                module = importlib.import_module(module_path)
                if hasattr(module, "router"):
                    app.include_router(module.router)

class MiddlewareConfig:
    @staticmethod
    def add_cors_middleware(app):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
class OnnxConfig:
    def __init__(self):
        self.weights_path = os.path.join(os.path.dirname(__file__), ".local", "share", "models", "onnx")
    
    def get_model(self,name):
        model_path = os.path.join(self.weights_path, f"{name}.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {name} not found at {model_path}")
        return model_path
class PthConfig:
    def __init__(self):
        self.weights_path = os.path.join(os.path.dirname(__file__), ".local", "share", "models", "pth")
    
    def get_model(self, name):
        model_path = os.path.join(self.weights_path, f"{name}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {name} not found at {model_path}")
        return model_path
class PtConfig:
    def __init__(self):
        self.weights_path = os.path.join(os.path.dirname(__file__), ".local", "share", "models", "pt")
    
    def get_model(self, name):
        model_path = os.path.join(self.weights_path, f"{name}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {name} not found at {model_path}")
        return model_path  
    def get_model_path(self):
        return os.path.join(os.path.dirname(__file__), ".local", "share", "models", "pt")
class TensorflowConfig:
    def __init__(self):
        self.weights_path = os.path.join(os.path.dirname(__file__), ".local", "share", "models", "tensorflow")
    
    def get_model(self, name):
        model_path = os.path.join(self.weights_path, f"{name}.pb")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {name} not found at {model_path}")
        return model_path
class ModelConfig:
    @staticmethod
    def check_file_name(file_name):
        allowed_extensions = ['.onnx', '.pth', '.pb',"/pt"]
        if not any(file_name.endswith(ext) for ext in allowed_extensions):
            raise ValueError("File must be an ONNX (.onnx), PyTorch (.pth), or TensorFlow (.pb) model file.")
        return True
    @staticmethod
    def get_base_path():
        return os.path.join(os.path.dirname(__file__), ".local", "share", "models")
    
class DBConfig:
    def __init__(self):
        self.db_path = os.path.join(os.path.dirname(__file__), ".local", "share", "database", "app.db")
        self.db_dir = os.path.dirname(self.db_path)
        
    def get_database_url(self):
        os.makedirs(self.db_dir, exist_ok=True)
        return f"sqlite:///{self.db_path}"

    def get_database_path(self):
        return self.db_path
    
    def create_connection(self):
        os.makedirs(self.db_dir, exist_ok=True)
        return sqlite3.connect(self.db_path)
    
    def execute_query(self, query, params=None):
        conn = self.create_connection()
        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor.fetchall()
        finally:
            conn.close()
    
    def execute_many(self, query, params_list):
        conn = self.create_connection()
        try:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()
    
    def fetch_one(self, query, params=None):
        conn = self.create_connection()
        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchone()
        finally:
            conn.close()
    
    def fetch_all(self, query, params=None):
        conn = self.create_connection()
        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()
        finally:
            conn.close()
    