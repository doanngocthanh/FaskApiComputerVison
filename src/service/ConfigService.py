from config import OnnxConfig, PthConfig, TensorflowConfig, ModelConfig,DBConfig
db = DBConfig()
class ConfigService:
            def __init__(self):
                self.db = DBConfig()
                self.db.execute_query(self.create_config_table())
            
            def create_config_table(self):
                return """
                CREATE TABLE IF NOT EXISTS config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_type VARCHAR(50) NOT NULL,
                    config_name VARCHAR(100) NOT NULL,
                    config_value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            
            def insert_config(self, config_type, config_name, config_value):
                query = "INSERT INTO config (config_type, config_name, config_value) VALUES (?, ?, ?)"
                self.db.execute_query(query, (config_type, config_name, config_value))
            
            def get_config(self, config_type, config_name):
                query = "SELECT config_value FROM config WHERE config_type = ? AND config_name = ?"
                result = self.db.fetch_one(query, (config_type, config_name))
                if result:
                    return result[0]
                return None
            
            def update_config(self, config_type, config_name, config_value):
                query = "UPDATE config SET config_value = ?, updated_at = CURRENT_TIMESTAMP WHERE config_type = ? AND config_name = ?"
                self.db.execute_query(query, (config_value, config_type, config_name))
            
            def delete_config(self, config_type, config_name):
                query = "DELETE FROM config WHERE config_type = ? AND config_name = ?"
                self.db.execute_query(query, (config_type, config_name))
            
            def get_all_configs(self, config_type=None):
                if config_type:
                    query = "SELECT * FROM config WHERE config_type = ?"
                    return self.db.fetch_all(query, (config_type,))
                else:
                    query = "SELECT * FROM config"
                    return self.db.fetch_all(query)