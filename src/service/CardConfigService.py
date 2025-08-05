import sqlite3
from typing import List, Dict, Optional
from config import DBConfig

class CardConfigService:
    """
    Service for managing card categories and types configuration in database
    """
    
    def __init__(self):
        self.db_config = DBConfig()
        self._initialize_tables()
    
    def _initialize_tables(self):
        """Initialize database tables for card configuration"""
        create_categories_table = """
        CREATE TABLE IF NOT EXISTS card_categories (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            nameEn TEXT NOT NULL,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        create_types_table = """
        CREATE TABLE IF NOT EXISTS card_types (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            nameEn TEXT NOT NULL,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Execute table creation
        self.db_config.execute_query(create_categories_table)
        self.db_config.execute_query(create_types_table)
        
        # Initialize default data if tables are empty
        self._initialize_default_data()
    
    def _initialize_default_data(self):
        """Initialize default card categories and types if not exists"""
        
        # Check if data already exists
        categories_count = self.db_config.fetch_one("SELECT COUNT(*) FROM card_categories")[0]
        types_count = self.db_config.fetch_one("SELECT COUNT(*) FROM card_types")[0]
        
        if categories_count == 0:
            # Insert default card categories
            default_categories = [
                (0, "Thẻ Căn Cước Công Dân", "Citizens Card", 1),
                (1, "Giấy Phép Lái Xe", "Driving License", 1),
                (2, "Thẻ Bảo Hiểm Y Tế", "Health Insurance Card", 1),
                (3, "Thẻ Ngân Hàng", "Bank Card", 1),
                (4, "Thẻ Sinh Viên", "Student Card", 1),
                (5, "Thẻ Căn Cước Công Dân Mới", "New Citizens Card", 1),
                (6, "Thẻ Căn Cước Công Dân Cũ", "Old Citizens Card", 1),
            ]
            
            insert_categories_query = """
            INSERT OR REPLACE INTO card_categories (id, name, nameEn, is_active) 
            VALUES (?, ?, ?, ?)
            """
            self.db_config.execute_many(insert_categories_query, default_categories)
            print("✅ Default card categories initialized")
        
        if types_count == 0:
            # Insert default card types
            default_types = [
                (0, "Mặt Trước", "Front", 1),
                (1, "Mặt Sau", "Back", 1)
            ]
            
            insert_types_query = """
            INSERT OR REPLACE INTO card_types (id, name, nameEn, is_active) 
            VALUES (?, ?, ?, ?)
            """
            self.db_config.execute_many(insert_types_query, default_types)
            print("✅ Default card types initialized")
    
    def get_card_categories(self, include_inactive: bool = False) -> List[Dict]:
        """Get all card categories from database"""
        if include_inactive:
            query = "SELECT id, name, nameEn, is_active FROM card_categories ORDER BY id"
        else:
            query = "SELECT id, name, nameEn, is_active FROM card_categories WHERE is_active = 1 ORDER BY id"
        
        rows = self.db_config.fetch_all(query)
        return [
            {
                "id": row[0],
                "name": row[1],
                "nameEn": row[2],
                "is_active": bool(row[3]) if len(row) > 3 else True
            }
            for row in rows
        ]
    
    def get_card_types(self, include_inactive: bool = False) -> List[Dict]:
        """Get all card types from database"""
        if include_inactive:
            query = "SELECT id, name, nameEn, is_active FROM card_types ORDER BY id"
        else:
            query = "SELECT id, name, nameEn, is_active FROM card_types WHERE is_active = 1 ORDER BY id"
        
        rows = self.db_config.fetch_all(query)
        return [
            {
                "id": row[0],
                "name": row[1],
                "nameEn": row[2],
                "is_active": bool(row[3]) if len(row) > 3 else True
            }
            for row in rows
        ]
    
    def get_card_category_by_id(self, category_id: int) -> Optional[Dict]:
        """Get specific card category by ID"""
        query = "SELECT id, name, nameEn, is_active FROM card_categories WHERE id = ? AND is_active = 1"
        row = self.db_config.fetch_one(query, (category_id,))
        
        if row:
            return {
                "id": row[0],
                "name": row[1],
                "nameEn": row[2],
                "is_active": bool(row[3]) if len(row) > 3 else True
            }
        return None
    
    def get_card_type_by_id(self, type_id: int) -> Optional[Dict]:
        """Get specific card type by ID"""
        query = "SELECT id, name, nameEn, is_active FROM card_types WHERE id = ? AND is_active = 1"
        row = self.db_config.fetch_one(query, (type_id,))
        
        if row:
            return {
                "id": row[0],
                "name": row[1],
                "nameEn": row[2],
                "is_active": bool(row[3]) if len(row) > 3 else True
            }
        return None
    
    def add_card_category(self, name: str, nameEn: str) -> int:
        """Add new card category"""
        # Get next available ID
        max_id_row = self.db_config.fetch_one("SELECT MAX(id) FROM card_categories")
        next_id = (max_id_row[0] + 1) if max_id_row[0] is not None else 0
        
        query = """
        INSERT INTO card_categories (id, name, nameEn, is_active) 
        VALUES (?, ?, ?, 1)
        """
        self.db_config.execute_query(query, (next_id, name, nameEn))
        return next_id
    
    def add_card_type(self, name: str, nameEn: str) -> int:
        """Add new card type"""
        # Get next available ID
        max_id_row = self.db_config.fetch_one("SELECT MAX(id) FROM card_types")
        next_id = (max_id_row[0] + 1) if max_id_row[0] is not None else 0
        
        query = """
        INSERT INTO card_types (id, name, nameEn, is_active) 
        VALUES (?, ?, ?, 1)
        """
        self.db_config.execute_query(query, (next_id, name, nameEn))
        return next_id
    
    def update_card_category(self, category_id: int, name: str = None, nameEn: str = None, is_active: bool = None) -> bool:
        """Update existing card category"""
        updates = []
        params = []
        
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        
        if nameEn is not None:
            updates.append("nameEn = ?")
            params.append(nameEn)
        
        if is_active is not None:
            updates.append("is_active = ?")
            params.append(1 if is_active else 0)
        
        if not updates:
            return False
        
        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(category_id)
        
        query = f"UPDATE card_categories SET {', '.join(updates)} WHERE id = ?"
        result = self.db_config.execute_query(query, params)
        return True
    
    def update_card_type(self, type_id: int, name: str = None, nameEn: str = None, is_active: bool = None) -> bool:
        """Update existing card type"""
        updates = []
        params = []
        
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        
        if nameEn is not None:
            updates.append("nameEn = ?")
            params.append(nameEn)
        
        if is_active is not None:
            updates.append("is_active = ?")
            params.append(1 if is_active else 0)
        
        if not updates:
            return False
        
        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(type_id)
        
        query = f"UPDATE card_types SET {', '.join(updates)} WHERE id = ?"
        result = self.db_config.execute_query(query, params)
        return True
    
    def delete_card_category(self, category_id: int) -> bool:
        """Soft delete card category (set is_active = 0)"""
        query = "UPDATE card_categories SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
        self.db_config.execute_query(query, (category_id,))
        return True
    
    def delete_card_type(self, type_id: int) -> bool:
        """Soft delete card type (set is_active = 0)"""
        query = "UPDATE card_types SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
        self.db_config.execute_query(query, (type_id,))
        return True
    
    def get_config_summary(self) -> Dict:
        """Get summary of current configuration"""
        categories = self.get_card_categories()
        types = self.get_card_types()
        
        return {
            "card_categories_count": len(categories),
            "card_types_count": len(types),
            "card_categories": categories,
            "card_types": types,
            "database_path": self.db_config.get_database_path()
        }

# Singleton instance for global use
card_config_service = CardConfigService()
