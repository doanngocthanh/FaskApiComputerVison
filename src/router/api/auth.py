from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
from config import DBConfig

router = APIRouter()
db_config = DBConfig()
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = "your-secret-key-here"  # Change this to a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Pydantic models for request/response
class User(BaseModel):
    name: str
    email: EmailStr

class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    name: str
    email: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Initialize database table
def init_db():
    """Initialize the users table if it doesn't exist"""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    db_config.execute_query(create_table_query)

# Password hashing functions
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# JWT token functions
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Initialize database when module loads
init_db()

@router.post("/register", response_model=UserResponse)
async def register_user(user: UserRegister):
    """Register a new user"""
    try:
        # Hash the password
        password_hash = hash_password(user.password)
        
        # Insert user
        db_config.execute_query(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (user.name, user.email, password_hash)
        )
        
        # Get the created user
        created_user = db_config.fetch_one(
            "SELECT id, name, email FROM users WHERE email = ?",
            (user.email,)
        )
        
        if created_user:
            return UserResponse(id=created_user[0], name=created_user[1], email=created_user[2])
        else:
            raise HTTPException(status_code=500, detail="Failed to create user")
            
    except Exception as e:
        if "UNIQUE constraint failed" in str(e):
            raise HTTPException(status_code=400, detail="Email already exists")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/login", response_model=Token)
async def login_user(user: UserLogin):
    """Login user and return access token"""
    # Get user from database
    db_user = db_config.fetch_one(
        "SELECT id, name, email, password_hash FROM users WHERE email = ?",
        (user.email,)
    )
    
    if not db_user or not verify_password(user.password, db_user[3]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Create access token
    access_token = create_access_token(data={"sub": db_user[0]})
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/profile", response_model=UserResponse)
async def get_profile(current_user_id: int = Depends(verify_token)):
    """Get current user profile"""
    user = db_config.fetch_one(
        "SELECT id, name, email FROM users WHERE id = ?",
        (current_user_id,)
    )
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(id=user[0], name=user[1], email=user[2])

@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, current_user_id: int = Depends(verify_token)):
    """Get a user by ID (requires authentication)"""
    user = db_config.fetch_one(
        "SELECT id, name, email FROM users WHERE id = ?",
        (user_id,)
    )
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(id=user[0], name=user[1], email=user[2])

@router.get("/users")
async def get_all_users(current_user_id: int = Depends(verify_token)):
    """Get all users (requires authentication)"""
    users = db_config.fetch_all("SELECT id, name, email FROM users")
    
    return [
        UserResponse(id=user[0], name=user[1], email=user[2])
        for user in users
    ]

@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: int, user: User, current_user_id: int = Depends(verify_token)):
    """Update a user (requires authentication)"""
    # Check if user exists
    existing_user = db_config.fetch_one(
        "SELECT id FROM users WHERE id = ?",
        (user_id,)
    )
    
    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        # Update user
        db_config.execute_query(
            "UPDATE users SET name = ?, email = ? WHERE id = ?",
            (user.name, user.email, user_id)
        )
        
        # Return updated user
        updated_user = db_config.fetch_one(
            "SELECT id, name, email FROM users WHERE id = ?",
            (user_id,)
        )
        
        return UserResponse(id=updated_user[0], name=updated_user[1], email=updated_user[2])
        
    except Exception as e:
        if "UNIQUE constraint failed" in str(e):
            raise HTTPException(status_code=400, detail="Email already exists")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/users/{user_id}")
async def delete_user(user_id: int, current_user_id: int = Depends(verify_token)):
    """Delete a user (requires authentication)"""
    # Check if user exists
    existing_user = db_config.fetch_one(
        "SELECT id FROM users WHERE id = ?",
        (user_id,)
    )
    
    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Delete user
    db_config.execute_query(
        "DELETE FROM users WHERE id = ?",
        (user_id,)
    )
    
    return {"message": "User deleted successfully"}
