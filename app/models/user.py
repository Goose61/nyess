import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

class User(UserMixin):
    """User model for authentication."""
    
    # Database path - uses the instance directory
    DB_PATH = os.path.join(os.getcwd(), 'instance', 'users.db')
    
    def __init__(self, id=None, username=None, email=None, password_hash=None):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
    
    @classmethod
    def init_db(cls):
        """Initialize the users database."""
        os.makedirs(os.path.dirname(cls.DB_PATH), exist_ok=True)
        
        conn = sqlite3.connect(cls.DB_PATH)
        cursor = conn.cursor()
        
        # Create users table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
        ''')
        
        conn.commit()
        conn.close()
    
    @classmethod
    def get_by_id(cls, user_id):
        """Get a user by ID."""
        conn = sqlite3.connect(cls.DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user_data = cursor.fetchone()
        
        conn.close()
        
        if user_data:
            return cls(
                id=user_data['id'],
                username=user_data['username'],
                email=user_data['email'],
                password_hash=user_data['password_hash']
            )
        return None
    
    @classmethod
    def get_by_username(cls, username):
        """Get a user by username."""
        conn = sqlite3.connect(cls.DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user_data = cursor.fetchone()
        
        conn.close()
        
        if user_data:
            return cls(
                id=user_data['id'],
                username=user_data['username'],
                email=user_data['email'],
                password_hash=user_data['password_hash']
            )
        return None
    
    @classmethod
    def get_by_email(cls, email):
        """Get a user by email."""
        conn = sqlite3.connect(cls.DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user_data = cursor.fetchone()
        
        conn.close()
        
        if user_data:
            return cls(
                id=user_data['id'],
                username=user_data['username'],
                email=user_data['email'],
                password_hash=user_data['password_hash']
            )
        return None
    
    def set_password(self, password):
        """Set the password hash."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check the password against the hash."""
        return check_password_hash(self.password_hash, password)
    
    def save(self):
        """Save the user to the database."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        
        if self.id is None:
            # New user
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (self.username, self.email, self.password_hash)
            )
            self.id = cursor.lastrowid
        else:
            # Update existing user
            cursor.execute(
                "UPDATE users SET username = ?, email = ?, password_hash = ? WHERE id = ?",
                (self.username, self.email, self.password_hash, self.id)
            )
        
        conn.commit()
        conn.close() 