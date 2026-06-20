import os
import sqlite3
from app.database import get_db_connection, init_db

def test_db_init_and_tables():
    db_path = "test_news_cache.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cursor.fetchall()}
    assert "news_annotations" in tables
    assert "jobs" in tables
    conn.close()
    
    if os.path.exists(db_path):
        os.remove(db_path)
