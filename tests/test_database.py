import os
import sqlite3
from app.database import get_db_connection, init_db

def test_db_init_and_tables(tmp_path):
    db_path = str(tmp_path / "test_news_cache.db")
    
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        assert "news_annotations" in tables
        assert "jobs" in tables
        
        # Verify column definitions for news_annotations
        cursor.execute("PRAGMA table_info(news_annotations);")
        columns = {row[1] for row in cursor.fetchall()}
        assert "dedupe_hash" in columns
        assert "sentiment_label" in columns
        assert "sentiment_score" in columns
    finally:
        conn.close()
