import sqlite3

def get_db_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str):
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        # News sentiment annotation cache table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_annotations (
            dedupe_hash TEXT PRIMARY KEY,
            model_id TEXT,
            sentiment_label TEXT,
            sentiment_score REAL,
            positive_score REAL,
            negative_score REAL,
            neutral_score REAL,
            annotated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Background training job tracking table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            model_id TEXT PRIMARY KEY,
            status TEXT,
            model_family TEXT,
            model_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            metrics_json TEXT,
            error_message TEXT
        );
        """)
        
        # Safety audit records tracking table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_records (
            request_id TEXT PRIMARY KEY,
            endpoint TEXT,
            model_id TEXT,
            input_hash TEXT,
            output_hash TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            status TEXT,
            error_message TEXT
        );
        """)
        conn.commit()
    finally:
        conn.close()
