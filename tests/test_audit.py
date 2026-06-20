import os
import sqlite3
import pytest
from app.database import init_db
from app.audit import log_audit_record, redact_secrets

def test_audit_records_creation(tmp_path):
    db_path = str(tmp_path / "test_audit.db")
    init_db(db_path)
    
    log_audit_record(
        db_path=db_path,
        request_id="req-123",
        endpoint="/score_daily_mover_candidates",
        model_id="m-123",
        input_data={"api_key": "secret-key", "model_id": "m-123"},
        output_data={"status": "completed"},
        status="completed"
    )
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM audit_records WHERE request_id = 'req-123'")
    row = cursor.fetchone()
    assert row is not None
    assert row["endpoint"] == "/score_daily_mover_candidates"
    assert "secret-key" not in row["input_hash"]
    conn.close()

def test_redact_secrets():
    data = {"api_key": "my-secret", "headers": {"X-API-Key": "my-secret"}, "other": "ok"}
    redacted = redact_secrets(data)
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["headers"]["X-API-Key"] == "[REDACTED]"
    assert redacted["other"] == "ok"
