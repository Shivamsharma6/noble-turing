import json
import hashlib
import sqlite3
import copy
from datetime import datetime, timezone
from typing import Any, Dict

def redact_secrets(data: Any) -> Any:
    if isinstance(data, dict):
        copy_dict = {}
        for k, v in data.items():
            if k.lower() in ("api_key", "x-api-key", "token", "secret", "password"):
                copy_dict[k] = "[REDACTED]"
            else:
                copy_dict[k] = redact_secrets(v)
        return copy_dict
    elif isinstance(data, list):
        return [redact_secrets(item) for item in data]
    return data

def hash_payload(data: Any) -> str:
    redacted = redact_secrets(data)
    dump = json.dumps(redacted, sort_keys=True)
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()

def log_audit_record(
    db_path: str,
    request_id: str,
    endpoint: str,
    model_id: str | None,
    input_data: Any,
    output_data: Any,
    status: str,
    error_message: str | None = None,
    started_at: datetime | None = None,
):
    if started_at is None:
        started_at = datetime.now(timezone.utc)
    completed_at = datetime.now(timezone.utc)
    
    in_hash = hash_payload(input_data)
    out_hash = hash_payload(output_data) if output_data is not None else None
    
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO audit_records 
            (request_id, endpoint, model_id, input_hash, output_hash, started_at, completed_at, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            request_id,
            endpoint,
            model_id,
            in_hash,
            out_hash,
            started_at.isoformat(),
            completed_at.isoformat(),
            status,
            error_message
        ))
        conn.commit()
    finally:
        conn.close()
