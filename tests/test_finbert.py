import os
import sqlite3
import pytest
from app.database import init_db
from app.models_lab.finbert import annotate_news_batch

def test_finbert_mock_annotation(tmp_path):
    db_path = str(tmp_path / "test_news_cache_finbert.db")
    init_db(db_path)
    
    news_items = [
        {
            "event_id": "e1",
            "dedupe_hash": "hash_1",
            "title": "Stock market surges",
            "snippet": "Strong earnings drive market higher.",
            "matched_symbols": ["SPY"],
            "observed_at": "2026-06-20T12:00:00",
            "source": "news"
        }
    ]
    
    # Run mock annotation (since downloading transformer model takes time)
    results = annotate_news_batch(news_items, db_path, use_mock=True)
    assert len(results) == 1
    assert results[0]["dedupe_hash"] == "hash_1"
    assert results[0]["sentiment_label"] in ["positive", "negative", "neutral"]
    
    # Check if cached in SQLite
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT sentiment_label, sentiment_score FROM news_annotations WHERE dedupe_hash='hash_1'")
    row = cursor.fetchone()
    assert row is not None
    assert row["sentiment_label"] == results[0]["sentiment_label"]
    assert row["sentiment_score"] == results[0]["sentiment_score"]
    conn.close()
