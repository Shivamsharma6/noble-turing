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
            "source": "news",
        }
    ]

    # Run mock annotation (since downloading transformer model takes time)
    results = annotate_news_batch(news_items, db_path, use_mock=True)
    assert len(results) == 1
    assert results[0]["dedupe_hash"] == "hash_1"
    assert results[0]["sentiment_label"] in ["positive", "negative", "neutral"]

    # Check that per-class scores are present and sum approximately to 1.0
    scores = [
        results[0]["positive_score"],
        results[0]["negative_score"],
        results[0]["neutral_score"],
    ]
    score_sum = sum(scores)
    assert 0.9 <= score_sum <= 1.1, f"Scores should sum to ~1.0, got {score_sum}"

    # Check if cached in SQLite
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT sentiment_label, sentiment_score FROM news_annotations WHERE dedupe_hash='hash_1'"
    )
    row = cursor.fetchone()
    assert row is not None
    assert row["sentiment_label"] == results[0]["sentiment_label"]
    assert row["sentiment_score"] == results[0]["sentiment_score"]
    conn.close()


def test_finbert_cache_hit(tmp_path):
    """Verify that a cached annotation is returned without re-computation."""
    db_path = str(tmp_path / "test_cache_hit.db")
    init_db(db_path)

    news_items = [
        {
            "event_id": "e1",
            "dedupe_hash": "hash_cached",
            "title": "Cached news",
            "snippet": "Already annotated.",
            "matched_symbols": ["AAPL"],
            "observed_at": "2026-06-20T12:00:00",
            "source": "news",
        }
    ]

    # First call: compute and cache
    results1 = annotate_news_batch(news_items, db_path, use_mock=True)
    assert results1[0]["model_id"] == "ProsusAI/finbert"

    # Second call: should hit cache
    results2 = annotate_news_batch(news_items, db_path, use_mock=True)
    assert results2[0]["model_id"] == "cached"
    assert results2[0]["sentiment_label"] == results1[0]["sentiment_label"]


def test_finbert_dedupe_hash_order_preserved(tmp_path):
    """Verify that results are returned in the same order as input."""
    db_path = str(tmp_path / "test_order.db")
    init_db(db_path)

    news_items = [
        {
            "event_id": f"e{i}",
            "dedupe_hash": f"hash_{i}",
            "title": f"News {i}",
            "snippet": f"Snippet {i}",
            "matched_symbols": ["SPY"],
            "observed_at": "2026-06-20T12:00:00",
            "source": "news",
        }
        for i in range(5)
    ]

    results = annotate_news_batch(news_items, db_path, use_mock=True)
    assert len(results) == 5
    for i, r in enumerate(results):
        assert r["dedupe_hash"] == f"hash_{i}"
