import sqlite3
from datetime import datetime
from typing import List, Dict, Any
import torch

# Global cache for pipeline to prevent reload
_pipeline = None

def get_finbert_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        # Determine device
        device = 0 if torch.cuda.is_available() else (-1 if not torch.backends.mps.is_available() else "mps")
        if device == "mps":
            device = "mps"
        _pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
    return _pipeline

def annotate_news_batch(news_items: List[Dict[str, Any]], db_path: str, use_mock: bool = False) -> List[Dict[str, Any]]:
    results = []
    to_compute = []
    to_compute_indices = []
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Step 1: Check cache
        for idx, item in enumerate(news_items):
            h = item["dedupe_hash"]
            cursor.execute(
                "SELECT sentiment_label, sentiment_score, positive_score, negative_score, neutral_score FROM news_annotations WHERE dedupe_hash = ?", 
                (h,)
            )
            row = cursor.fetchone()
            if row:
                results.append({
                    "dedupe_hash": h,
                    "model_id": "cached",
                    "sentiment_label": row["sentiment_label"],
                    "sentiment_score": row["sentiment_score"],
                    "positive_score": row["positive_score"],
                    "negative_score": row["negative_score"],
                    "neutral_score": row["neutral_score"],
                    "annotated_at": datetime.utcnow().isoformat()
                })
            else:
                # Placeholder to keep ordering
                results.append(None)
                to_compute.append(f"{item['title']}. {item['snippet']}")
                to_compute_indices.append(idx)
                
        # Step 2: Run inference on misses
        if to_compute:
            if use_mock:
                # Mock outputs for testing
                computed_results = [
                    {"label": "positive", "score": 0.95} for _ in to_compute
                ]
            else:
                pipe = get_finbert_pipeline()
                # FinBERT model outputs labels: positive, negative, neutral
                computed_results = pipe(to_compute)
                
            for idx_in_batch, out in enumerate(computed_results):
                orig_idx = to_compute_indices[idx_in_batch]
                item = news_items[orig_idx]
                h = item["dedupe_hash"]
                
                # Map scores
                label = out["label"]
                score = out["score"]
                
                pos, neg, neu = 0.0, 0.0, 0.0
                if label == "positive":
                    pos = score
                    neg = (1.5 - score) * 0.1  # ensure sum is not strict but sensible
                    neu = (1.5 - score) * 0.1
                elif label == "negative":
                    neg = score
                    pos = (1.5 - score) * 0.1
                    neu = (1.5 - score) * 0.1
                else:
                    neu = score
                    pos = (1.5 - score) * 0.1
                    neg = (1.5 - score) * 0.1
                    
                res = {
                    "dedupe_hash": h,
                    "model_id": "ProsusAI/finbert",
                    "sentiment_label": label,
                    "sentiment_score": score,
                    "positive_score": pos,
                    "negative_score": neg,
                    "neutral_score": neu,
                    "annotated_at": datetime.utcnow().isoformat()
                }
                
                # Save to DB cache
                cursor.execute("""
                    INSERT OR REPLACE INTO news_annotations 
                    (dedupe_hash, model_id, sentiment_label, sentiment_score, positive_score, negative_score, neutral_score, annotated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (h, res["model_id"], res["sentiment_label"], res["sentiment_score"], res["positive_score"], res["negative_score"], res["neutral_score"], res["annotated_at"]))
                
                results[orig_idx] = res
        conn.commit()
    finally:
        conn.close()
        
    return results
