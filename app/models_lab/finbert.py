import sqlite3
from datetime import datetime, timezone
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
        # return_all_scores returns probabilities for ALL classes, not just the dominant one
        _pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=device,
            return_all_scores=True,
        )
    return _pipeline


def annotate_news_batch(
    news_items: List[Dict[str, Any]],
    db_path: str,
    use_mock: bool = False,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    to_compute: List[str] = []
    to_compute_indices: List[int] = []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Step 1: Check cache
        for idx, item in enumerate(news_items):
            h = item["dedupe_hash"]
            cursor.execute(
                "SELECT sentiment_label, sentiment_score, positive_score, negative_score, neutral_score FROM news_annotations WHERE dedupe_hash = ?",
                (h,),
            )
            row = cursor.fetchone()
            if row:
                results.append(
                    {
                        "dedupe_hash": h,
                        "model_id": "cached",
                        "sentiment_label": row["sentiment_label"],
                        "sentiment_score": row["sentiment_score"],
                        "positive_score": row["positive_score"],
                        "negative_score": row["negative_score"],
                        "neutral_score": row["neutral_score"],
                        "annotated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            else:
                results.append(None)
                to_compute.append(f"{item['title']}. {item['snippet']}")
                to_compute_indices.append(idx)

        # Step 2: Run inference on misses
        if to_compute:
            if use_mock:
                # Mock outputs for testing
                computed_results = [
                    [
                        {"label": "positive", "score": 0.95},
                        {"label": "negative", "score": 0.03},
                        {"label": "neutral", "score": 0.02},
                    ]
                    for _ in to_compute
                ]
            else:
                pipe = get_finbert_pipeline()
                # FinBERT with return_all_scores returns list of lists of dicts,
                # one dict per class: positive, negative, neutral
                computed_results = pipe(to_compute)

            for idx_in_batch, out in enumerate(computed_results):
                orig_idx = to_compute_indices[idx_in_batch]
                item = news_items[orig_idx]
                h = item["dedupe_hash"]

                # Map per-class probabilities from FinBERT output
                # FinBERT returns classes in order: positive, negative, neutral
                scores_map: Dict[str, float] = {}
                for cls_entry in out:
                    scores_map[cls_entry["label"]] = cls_entry["score"]

                positive_score = scores_map.get("positive", 0.0)
                negative_score = scores_map.get("negative", 0.0)
                neutral_score = scores_map.get("neutral", 0.0)

                # Dominant class is the one with highest probability
                label = max(scores_map, key=scores_map.get)
                sentiment_score = scores_map[label]

                now = datetime.now(timezone.utc).isoformat()

                res = {
                    "dedupe_hash": h,
                    "model_id": "ProsusAI/finbert",
                    "sentiment_label": label,
                    "sentiment_score": sentiment_score,
                    "positive_score": positive_score,
                    "negative_score": negative_score,
                    "neutral_score": neutral_score,
                    "annotated_at": now,
                }

                # Save to DB cache
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO news_annotations
                    (dedupe_hash, model_id, sentiment_label, sentiment_score, positive_score, negative_score, neutral_score, annotated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        h,
                        res["model_id"],
                        res["sentiment_label"],
                        res["sentiment_score"],
                        res["positive_score"],
                        res["negative_score"],
                        res["neutral_score"],
                        res["annotated_at"],
                    ),
                )

                results[orig_idx] = res

        conn.commit()
    finally:
        conn.close()

    return results
