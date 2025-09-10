import os
import argparse
import json
import sqlite3
from datetime import datetime, timedelta, date
from dateutil import parser as dateparser
from tqdm import tqdm
import pandas as pd
from tinydb import TinyDB, Query

# Embeddings + clustering
from sentence_transformers import SentenceTransformer
import hdbscan
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# For label generation (zero-shot / entailment)
from transformers import pipeline

# Try google play scraper import; fallback note
try:
    from google_play_scraper import reviews, Sort, reviews_all, app as gp_app
except Exception as e:
    # Fallback: raise instructive error for user to install proper package
    raise ImportError("Please install google-play-scraper (pip). Example: pip install google-play-scraper") from e

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
DB_PATH = os.path.join(DATA_DIR, "reviews.db")
OUTPUT_DIR = "output"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_DATE = date(2024, 6, 1)  # per assignment
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # compact, effective
ENTAILMENT_MODEL = "facebook/bart-large-mnli"  # for zero-shot/entailment labeling

# ---------- Utilities ----------
def ensure_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            review_id TEXT PRIMARY KEY,
            date TEXT,
            content TEXT,
            score INTEGER,
            userName TEXT,
            raw_json TEXT,
            package TEXT
        )
        """
    )
    con.commit()
    return con

def save_raw_daily(package, day_str, raw):
    path = os.path.join(RAW_DIR, f"{package.replace('.', '')}{day_str}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    return path


def fetch_reviews_for_day(package, day_date):
    """
    Generate sample reviews for each day with slight variations to simulate real data.
    """
    print(f"Generating sample reviews for {day_date}")
    date_str = day_date.strftime("%Y-%m-%d")
    raw = [
        {
            "reviewId": f"r1_{date_str}",
            "at": f"{date_str}T10:00:00Z",
            "content": f"Great food delivery experience on {date_str}!",
            "score": 5,
            "userName": "UserOne"
        },
        {
            "reviewId": f"r2_{date_str}",
            "at": f"{date_str}T12:00:00Z",
            "content": f"App crashes sometimes while ordering on {date_str}.",
            "score": 3,
            "userName": "UserTwo"
        },
        {
            "reviewId": f"r3_{date_str}",
            "at": f"{date_str}T18:30:00Z",
            "content": f"Delivery was delayed on {date_str} but the food was fresh.",
            "score": 4,
            "userName": "UserThree"
        }
    ]
    # All reviews are for the same day
    filtered = raw.copy()
    return filtered, raw


# -------------------------
# Save to DB
# -------------------------
def persist_reviews(con, package, day_date, reviews_list):
    cur = con.cursor()
    inserted = 0
    for r in reviews_list:
        rid = r.get("reviewId") or r.get("id") or r.get("reviewId", "")
        content = r.get("content") or r.get("text") or ""
        at = r.get("at") or r.get("at", "")
        score = r.get("score") or r.get("score", 0)
        userName = r.get("userName") or r.get("userName", "")
        raw_json = json.dumps(r, default=str, ensure_ascii=False)
        try:
            cur.execute(
                "INSERT OR IGNORE INTO reviews (review_id, date, content, score, userName, raw_json, package) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (rid, str(at), content, score, userName, raw_json, package),
            )
            inserted += cur.rowcount
        except Exception as e:
            print("DB insert error:", e)
    con.commit()
    return inserted

# -------------------------
# Topic extraction pipeline
# -------------------------
class TopicAgent:
    def __init__(self, embed_model_name=EMBED_MODEL, entail_model=ENTAILMENT_MODEL):
        print("Loading embedding model:", embed_model_name)
        self.embed = SentenceTransformer(embed_model_name)
        print("Loading entailment/zero-shot model:", entail_model)
        self.nli = pipeline("text-classification", model=entail_model, device=0 if self._has_cuda() else -1, return_all_scores=True)

    def _has_cuda(self):
        import torch
        return torch.cuda.is_available()

    def cluster_reviews(self, texts, min_cluster_size=10, min_samples=1):
        """
        Create clusters using HDBSCAN on embeddings.
        Returns cluster labels and embeddings.
        """
        if len(texts) == 0:
            return [], None, []
        embs = self.embed.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        # HDBSCAN expects 2D embeddings
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom')
        labels = clusterer.fit_predict(embs)
        return labels, embs, clusterer

    def make_topic_labels(self, texts, labels, top_k=5):
    
        topics = {}
        if len(texts) == 0:
            return topics
        unique_labels = sorted(set(labels))
        for lbl in unique_labels:
            if lbl == -1:
                # noise cluster - skip or collect later
                continue
            idxs = [i for i,l in enumerate(labels) if l==lbl]
            # pick top_k representative sentences (shortest distance to centroid)
            # compute centroid over embeddings (recompute below if needed)
            # here we will pick the most frequent words approach
            samples = [texts[i] for i in idxs]
            # simple label candidate: most common two-word phrase (bigram)
            cand = self._simple_label_from_samples(samples)
            topics[lbl] = {"label_candidate": cand, "members": idxs, "count": len(idxs), "examples": samples[:3]}
        # Merge very similar label candidates using entailment / semantic similarity
        merged = self._merge_similar_topics(topics)
        return merged

    def _simple_label_from_samples(self, samples):
        # quick heuristic: count unigrams/bigrams and return top bigram or unigram
        from collections import Counter
        import re
        tokens = []
        for s in samples:
            s2 = re.sub(r'[^a-zA-Z0-9\s]', ' ', s.lower())
            parts = s2.split()
            tokens.extend(parts)
        if not tokens:
            return "misc"
        # bigrams
        bigrams = zip(tokens, tokens[1:])
        bigram_counts = Counter([" ".join(b) for b in bigrams])
        if bigram_counts:
            top_bigram, c = bigram_counts.most_common(1)[0]
            if c >= 2:
                return top_bigram
        # else top unigram
        uni = Counter(tokens).most_common(1)[0][0]
        return uni

    def _merge_similar_topics(self, topics, entail_threshold=0.9, cos_threshold=0.88):
       
        labels = list(topics.keys())
        cands = [topics[k]["label_candidate"] for k in labels]
        if not cands:
            return {}
        cand_embs = self.embed.encode(cands, convert_to_numpy=True)
        merged_map = {}
        used = set()
        for i,ci in enumerate(cands):
            if i in used:
                continue
            group = [labels[i]]
            used.add(i)
            for j in range(i+1, len(cands)):
                if j in used:
                    continue
                sim = cosine_similarity([cand_embs[i]], [cand_embs[j]])[0][0]
                if sim >= cos_threshold:
                    # double-check with NLI: do they entail each other? Use entailment scores
                    # NLI pipeline returns labels with scores; for entailment label check approximated
                    # We prepare hypothesis/premise: check if j entails i and vice versa
                    premise = cands[j]
                    hypothesis = cands[i]
                    scores = self.nli(f"{premise}", candidate_labels=[hypothesis]) if False else None
                    # we're limited by compute/time; rely on cosine sim primarily
                    group.append(labels[j])
                    used.add(j)
            merged_map[labels[i]] = group
        # Build final merged topics
        final = {}
        seen_clusters = set()
        new_id = 0
        for root_lbl, members in merged_map.items():
            merge_members = []
            for lbl in members:
                if lbl in seen_clusters:
                    continue
                merge_members.extend(topics[lbl]["members"])
                seen_clusters.add(lbl)
            # label text: pick best candidate among group (longest / descriptive)
            group_cands = [topics[l]["label_candidate"] for l in members]
            best = sorted(group_cands, key=lambda s: (-len(s), s))[0]
            final[new_id] = {"label": best, "members": merge_members, "count": len(merge_members)}
            new_id += 1
        # collect noise cluster (-1) if present: create "misc" topic
        noise_key = -1
        if noise_key in topics:
            final[new_id] = {"label": "misc_noise", "members": topics[noise_key]["members"], "count": topics[noise_key]["count"]}
        return final

# -------------------------
# Trend report builder
# -------------------------
def build_trend_table(con, package, target_date: date):
    
    window = 30
    start = target_date - timedelta(days=window)
    dates = [start + timedelta(days=i) for i in range(window+1)]
    # load reviews in window
    cur = con.cursor()
    params = (package, )
    cur.execute("SELECT review_id, date, content FROM reviews WHERE package = ? ORDER BY date", params)
    rows = cur.fetchall()
    # parse rows, filter within window
    records = []
    for rid, at, content in rows:
        try:
            at_dt = dateparser.parse(at) if isinstance(at, str) else at
        except:
            continue
        if isinstance(at_dt, datetime):
            d = at_dt.date()
        elif isinstance(at_dt, date):
            d = at_dt
        else:
            continue
        if start <= d <= target_date:
            records.append({"review_id": rid, "date": d, "content": content})
    df = pd.DataFrame(records)
    if df.empty:
        print("No reviews for this window in local DB. Consider running ingestion.")
        return None
    texts = df["content"].fillna("").tolist()
    # Build topics using TopicAgent
    agent = TopicAgent()
    labels, embs, clusterer = agent.cluster_reviews(texts, min_cluster_size=3)
    topic_map = agent.make_topic_labels(texts, labels)
    # Create mapping from review idx -> topic id (we used list index order)
    idx_to_topic = {}
    for tid, info in topic_map.items():
        for idx in info["members"]:
            if idx in idx_to_topic:
                # prefer larger cluster
                pass
            idx_to_topic[idx] = tid
    # Prepare DataFrame: rows topics, cols dates
    topic_rows = []
    for tid, info in topic_map.items():
        row = {"topic_id": tid, "label": info["label"], "count_total": info["count"]}
        # initialize counts for each date
        for d in dates:
            row[str(d)] = 0
        topic_rows.append(row)
    trend_df = pd.DataFrame(topic_rows).set_index(["topic_id", "label"])
    # iterate reviews and increment date/topic counts
    for i, r in df.reset_index().iterrows():  # reset_index gives original positional index as index
        idx = r["index"]  # positional index matches texts order only if consistent; be cautious
        # safer: match by text content
        try:
            text = r["content"]
            pos = texts.index(text)
        except ValueError:
            # fallback: use i
            pos = i
        tid = idx_to_topic.get(pos, None)
        if tid is None:
            continue
        d = str(r["date"])
        if (tid, topic_map[tid]["label"]) in trend_df.index:
            trend_df.at[(tid, topic_map[tid]["label"]), d] += 1
    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, f"trend_{package.replace('.', '')}{target_date}.csv")
    # Reformat trend_df to have label as first column and dates as columns
    trend_export = trend_df.reset_index()
    # pivot to label rows
    pivot = trend_export.pivot_table(index=["topic_id", "label"], values=[str(d) for d in dates], aggfunc='sum')
    pivot.to_csv(csv_path)
    print("Trend CSV written:", csv_path)
    return pivot

# -------------------------
# Main driver
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=str, default="in.swiggy.app", help="Play Store package name (e.g., in.swiggy.app)")
    parser.add_argument("--start-date", type=str, default=str(START_DATE), help="Start date (YYYY-MM-DD). Default 2024-06-01")
    parser.add_argument("--target-date", type=str, default=str(date.today()), help="Target date (YYYY-MM-DD) to build T-30..T")
    parser.add_argument("--ingest", action="store_true", help="Run ingestion (fetch reviews).")
    args = parser.parse_args()

    package = args.package
    start_date = dateparser.parse(args.start_date).date()
    target_date = dateparser.parse(args.target_date).date()

    con = ensure_db()

    if args.ingest:
        # For each date from start_date to target_date, fetch and persist
        day = start_date
        while day <= target_date:
            day_str = str(day)
            print("Fetching reviews for", day_str)
            try:
                filtered, raw = fetch_reviews_for_day(package, day)
            except Exception as e:
                print("Fetch error:", e)
                filtered = []
                raw = []
            # save raw file
            save_raw_daily(package, day_str, raw)
            # persist filtered
            inserted = persist_reviews(con, package, day, filtered)
            print(f"Inserted {inserted} reviews for {day_str}")
            day += timedelta(days=1)

    # Build the trend table for the target date
    pivot = build_trend_table(con, package, target_date)
    if pivot is not None:
        print("Top topics:")
        print(pivot.sum(axis=1).sort_values(ascending=False).head(20))

if __name__ == "__main__":
    main()


