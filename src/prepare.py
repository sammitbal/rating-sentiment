"""Convert the raw All_Beauty jsonl.gz files into cleaned parquet tables.

Usage:  python src/prepare.py
"""
import gzip
import html
import json
import re

import numpy as np
import pandas as pd

from config import DATA_URL, META, META_RAW, PROCESSED, RAW_SUMMARY, REVIEWS, REVIEWS_RAW

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
# Amazon auto-filled empty titles with the star rating ("Five Stars"). Those titles leak the
# rating into the text, which would make ratings and text look more consistent than they are.
AUTO_TITLE_RE = re.compile(r"^(one|two|three|four|five) stars?$", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def clean_text(s):
    if not s:
        return ""
    s = TAG_RE.sub(" ", html.unescape(s))
    return WS_RE.sub(" ", s).strip()


def join_title_text(title, text):
    if not title:
        return text
    if not text:
        return title
    sep = " " if title[-1] in ".!?" else ". "
    return title + sep + text


def read_jsonl(path):
    if not path.exists():
        raise SystemExit(f"Missing {path}. Download it from {DATA_URL}")
    with open(path, "rb") as fp:
        if fp.read(2) != b"\x1f\x8b":
            raise SystemExit(f"{path} is not a valid gzip file (it may have been re-saved as text). "
                             f"Download it again from {DATA_URL}")
    with gzip.open(path, "rt", encoding="utf-8") as fp:
        for line in fp:
            yield json.loads(line)


def parse_price(p):
    if p is None or isinstance(p, (int, float)):
        return np.nan if p is None else float(p)
    m = NUMBER_RE.search(str(p).replace(",", ""))
    return float(m.group()) if m else np.nan


def build_reviews():
    rows = []
    for i, r in enumerate(read_jsonl(REVIEWS_RAW)):
        raw_title = clean_text(r.get("title"))
        auto_title = bool(AUTO_TITLE_RE.match(raw_title))
        rows.append({
            "review_id": i,
            "rating": float(r["rating"]),
            "title": "" if auto_title else raw_title,
            "auto_title": auto_title,
            "text": clean_text(r.get("text")),
            "n_images": len(r.get("images") or []),
            "asin": r.get("asin"),
            "parent_asin": r.get("parent_asin"),
            "user_id": r.get("user_id"),
            "timestamp": r.get("timestamp"),
            "verified_purchase": bool(r.get("verified_purchase")),
            "helpful_vote": int(r.get("helpful_vote") or 0),
        })
    df = pd.DataFrame(rows)
    n_raw = len(df)
    # Pre-deduplication counts, for comparison with Kung et al. (2024), who used the raw file.
    (df.groupby("rating").helpful_vote.agg(reviews="size", mean_helpful="mean")
       .to_csv(RAW_SUMMARY))

    key = ["user_id", "asin", "timestamp", "rating", "title", "text"]
    df = df.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
    print(f"reviews: {n_raw:,} raw, {n_raw - len(df):,} exact duplicates dropped, {len(df):,} kept")
    print(f"auto-generated star titles removed: {df.auto_title.sum():,}")

    df["time"] = pd.to_datetime(df.timestamp, unit="ms", utc=True)
    df["full_text"] = [join_title_text(a, b) for a, b in zip(df.title, df.text)]
    df["word_count"] = df.full_text.str.split().str.len().fillna(0).astype(int)
    return df


def build_meta():
    rows = []
    for m in read_jsonl(META_RAW):
        details = m.get("details") or {}
        rows.append({
            "parent_asin": m.get("parent_asin"),
            "product_title": m.get("title"),
            "store": m.get("store"),
            "brand": details.get("Brand"),
            "avg_rating": m.get("average_rating"),
            "rating_number": m.get("rating_number"),
            "price": parse_price(m.get("price")),
            "n_meta_images": len(m.get("images") or []),
            "n_features": len(m.get("features") or []),
            "description_chars": sum(len(d) for d in (m.get("description") or [])),
            "categories": " > ".join(m.get("categories") or []),
        })
    meta = pd.DataFrame(rows).drop_duplicates("parent_asin").reset_index(drop=True)
    print(f"products with metadata: {len(meta):,}")
    return meta


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    build_reviews().to_parquet(REVIEWS, index=False)
    build_meta().to_parquet(META, index=False)
    print(f"wrote {REVIEWS.name}, {META.name} to {PROCESSED}")


if __name__ == "__main__":
    main()
