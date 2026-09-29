"""Paths and constants shared by the pipeline scripts and notebooks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PRECOMPUTED = ROOT / "data" / "precomputed"  # shipped with the repo (RoBERTa scores)
FIGURES = ROOT / "reports" / "figures"

CATEGORY = "All_Beauty"
DATA_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
REVIEWS_RAW = RAW / f"{CATEGORY}.jsonl.gz"
META_RAW = RAW / f"meta_{CATEGORY}.jsonl.gz"

RAW_SUMMARY = PROCESSED / "raw_rating_summary.csv"
REVIEWS = PROCESSED / "reviews.parquet"
META = PROCESSED / "meta.parquet"
SENT_LEXICON = PROCESSED / "sentiment_lexicon.parquet"
SENT_ROBERTA = PROCESSED / "sentiment_roberta.parquet"
ROBERTA_CHUNKS = PROCESSED / "roberta_chunks"
ANALYSIS = PROCESSED / "analysis.parquet"

# Trained on tweets rather than Amazon star ratings, so its labels are independent of the
# ratings we compare them against (a star-trained model would learn to hide inconsistency).
ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
