"""Install (or export) the shipped RoBERTa sentiment scores.

Scoring 694K reviews with RoBERTa takes minutes on a GPU but hours on a CPU. The scores therefore
ship with the repository in data/precomputed/, so the project runs without PyTorch or a GPU.
They are keyed by review_id, which makes them valid only for the exact cleaned data they were
computed from. `install` checks that (every review id and review text must match) before copying
them into data/processed/.

Usage:
    python src/precomputed.py install                  # what run_pipeline.py does
    python src/precomputed.py export --note "..."      # maintainers: ship freshly computed scores
"""
import argparse
import hashlib
import json
import shutil
from datetime import date

import pandas as pd

from config import PRECOMPUTED, REVIEWS, ROBERTA_MODEL, SENT_ROBERTA

SHIPPED = PRECOMPUTED / "sentiment_roberta.parquet"
MANIFEST = PRECOMPUTED / "manifest.json"


def reviews_digest():
    """SHA-256 over every (review_id, cleaned text) pair, in review_id order."""
    df = pd.read_parquet(REVIEWS, columns=["review_id", "full_text"]).sort_values("review_id")
    h = hashlib.sha256()
    for rid, text in zip(df.review_id, df.full_text):
        h.update(f"{rid}\x1f{text}\x1e".encode("utf-8"))
    return h.hexdigest(), len(df)


def install():
    if not REVIEWS.exists():
        raise SystemExit(f"{REVIEWS} is missing; run prepare.py first.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    digest, n = reviews_digest()
    if digest != manifest["reviews_sha256"]:
        raise SystemExit(
            "The shipped RoBERTa scores were computed on different cleaned data "
            f"({manifest['reviews']:,} reviews; you have {n:,}), so they cannot be used.\n"
            "Recompute them instead:  python run_pipeline.py --recompute-roberta")
    shutil.copyfile(SHIPPED, SENT_ROBERTA)
    print(f"installed shipped RoBERTa scores ({manifest['reviews']:,} reviews, "
          f"computed with {manifest['computed_with']})")


def export(note):
    scores = pd.read_parquet(SENT_ROBERTA)
    digest, n = reviews_digest()
    if len(scores) != n:
        raise SystemExit(f"{SENT_ROBERTA} has {len(scores):,} rows but there are {n:,} reviews")
    PRECOMPUTED.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(SHIPPED, index=False, compression="zstd")
    MANIFEST.write_text(json.dumps({
        "model": ROBERTA_MODEL, "computed_with": note, "created": date.today().isoformat(),
        "reviews": n, "reviews_sha256": digest,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {SHIPPED} ({SHIPPED.stat().st_size / 1e6:.1f} MB) and {MANIFEST.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["install", "export"])
    ap.add_argument("--note", default="", help="export: hardware/software the scores came from")
    args = ap.parse_args()
    if args.command == "install":
        install()
    else:
        export(args.note)


if __name__ == "__main__":
    main()
