"""Score every review with the two lexicon-based models: VADER and TextBlob.

VADER is scored on the combined title + body and on each part separately (so the notebooks
can look at title/body disagreement). TextBlob is the tool used by Kung et al. (2024).

Usage:  python src/sentiment_lexicon.py [--workers N]
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from textblob import TextBlob
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import REVIEWS, SENT_LEXICON

_vader = None


def _init_worker():
    global _vader
    _vader = SentimentIntensityAnalyzer()


def _score(row):
    full, title, body = row
    tb = TextBlob(full).sentiment
    return (
        _vader.polarity_scores(full)["compound"],
        _vader.polarity_scores(title)["compound"] if title else np.nan,
        _vader.polarity_scores(body)["compound"] if body else np.nan,
        tb.polarity,
        tb.subjectivity,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    df = pd.read_parquet(REVIEWS, columns=["review_id", "full_text", "title", "text"])
    rows = list(zip(df.full_text, df.title, df.text))
    with Pool(args.workers, initializer=_init_worker) as pool:
        scores = list(tqdm(pool.imap(_score, rows, chunksize=2000), total=len(rows),
                           desc="VADER + TextBlob", mininterval=10))

    cols = ["vader", "vader_title", "vader_body", "textblob", "textblob_subjectivity"]
    out = pd.DataFrame(scores, columns=cols)
    out.insert(0, "review_id", df.review_id.to_numpy())
    out.to_parquet(SENT_LEXICON, index=False)
    print(f"wrote {SENT_LEXICON} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
