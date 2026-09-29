"""Join reviews, sentiment scores and product metadata, and define rating–text (in)consistency.

Definitions
  rating polarity    1–2 stars negative, 3 stars neutral, 4–5 stars positive
  text polarity      VADER, TextBlob: score > +0.05 positive, < -0.05 negative, otherwise neutral
                     RoBERTa: most probable class
  reversal (model)   a 1–2 or 4–5 star review whose text polarity, per that model, is the opposite
                     of its rating polarity. 3-star reviews cannot reverse and are excluded.
  inconsistent       RoBERTa reverses AND at least one lexicon model agrees ("RoBERTa-anchored").
                     A plain 2-of-3 majority is kept as `inconsistent_majority` for comparison:
                     VADER and TextBlob share failure modes (e.g. "Don't waste your money" reads
                     positive to both), so letting them outvote RoBERTa mostly adds errors. The
                     manual audit in reports/audit_labels.csv found 0/30 true cases among them.
  confident          inconsistent, and RoBERTa gives the opposite class probability >= 0.8
  unanimous          all three models reverse
  discrepancy        stars minus the rating typically given to text with this sentiment
                     (isotonic fit of rating on the averaged, z-scored model scores)

Reviewer and product history features are leave-one-out or prior-only, so none of them
contains the review's own outcome.

Usage:  python src/build_features.py [--roberta-file PATH] [--out PATH]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from config import ANALYSIS, META, REVIEWS, SENT_LEXICON, SENT_ROBERTA

LEXICON_BAND = 0.05  # VADER authors' recommended neutral band, applied to TextBlob too
CONFIDENT = 0.8  # RoBERTa probability of the opposite class for the high-precision flag
MODELS = ["vader", "textblob", "roberta"]
CONTRAST_RE = r"\b(?:but|however|although|though|except|yet)\b"
LOGISTICS_RE = r"\b(?:ship\w*|deliver\w*|arriv\w*|packag\w*|seller|vendor)\b"


def polarity(score, band=LEXICON_BAND):
    return np.select([score > band, score < -band], [1, -1], 0).astype(np.int8)


def majority(*pols):
    """The polarity at least two of the three models agree on; NaN when all three differ."""
    votes = np.column_stack(pols)
    out = np.full(len(votes), np.nan)
    for v in (-1, 0, 1):
        out[(votes == v).sum(axis=1) >= 2] = v
    return out


def add_consistency(df):
    df["roberta"] = df.rb_pos - df.rb_neg
    df["roberta_pol"] = (df[["rb_neg", "rb_neu", "rb_pos"]].to_numpy().argmax(axis=1) - 1).astype(np.int8)
    df["vader_pol"] = polarity(df.vader)
    df["textblob_pol"] = polarity(df.textblob)

    df["rating_pol"] = np.select([df.rating >= 4, df.rating <= 2], [1, -1], 0).astype(np.int8)
    df["polar_rating"] = df.rating_pol != 0
    for m in MODELS:
        df[f"rev_{m}"] = df.polar_rating & (df[f"{m}_pol"] == -df.rating_pol)
    df["n_models_reversed"] = df[[f"rev_{m}" for m in MODELS]].sum(axis=1).astype(np.int8)
    df["flag_pattern"] = (df.rev_vader.map({True: "V", False: "-"})
                          + df.rev_textblob.map({True: "T", False: "-"})
                          + df.rev_roberta.map({True: "R", False: "-"}))
    df["text_pol"] = majority(df.vader_pol, df.textblob_pol, df.roberta_pol)
    df["p_opposite"] = np.select([df.rating_pol > 0, df.rating_pol < 0], [df.rb_neg, df.rb_pos], np.nan)

    df["inconsistent"] = df.rev_roberta & (df.rev_vader | df.rev_textblob)
    df["inconsistent_majority"] = df.polar_rating & (df.text_pol == -df.rating_pol)
    df["inconsistent_unanimous"] = df.n_models_reversed == 3
    df["inconsistent_confident"] = df.inconsistent & (df.p_opposite >= CONFIDENT)
    df["direction"] = np.select(
        [~df.polar_rating, df.inconsistent & (df.rating_pol > 0), df.inconsistent & (df.rating_pol < 0)],
        ["3 stars (excluded)", "negative text, high stars", "positive text, low stars"],
        "consistent")

    z = [(df[m] - df[m].mean()) / df[m].std() for m in MODELS]
    df["text_score"] = np.mean(z, axis=0)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(df.text_score, df.rating)
    df["implied_rating"] = iso.predict(df.text_score)
    df["discrepancy"] = df.rating - df.implied_rating
    return df


def add_review_features(df):
    lower = df.full_text.str.lower()
    df["log_words"] = np.log1p(df.word_count)
    df["has_images"] = df.n_images > 0
    df["has_title"] = df.title.str.len() > 0
    df["n_contrast"] = lower.str.count(CONTRAST_RE)
    df["has_contrast"] = df.n_contrast > 0
    df["mentions_logistics"] = lower.str.contains(LOGISTICS_RE)
    df["has_exclaim"] = df.full_text.str.contains("!", regex=False)
    df["year"] = df.time.dt.year
    df["age_years"] = (df.time.max() - df.time).dt.days / 365.25
    df["has_helpful"] = df.helpful_vote > 0
    df["log_helpful"] = np.log1p(df.helpful_vote)
    return df


def add_product_features(df, meta):
    df = df.merge(meta, on="parent_asin", how="left", validate="many_to_one")
    g = df.groupby("parent_asin").rating
    df["prod_n_reviews"] = g.transform("size")
    loo = (g.transform("sum") - df.rating) / (df.prod_n_reviews - 1)
    df["prod_loo_mean_rating"] = loo.where(df.prod_n_reviews > 1)
    df["rating_dev"] = df.rating - df.prod_loo_mean_rating
    df["log_rating_number"] = np.log1p(df.rating_number)
    df["log_price"] = np.log1p(df.price)
    return df


def add_reviewer_features(df):
    df = df.sort_values(["user_id", "timestamp", "review_id"]).reset_index(drop=True)
    g = df.groupby("user_id")
    df["user_n_reviews"] = g.rating.transform("size")
    df["user_seq"] = g.cumcount() + 1  # 1 = the user's first review in this category
    n_other = df.user_n_reviews - 1

    df["user_loo_mean_rating"] = ((g.rating.transform("sum") - df.rating) / n_other).where(n_other > 0)
    polar = df.polar_rating.astype(int)
    incons = df.inconsistent.astype(int)
    other_polar = polar.groupby(df.user_id).transform("sum") - polar
    other_incons = incons.groupby(df.user_id).transform("sum") - incons
    df["user_other_polar"] = other_polar
    df["user_loo_incons_rate"] = (other_incons / other_polar).where(other_polar > 0)

    prior_polar = polar.groupby(df.user_id).cumsum() - polar
    prior_incons = incons.groupby(df.user_id).cumsum() - incons
    df["user_prior_incons_rate"] = (prior_incons / prior_polar).where(prior_polar > 0)

    # Prior-only helpfulness history. The all-reviews version includes the review's own votes
    # (the target) and is kept only to show the leakage in Kung et al. (2024).
    prior_votes = g.helpful_vote.cumsum() - df.helpful_vote
    df["user_prior_mean_helpful"] = (prior_votes / (df.user_seq - 1)).where(df.user_seq > 1)
    df["user_mean_helpful_all"] = g.helpful_vote.transform("mean")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roberta-file", default=str(SENT_ROBERTA))
    ap.add_argument("--out", default=str(ANALYSIS))
    args = ap.parse_args()

    df = (pd.read_parquet(REVIEWS)
            .merge(pd.read_parquet(SENT_LEXICON), on="review_id", validate="one_to_one")
            .merge(pd.read_parquet(args.roberta_file), on="review_id", validate="one_to_one"))
    df = add_consistency(df)
    df = add_review_features(df)
    df = add_product_features(df, pd.read_parquet(META))
    df = add_reviewer_features(df)
    df = df.sort_values("review_id").reset_index(drop=True)
    df.to_parquet(args.out, index=False)

    polar = df[df.polar_rating]
    print(f"wrote {args.out}: {len(df):,} reviews, {len(polar):,} with 1-2 or 4-5 stars")
    for m in MODELS:
        print(f"  reversal rate, {m:<8}: {polar[f'rev_{m}'].mean():.2%}")
    for c in ["inconsistent_majority", "inconsistent", "inconsistent_confident", "inconsistent_unanimous"]:
        print(f"  {c:<24}: {polar[c].mean():.2%}")


if __name__ == "__main__":
    main()
