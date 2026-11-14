"""Train / validation / test split of the reviews, shared by every model in the project.

70% train, 10% validation, 20% test. The split is stratified on the outcome, so each part has the same
share of positives, and it is reproducible: the seed is fixed and the assignment is written to disk.

    train       fit the models
    validation  choose model settings (no test-set numbers are used for this)
    test        report results, once, with the settings chosen on validation

Usage:
    python src/splits.py        # writes data/splits/{train,validation,test}.parquet
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import ANALYSIS, SPLITS

SEED = 0
TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.10  # of the whole dataset, so train = 70%
PART_NAMES = ("train", "validation", "test")
ID_COLUMN = "review_id"


def assign(df, outcome="has_helpful", seed=SEED):
    """Return a Series of 'train' / 'validation' / 'test' labels, indexed like `df`."""
    rest, test = train_test_split(df.index, test_size=TEST_FRACTION, random_state=seed,
                                  stratify=df[outcome])
    # The validation share of the whole dataset, expressed as a share of what is left after the test cut.
    val_of_rest = VALIDATION_FRACTION / (1 - TEST_FRACTION)
    train, validation = train_test_split(rest, test_size=val_of_rest, random_state=seed,
                                         stratify=df.loc[rest, outcome])
    labels = pd.Series(index=df.index, dtype="object")
    labels.loc[train], labels.loc[validation], labels.loc[test] = "train", "validation", "test"
    return labels


def write(df=None, outcome="has_helpful"):
    """Write one parquet file per part, holding each part's review ids and its outcome."""
    if df is None:
        df = pd.read_parquet(ANALYSIS, columns=[ID_COLUMN, outcome])
    labels = assign(df, outcome)
    SPLITS.mkdir(parents=True, exist_ok=True)
    counts = {}
    for part in PART_NAMES:
        rows = df.loc[labels == part, [ID_COLUMN, outcome]].reset_index(drop=True)
        rows.to_parquet(SPLITS / f"{part}.parquet", index=False)
        counts[part] = (len(rows), rows[outcome].mean())
    return counts


def load_split(part):
    """Review ids in one part, as a numpy array (use with the analysis table)."""
    return pd.read_parquet(SPLITS / f"{part}.parquet")[ID_COLUMN].to_numpy()


if __name__ == "__main__":
    counts = write()
    total = sum(n for n, _ in counts.values())
    for part, (n, rate) in counts.items():
        print(f"{part:<11} {n:>9,}  ({n / total:.1%})   share with >=1 helpful vote: {rate:.3f}")
    print(f"wrote {len(PART_NAMES)} files to {SPLITS}")
