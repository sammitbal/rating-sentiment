"""Checks for src/timeseries.py on synthetic reviews: no download or sentiment models needed.

    python tests/test_timeseries.py          # or: python -m pytest tests
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import timeseries as ts  # noqa: E402


def make_reviews(n_products=300, improving=0.0, seed=0):
    """Products with 40-120 reviews over four years. A fraction `improving` of them really get
    better (underlying quality climbs from poor to good); the rest stay where they are."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2018-01-01", tz="UTC")
    rows, planted = [], set()
    for p in range(n_products):
        asin = f"P{p:04d}"
        n = int(rng.integers(40, 120))
        frac = np.sort(rng.uniform(0, 1, n))
        if rng.uniform() < improving:
            planted.add(asin)
            quality = 0.2 + 0.7 * frac
        else:
            quality = np.full(n, rng.beta(5, 2))
        rating = np.clip(np.round(1 + 4 * quality + rng.normal(0, 0.9, n)), 1, 5)
        times = start + pd.to_timedelta(frac * 4 * 365.25, unit="D")
        rows.append(pd.DataFrame({"parent_asin": asin, "time": times, "rating": rating,
                                  "product_title": f"Product {p}"}))
    df = pd.concat(rows, ignore_index=True)
    df["review_id"] = np.arange(len(df))
    df["roberta"] = (df.rating - 3) / 2 + rng.normal(0, 0.2, len(df))
    return df.sample(frac=1, random_state=seed).reset_index(drop=True), planted  # unsorted on purpose


def test_ordering_and_filters():
    df, _ = make_reviews(60, seed=1)
    products = ts.ordered_products(df, min_reviews=60)
    assert len(products.asins) == len(products.ratings) == len(products.years) > 0
    for asin, r, y in zip(products.asins, products.ratings, products.years):
        mine = df[df.parent_asin == asin].sort_values(["time", "review_id"])
        assert np.array_equal(r, mine.rating.to_numpy(float)), "ratings must be in time order"
        assert len(r) >= 60 and y[0] == 0 and np.all(np.diff(y) >= 0)


def test_early_late_statistics():
    r = np.array([1.0] * 10 + [3.0] * 10 + [5.0] * 10)
    s = ts.stat_arrays([r], frac=1 / 3, bins=10)
    assert s["early_mean"][0] == 1 and s["late_mean"][0] == 5
    assert s["early_neg"][0] == 1 and s["late_neg"][0] == 0
    assert np.allclose(s["curve"][0], [r[i * 3:(i + 1) * 3].mean() for i in range(10)])
    rough, improved, faded = ts.classify(s)
    assert rough[0] and improved[0] and faded[0]


def test_finds_planted_recoveries():
    df, planted = make_reviews(300, improving=0.3, seed=2)
    products = ts.ordered_products(df)
    obs = ts.summarise(products.ratings)
    null, curves, neg_curves = ts.permutation_null(products.ratings, n_perm=60, seed=3)
    cmp = ts.compare_to_null(obs, null)
    assert cmp.loc["n_improved", "p_value"] < 0.05, cmp
    assert cmp.loc["n_faded", "p_value"] < 0.05, cmp
    table = ts.product_table(products)
    found = set(table[table.improved].parent_asin)
    assert len(found & planted) / len(found) > 0.9, "flagged products should be the planted ones"
    # the recovery curve rises over the history, far above anything the shuffled data produces
    assert obs["curve"][-1] - obs["curve"][0] > np.nanpercentile(curves[:, -1] - curves[:, 0], 97.5)
    assert curves.shape == neg_curves.shape == (60, ts.BINS)
    assert "text_gain" in table and table.loc[table.improved, "text_gain"].mean() > 0


def test_no_false_alarm_without_change():
    df, _ = make_reviews(300, improving=0.0, seed=4)
    products = ts.ordered_products(df)
    obs = ts.summarise(products.ratings)
    null, *_ = ts.permutation_null(products.ratings, n_perm=60, seed=5)
    cmp = ts.compare_to_null(obs, null)
    assert cmp.loc["n_improved", "p_value"] > 0.02, cmp
    # regression to the mean: even shuffled data shows rough-start products "recovering"
    assert null.mean_gain_rough.mean() > 0


def test_permutation_keeps_each_products_ratings():
    df, _ = make_reviews(40, seed=6)
    products = ts.ordered_products(df)
    rng = np.random.default_rng(0)
    for r in products.ratings:
        assert np.array_equal(np.sort(rng.permutation(r)), np.sort(r))


def test_period_tables_handle_time_zones():
    df, _ = make_reviews(20, seed=7)
    for freq in ts.MONTHS:
        t = ts.period_table(df, freq)
        assert t.n.sum() == len(df) and t.k_neg.sum() == (df.rating <= 2).sum()
        assert (t.neg_lo <= t.neg_rate).all() and (t.neg_rate <= t.neg_hi).all()
    assert ts.period_start(df.time, "H").dt.month.isin([1, 7]).all()
    assert ts.period_start(df.time.dt.tz_localize(None), "Y").dt.month.eq(1).all()


def test_figures_draw():
    import matplotlib.pyplot as plt
    ts.set_style()
    df, _ = make_reviews(120, improving=0.3, seed=8)
    products = ts.ordered_products(df)
    table = ts.product_table(products)
    fig, picked = ts.plot_improvers(products, table, n=4)
    assert len(picked) == 4 and len(fig.axes) == 8
    obs = ts.summarise(products.ratings)
    _, curves, neg_curves = ts.permutation_null(products.ratings, n_perm=10)
    assert len(ts.plot_recovery(obs, curves, neg_curves).axes) == 2
    plt.close("all")


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
