"""Review trajectories over time: do products recover, and do their negative reviews fade?

The question: a product may collect poor reviews early on (say in 2020), then improve, so that
later reviews are better and the negative ones disappear. This module measures that and draws it.

Method
  order        each product's reviews are put in time order
  early/late   the first and last third (FRAC) of that product's reviews
  rough start  the early third averages ROUGH_START stars or less
  improved     rough start, and the late third averages at least GAIN stars more
  faded        at least NEG_EARLY of the early reviews are 1-2 stars, and at most NEG_LATE of the
               late ones ("the negative reviews disappeared")
  null         shuffle each product's ratings across its own review order and redo everything.
               Rough-start products are picked on a noisy early average, so some "recover" by
               chance alone (regression to the mean). The shuffled data shows how many, and how
               much; the real data has to beat that to count as evidence of genuine improvement.

Only products with at least MIN_REVIEWS reviews spread over MIN_SPAN_YEARS are analysed.

Usage:
    python src/timeseries.py --list 15              # the most-reviewed products, to pick from
    python src/timeseries.py --asin B07XXXXXXX      # timeline figure for one product
    python src/timeseries.py --top 6                # the strongest recoveries, side by side
"""
import argparse
import textwrap
from collections import namedtuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BLUE, INK, INK_2, NEUTRAL, RED, load, savefig, set_style, wilson

MIN_REVIEWS = 30
MIN_SPAN_YEARS = 1.0
FRAC = 1 / 3
ROUGH_START = 3.5
GAIN = 1.0
NEG_EARLY = 0.30
NEG_LATE = 0.05
NEG_MAX_STAR = 2  # 1-2 stars count as a negative review
BINS = 10  # slices of each product's review history, for the recovery curve
MONTHS = {"Y": 12, "H": 6, "Q": 3, "M": 1}
BAR_DAYS = {"Y": 300, "H": 150, "Q": 75, "M": 25}

Products = namedtuple("Products", "frame asins ratings years")
PARAMS = dict(frac=FRAC, rough_start=ROUGH_START, gain=GAIN, neg_early=NEG_EARLY,
              neg_late=NEG_LATE, bins=BINS)


def _naive(time):
    """Datetimes without a time zone, so reviews and period starts can share one axis."""
    return time.dt.tz_localize(None) if time.dt.tz is not None else time


# ---------------------------------------------------------------------------------------------
# Selecting and ordering products
# ---------------------------------------------------------------------------------------------
def ordered_products(df, min_reviews=MIN_REVIEWS, min_span_years=MIN_SPAN_YEARS):
    """Products with enough reviews over enough time, each with its ratings in time order.

    Returns Products(frame, asins, ratings, years): `frame` holds the kept reviews sorted by
    product and time; `ratings[i]` and `years[i]` (years since the product's first review in
    this data) belong to `asins[i]`.
    """
    if min_reviews < BINS:
        raise ValueError(f"min_reviews must be at least {BINS} (the number of recovery-curve bins)")
    d = df.sort_values(["parent_asin", "time", "review_id"], kind="stable")
    g = d.groupby("parent_asin", sort=False)
    d = d[g.rating.transform("size") >= min_reviews]
    ends = d.groupby("parent_asin", sort=False).time.agg(first="first", last="last")
    ok = ends.index[(ends["last"] - ends["first"]).dt.days / 365.25 >= min_span_years]
    d = d[d.parent_asin.isin(ok)]
    if d.empty:
        raise ValueError(f"No product has {min_reviews}+ reviews spread over {min_span_years}+ years; "
                         "lower min_reviews or min_span_years.")
    asins, ratings, years = [], [], []
    for asin, grp in d.groupby("parent_asin", sort=False):
        asins.append(asin)
        ratings.append(grp.rating.to_numpy(float))
        years.append((grp.time - grp.time.iloc[0]).dt.total_seconds().to_numpy() / (365.25 * 86400))
    return Products(d, asins, ratings, years)


# ---------------------------------------------------------------------------------------------
# Early-versus-late statistics (fast: prefix sums, no per-review loops)
# ---------------------------------------------------------------------------------------------
def stat_arrays(ratings, frac=FRAC, bins=BINS):
    """Early/late mean rating, early/late 1-2 star share and the per-slice curves, per product."""
    out = {k: [] for k in ("early_mean", "late_mean", "early_neg", "late_neg")}
    curve, neg_curve = [], []
    for r in ratings:
        n = len(r)
        k = max(1, int(round(n * frac)))
        cs = np.concatenate(([0.0], np.cumsum(r)))
        ng = np.concatenate(([0.0], np.cumsum(r <= NEG_MAX_STAR)))
        edges = np.linspace(0, n, bins + 1).round().astype(int)
        width = np.diff(edges)
        out["early_mean"].append(cs[k] / k)
        out["late_mean"].append((cs[n] - cs[n - k]) / k)
        out["early_neg"].append(ng[k] / k)
        out["late_neg"].append((ng[n] - ng[n - k]) / k)
        curve.append(np.diff(cs[edges]) / width)
        neg_curve.append(np.diff(ng[edges]) / width)
    out = {k: np.asarray(v) for k, v in out.items()}
    out["curve"], out["neg_curve"] = np.asarray(curve), np.asarray(neg_curve)
    return out


def classify(s, rough_start=ROUGH_START, gain=GAIN, neg_early=NEG_EARLY, neg_late=NEG_LATE):
    """Boolean flags per product from stat_arrays output."""
    rough = s["early_mean"] <= rough_start
    improved = rough & (s["late_mean"] - s["early_mean"] >= gain)
    faded = (s["early_neg"] >= neg_early) & (s["late_neg"] <= neg_late)
    return rough, improved, faded


def summarise(ratings, frac=FRAC, rough_start=ROUGH_START, gain=GAIN, neg_early=NEG_EARLY,
              neg_late=NEG_LATE, bins=BINS):
    """Headline numbers plus the average recovery curve of the rough-start products."""
    s = stat_arrays(ratings, frac, bins)
    rough, improved, faded = classify(s, rough_start, gain, neg_early, neg_late)
    nan = np.full(bins, np.nan)
    return {
        "n_products": len(ratings),
        "n_rough": int(rough.sum()),
        "n_improved": int(improved.sum()),
        "n_faded": int(faded.sum()),
        "mean_gain_rough": float((s["late_mean"] - s["early_mean"])[rough].mean()) if rough.any() else np.nan,
        "curve": s["curve"][rough].mean(axis=0) if rough.any() else nan,
        "neg_curve": s["neg_curve"][rough].mean(axis=0) if rough.any() else nan,
    }


def permutation_null(ratings, n_perm=200, seed=0, **params):
    """Redo summarise() on data where each product's ratings are shuffled across its own timeline.

    Selection (rough start) is redone on every shuffle, so the null includes regression to the
    mean. Returns (scalar statistics per shuffle as a DataFrame, curve array, neg_curve array).
    """
    rng = np.random.default_rng(seed)
    rows, curves, neg_curves = [], [], []
    for _ in range(n_perm):
        res = summarise([rng.permutation(r) for r in ratings], **params)
        curves.append(res.pop("curve"))
        neg_curves.append(res.pop("neg_curve"))
        rows.append(res)
    return pd.DataFrame(rows), np.asarray(curves), np.asarray(neg_curves)


def compare_to_null(observed, null):
    """Observed headline numbers next to the shuffle distribution, with one-sided p-values."""
    rows = []
    for key in ("n_rough", "n_improved", "n_faded", "mean_gain_rough"):
        draws = null[key].to_numpy(float)
        draws = draws[~np.isnan(draws)]
        rows.append({
            "statistic": key,
            "observed": observed[key],
            "null_mean": draws.mean(),
            "null_lo": np.percentile(draws, 2.5),
            "null_hi": np.percentile(draws, 97.5),
            "p_value": (1 + (draws >= observed[key]).sum()) / (1 + len(draws)),
        })
    return pd.DataFrame(rows).set_index("statistic")


def product_table(products, **params):
    """One row per analysed product: early/late means, change, trend and the three flags."""
    p = {**PARAMS, **params}
    s = stat_arrays(products.ratings, p["frac"], p["bins"])
    rough, improved, faded = classify(s, p["rough_start"], p["gain"], p["neg_early"], p["neg_late"])
    slope = []
    for r, y in zip(products.ratings, products.years):
        yc = y - y.mean()
        slope.append((yc * (r - r.mean())).sum() / (yc**2).sum() if (yc**2).sum() > 0 else np.nan)
    t = pd.DataFrame({
        "parent_asin": products.asins,
        "n_reviews": [len(r) for r in products.ratings],
        "span_years": [y[-1] for y in products.years],
        "early_mean": s["early_mean"], "late_mean": s["late_mean"],
        "gain": s["late_mean"] - s["early_mean"],
        "early_neg": s["early_neg"], "late_neg": s["late_neg"],
        "slope_per_year": slope,
        "rough": rough, "improved": improved, "faded": faded,
    })
    d = products.frame
    if "product_title" in d:
        t["product_title"] = d.groupby("parent_asin", sort=False).product_title.first().to_numpy()
    if "roberta" in d:  # does the wording improve too, or only the stars?
        text = [r.to_numpy() for _, r in d.groupby("parent_asin", sort=False).roberta]
        k = [max(1, int(round(len(x) * p["frac"]))) for x in text]
        t["text_gain"] = [x[-j:].mean() - x[:j].mean() for x, j in zip(text, k)]
    return t


# ---------------------------------------------------------------------------------------------
# Period tables
# ---------------------------------------------------------------------------------------------
def period_start(time, freq="H"):
    """First day of the year (Y), half-year (H), quarter (Q) or month (M) each review falls in."""
    t = _naive(time)
    step = MONTHS[freq]
    month = ((t.dt.month - 1) // step) * step + 1
    return pd.to_datetime(pd.DataFrame({"year": t.dt.year, "month": month, "day": 1}))


def period_table(df, freq="Y", min_n=1):
    """Reviews, mean rating and 1-2 star share per period, with a Wilson 95% interval."""
    d = df.assign(period=period_start(df.time, freq), is_neg=(df.rating <= NEG_MAX_STAR))
    t = d.groupby("period").agg(n=("rating", "size"), mean_rating=("rating", "mean"),
                                k_neg=("is_neg", "sum")).reset_index()
    t = t[t.n >= min_n].copy()
    t["neg_rate"], t["neg_lo"], t["neg_hi"] = wilson(t.k_neg, t.n)
    return t


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
def format_years(ax):
    """One tick per calendar year, labelled with the year only."""
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _label(row_or_title, asin, width=44):
    title = row_or_title if isinstance(row_or_title, str) and row_or_title else asin
    return textwrap.shorten(title, width=width, placeholder="…")


def plot_product(reviews, ax, ax_neg, freq="H", min_n=3, title=None, seed=0):
    """One product's history: every review (top), its period mean, and its 1-2 star share (bottom)."""
    rng = np.random.default_rng(seed)
    t = _naive(reviews.time)
    colour = np.where(reviews.rating <= NEG_MAX_STAR, RED, np.where(reviews.rating >= 4, BLUE, NEUTRAL))
    ax.scatter(t, reviews.rating + rng.uniform(-0.18, 0.18, len(reviews)), s=14, c=colour,
               alpha=0.45, linewidths=0)
    pt = period_table(reviews, freq, min_n=min_n)
    ax.plot(pt.period, pt.mean_rating, color=INK, marker="o", markersize=4, linewidth=1.8,
            label="mean rating per period")
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks(range(1, 6))
    ax.set_ylabel("stars")
    ax.tick_params(labelbottom=False)
    if title:
        ax.set_title(title, fontsize=10)
    ax_neg.bar(pt.period, 100 * pt.neg_rate, width=BAR_DAYS[freq], align="edge", color=RED, alpha=0.85)
    ax_neg.set_ylim(0, 100)
    ax_neg.set_yticks([0, 50, 100])
    ax_neg.set_ylabel("1–2★ (%)")
    ax.sharex(ax_neg)
    format_years(ax_neg)
    ax.grid(axis="x", visible=False)
    ax_neg.grid(axis="x", visible=False)


def plot_improvers(products, table, n=6, freq="H", min_n=3):
    """Small multiples of the strongest recoveries, ranked by how much the late reviews beat the early ones."""
    pool = table[table.improved]
    if pool.empty:
        pool = table[table.rough]
    if pool.empty:
        pool = table
    top = pool.sort_values(["gain", "n_reviews"], ascending=False).head(n)
    rows = int(np.ceil(len(top) / 2))
    fig = plt.figure(figsize=(12, 4.2 * rows))
    outer = fig.add_gridspec(rows, 2, hspace=0.42, wspace=0.18)
    for i, (_, row) in enumerate(top.iterrows()):
        r, c = divmod(i, 2)
        inner = outer[r, c].subgridspec(2, 1, height_ratios=[3, 1.2], hspace=0.08)
        ax, ax_neg = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        reviews = products.frame[products.frame.parent_asin == row.parent_asin]
        name = _label(row.get("product_title"), row.parent_asin)
        plot_product(reviews, ax, ax_neg, freq, min_n,
                     title=f"{name}  ·  {row.early_mean:.1f}★ → {row.late_mean:.1f}★  (n={row.n_reviews})")
    fig.suptitle("Products whose reviews got better over time", x=0.01, ha="left", fontsize=13,
                 fontweight="semibold")
    return fig, top


def plot_recovery(observed, null_curve, null_neg_curve, bins=BINS, frac=FRAC):
    """Average rating and 1-2 star share across a product's review history, against the shuffled null."""
    x = (np.arange(bins) + 0.5) * 100 / bins
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, obs, null, ylabel, scale in (
            (axes[0], observed["curve"], null_curve, "mean rating (stars)", 1),
            (axes[1], observed["neg_curve"], null_neg_curve, "1–2★ reviews (%)", 100)):
        lo, hi = np.nanpercentile(null, [2.5, 97.5], axis=0)
        ax.axvspan(0, 100 * frac, color=NEUTRAL, alpha=0.25, linewidth=0)
        ax.fill_between(x, scale * lo, scale * hi, color=NEUTRAL, alpha=0.6, linewidth=0,
                        label="shuffled data (95% range)")
        ax.plot(x, scale * np.nanmedian(null, axis=0), color=INK_2, linestyle="--", linewidth=1.2)
        ax.plot(x, scale * obs, color=BLUE, marker="o", label="observed")
        ax.set_xlabel("position in the product's review history (%)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="lower right")
    axes[0].text(1.5, 0.97, "selection window:\nfirst third", transform=axes[0].get_xaxis_transform(),
                 va="top", fontsize=8.5, color=INK_2)
    axes[0].set_title("Rough-start products: rating over their history")
    axes[1].set_title("…and the share of negative reviews")
    return fig


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asin", help="parent_asin of one product to draw")
    ap.add_argument("--top", type=int, help="draw the N strongest recoveries")
    ap.add_argument("--list", type=int, metavar="N", help="print the N most-reviewed products")
    ap.add_argument("--freq", default="H", choices=sorted(MONTHS), help="period: Y, H (half-year), Q or M")
    ap.add_argument("--min-reviews", type=int, default=MIN_REVIEWS)
    ap.add_argument("--min-span-years", type=float, default=MIN_SPAN_YEARS)
    args = ap.parse_args()
    if not (args.asin or args.top or args.list):
        ap.error("choose one of --asin, --top or --list")

    set_style()
    cols = ["review_id", "parent_asin", "product_title", "time", "rating", "roberta"]
    df = load(columns=cols)

    if args.list:
        top = (df.groupby("parent_asin").agg(reviews=("rating", "size"), mean=("rating", "mean"),
                                             title=("product_title", "first"))
                 .sort_values("reviews", ascending=False).head(args.list))
        top["title"] = top.title.fillna("").map(lambda s: textwrap.shorten(s, 60, placeholder="…"))
        print(top.round({"mean": 2}).to_string())
    if args.asin:
        reviews = df[df.parent_asin == args.asin].sort_values("time")
        if reviews.empty:
            raise SystemExit(f"No reviews for parent_asin {args.asin!r}. Try --list 15 to see products.")
        fig = plt.figure(figsize=(9, 5))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], hspace=0.1)
        name = _label(reviews.product_title.iloc[0], args.asin, 70)
        plot_product(reviews, fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), args.freq,
                     title=f"{name}  (n={len(reviews)})")
        savefig(fig, f"07_product_{args.asin}")
        print(f"wrote reports/figures/07_product_{args.asin}.png")
    if args.top:
        products = ordered_products(df, args.min_reviews, args.min_span_years)
        table = product_table(products)
        fig, picked = plot_improvers(products, table, args.top, args.freq)
        savefig(fig, "07_top_improvers")
        print(picked[["parent_asin", "n_reviews", "early_mean", "late_mean", "gain"]]
              .round(2).to_string(index=False))
        print("wrote reports/figures/07_top_improvers.png")


if __name__ == "__main__":
    main()
