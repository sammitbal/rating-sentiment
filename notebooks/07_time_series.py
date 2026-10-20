# %% [markdown]
# # 7 · Do products get better over time, and do their negative reviews fade?
#
# The first six notebooks treat every review as a separate observation. This one follows **each
# product through time**. A product can start badly, say with a faulty first batch in 2020, then
# improve. If it does, its later reviews should be better and the 1–2★ reviews should thin out.
#
# **Approach**
#
# 1. **The platform as a whole.** Mean rating and the share of 1–2★ reviews per year.
# 2. **Single products.** Every review of a product on a timeline, with its period mean and its
#    1–2★ share underneath. First the most-reviewed product, then the strongest recoveries.
# 3. **Is it more than chance?** A product picked because its early reviews were poor will tend to
#    look better later even if nothing changed (*regression to the mean*). To see how much, we shuffle
#    each product's ratings across its own timeline and repeat the whole analysis. Only what the real
#    data shows **beyond** the shuffled data counts as evidence of improvement.
# 4. **Do the words improve too, or only the stars?** Using the RoBERTa text score from notebook 2.
# 5. **Sensitivity.** Do the conclusions hold under other definitions?
#
# **Definitions** (implemented in `src/timeseries.py`)
#
# * A product is analysed when it has at least 30 reviews spread over at least one year.
# * Its **early** reviews are the first third in time order, its **late** reviews the last third.
# * **Rough start**: the early reviews average 3.5★ or less.
# * **Improved**: rough start, and the late reviews average at least 1★ more.
# * **Faded**: at least 30% of the early reviews are 1–2★, and at most 5% of the late ones.

# %%
import sys

sys.path.insert(0, "../src")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import timeseries as ts
from common import BLUE, INK_2, NEUTRAL, RED, load, savefig, set_style, spearman

set_style()
df = load(columns=["review_id", "parent_asin", "product_title", "time", "rating", "roberta"])
print(f"{len(df):,} reviews of {df.parent_asin.nunique():,} products, "
      f"{df.time.min():%Y-%m-%d} to {df.time.max():%Y-%m-%d}")

# %% [markdown]
# ## 1 · The platform as a whole
#
# Years with fewer than 200 reviews are left out, because their rates are too noisy to read. The shaded
# band is a Wilson 95% interval.

# %%
yearly = ts.period_table(df, "Y")
yearly = yearly[yearly.n >= 200]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].plot(yearly.period, yearly.mean_rating, marker="o", color=BLUE)
axes[0].set_ylabel("mean rating (stars)")
axes[0].set_title("Mean rating per year")
axes[1].fill_between(yearly.period, 100 * yearly.neg_lo, 100 * yearly.neg_hi, color=RED, alpha=0.2,
                     linewidth=0)
axes[1].plot(yearly.period, 100 * yearly.neg_rate, marker="o", color=RED)
axes[1].set_ylabel("1–2★ reviews (%)")
axes[1].set_title("Share of negative reviews per year")
for ax in axes:
    ax.grid(axis="x", visible=False)
    ts.format_years(ax)
savefig(fig, "07_overall_trend")
yearly.assign(year=yearly.period.dt.year).set_index("year")[["n", "mean_rating", "neg_rate"]].round(3)

# %% [markdown]
# ## 2 · Which products have a history we can follow?
#
# Most products have only a handful of reviews, which is not enough to see a trend. The analysis keeps
# products with a real history.

# %%
products = ts.ordered_products(df)
table = ts.product_table(products)
covered = len(products.frame)
print(f"{len(table):,} of {df.parent_asin.nunique():,} products have 30+ reviews over 1+ years; "
      f"together they hold {covered:,} reviews ({covered / len(df):.0%} of all reviews)")
print(f"  rough start : {table.rough.sum():,}")
print(f"  improved    : {table.improved.sum():,}   (rough start, and the late reviews are 1★+ better)")
print(f"  faded       : {table.faded.sum():,}   (negative reviews fell from 30%+ to 5% or less)")

# %% [markdown]
# ## 3 · Product timelines
#
# Each dot is a review: red for 1–2★, grey for 3★, blue for 4–5★, spread vertically a little so that
# they do not overlap. The black line is the mean rating per half-year. The red bars underneath are the
# share of 1–2★ reviews in each half-year. **First, the most-reviewed product**, whatever its trend:

# %%
biggest = table.sort_values("n_reviews", ascending=False).iloc[0]
reviews = products.frame[products.frame.parent_asin == biggest.parent_asin]
fig = plt.figure(figsize=(9, 5))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], hspace=0.1)
ts.plot_product(reviews, fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                title=f"{ts._label(biggest.get('product_title'), biggest.parent_asin, 70)}  (n={biggest.n_reviews})")
savefig(fig, "07_most_reviewed_product")
biggest[["parent_asin", "n_reviews", "early_mean", "late_mean", "early_neg", "late_neg"]]

# %% [markdown]
# **The strongest recoveries**: products that started rough and improved the most. This figure is
# chosen to show the pattern, so it says nothing about *how common* the pattern is. Section 4 answers that.

# %%
fig, top = ts.plot_improvers(products, table, n=6)
savefig(fig, "07_top_improvers")
cols = [c for c in ["parent_asin", "product_title", "n_reviews", "early_mean", "late_mean", "gain",
                    "early_neg", "late_neg"] if c in top]
top[cols].round(2)

# %% [markdown]
# ## 4 · Is that more than chance?
#
# Each product's ratings are shuffled across its own timeline 200 times, and the whole procedure (pick
# the rough starts, measure the recovery) is repeated on every shuffle. The shuffled data has the same
# products, the same ratings and the same selection effect. It lacks only the real ordering in time.
#
# * `n_improved`, `n_faded`: how many products recover or lose their negative reviews
# * `mean_gain_rough`: the average gain of the rough-start products, in stars
# * `p_value`: the share of shuffles that did at least as well as the real data

# %%
N_PERM = 200
observed = ts.summarise(products.ratings)
null, null_curve, null_neg_curve = ts.permutation_null(products.ratings, n_perm=N_PERM, seed=0)
ts.compare_to_null(observed, null).round(3)

# %% [markdown]
# **The recovery curve.** Each rough-start product's history is cut into ten equal slices of its
# reviews. The lines show the average rating and the average 1–2★ share per slice. The grey band is
# where shuffled data lands. The first third (shaded) is where products were *selected* for being
# rough, so it is low by construction. The test is what happens **afterwards**: only the real data
# should leave the grey band.

# %%
fig = ts.plot_recovery(observed, null_curve, null_neg_curve)
savefig(fig, "07_recovery_curve")

# %% [markdown]
# ## 5 · Do the words improve along with the stars?
#
# A rating can rise because the product got better, or because the reviewers got more lenient. If the
# **text** gets more positive too (RoBERTa positive minus negative probability, notebook 2), the
# improvement is in what people say and not only in the stars they click.

# %%
if "text_gain" in table:
    rough = table[table.rough]
    rho = spearman(rough.gain, rough.text_gain)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(rough.gain, rough.text_gain, s=14, color=BLUE, alpha=0.5, linewidths=0)
    ax.axhline(0, color=NEUTRAL, linewidth=1)
    ax.axvline(0, color=NEUTRAL, linewidth=1)
    ax.set_xlabel("change in mean rating, early to late third (stars)")
    ax.set_ylabel("change in text sentiment (RoBERTa, −1 to +1)")
    ax.set_title("Rough-start products: stars vs words")
    ax.text(0.97, 0.04, f"Spearman ρ = {rho:.2f}  (n = {len(rough):,})", transform=ax.transAxes,
            ha="right", color=INK_2)
    savefig(fig, "07_stars_vs_words")
    print(f"Rough-start products: mean text change {rough.text_gain.mean():+.3f}; "
          f"among the {int(rough.improved.sum())} improved: {rough[rough.improved].text_gain.mean():+.3f}")

# %% [markdown]
# ## 6 · Do the conclusions depend on the definitions?
#
# The same test under other cut-offs, with 50 shuffles each. A real effect should survive reasonable
# changes. A result that appears for only one set of cut-offs should not be trusted.

# %%
settings = [
    dict(label="main (first/last third, ≤3.5★, +1★)"),
    dict(label="first/last quarter", frac=1 / 4),
    dict(label="first/last half", frac=1 / 2),
    dict(label="rough start ≤ 3.0★", rough_start=3.0),
    dict(label="rough start ≤ 4.0★", rough_start=4.0),
    dict(label="gain ≥ 0.5★", gain=0.5),
    dict(label="gain ≥ 1.5★", gain=1.5),
]
rows = []
for s in settings:
    params = {k: v for k, v in s.items() if k != "label"}
    obs = ts.summarise(products.ratings, **params)
    nul, *_ = ts.permutation_null(products.ratings, n_perm=50, seed=1, **params)
    rows.append({"definition": s["label"], "rough_start": obs["n_rough"], "improved": obs["n_improved"],
                 "improved, shuffled": round(nul.n_improved.mean(), 1),
                 "mean gain, rough": round(obs["mean_gain_rough"], 2),
                 "mean gain, shuffled": round(nul.mean_gain_rough.mean(), 2),
                 "p (improved)": round((1 + (nul.n_improved >= obs["n_improved"]).sum()) / (1 + len(nul)), 3)})
pd.DataFrame(rows).set_index("definition")

# %% [markdown]
# ## What to take from this notebook
#
# * **Platform trend (section 1).** Whether ratings fall or rise across the years, and whether the
#   1–2★ share drifts.
# * **Single products (section 3).** What a recovery looks like when it happens: the black line climbs
#   while the red bars shrink. The figure shows examples. It does not show how common they are.
# * **The test (section 4).** The number that matters is `n_improved` against its shuffled range, and the
#   recovery curve leaving the grey band after the shaded selection window. If the observed values sit
#   inside the shuffled range, the apparent recoveries are what chance and regression to the mean alone
#   would produce.
# * **Stars versus words (section 5).** A positive relationship means the improvement is in the text too.
# * **Limits.** The analysis follows products through time, not the same reviewers, so a change in who
#   reviews (for instance, more verified buyers, or a seller starting to ask for reviews) can look like a
#   change in the product. Reviews are also a self-selected sample of buyers, and a product's early
#   reviews are few, so the early third is noisy for products near the 30-review minimum.
