# %% [markdown]
# # 4 · What predicts a disagreement between stars and text?
#
# The two directions of inconsistency are modelled **separately**:
#
# * among **4–5★** reviews, what predicts *negative* text?
# * among **1–2★** reviews, what predicts *positive* text?
#
# Pooling them would average two different phenomena whose predictors can point in opposite
# directions.
#
# The notebook proceeds in four steps:
#
# 1. inconsistency rates by single factors, with no controls
# 2. multivariable logistic regressions, reported as odds ratios with 95% CIs. Continuous predictors
#    are standardised, so an odds ratio is the effect of +1 SD.
# 3. a gradient-boosted model with permutation importance, which captures non-linear effects
# 4. the words and phrases that distinguish inconsistent from consistent reviews
#
# **Caveat.** Text-derived predictors such as length and contrast words can also change how well the
# sentiment models work (notebook 2). An association can therefore reflect true reviewer behaviour,
# measurement error, or both.

# %%
import sys

sys.path.insert(0, "../src")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from common import (BLUE, HIGH_NEG, INK_2, LOW_POS, RED, SURFACE, forest, load, rate_table,
                    ratio_table, savefig, set_style)

set_style()
df = load()
polar = df[df.polar_rating & df.avg_rating.notna()].copy()
polar["price_missing"] = polar.price.isna()
polar["log_price"] = polar.log_price.fillna(polar.log_price.median())
polar["log_user_reviews"] = np.log(polar.user_n_reviews)
polar["extreme"] = polar.rating.isin([1, 5])
GROUPS = {HIGH_NEG: polar[polar.rating >= 4].copy(), LOW_POS: polar[polar.rating <= 2].copy()}
COLOR = {HIGH_NEG: RED, LOW_POS: BLUE}
for name, g in GROUPS.items():
    print(f"{name}: {len(g):,} reviews, {g.inconsistent.mean():.2%} inconsistent")

# %% [markdown]
# ## Single-factor views

# %%
def binned(frame):
    return {
        "review length (words)": pd.cut(frame.word_count, [-1, 5, 15, 30, 60, 120, np.inf],
                                        labels=["≤5", "6–15", "16–30", "31–60", "61–120", "121+"]),
        "product average rating": pd.cut(frame.avg_rating, [0, 3, 3.5, 4, 4.5, 5],
                                         labels=["≤3", "3–3.5", "3.5–4", "4–4.5", "4.5–5"]),
        "reviews the reviewer wrote": pd.cut(frame.user_n_reviews, [0, 1, 2, 4, np.inf],
                                             labels=["1", "2", "3–4", "5+"]),
    }


factor_names = list(binned(polar))
fig, axes = plt.subplots(2, len(factor_names), figsize=(12, 6), sharey="row")
tables = {}
for row, (name, g) in enumerate(GROUPS.items()):
    for col, (factor, bins) in enumerate(binned(g).items()):
        t = rate_table(g.assign(_bin=bins), "_bin")
        tables[(name, factor)] = t
        ax = axes[row, col]
        x = np.arange(len(t))
        ax.vlines(x, 100 * t.lo, 100 * t.hi, color=COLOR[name], linewidth=2)
        ax.scatter(x, 100 * t.rate, color=COLOR[name], s=45, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.set_xticks(x, t._bin.astype(str))
        ax.grid(axis="x", visible=False)
        if row == 1:
            ax.set_xlabel(factor, color=INK_2)
        if col == 0:
            ax.set_ylabel(f"{'4–5★' if name == HIGH_NEG else '1–2★'}: inconsistent (%)")
    top = max(100 * tables[(name, f)].hi.max() for f in factor_names)
    axes[row, 0].set_ylim(0, top * 1.08)  # shared across the row
    axes[row, 0].set_title(name.capitalize(), loc="left")
fig.tight_layout()
savefig(fig, "04_single_factors")

binary = ["verified_purchase", "has_images", "has_title", "has_contrast", "mentions_logistics", "has_exclaim"]
pd.DataFrame({name: {f: f"{g.loc[g[f], 'inconsistent'].mean():.1%} vs {g.loc[~g[f], 'inconsistent'].mean():.1%}"
                     for f in binary} for name, g in GROUPS.items()}).rename_axis("rate with vs without")

# %% [markdown]
# ## Multivariable logistic regression

# %%
CONT = ["log_words", "year", "avg_rating", "log_rating_number", "log_price", "log_user_reviews"]
BIN = ["extreme", "has_title", "has_images", "verified_purchase", "has_contrast", "mentions_logistics",
       "has_exclaim", "price_missing"]
LABELS = {
    "extreme": "extreme rating (5★ vs 4★; 1★ vs 2★)",
    "z_log_words": "longer review (+1 SD log words)",
    "has_title": "user-written title",
    "has_images": "includes photos",
    "verified_purchase": "verified purchase",
    "has_contrast": "contrast word (but, however, though…)",
    "mentions_logistics": "mentions shipping / delivery / seller",
    "has_exclaim": "exclamation mark",
    "z_year": "posted later (+1 SD year)",
    "z_avg_rating": "higher product avg rating (+1 SD)",
    "z_log_rating_number": "more popular product (+1 SD log #ratings)",
    "z_log_price": "higher price (+1 SD log)",
    "price_missing": "price missing",
    "z_log_user_reviews": "more active reviewer (+1 SD log #reviews)",
}


def design(g):
    d = g[["inconsistent"] + BIN + CONT].copy()
    for c in ["inconsistent"] + BIN:
        d[c] = d[c].astype(int)
    for c in CONT:
        d[f"z_{c}"] = (d[c] - d[c].mean()) / d[c].std()
    return d


formula = "inconsistent ~ " + " + ".join(BIN + [f"z_{c}" for c in CONT])
fits = {name: smf.logit(formula, design(g)).fit(disp=0) for name, g in GROUPS.items()}
ors = {name: ratio_table(f).reindex(list(LABELS)) for name, f in fits.items()}

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
for ax, (name, t) in zip(axes, ors.items()):
    forest(ax, t, labels=[LABELS[i] for i in t.index], color=COLOR[name])
    ax.set_title(f"{'4–5★' if name == HIGH_NEG else '1–2★'} reviews: odds of {name.split(',')[0]}")
    ax.set_xlabel("odds ratio (log scale), 95% CI")
    ticks = [0.25, 0.5, 1, 2, 4]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
fig.tight_layout()
savefig(fig, "04_odds_ratios")

pd.concat({name: t.round(3) for name, t in ors.items()}, axis=1).set_axis(
    [LABELS[i] for i in ors[HIGH_NEG].index])

# %%
for name, f in fits.items():
    print(f"{name}: n = {int(f.nobs):,}, pseudo-R² = {f.prsquared:.3f}")

# %% [markdown]
# ### Robustness: the high-precision flag
#
# The working flag includes many mixed reviews (audited precision ≈41% for 4–5★ and ≈14% for 1–2★).
# The models are refit with `inconsistent_confident` (RoBERTa P(opposite) ≥ 0.8), which has higher
# precision. A predictor whose odds ratio holds up, or grows, under the stricter flag is more likely to
# track genuine inconsistency than mixed reviews.

# %%
def odds_ratios(g, outcome):
    return ratio_table(smf.logit(formula, design(g.assign(inconsistent=g[outcome]))).fit(disp=0)).odds_ratio


robust = pd.DataFrame({(name, flag): (ors[name].odds_ratio if flag == "working"
                                      else odds_ratios(g, "inconsistent_confident"))
                       for name, g in GROUPS.items() for flag in ["working", "confident"]}).reindex(list(LABELS))
robust.index = [LABELS[i] for i in robust.index]
robust.round(2)

# %% [markdown]
# ## Gradient boosting: how predictable is inconsistency, and from what?
#
# The model uses the same features in unstandardised form. Importance is the drop in held-out AUC when
# a feature is shuffled.

# %%
FEATURES = BIN + CONT
importances, aucs = {}, {}
for name, g in GROUPS.items():
    X = g[FEATURES].astype(float)
    y = g.inconsistent.astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0).fit(X_tr, y_tr)
    aucs[name] = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    sub = X_te.sample(min(len(X_te), 40000), random_state=0)
    pi = permutation_importance(model, sub, y_te.loc[sub.index], scoring="roc_auc", n_repeats=5,
                                random_state=0)
    importances[name] = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
print({k: round(v, 3) for k, v in aucs.items()})

names = {"log_words": "review length", "year": "year", "avg_rating": "product avg rating",
         "log_rating_number": "product popularity", "log_price": "price", "log_user_reviews": "reviewer activity",
         "extreme": "extreme rating", "has_title": "user title", "has_images": "photos",
         "verified_purchase": "verified", "has_contrast": "contrast word", "mentions_logistics": "logistics words",
         "has_exclaim": "exclamation", "price_missing": "price missing"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (name, imp) in zip(axes, importances.items()):
    ax.barh([names[i] for i in imp.index], imp.to_numpy(), height=0.55, color=COLOR[name])
    ax.set_title(f"{'4–5★' if name == HIGH_NEG else '1–2★'} reviews (test AUC {aucs[name]:.2f})")
    ax.set_xlabel("drop in AUC when shuffled")
    ax.grid(axis="y", visible=False)
fig.tight_layout()
savefig(fig, "04_permutation_importance")
pd.DataFrame(importances).round(4).sort_values(HIGH_NEG, ascending=False)

# %% [markdown]
# ## What inconsistent reviews talk about
#
# The tables give terms that are over-represented in inconsistent versus consistent reviews of the same
# star group. The measure is the log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi &
# Quinn, 2008), and **z > 0 means more typical of inconsistent reviews**. Consistent reviews are
# subsampled to 100,000 per group.

# %%
def distinctive_terms(a, b, n=20):
    vec = CountVectorizer(ngram_range=(1, 2), min_df=25, token_pattern=r"(?u)\b[a-z][a-z']+\b")
    X = vec.fit_transform(pd.concat([a, b]))
    ya = np.asarray(X[:len(a)].sum(axis=0)).ravel().astype(float)
    yb = np.asarray(X[len(a):].sum(axis=0)).ravel().astype(float)
    prior = 0.01 * (ya + yb) + 0.01
    a0, na, nb = prior.sum(), ya.sum(), yb.sum()
    delta = (np.log((ya + prior) / (na + a0 - ya - prior)) - np.log((yb + prior) / (nb + a0 - yb - prior)))
    z = delta / np.sqrt(1 / (ya + prior) + 1 / (yb + prior))
    t = pd.DataFrame({"term": vec.get_feature_names_out(), "z": z,
                      "per 10k words, inconsistent": 1e4 * ya / na, "per 10k words, consistent": 1e4 * yb / nb})
    return t.nlargest(n, "z").round(2).reset_index(drop=True)


terms = {}
for name, g in GROUPS.items():
    inc = g.loc[g.inconsistent, "full_text"]
    con = g.loc[~g.inconsistent, "full_text"].sample(100_000, random_state=0)
    terms[name] = distinctive_terms(inc, con)
    print(f"\n=== {name}: most distinctive of inconsistent reviews ===")
    display(terms[name])
