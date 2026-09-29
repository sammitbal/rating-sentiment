"""Shared loading, plot styling and small statistics helpers for the notebooks."""
import logging
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ANALYSIS, FIGURES  # noqa: E402

# Chart colours: a validated colour-blind-safe palette (light surface).
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, NEUTRAL = "#e1e0d9", "#c3c2b7", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, RED = "#2a78d6", "#e34948"  # diverging poles: positive text / negative text
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

MODELS = ["vader", "textblob", "roberta"]
MODEL_LABELS = {"vader": "VADER", "textblob": "TextBlob", "roberta": "RoBERTa"}
MODEL_COLORS = dict(zip(MODELS, SERIES[:3]))
HIGH_NEG, LOW_POS = "negative text, high stars", "positive text, low stars"
DIRECTION_COLORS = {HIGH_NEG: RED, LOW_POS: BLUE}


def set_style():
    # Silence warnings about fonts missing on this OS and DejaVu Sans lacking semibold.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11.5,
        "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 10,
        # First installed font wins: Windows, macOS, then matplotlib's bundled fallback.
        "font.family": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9.5,
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 7,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def savefig(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")


def load(columns=None):
    """The analysis table (override the path with the ANALYSIS_PATH environment variable)."""
    return pd.read_parquet(Path(os.environ.get("ANALYSIS_PATH", ANALYSIS)), columns=columns)


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion; returns (p, lo, hi)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return p, np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def rate_table(df, by, flag="inconsistent", min_n=1):
    """Rate of a boolean flag per group, with Wilson 95% CIs."""
    g = df.groupby(by, observed=True)[flag].agg(k="sum", n="size").reset_index()
    g = g[g.n >= min_n]
    g["rate"], g["lo"], g["hi"] = wilson(g.k, g.n)
    return g


def fmt_pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def fleiss_kappa(labels):
    """Fleiss' kappa for an (n_items, n_raters) array of category labels."""
    cats = np.unique(labels)
    counts = np.stack([(labels == c).sum(axis=1) for c in cats], axis=1)
    n_raters = labels.shape[1]
    p_j = counts.sum(axis=0) / counts.sum()
    p_i = ((counts * (counts - 1)).sum(axis=1)) / (n_raters * (n_raters - 1))
    p_bar, p_e = p_i.mean(), (p_j**2).sum()
    return (p_bar - p_e) / (1 - p_e)


def ratio_table(result, name="odds_ratio"):
    """Exponentiated coefficients (odds or rate ratios) with 95% CIs, intercept dropped."""
    ci = result.conf_int()
    return pd.DataFrame({name: np.exp(result.params), "lo": np.exp(ci[0]), "hi": np.exp(ci[1]),
                         "p": result.pvalues}).drop(index="Intercept")


def forest(ax, table, col="odds_ratio", labels=None, color=BLUE, offset=0.0, label=None):
    """Dot-and-whisker plot of ratios on a log axis; 1.0 (no effect) is the reference line."""
    y = np.arange(len(table))[::-1] + offset
    ax.hlines(y, table.lo, table.hi, color=color, linewidth=2)
    ax.scatter(table[col], y, color=color, s=40, zorder=3, edgecolor=SURFACE, linewidth=1.5, label=label)
    ax.axvline(1, color=AXIS, linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(np.arange(len(table))[::-1], labels if labels is not None else table.index)
    ax.grid(axis="y", visible=False)


def spearman(x, y):
    return stats.spearmanr(x, y, nan_policy="omit").statistic
