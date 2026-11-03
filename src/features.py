"""Strictly-past trailing-window features.

Per unit-cycle: cycle index + per-channel (3 settings + 21 sensors) window
mean, std (ddof=0 -> finite 0 for a single cycle) and slope over the past
WINDOW cycles including the current one. Short histories use the available
cycles (slope needs >= 2 points, else 0). No feature touches any cycle after t.

Constant channels (zero std over fit rows) are excluded as uninformative -
a validated per-dataset rule, not a sensor-fault flag.
Normalizer statistics and the per-feature min/max envelope come from
fit-split rows ONLY.
"""
import numpy as np
import pandas as pd
from . import config
from .data import CHANNELS

FEAT_STATS = ("mean", "std", "slope")


def _window_stats_unit(x, w):
    """Vectorized: window mean, std (ddof=0), slope for windows ending at each i.
    Short histories use available points; slope 0 when fewer than 2 points."""
    x = np.asarray(x, float)
    L = len(x)
    idx = np.arange(L)
    lo = np.maximum(0, idx - w + 1)
    k = (idx - lo + 1).astype(float)
    c = np.concatenate([[0.0], np.cumsum(x)])
    cx = np.concatenate([[0.0], np.cumsum(x * idx)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    sx = c[idx + 1] - c[lo]
    sxx = c2[idx + 1] - c2[lo]
    stx = (cx[idx + 1] - cx[lo]) - lo * sx
    mean = sx / k
    var = sxx / k - mean ** 2
    std = np.sqrt(np.clip(var, 0, None))
    st = k * (k - 1) / 2.0
    stt = (k - 1) * k * (2 * k - 1) / 6.0
    den = k * stt - st * st
    slope = np.where((k >= 2) & (den > 0), (k * stx - st * sx) / np.where(den > 0, den, 1.0), 0.0)
    return mean, std, slope


def build_features(df, w=config.WINDOW):
    """Returns DataFrame with unit, cycle and 1 + 24*3 feature columns."""
    df = df.sort_values(["unit", "cycle"])
    out = df[["unit", "cycle"]].copy()
    for ch in CHANNELS:
        parts_m, parts_s, parts_l = [], [], []
        for _, g in df.groupby("unit", sort=False):
            m, s, l = _window_stats_unit(g[ch].to_numpy(dtype=float), w)
            parts_m.append(m); parts_s.append(s); parts_l.append(l)
        out[f"{ch}_mean"] = np.concatenate(parts_m)
        out[f"{ch}_std"] = np.concatenate(parts_s)
        out[f"{ch}_slope"] = np.concatenate(parts_l)
    return out


FEATURE_COLS_CACHE = None


def feature_columns():
    global FEATURE_COLS_CACHE
    if FEATURE_COLS_CACHE is None:
        FEATURE_COLS_CACHE = ["cycle"] + [f"{ch}_{st}" for ch in CHANNELS for st in FEAT_STATS]
    return FEATURE_COLS_CACHE


def find_constant_channels(fit_raw):
    """Channels with zero variance over fit rows: excluded as uninformative."""
    const = [ch for ch in CHANNELS if fit_raw[ch].std(ddof=0) == 0]
    return const


class Preprocessor:
    """Fit-split-only normalizer + envelope. fit() on fit rows; transform() anywhere."""

    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.mean_ = None
        self.std_ = None
        self.env_min_ = None
        self.env_max_ = None

    def fit(self, feat_df):
        X = feat_df[self.feature_cols].to_numpy(dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        std[std == 0] = 1.0
        self.std_ = std
        self.env_min_ = X.min(axis=0)
        self.env_max_ = X.max(axis=0)
        return self

    def transform(self, feat_df):
        X = feat_df[self.feature_cols].to_numpy(dtype=float)
        return (X - self.mean_) / self.std_

    def envelope_flags(self, feat_df):
        """Descriptive screen: rows outside fit min/max per feature.
        NOT a validated out-of-distribution detector."""
        X = feat_df[self.feature_cols].to_numpy(dtype=float)
        return ((X < self.env_min_) | (X > self.env_max_)).any(axis=1)
