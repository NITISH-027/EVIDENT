"""Frozen trivial baselines, computed on fit-split rows only."""
import numpy as np
from . import config


class Baselines:
    def __init__(self):
        self.b1_constant = None   # pooled unweighted mean true RUL over fit rows
        self.b2_median_lifetime = None  # median fit-unit lifetime

    def fit(self, fit_df_with_rul):
        self.b1_constant = float(fit_df_with_rul["true_rul"].mean())
        lifetimes = fit_df_with_rul.groupby("unit")["cycle"].max()
        self.b2_median_lifetime = float(lifetimes.median())
        return self

    def predict(self, cycles):
        cycles = np.asarray(cycles, dtype=float)
        b1 = np.full_like(cycles, self.b1_constant)
        b2 = np.maximum(0.0, self.b2_median_lifetime - cycles)
        return b1, b2
