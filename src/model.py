"""Per-subset gradient-boosted quantile model: three DISTINCT heads
(q0.1, q0.5, q0.9) fitted separately with the quantile loss.

Fixed post-processing, part of the model definition and applied identically
on every split: sort the three head outputs into q0.1<=q0.5<=q0.9 order,
then clip at 0. Raw negative-prediction and quantile-crossing rates are
recorded, never hidden. Training targets are never capped; predictions are
never clipped to the training target range.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from . import config

DEFAULT_PARAMS = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
                      min_samples_leaf=20, random_state=config.SEED)


def unit_weights(df):
    """Unit-balanced: each unit carries total weight 1 across its cycle rows."""
    counts = df.groupby("unit")["cycle"].transform("count")
    return (1.0 / counts).to_numpy()


class QuantileRULModel:
    def __init__(self, params=None):
        self.params = params or dict(DEFAULT_PARAMS)
        self.heads = {}
        self.fit_stats = {}

    def fit(self, X, y, sample_weight=None):
        for q in config.QUANTILES:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, **self.params)
            m.fit(X, y, sample_weight=sample_weight)
            self.heads[q] = m
        return self

    def predict_raw(self, X):
        return {q: self.heads[q].predict(X) for q in config.QUANTILES}

    def predict(self, X):
        """Fixed post-processing: order quantiles, clip at 0. Returns
        (q10, q50, q90, diagnostics) with raw rates measured pre-processing."""
        raw = self.predict_raw(X)
        cols = np.column_stack([raw[q] for q in config.QUANTILES])
        neg_rate = float((cols < 0).any(axis=1).mean()) if len(cols) else 0.0
        crossing = float(((cols[:, 0] > cols[:, 1]) | (cols[:, 1] > cols[:, 2])).mean()) if len(cols) else 0.0
        fixed = np.sort(cols, axis=1)
        fixed = np.maximum(fixed, 0.0)
        diag = {"raw_negative_row_rate": neg_rate, "raw_crossing_row_rate": crossing}
        return fixed[:, 0], fixed[:, 1], fixed[:, 2], diag
