"""Split-conformal band calibration (v2 improvement track).

Conformalized-quantile-regression style calibration of the frozen v1
[q0.1, q0.9] band. Conformity scores and offsets are computed on the TUNE
split ONLY - never on replay-holdout or official-test rows. Any replay/test
coverage measured afterwards is a post-hoc evaluation of a method chosen
after v1 results were known, and is labeled as such wherever reported.

No coverage guarantee applies here: the tune split used for calibration is
the same sample that selected the v1 hyperparameters, calibration variants are
compared on those same rows, and rows within a unit are correlated. Every
coverage number this module produces is an EMPIRICAL measurement, labeled by
the split it was measured on - never a guaranteed rate.
"""
import numpy as np
from . import config

NOMINAL = config.QUANTILES[2] - config.QUANTILES[0]  # 0.8


def conformity_scores(q10, q90, y):
    """CQR score: distance of y outside [q10, q90]; negative when inside."""
    q10, q90, y = np.asarray(q10, float), np.asarray(q90, float), np.asarray(y, float)
    return np.maximum(q10 - y, y - q90)


def conformal_offset(scores, nominal=NOMINAL):
    """Finite-sample quantile of scores at level ceil((n+1)*nominal)/n."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = min(n, int(np.ceil((n + 1) * nominal))) - 1
    return float(s[max(0, k)])


def strata_by_q50(q50):
    """Stratum index per row from PREDICTED median (known at deployment,
    unlike true RUL). Uses the protocol's RUL bands."""
    q50 = np.asarray(q50, float)
    out = np.zeros(len(q50), dtype=int)
    for i, (lo, hi) in enumerate(config.RUL_BANDS):
        m = (q50 >= lo) if hi is None else ((q50 >= lo) & (q50 <= hi))
        out[m] = i
    return out


STRATUM_LABELS = [f"{lo}-{hi}" if hi else f">{lo - 1}" for lo, hi in config.RUL_BANDS]


def fit_calibration(q10, q50, q90, y):
    """Fit global and q50-stratified offsets on the given (tune-split) rows.
    Returns a dict serializable to JSON."""
    scores = conformity_scores(q10, q90, y)
    strata = strata_by_q50(q50)
    strat = {}
    for i, label in enumerate(STRATUM_LABELS):
        s = scores[strata == i]
        strat[label] = {"n": int(len(s)),
                        "offset": conformal_offset(s) if len(s) else None}
    return {"method": "empirical split-conformal-style calibration on tune rows",
            "nominal": NOMINAL,
            "global_offset": conformal_offset(scores),
            "stratified_offsets": strat,
            "n_tune_rows": int(len(scores)),
            "caveat": "no coverage guarantee: calibration shares the tune split "
                      "with v1 hyperparameter selection; all coverage numbers are "
                      "empirical measurements on named splits, not guaranteed rates"}


def apply_calibration(q10, q50, q90, calib, stratified):
    """Apply the fitted offset(s). Returns (lo, hi)."""
    q10, q50, q90 = (np.asarray(q10, float), np.asarray(q50, float), np.asarray(q90, float))
    if stratified:
        strata = strata_by_q50(q50)
        lo, hi = np.empty_like(q10), np.empty_like(q10)
        for i, label in enumerate(STRATUM_LABELS):
            m = strata == i
            off = calib["stratified_offsets"][label]["offset"]
            off = calib["global_offset"] if off is None else off
            lo[m], hi[m] = q10[m] - off, q90[m] + off
    else:
        off = calib["global_offset"]
        lo, hi = q10 - off, q90 + off
    return lo, hi


def coverage_by_stratum(y, lo, hi, q50):
    """Marginal and per-stratum coverage/width for selection and reporting."""
    y, lo, hi, q50 = map(lambda a: np.asarray(a, float), (y, lo, hi, q50))
    strata = strata_by_q50(q50)
    per = {}
    for i, label in enumerate(STRATUM_LABELS):
        m = strata == i
        if m.any():
            per[label] = {"n": int(m.sum()),
                          "coverage": float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))),
                          "width": float(np.mean(hi[m] - lo[m]))}
    return {"coverage": float(np.mean((y >= lo) & (y <= hi))),
            "width": float(np.mean(hi - lo)), "per_stratum": per}


def cross_fitted_tune_coverage(tune_df, stratified, nominal=NOMINAL):
    """Honest empirical estimate on UNSEEN units from the same development
    split: leave-one-unit-out offsets, per-unit coverage on the held-out tune
    unit, equal-unit average. Small tune splits make this noisy - report n."""
    import numpy as np
    units = np.unique(tune_df["unit"])
    covs, widths = [], []
    for u in units:
        rest = tune_df[tune_df["unit"] != u]
        held = tune_df[tune_df["unit"] == u]
        cal = fit_calibration(rest["q10"], rest["q50"], rest["q90"], rest["true_rul"])
        lo, hi = apply_calibration(held["q10"], held["q50"], held["q90"], cal, stratified)
        covs.append(float(np.mean((held["true_rul"] >= lo) & (held["true_rul"] <= hi))))
        widths.append(float(np.mean(hi - lo)))
    return {"n_units": int(len(units)),
            "coverage_equal_unit": float(np.mean(covs)) if covs else None,
            "width_equal_unit": float(np.mean(widths)) if widths else None,
            "design": "leave-one-unit-out within the tune split; empirical",
            "caveat": ("NOT a clean generalization estimate: these tune rows also "
                       "selected the v1 hyperparameters and were used to compare "
                       "calibration variants, so selection reuse inflates this "
                       "estimate in an unquantified way; treat as a development "
                       "diagnostic only")}
