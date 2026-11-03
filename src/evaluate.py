"""Evaluation contract implementation. Everything here follows the frozen
protocol: tune split for selection only; replay-holdout for trajectory/band/
warning claims; official test scored once at endpoints; paired unit bootstrap
on the RMSE difference; inconclusive reported as inconclusive.
"""
import numpy as np
from . import config


def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def phm_score(pred, true):
    """Asymmetric PHM-style score: late predictions penalized more."""
    d = np.asarray(pred, float) - np.asarray(true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def pinball(y, pred, q):
    d = np.asarray(y, float) - np.asarray(pred, float)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def banded_rmse(pred, true):
    """RMSE per true-RUL band, including early life."""
    true = np.asarray(true, float); pred = np.asarray(pred, float)
    out = {}
    for lo, hi in config.RUL_BANDS:
        m = (true >= lo) if hi is None else ((true >= lo) & (true <= hi))
        label = f"{lo}-{hi}" if hi else f">{lo - 1}"
        out[label] = {"n": int(m.sum()), "rmse": rmse(pred[m], true[m]) if m.any() else None}
    return out


def per_unit_coverage_width(units, true, lo, hi):
    """Coverage and width per unit first; equal-unit average after."""
    units = np.asarray(units)
    cov, wid = {}, {}
    for u in np.unique(units):
        m = units == u
        cov[u] = float(np.mean((true[m] >= lo[m]) & (true[m] <= hi[m])))
        wid[u] = float(np.mean(hi[m] - lo[m]))
    return cov, wid


def unit_bootstrap_ci(values_by_unit, resamples=config.BOOTSTRAP_RESAMPLES,
                      level=config.CI_LEVEL, seed=config.SEED):
    """CI for the equal-unit average by resampling units."""
    units = list(values_by_unit)
    vals = np.array([values_by_unit[u] for u in units])
    if len(units) < 2:
        v = float(vals.mean()) if len(vals) else None
        return v, (None, None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(resamples, len(vals)))
    means = vals[idx].mean(axis=1)
    alpha = (1 - level) / 2
    return float(vals.mean()), (float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)))


def paired_bootstrap_rmse_diff(pred_model, pred_base, true,
                               resamples=config.BOOTSTRAP_RESAMPLES,
                               level=config.CI_LEVEL, seed=config.SEED):
    """Paired unit-resampling bootstrap CI on (RMSE_base - RMSE_model).
    Positive means the model is better. Acceptance needs CI excluding 0."""
    pred_model = np.asarray(pred_model, float); pred_base = np.asarray(pred_base, float)
    true = np.asarray(true, float)
    n = len(true)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    se_m = (pred_model - true) ** 2
    se_b = (pred_base - true) ** 2
    diffs = np.sqrt(se_b[idx].mean(axis=1)) - np.sqrt(se_m[idx].mean(axis=1))
    alpha = (1 - level) / 2
    return {"rmse_model": rmse(pred_model, true), "rmse_baseline": rmse(pred_base, true),
            "diff_mean": float(diffs.mean()),
            "ci": (float(np.quantile(diffs, alpha)), float(np.quantile(diffs, 1 - alpha))),
            "excludes_zero": bool(np.quantile(diffs, alpha) > 0)}


def warning_episode_eval(df_replay):
    """Declared episode policy on untouched replay-holdout units (full
    trajectories to failure). df_replay: unit, cycle, true_rul, q50.

    Exclusive first-alert classification per unit:
      catch     - first flag at a cycle with 0 < true_rul <= N
      premature - first flag while true_rul > N (never also a catch)
      miss      - no flag before the failure cycle (first flag at true_rul == 0
                  is too late and does not count as a catch)
    Reported separately: catch fraction; lead time over catches; raw per-cycle
    FPR (flagged cycles with true_rul > N over all cycles with true_rul > N);
    latched unit alert burden. No downtime/cost claims.
    """
    N = config.HORIZON_N
    per_unit, lead_times = {}, []
    for u, g in df_replay.groupby("unit"):
        g = g.sort_values("cycle")
        flag = (g["q50"] <= N).to_numpy()
        tr = g["true_rul"].to_numpy()
        cyc = g["cycle"].to_numpy()
        firsts = np.nonzero(flag)[0]
        if len(firsts) == 0:
            per_unit[u] = {"class": "miss", "first_flag_cycle": None}
            continue
        i = firsts[0]
        if tr[i] <= 0:
            per_unit[u] = {"class": "miss", "first_flag_cycle": int(cyc[i]),
                           "note": "first flag at failure cycle (too late)"}
        elif tr[i] <= N:
            per_unit[u] = {"class": "catch", "first_flag_cycle": int(cyc[i])}
            lead_times.append(float(g["cycle"].max() - cyc[i]))
        else:
            per_unit[u] = {"class": "premature", "first_flag_cycle": int(cyc[i]),
                           "true_rul_at_flag": float(tr[i])}
    n_units = len(per_unit)
    n_catch = sum(1 for v in per_unit.values() if v["class"] == "catch")
    n_prem = sum(1 for v in per_unit.values() if v["class"] == "premature")
    n_miss = n_units - n_catch - n_prem
    flag_all = (df_replay["q50"] <= N).to_numpy()
    tr_all = df_replay["true_rul"].to_numpy()
    neg = tr_all > N
    fpr = float((flag_all & neg).sum() / neg.sum()) if neg.sum() else None
    alerted_cycles = df_replay.assign(flag=flag_all).groupby("unit")["flag"].sum()
    lead = np.array(lead_times)
    return {
        "horizon_N": N,
        "units": n_units,
        "catch_fraction": n_catch / n_units if n_units else None,
        "premature_unit_fraction": n_prem / n_units if n_units else None,
        "miss_fraction": n_miss / n_units if n_units else None,
        "lead_time_cycles": {
            "n": int(len(lead)),
            "median": float(np.median(lead)) if len(lead) else None,
            "p10": float(np.quantile(lead, 0.1)) if len(lead) else None,
            "p90": float(np.quantile(lead, 0.9)) if len(lead) else None},
        "per_cycle_fpr": fpr,
        "latched_alert_burden": {
            "units_ever_alerting_fraction": float((alerted_cycles > 0).mean()),
            "mean_alerted_cycles_per_unit": float(alerted_cycles.mean())},
        "policy": "exclusive first-alert episode policy (declared, not universal); "
                  "per-cycle FPR and latched burden are different quantities",
    }
