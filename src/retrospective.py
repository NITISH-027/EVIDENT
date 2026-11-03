"""Retrospective model/decision evaluation against RECORDED simulator truth.

Separate estimand, predefined before computation: cycle-local decision quality
of the frozen model on simulator replay units, using the recorded remaining
life at each cycle as truth. The decision-relevant window is the same horizon
the flag claims (predicted remaining life <= HORIZON_N cycles), so the truth
check is recorded RUL <= HORIZON_N at that cycle.

This is NOT the first-alert unit policy (the scoreboard's latched unit
classification is untouched and stays primary), NOT machine-health validation
beyond the simulator's recorded outcomes, and NOT reviewer-agreement scoring
(reviewer opinion vs flag state is reported separately, optionally, and is not
outcome scoring). Imported units have no recorded truth and are always
excluded; there is no imported-truth invention.

Equal-unit weighting: per-unit fractions averaged over units, so long-lived
units do not dominate.
"""
import numpy as np


def policy_flags(df, horizon, policy, persistence_k=3):
    """Boolean flag series per policy (row order of df sorted by unit, cycle).
    v1_q50: q50<=horizon. q90: raw q90<=horizon. persistence_3: q50<=horizon for
    3 consecutive cycles of the same unit, first alert at the 3rd confirming
    cycle."""
    d = df.sort_values(["unit", "cycle"])
    if policy == "v1_q50":
        return (d.q50 <= horizon).to_numpy(bool)
    if policy == "q90":
        return (d.q90 <= horizon).to_numpy(bool)
    if policy == "persistence_3":
        raw = (d.q50 <= horizon).astype(int)
        run = raw.groupby(d["unit"]).rolling(persistence_k, min_periods=persistence_k).sum()
        return (run.reset_index(level=0, drop=True) >= persistence_k).reindex(d.index).fillna(False).to_numpy(bool)
    raise ValueError(policy)


def _bootstrap_ci(vals, n_boot=2000, seed=20261102):
    vals = np.asarray([v for v in vals if v is not None], float)
    if len(vals) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, (n_boot, len(vals))).mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _iter_units(df, fl_all):
    d = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    d["_fl"] = fl_all
    for u, g in d.groupby("unit"):
        yield u, g, g["_fl"].to_numpy(bool)


def decision_vs_truth(df, horizon, policy="v1_q50"):
    """df: replay rows with columns unit, cycle, q50, q90, true_rul.
    Per-unit and equal-unit cycle-local decision-quality fractions with
    unit-bootstrap CIs. Descriptive statistics on correlated simulator data -
    NOT first-alert policy effectiveness (that is the scoreboard's latched
    unit classification, which is untouched)."""
    fl_all = policy_flags(df, horizon, policy)
    per_unit = {}
    for u, g, fl in _iter_units(df, fl_all):
        rul = g.true_rul.to_numpy(float)
        inw = (rul > 0) & (rul <= horizon)
        after = rul <= 0
        early = rul > horizon
        nf = int(fl.sum())
        nn = int((~fl).sum())
        per_unit[int(u)] = {
            "n_cycles": int(len(g)),
            "n_flagged": nf,
            "flag_in_window_frac": float((fl & inw).sum() / nf) if nf else None,
            "flag_early_frac": float((fl & early).sum() / nf) if nf else None,
            "flag_after_failure_frac": float((fl & after).sum() / nf) if nf else None,
            "noflag_correct_frac": float((~fl & early).sum() / nn) if nn else None,
            "noflag_in_window_frac": float((~fl & inw).sum() / nn) if nn else None,
            "noflag_after_failure_frac": float((~fl & after).sum() / nn) if nn else None,
        }
    def _eq(key):
        vals = [v[key] for v in per_unit.values() if v[key] is not None]
        return float(np.mean(vals)) if vals else None
    def _ci(key):
        return _bootstrap_ci([v[key] for v in per_unit.values()])
    return {
        "estimand": ("cycle-local decision quality vs recorded simulator truth: among cycles "
                     "with a flag shown, fractions with recorded remaining life in-window "
                     "(0 < RUL <= horizon), early (RUL > horizon), and after failure (RUL <= 0); "
                     "among unflagged cycles, fractions correct (RUL > horizon), in-window "
                     "(0 < RUL <= horizon), and after failure. Equal-unit averages with "
                     "unit-bootstrap 95% CIs. Descriptive statistics on correlated simulator "
                     "data - NOT first-alert policy effectiveness; separate from the scoreboard's "
                     "latched unit classification; simulator truth only; imported units excluded."),
        "horizon": horizon,
        "policy": policy,
        "units": len(per_unit),
        "flag_in_window_equal_unit": _eq("flag_in_window_frac"),
        "flag_in_window_ci": _ci("flag_in_window_frac"),
        "flag_early_equal_unit": _eq("flag_early_frac"),
        "flag_after_failure_equal_unit": _eq("flag_after_failure_frac"),
        "noflag_correct_equal_unit": _eq("noflag_correct_frac"),
        "noflag_in_window_equal_unit": _eq("noflag_in_window_frac"),
        "noflag_after_failure_equal_unit": _eq("noflag_after_failure_frac"),
        "per_unit": per_unit,
    }
