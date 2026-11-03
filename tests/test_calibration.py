"""Tests for the v2 empirical tune-split calibration module and the v2
evaluation plumbing. All calibration claims are empirical - tests check
mechanics and labeling, never guaranteed coverage."""
import os, sys, json
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import calibration as cal


def _toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.uniform(0, 150, n)
    q50 = true + rng.normal(0, 12, n)
    q90 = q50 + 15 + rng.normal(0, 3, n)
    q10 = q50 - 15 + rng.normal(0, 3, n)
    unit = np.repeat(np.arange(1, n // 10 + 1), 10)[:n]
    return pd.DataFrame({"unit": unit, "true_rul": true, "q10": q10, "q50": q50, "q90": q90})


def test_conformal_offset_in_sample_coverage_at_least_nominal():
    df = _toy()
    scores = cal.conformity_scores(df.q10, df.q90, df.true_rul)
    off = cal.conformal_offset(scores)
    lo, hi = df.q10 - off, df.q90 + off
    cov = np.mean((df.true_rul >= lo) & (df.true_rul <= hi))
    assert cov >= cal.NOMINAL  # in-sample construction property, not a guarantee elsewhere


def test_apply_calibration_widens_intervals():
    df = _toy()
    calib = cal.fit_calibration(df.q10, df.q50, df.q90, df.true_rul)
    lo, hi = cal.apply_calibration(df.q10, df.q50, df.q90, calib, False)
    assert np.all(lo <= df.q10.to_numpy() + 1e-9)
    assert np.all(hi >= df.q90.to_numpy() - 1e-9)


def test_stratified_offsets_vary_by_band_when_errors_differ():
    # errors larger for high-RUL band -> stratified offset larger there
    n = 300
    true = np.concatenate([np.full(n, 20.0), np.full(n, 120.0)])
    err = np.concatenate([np.full(n, 5.0), np.full(n, 40.0)])
    q50 = true + err
    q10, q90 = q50 - 10, q50 + 10
    calib = cal.fit_calibration(q10, q50, q90, true)
    strat = calib["stratified_offsets"]
    offs = sorted(v["offset"] for v in strat.values() if v["offset"] is not None)
    assert offs[0] < offs[-1]


def test_cross_fitted_coverage_is_empirical_and_labelled():
    df = _toy(120)
    out = cal.cross_fitted_tune_coverage(df, stratified=False)
    assert out["n_units"] == 12
    assert 0.0 <= out["coverage_equal_unit"] <= 1.0
    assert "empirical" in out["design"]
    calib = cal.fit_calibration(df.q10, df.q50, df.q90, df.true_rul)
    assert "empirical" in calib["method"]
    assert "no coverage guarantee" in calib["caveat"]


def test_results_v2_files_exist_and_carry_post_hoc_labels():
    for ds in ("FD001", "FD002", "FD003", "FD004"):
        p = os.path.join(ROOT, "results", f"results_v2_{ds}.json")
        assert os.path.exists(p), f"{p} missing - run evaluate_v2.py"
        o = json.load(open(p))
        assert "post-hoc" in o["replay_band_v2"]["label"]
        assert "post-hoc" in o["replay_warning_v2"]["label"]
        assert "post-hoc" in o["test_endpoint_band_v2"]["label"]
        assert o["selected_variant"] in ("global", "stratified")
        assert o["warning_policy_selection"]["selected"] in ("v1_q50", "q90", "calibrated_lower", "persistence_3")
        # v1 baselines preserved alongside v2 for comparison
        assert "v1_coverage_equal_unit" in o["replay_band_v2"]
        assert "v1_width_equal_unit" in o["replay_band_v2"]
        assert "v1_warning" in o["replay_warning_v2"]


def test_calibration_artifacts_fitted_on_tune_rows_only():
    for ds in ("FD001", "FD002", "FD003", "FD004"):
        calib = json.load(open(os.path.join(ROOT, "results", f"calibration_{ds}.json")))
        assert calib["n_tune_rows"] > 0
        assert calib["nominal"] == 0.8
