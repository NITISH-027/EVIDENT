import joblib
"""Descriptive metrics computed from FROZEN stored predictions only.
Additive fields, no retraining, no reselection, test not re-scored.
Also verifies saved models reproduce frozen test endpoint predictions
(cheap inference equality check).
"""
import json, os, pickle, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config, data, features, evaluate as ev


def augment(subset):
    with open(os.path.join(config.RESULTS_DIR, f"results_{subset}.json")) as f:
        res = json.load(f)
    rp = joblib.load(os.path.join(config.RESULTS_DIR, f"replay_{subset}.pkl"))
    tp = joblib.load(os.path.join(config.RESULTS_DIR, f"test_endpoint_{subset}.pkl"))
    add = {}
    add["replay_baselines_same_population"] = {
        "B1_rmse": ev.rmse(rp.b1 if "b1" in rp else np.full(len(rp), res["baselines"]["B1_constant"]), rp.true_rul),
        "B2_rmse": ev.rmse(np.maximum(0.0, res["baselines"]["B2_median_lifetime"] - rp.cycle.to_numpy()), rp.true_rul),
        "model_rmse": res["replay"]["rmse_all"],
        "population": "replay-holdout rows, identical for all three"}
    cov_u = {int(u): float(v) for u, v in
             tp.assign(c=(tp.true_rul >= tp.q10) & (tp.true_rul <= tp.q90)).groupby("unit")["c"].mean().items()}
    wid_u = {int(u): float(v) for u, v in (tp.q90 - tp.q10).groupby(tp.unit).mean().items()}
    cm, cci = ev.unit_bootstrap_ci(cov_u); wm, wci = ev.unit_bootstrap_ci(wid_u)
    add["endpoint_band"] = {"coverage_mean": cm, "coverage_ci": cci,
                            "width_mean": wm, "width_ci": wci,
                            "population": "one endpoint per unit, unit bootstrap CI"}
    res["descriptive_additions"] = add
    res["descriptive_additions_note"] = ("computed from frozen stored predictions only; "
                                         "no retraining, no reselection, official test not re-scored")
    with open(os.path.join(config.RESULTS_DIR, f"results_{subset}.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    # cheap equality check: saved model reproduces frozen test endpoint predictions
    with open(os.path.join(config.RESULTS_DIR, f"model_{subset}.pkl"), "rb") as f:
        bundle = joblib.load(f)
    test, _ = data.load_subset(subset, "test")
    tf = features.build_features(test)
    last = tf.sort_values("cycle").groupby("unit").tail(1)
    q10, q50, q90, _ = bundle["model"].predict(bundle["preprocessor"].transform(last))
    ok = (np.allclose(q50, tp.q50.to_numpy(), rtol=1e-6, atol=1e-6)
          and np.allclose(q10, tp.q10.to_numpy(), rtol=1e-6, atol=1e-6)
          and np.allclose(q90, tp.q90.to_numpy(), rtol=1e-6, atol=1e-6))
    print(subset, "replay B1/B2:", round(add["replay_baselines_same_population"]["B1_rmse"], 1),
          round(add["replay_baselines_same_population"]["B2_rmse"], 1),
          "| endpoint coverage CI:", [round(x, 3) for x in cci],
          "| saved-model==frozen test predictions:", ok)
    assert ok


for s in config.SUBSETS:
    augment(s)
