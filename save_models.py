"""Persist trained models for reproducible inference.
Refits with the FROZEN configuration (splits.json + selected params from the
original run's results JSON), then ASSERTS the refit model reproduces the
stored replay predictions exactly (within float tolerance). This is model
persistence, not retraining: the frozen results stand from the original run.
"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config, data, features, model as mm


def save_subset(subset):
    with open(os.path.join(config.RESULTS_DIR, f"results_{subset}.json")) as f:
        res = json.load(f)
    with open(os.path.join(config.RESULTS_DIR, "splits.json")) as f:
        sp = json.load(f)["subsets"][subset]
    train, _ = data.load_subset(subset, "train")
    train = data.true_rul_train(train)
    feat = features.build_features(train)
    feat = feat.merge(train[["unit", "cycle", "true_rul"]], on=["unit", "cycle"])
    fit_f = feat[feat.unit.isin(sp["fit"])]
    replay_f = feat[feat.unit.isin(sp["replay"])]
    const_ch = res["constant_channels_excluded"]
    cols = [c for c in features.feature_columns()
            if not any(c.startswith(ch + "_") for ch in const_ch)]
    pre = features.Preprocessor(cols).fit(fit_f)
    sel = {k: v for k, v in res["tuning"]["selected"]["params"].items()}
    params = {**mm.DEFAULT_PARAMS, **sel}
    m = mm.QuantileRULModel(params=params)
    m.fit(pre.transform(fit_f), fit_f["true_rul"].to_numpy(),
          sample_weight=mm.unit_weights(fit_f))
    q10, q50, q90, _ = m.predict(pre.transform(replay_f))
    stored = __import__("joblib").load(os.path.join(config.RESULTS_DIR, f"replay_{subset}.pkl"))
    ok = (np.allclose(q10, stored.q10, rtol=1e-6, atol=1e-6)
          and np.allclose(q50, stored.q50, rtol=1e-6, atol=1e-6)
          and np.allclose(q90, stored.q90, rtol=1e-6, atol=1e-6))
    if not ok:
        raise SystemExit(f"{subset}: refit predictions DO NOT match frozen results - not saving")
    with open(os.path.join(config.RESULTS_DIR, f"model_{subset}.pkl"), "wb") as f:
        pickle.dump({"model": m, "preprocessor": pre, "feature_cols": cols,
                     "constant_channels": const_ch, "params": params,
                     "matches_frozen_results": True}, f)
    print(f"{subset}: saved, predictions identical to frozen results")


if __name__ == "__main__":
    save_subset(sys.argv[1])
