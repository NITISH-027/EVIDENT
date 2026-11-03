"""EVIDENT pipeline: frozen splits -> features -> tune -> train -> replay eval
-> official test endpoints -> results. FD002-FD004 scored exactly once with this
final code; FD001 was additionally scored once by a pre-refactor smoke run
(identical result; see README provenance note). Run: python3 run_pipeline.py [FD001 ...]
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config, data, features, splits as splits_mod, baselines as bl, model as mm, evaluate as ev

TUNE_GRID = [
    dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31),
    dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=31),
    dict(max_iter=300, learning_rate=0.10, max_leaf_nodes=31),
    dict(max_iter=600, learning_rate=0.10, max_leaf_nodes=31),
    dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63),
]


def run_subset(subset):
    t0 = time.time()
    res = {"subset": subset, "constants": {
        "window": config.WINDOW, "horizon_N": config.HORIZON_N,
        "quantiles": config.QUANTILES, "split": config.SPLIT_FRACTIONS,
        "seed": config.SEED, "bootstrap": config.BOOTSTRAP_RESAMPLES}}
    with open(os.path.join(config.RESULTS_DIR, "splits.json")) as f:
        sp = json.load(f)["subsets"][subset]

    train, schema_train = data.load_subset(subset, "train")
    test, schema_test = data.load_subset(subset, "test")
    endpoints = data.load_test_endpoints(subset)
    res["schema"] = {"train": schema_train, "test": schema_test}
    train = data.true_rul_train(train)

    feat = features.build_features(train)
    feat = feat.merge(train[["unit", "cycle", "true_rul"]], on=["unit", "cycle"])
    fit_ids, tune_ids, replay_ids = sp["fit"], sp["tune"], sp["replay"]
    fit_f = feat[feat.unit.isin(fit_ids)]
    tune_f = feat[feat.unit.isin(tune_ids)]
    replay_f = feat[feat.unit.isin(replay_ids)]

    const_ch = features.find_constant_channels(fit_f.assign(**{c: fit_f[c] for c in []}) if False else
                                               train[train.unit.isin(fit_ids)])
    res["constant_channels_excluded"] = const_ch
    cols = [c for c in features.feature_columns()
            if not any(c.startswith(ch + "_") for ch in const_ch)]
    res["n_features"] = len(cols)

    pre = features.Preprocessor(cols).fit(fit_f)
    base = bl.Baselines().fit(fit_f)
    res["baselines"] = {"B1_constant": base.b1_constant, "B2_median_lifetime": base.b2_median_lifetime}

    X_fit, y_fit = pre.transform(fit_f), fit_f["true_rul"].to_numpy()
    X_tune, y_tune = pre.transform(tune_f), tune_f["true_rul"].to_numpy()
    w_fit = mm.unit_weights(fit_f)

    # Hyperparameter selection on the TUNE split only, median head; the
    # selected config is shared across the three quantile heads (predeclared).
    tuning = []
    for params in TUNE_GRID:
        m = mm.QuantileRULModel(params={**mm.DEFAULT_PARAMS, **params})
        m.heads[0.5] = m.heads.get(0.5)
        from sklearn.ensemble import HistGradientBoostingRegressor
        h = HistGradientBoostingRegressor(loss="quantile", quantile=0.5,
                                          **{**mm.DEFAULT_PARAMS, **params})
        h.fit(X_fit, y_fit, sample_weight=w_fit)
        p = h.predict(X_tune)
        tuning.append({"params": params, "tune_pinball_q50": ev.pinball(y_tune, p, 0.5),
                       "tune_rmse_q50": ev.rmse(p, y_tune)})
    best = min(tuning, key=lambda r: r["tune_pinball_q50"])
    res["tuning"] = {"candidates": tuning, "selected": best,
                     "note": "selected on tune split only; shared across quantile heads"}

    sel = {**mm.DEFAULT_PARAMS, **best["params"]}
    model = mm.QuantileRULModel(params=sel).fit(X_fit, y_fit, sample_weight=w_fit)

    # Weighting sensitivity (tune split, median head): unit-balanced vs unweighted.
    from sklearn.ensemble import HistGradientBoostingRegressor
    h_unw = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, **sel).fit(X_fit, y_fit)
    res["weighting_sensitivity"] = {
        "unit_balanced_tune_rmse": ev.rmse(model.heads[0.5].predict(X_tune), y_tune),
        "unweighted_tune_rmse": ev.rmse(h_unw.predict(X_tune), y_tune),
        "note": "sensitivity run on tune split; unit-balanced kept as proposed default"}

    # ---- Replay-holdout: trajectory/band/warning claims (untouched) ----
    X_re = pre.transform(replay_f)
    q10, q50, q90, diag_re = model.predict(X_re)
    replay_f = replay_f.assign(q10=q10, q50=q50, q90=q90,
                               envelope_flag=pre.envelope_flags(replay_f))
    y_re = replay_f["true_rul"].to_numpy()
    res["postprocess_replay"] = diag_re
    res["replay"] = {
        "units": int(replay_f.unit.nunique()), "rows": int(len(replay_f)),
        "rmse_all": ev.rmse(q50, y_re),
        "banded_rmse": ev.banded_rmse(q50, y_re),
        "envelope_flag_rate": float(replay_f.envelope_flag.mean()),
    }
    cov_u, wid_u = ev.per_unit_coverage_width(replay_f.unit.to_numpy(), y_re, q10, q90)
    cov_mean, cov_ci = ev.unit_bootstrap_ci(cov_u)
    wid_mean, wid_ci = ev.unit_bootstrap_ci(wid_u)
    res["replay"]["band"] = {"nominal": 0.80, "coverage_mean": cov_mean, "coverage_ci": cov_ci,
                             "width_mean": wid_mean, "width_ci": wid_ci,
                             "population": "per unit first, equal-unit average, unit bootstrap CI"}
    res["replay"]["warning"] = ev.warning_episode_eval(replay_f)
    replay_f.to_pickle(os.path.join(config.RESULTS_DIR, f"replay_{subset}.pkl"))

    # ---- Official test: scored ONCE at endpoints (independent point/interval) ----
    test_feat = features.build_features(test)
    last = test_feat.sort_values("cycle").groupby("unit").tail(1)
    X_te = pre.transform(last)
    tq10, tq50, tq90, diag_te = model.predict(X_te)
    true_end = endpoints[: len(last)] if len(endpoints) >= len(last) else endpoints
    order = last["unit"].to_numpy() - 1
    true_end = endpoints[order]
    b1, b2 = base.predict(last["cycle"].to_numpy())
    res["postprocess_test"] = diag_te
    res["test"] = {
        "units": int(len(last)),
        "model_endpoint_rmse": ev.rmse(tq50, true_end),
        "model_endpoint_mae": ev.mae(tq50, true_end),
        "model_phm_score": ev.phm_score(tq50, true_end),
        "vs_B1": ev.paired_bootstrap_rmse_diff(tq50, b1, true_end),
        "vs_B2": ev.paired_bootstrap_rmse_diff(tq50, b2, true_end),
        "endpoint_coverage": float(np.mean((true_end >= tq10) & (true_end <= tq90))),
        "endpoint_width_mean": float(np.mean(tq90 - tq10)),
    }
    last.assign(q10=tq10, q50=tq50, q90=tq90, true_rul=true_end, b1=b1, b2=b2).to_pickle(
        os.path.join(config.RESULTS_DIR, f"test_endpoint_{subset}.pkl"))

    ok_b1 = res["test"]["vs_B1"]["excludes_zero"]
    ok_b2 = res["test"]["vs_B2"]["excludes_zero"]
    res["verdicts"] = {
        "model_vs_baselines": "accepted" if (ok_b1 and ok_b2) else "inconclusive / no demonstrated improvement",
        "band_quality": "see replay band coverage vs nominal 0.80 - reported separately, not implied by RMSE",
        "warning_quality": "see replay warning episode metrics - reported separately, not implied by RMSE",
        "note": "three verdicts stand separately; FD002-FD004 official test scored once, FD001 twice (identical; reproducibility repeat - see README); "
                "post-test changes would be exploratory only",
    }
    res["runtime_seconds"] = round(time.time() - t0, 1)
    out = os.path.join(config.RESULTS_DIR, f"results_{subset}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    return res


if __name__ == "__main__":
    subsets = sys.argv[1:] or config.SUBSETS
    for s in subsets:
        r = run_subset(s)
        print(f"{s}: test RMSE {r['test']['model_endpoint_rmse']:.2f} "
              f"(B1 {r['test']['vs_B1']['rmse_baseline']:.2f}, B2 {r['test']['vs_B2']['rmse_baseline']:.2f}) "
              f"verdict={r['verdicts']['model_vs_baselines']} "
              f"coverage={r['replay']['band']['coverage_mean']:.3f} "
              f"catch={r['replay']['warning']['catch_fraction']:.3f} "
              f"({r['runtime_seconds']}s)")
