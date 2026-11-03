"""EVIDENT v2 improvement evaluation.

Empirical tune-split band calibration + warning-policy candidates.
Fits NOTHING on replay-holdout or official test; every replay/test number is a
post-hoc evaluation of a method chosen after v1 results were known, and every
output record says so. v1 artifacts are read, never modified.

CANDIDATE REGISTRY (warning policy), with timestamps (IST, 2026-11-02):
- Internal design note 14:15:53 and 14:16:22: P0 v1 q50<=30; P1 q90<=30;
  P2 conformal lower bound <=30; P3 q50<=30 for 3 consecutive cycles.
- Methodology note sent to parent 14:15:53 listed P0, P2, P3 and OMITTED P1
  (compression error in the note; P1 was in the design registry before the
  note was sent and before any measurement). Expansion disclosed to parent
  14:2x; the note's list was not the full registry.
- Run 1 (14:17:04): implemented P0, P1, P2. P3 declared but not implemented.
- Run 2 (later same day): P3 implemented and added; this addition was made
  AFTER run-1 replay numbers had been seen, so the candidate SET is itself
  post-hoc. The selection RULE is unchanged and uses tune-split scores only.
Selection rule (fixed since run 1): among candidates whose TUNE catch fraction
is >= v1's tune catch fraction, minimize TUNE premature fraction; tie-break:
higher median lead time; if no candidate qualifies, keep v1 and say so.
Replay/test were never inspected to select candidates or variants: run 1
produced tune selection and replay/test measurement in a single execution.

Run: python3 evaluate_v2.py
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config, data, features, evaluate as ev, calibration as cal

RESULTS = config.RESULTS_DIR
POST_HOC = ("post-hoc evaluation on {split}: method chosen after v1 results on "
            "this split were known; not an untouched confirmation")

# Predeclared warning-policy candidates (first-alert policy, same exclusive
# classification as v1). Selection criterion fixed before looking at results:
# among candidates whose TUNE catch fraction is not worse than v1's tune catch
# fraction, minimize TUNE premature fraction; tie-break: higher median lead
# time; if no candidate satisfies the constraint, keep v1 and say so.
def policy_flags(kind, q10, q50, q90, lo_cal=None):
    if kind == "v1_q50":
        return q50 <= config.HORIZON_N
    if kind == "q90":
        return q90 <= config.HORIZON_N
    if kind == "calibrated_lower":
        return lo_cal <= config.HORIZON_N
    raise ValueError(kind)


PERSISTENCE_K = 3


def warning_eval_with_policy(df, kind, lo_cal=None):
    d = df.sort_values(["unit", "cycle"]).copy()
    if kind == "persistence_3":
        # flag shown at cycle c iff q50<=N at c and at the K-1 preceding cycles
        # of the same unit: first alert lands at the K-th confirming cycle
        raw = (d.q50 <= config.HORIZON_N).astype(int)
        run = raw.groupby(d["unit"]).rolling(PERSISTENCE_K, min_periods=PERSISTENCE_K).sum()
        d["flag_policy"] = (run.reset_index(level=0, drop=True) >= PERSISTENCE_K).reindex(d.index).fillna(False).to_numpy(bool)
    else:
        d["flag_policy"] = policy_flags(kind, d.q10.to_numpy(), d.q50.to_numpy(),
                                        d.q90.to_numpy(),
                                        None if lo_cal is None else lo_cal)
    out = ev.warning_episode_eval(d.assign(q50=np.where(d["flag_policy"], 0.0, 1e9)))
    return out


def run_subset(ds, tune_feat):
    bundle = joblib.load(os.path.join(RESULTS, f"model_{ds}.pkl"))
    pre, mdl = bundle["preprocessor"], bundle["model"]
    v1 = json.load(open(os.path.join(RESULTS, f"results_{ds}.json")))

    tune = tune_feat
    q10, q50, q90, _ = mdl.predict(pre.transform(tune))
    tune = tune.assign(q10=q10, q50=q50, q90=q90)

    # --- band calibration fitted on tune rows only ---
    calib = cal.fit_calibration(tune.q10, tune.q50, tune.q90, tune.true_rul)
    with open(os.path.join(RESULTS, f"calibration_{ds}.json"), "w") as f:
        json.dump(calib, f, indent=1)

    out = {"subset": ds, "v2": True, "calibration": calib,
           "post_hoc_notice": POST_HOC.format(split="replay/test")}

    # tune in-sample calibration coverage (optimistic; development sample)
    for stratified, key in ((False, "global"), (True, "stratified")):
        lo, hi = cal.apply_calibration(tune.q10, tune.q50, tune.q90, calib, stratified)
        cov, wid = ev.per_unit_coverage_width(tune.unit.to_numpy(), tune.true_rul.to_numpy(), lo, hi)
        out[f"tune_calibration_{key}"] = {
            "coverage_equal_unit": float(np.mean(list(cov.values()))),
            "width_equal_unit": float(np.mean(list(wid.values()))),
            "label": "in-sample on the calibration rows; optimistic by construction"}
        out[f"tune_crossfitted_{key}"] = cal.cross_fitted_tune_coverage(
            tune[["unit", "q10", "q50", "q90", "true_rul"]], stratified)

    # variant selection: predeclared - pick stratified only if its cross-fitted
    # equal-unit coverage is at least 2 points better than global's AND its
    # cross-fitted width is no more than 10% larger.
    g, s = out["tune_crossfitted_global"], out["tune_crossfitted_stratified"]
    use_strat = (s["coverage_equal_unit"] is not None and g["coverage_equal_unit"] is not None
                 and s["coverage_equal_unit"] >= g["coverage_equal_unit"] + 0.02
                 and s["width_equal_unit"] <= g["width_equal_unit"] * 1.10)
    out["selected_variant"] = "stratified" if use_strat else "global"
    out["selection_rule"] = ("stratified only if cross-fitted coverage >= global + 2pts "
                             "and width <= global * 1.10; fixed before measurement")

    # --- replay-holdout: post-hoc band measurement ---
    replay = joblib.load(os.path.join(RESULTS, f"replay_{ds}.pkl"))
    lo, hi = cal.apply_calibration(replay.q10, replay.q50, replay.q90, calib, use_strat)
    cov, wid = ev.per_unit_coverage_width(replay.unit.to_numpy(), replay.true_rul.to_numpy(), lo, hi)
    m, ci = ev.unit_bootstrap_ci(cov)
    _, wci = ev.unit_bootstrap_ci(wid)
    out["replay_band_v2"] = {
        "variant": out["selected_variant"],
        "coverage_equal_unit": m, "coverage_ci": list(ci),
        "width_equal_unit": float(np.mean(list(wid.values()))), "width_ci": list(wci),
        "v1_coverage_equal_unit": v1["replay"]["band"]["coverage_mean"],
        "v1_width_equal_unit": v1["replay"]["band"]["width_mean"],
        "label": POST_HOC.format(split="replay-holdout")}

    # --- official test endpoints: post-hoc band measurement from stored predictions ---
    ep = joblib.load(os.path.join(RESULTS, f"test_endpoint_{ds}.pkl"))
        # ep has one row per unit with q10/q50/q90/true_rul
    lo, hi = cal.apply_calibration(ep.q10, ep.q50, ep.q90, calib, use_strat)
    ecov = (ep.true_rul.to_numpy() >= lo) & (ep.true_rul.to_numpy() <= hi)
    out["test_endpoint_band_v2"] = {
        "coverage": float(np.mean(ecov)),
        "width_mean": float(np.mean(hi - lo)),
        "v1_coverage": v1["test"]["endpoint_coverage"],
        "v1_width_mean": v1["test"]["endpoint_width_mean"],
        "label": POST_HOC.format(split="official test endpoints")}

    # --- warning-policy candidates on tune (selection), replay post-hoc ---
    cands = {}
    lo_tune, _ = cal.apply_calibration(tune.q10, tune.q50, tune.q90, calib, use_strat)
    lo_rep, _ = cal.apply_calibration(replay.q10, replay.q50, replay.q90, calib, use_strat)
    for kind in ("v1_q50", "q90", "calibrated_lower", "persistence_3"):
        tdf = tune[["unit", "cycle", "true_rul", "q10", "q50", "q90"]].copy()
        lt = lo_tune if kind == "calibrated_lower" else None
        cands[kind] = {"tune": warning_eval_with_policy(tdf, kind, lt)}
    v1_tune_catch = cands["v1_q50"]["tune"]["catch_fraction"]
    v1_tune_prem = cands["v1_q50"]["tune"]["premature_unit_fraction"]
    eligible = [k for k in cands if k != "v1_q50"
                and cands[k]["tune"]["catch_fraction"] is not None
                and v1_tune_catch is not None
                and cands[k]["tune"]["catch_fraction"] >= v1_tune_catch]
    if eligible:
        best = min(eligible, key=lambda k: (cands[k]["tune"]["premature_unit_fraction"],
                                            -(cands[k]["tune"]["lead_time_cycles"]["median"] or 0)))
        selected, note = best, "selected on tune by predeclared rule"
    else:
        selected, note = "v1_q50", "no candidate matched v1 tune catch fraction; v1 policy kept"
    out["warning_policy_selection"] = {"registry": "P0 v1_q50, P1 q90 (raw, uncalibrated), P2 calibrated_lower, P3 persistence_3; "
                                       "registry timeline in evaluate_v2.py docstring; P1 omitted from the note sent to parent - "
                                       "disclosed expansion; P3 added in run 2 after run-1 replay numbers were seen",
                                       "candidates_tune": {k: v["tune"] for k, v in cands.items()},
                                       "v1_tune_catch": v1_tune_catch, "v1_tune_premature": v1_tune_prem,
                                       "selected": selected, "note": note,
                                       "rule": "tune catch >= v1 tune catch, then min tune premature, tie-break max median lead"}
    rdf = replay[["unit", "cycle", "true_rul", "q10", "q50", "q90"]].copy()
    lr = lo_rep if selected == "calibrated_lower" else None
    out["replay_warning_v2"] = warning_eval_with_policy(rdf, selected, lr)
    out["replay_warning_v2"]["policy"] = selected
    out["replay_warning_v2"]["v1_warning"] = v1["replay"]["warning"]
    out["replay_warning_v2"]["label"] = POST_HOC.format(split="replay-holdout")

    with open(os.path.join(RESULTS, f"results_v2_{ds}.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def main():
    t0 = time.time()
    for ds in config.SUBSETS:
        train, _ = data.load_subset(ds, "train")
        train = data.true_rul_train(train)
        feat = features.build_features(train)
        feat = feat.merge(train[["unit", "cycle", "true_rul"]], on=["unit", "cycle"])
        sp = json.load(open(os.path.join(RESULTS, "splits.json")))["subsets"][ds]
        tune_feat = feat[feat.unit.isin(sp["tune"])].reset_index(drop=True)
        out = run_subset(ds, tune_feat)
        b = out["replay_band_v2"]
        w = out["replay_warning_v2"]
        print(f"{ds}: band v1 {b['v1_coverage_equal_unit']:.2f} -> v2 {b['coverage_equal_unit']:.2f} "
              f"(width {b['v1_width_equal_unit']:.0f} -> {b['width_equal_unit']:.0f}, {out['selected_variant']}); "
              f"warning policy {w['policy']}: catch {w['catch_fraction']:.2f} premature {w['premature_unit_fraction']:.2f} "
              f"(v1 {w['v1_warning']['catch_fraction']:.2f}/{w['v1_warning']['premature_unit_fraction']:.2f}) "
              f"[both POST-HOC]")
    print(f"done in {time.time() - t0:.0f}s - all replay/test numbers are post-hoc")


if __name__ == "__main__":
    main()
