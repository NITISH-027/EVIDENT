"""EVIDENT (working name) - advisory replay platform.
All predictions shown are the precomputed outputs of the frozen pipeline run
(results/*.pkl) - the same numbers the evaluation contract produced.
"""
import json, os, io, glob, hashlib, datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, abort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from src import config as _cfg
from src import retrospective as _retro
from src import data as _data
from src import features as _feat
RESULTS = os.path.join(BASE, "results")
DATA = os.path.join(BASE, "data_raw")
AUDIT = os.path.join(RESULTS, "audit.jsonl")

app = Flask(__name__)
SUBSET_WHITELIST = {"FD001", "FD002", "FD003", "FD004"}


@app.before_request
def _guard_subset():
    for v in (request.view_args or {}).values():
        if isinstance(v, str) and v.startswith("FD") and v not in SUBSET_WHITELIST:
            return "unknown subset", 404
HORIZON_N = 30

COLUMNS = (["unit", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)])


_MODELS = {}
def load_model(subset):
    if subset not in _MODELS:
        with open(os.path.join(RESULTS, "model_" + subset + ".pkl"), "rb") as f:
            _MODELS[subset] = joblib.load(f)
    return _MODELS[subset]


def load_results(subset):
    with open(os.path.join(RESULTS, f"results_{subset}.json")) as f:
        return json.load(f)


def load_replay(subset):
    return joblib.load(os.path.join(RESULTS, f"replay_{subset}.pkl"))


_POLICY_CACHE = {}


POLICY_CONDITIONS = {
    "v1_q50": "predicted median remaining life <= {N} cycles",
    "q90": "predicted 90th-percentile remaining life <= {N} cycles (upper band end within horizon)",
    "calibrated_lower": "calibrated lower band end <= {N} cycles",
    "persistence_3": "predicted median remaining life <= {N} cycles for 3 consecutive cycles (first alert at the 3rd)",
}


def policy_condition(policy):
    return POLICY_CONDITIONS[policy].format(N=HORIZON_N)


def selected_policy(subset, version):
    """(policy_name, human label, exploratory). v1 is the default and the
    stable shipped policy; v2 is the tune-selected exploratory policy."""
    if version == "v2":
        v2 = load_v2(subset)
        if v2 is None:
            return "UNAVAILABLE", f"V2 RESULTS UNAVAILABLE for {subset} - results_v2_{subset}.json missing; displaying v1 stable policy as an EXPLICIT fallback, not v2", False
        p = v2["replay_warning_v2"]["policy"]
        return p, f"v2 exploratory ({p}): flag = {policy_condition(p)} - post-hoc tune-selected, not a validated improvement", True
    return "v1_q50", f"v1 policy (q50 <= horizon): flag = {policy_condition('v1_q50')} - stable default", False


def current_version():
    return "v2" if request.args.get("version") == "v2" else "v1"


def replay_with_flags(subset, version):
    """Replay df with a per-cycle 'flag' column under the selected policy.
    persistence_3 confirms on the 3rd consecutive q50<=horizon cycle of the
    same unit (past-only: each cycle uses only itself and earlier cycles)."""
    key = (subset, version)
    if key not in _POLICY_CACHE:
        df = load_replay(subset).sort_values(["unit", "cycle"]).copy()
        policy, _, _ = selected_policy(subset, version)
        if policy == "UNAVAILABLE":
            policy = "v1_q50"  # explicit fallback; the UNAVAILABLE label is shown by callers
        df["flag"] = _retro.policy_flags(df, HORIZON_N, policy)
        _POLICY_CACHE[key] = df
    return _POLICY_CACHE[key]


def raw_unit(subset, unit):
    df = pd.read_csv(os.path.join(DATA, f"train_{subset}.txt"), sep=r"\s+",
                     header=None, names=COLUMNS)
    return df[df.unit == unit].sort_values("cycle")


def subsets():
    return sorted(p.split("results_")[1].split(".json")[0]
                  for p in glob.glob(os.path.join(RESULTS, "results_FD*.json")))


def matched_baseline(vals, ops):
    """Baseline at cycle i uses ONLY cycles before i: mean of past readings whose
    operating-setting vector is within a tolerance of cycle i's settings.
    Settings are normalized per point by the mean/std of past rows only, and the
    tolerance is the 10th percentile of past rows' distances to the past
    operating centroid - no future row influences any baseline point, so the
    trace at cycle i is identical no matter how much later history is shown.
    Falls back to the plain past mean when the unit shows no operating-regime
    variation (single-regime subsets) or fewer than 3 past cycles match.
    Returns (baseline, n_matched) so the chart can state the matched count."""
    L = len(vals)
    out = np.empty(L)
    nmatch = np.zeros(L, int)
    for i in range(L):
        if i == 0:
            out[i] = vals[0]; continue
        past = ops[:i]
        mu = past.mean(axis=0)
        sd = past.std(axis=0); sd[sd == 0] = 1.0
        pn = (past - mu) / sd
        ci = (ops[i] - mu) / sd
        regime_var = bool((past.std(axis=0) > 1e-6).any())
        if regime_var and i >= 4:
            cen = pn.mean(axis=0)
            tol = np.quantile(np.sqrt(((pn - cen) ** 2).sum(axis=1)), 0.1)
            d = np.sqrt(((pn - ci) ** 2).sum(axis=1))
            m = d <= max(tol, 1e-12)
            if m.sum() >= 3:
                out[i] = vals[:i][m].mean(); nmatch[i] = int(m.sum()); continue
        out[i] = vals[:i].mean(); nmatch[i] = i
    return out, nmatch


@app.route("/")
def fleet():
    subset = request.args.get("subset", "FD001")
    if subset not in SUBSET_WHITELIST:
        abort(404)
    cycle = int(request.args.get("cycle", 50))
    version = current_version()
    policy, policy_label, exploratory = selected_policy(subset, version)
    df = replay_with_flags(subset, version)
    cur = df[df.cycle <= cycle].sort_values("cycle").groupby("unit").tail(1)
    rows = [{
        "unit": int(r.unit), "cycle": int(r.cycle), "q50": round(r.q50, 1),
        "band": f"[{r.q10:.0f}, {r.q90:.0f}]",
        "flag": bool(r.flag),
        "envelope": bool(r.envelope_flag),
        "limited": bool(r.cycle < 30),
    } for r in cur.itertuples()]
    rows.sort(key=lambda r: r["q50"])
    cov = round(100 * load_results(subset)["replay"]["band"]["coverage_mean"])
    # demo unit: a replay unit whose first flag is a catch
    demo = None
    for u, g in df.groupby("unit"):
        f = g[g.flag]
        if len(f) and 0 < f.iloc[0].true_rul <= HORIZON_N:
            demo = {"unit": int(u), "first_flag_cycle": int(f.iloc[0].cycle),
                    "failure_cycle": int(g.cycle.max())}
            break
    return render_template("fleet.html", subset=subset, subsets=subsets(),
                           cycle=cycle, rows=rows, demo=demo, horizon=HORIZON_N, cov=cov,
                           version=version, policy_label=policy_label, exploratory=exploratory)


def _fig_to_response(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/chart/rul/<subset>/<int:unit>/<int:cycle>.png")
def chart_rul(subset, unit, cycle):
    version = current_version()
    policy, policy_label, _ = selected_policy(subset, version)
    df = replay_with_flags(subset, version)
    if len(df[df.unit == unit]) == 0:
        abort(404)
    g = df[(df.unit == unit) & (df.cycle <= cycle)].sort_values("cycle")
    show_truth = request.args.get("show_truth") == "1"
    fig, ax = plt.subplots(figsize=(7, 3.2))
    if request.args.get("band") == "v2" and load_calibration(subset) and load_v2(subset):
        from src import calibration as _cal
        calib = load_calibration(subset)
        v2 = load_v2(subset)
        lo, hi = _cal.apply_calibration(g.q10.to_numpy(), g.q50.to_numpy(), g.q90.to_numpy(),
                                        calib, v2["selected_variant"] == "stratified")
        ax.fill_between(g.cycle, lo, hi, alpha=0.25, color="C2",
                        label="calibrated band (v2): empirical tune calibration, no guarantee; "
                              "post-hoc replay coverage %.0f%%, width %.0f"
                              % (100 * v2["replay_band_v2"]["coverage_equal_unit"],
                                 v2["replay_band_v2"]["width_equal_unit"]))
        ax.fill_between(g.cycle, g.q10, g.q90, alpha=0.15, color="C0",
                        label="v1 band: post-hoc replay coverage %.0f%%, width %.0f"
                              % (100 * v2["replay_band_v2"]["v1_coverage_equal_unit"],
                                 v2["replay_band_v2"]["v1_width_equal_unit"]))
    else:
        ax.fill_between(g.cycle, g.q10, g.q90, alpha=0.25, label="band: nominal 80%, measured coverage below nominal - see scoreboard")
    ax.plot(g.cycle, g.q50, label="predicted RUL (median)", color="C0")
    if show_truth:
        ax.plot(g.cycle, g.true_rul, label="RETROSPECTIVE ground truth (evaluation overlay - hidden during engineer review)", color="C3", ls="--")
    ax.axhline(HORIZON_N, color="gray", ls=":", lw=1)
    ax.text(g.cycle.min(), HORIZON_N + 2, f"warning threshold {HORIZON_N}", fontsize=7, color="gray")
    fl = g[g.flag]
    if len(fl):
        ax.axvline(fl.iloc[0].cycle, color="orange", lw=1)
        fc = fl.iloc[0].cycle
        xmin, xmax = g.cycle.min(), g.cycle.max()
        right_side = fc > xmin + 0.6 * (xmax - xmin)
        ax.text(fc - 1 if right_side else fc + 1, ax.get_ylim()[1] * 0.95,
                "first flag", fontsize=7, color="orange",
                ha="right" if right_side else "left", va="top")
    ax.set_xlabel("cycle"); ax.set_ylabel("remaining cycles")
    ax.legend(fontsize=7)
    # compact title: full trigger condition lives in the HTML policy banner;
    # a long title stretches the canvas and shrinks the actual plot
    ax.set_title(f"{subset} unit {unit} - replay to cycle {cycle} - flags: {policy}", fontsize=9)
    return _fig_to_response(fig)


SINGLE_REGIME_SUBSETS = {"FD001", "FD003"}  # one operating condition by dataset design


def _sensors_fig(df, raw, cycle, title, subset):
    g = df[(df.cycle <= cycle)].sort_values("cycle")
    slope_cols = [c for c in df.columns if c.endswith("_slope") and c.startswith("s")]
    last = g.iloc[-1]
    # context-signal selection: channels whose current window slope deviates most
    # from this unit's own median slope over the shown history. Model association, not cause.
    dev = {c: abs(float(last[c]) - float(g[c].median())) for c in slope_cols}
    top = sorted(dev, key=dev.get, reverse=True)[:4]
    raw = raw[raw.cycle <= cycle]
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.2))
    ops = raw[["op1", "op2", "op3"]].to_numpy(float)
    single = subset in SINGLE_REGIME_SUBSETS
    fb_total = 0
    for ax, col in zip(axes.ravel(), top):
        ch = col.replace("_slope", "")
        vals = raw[ch].to_numpy(float)
        if single:
            c = np.cumsum(vals)
            base = np.concatenate([[vals[0]], c[:-1] / np.arange(1, len(vals))]) if len(vals) > 1 else vals.copy()
            ax.plot(raw.cycle, vals, lw=0.7, color="C0")
            ax.plot(raw.cycle, base, lw=1.2, color="C3")
        else:
            base, nm = matched_baseline(vals, ops)
            past = np.arange(1, len(vals) + 1)
            fb = nm == 0
            fb[1:] = nm[1:] == past[1:]  # plain-past-mean fallback points
            fb_total += int(fb.sum())
            ax.axhline(0, color="gray", lw=0.6, ls=":")
            ax.plot(raw.cycle, vals - base, lw=0.8, color="C0")
            if fb.any():
                ax.plot(raw.cycle[fb], (vals - base)[fb], ".", ms=2.5, color="C3")
        ax.set_title(ch + " (selected trend signal)", fontsize=8)
        ax.tick_params(labelsize=7)
    if single:
        labels = ["reading", "own past mean (strictly earlier cycles)"]
        handles = [plt.Line2D([], [], color="C0", lw=0.7), plt.Line2D([], [], color="C3", lw=1.2)]
        method = ("Single operating regime for this subset: baseline is the plain mean of strictly "
                  "earlier cycles of this same unit.")
    else:
        labels = ["reading minus matched past baseline", "cycles on plain-past-mean fallback"]
        handles = [plt.Line2D([], [], color="C0", lw=0.8), plt.Line2D([], [], color="C3", marker=".", ls="", ms=4)]
        method = ("Deviation view: reading minus this unit's own past baseline matched on operating "
                  "settings (strictly earlier cycles; tolerance = 10th pct of past distances to the past "
                  "centroid; plain past mean where <3 past cycles match). Level shifts between operating "
                  "regimes are expected context, NOT degradation evidence; only sustained deviation matters.")
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 0.985))
    fig.text(0.5, 0.005, title + ". " + method + " Model association, not proof of physical cause.",
             ha="center", fontsize=6.5, wrap=True)
    fig.suptitle("Trend signals vs the unit OWN past baseline", fontsize=9, y=1.0)
    fig.tight_layout(rect=[0, 0.045, 1, 0.93])
    return fig


@app.route("/chart/sensors/<subset>/<int:unit>/<int:cycle>.png")
def chart_sensors(subset, unit, cycle):
    df = load_replay(subset)
    if len(df[df.unit == unit]) == 0:
        abort(404)
    raw = raw_unit(subset, unit)
    return _fig_to_response(_sensors_fig(df[df.unit == unit], raw, cycle,
                                         "Context for the warning flag", subset))


@app.route("/unit/<subset>/<int:unit>")
def unit_view(subset, unit):
    version = current_version()
    policy, policy_label, exploratory = selected_policy(subset, version)
    df = replay_with_flags(subset, version)
    g = df[df.unit == unit].sort_values("cycle")
    if len(g) == 0:
        return ("This unit is not in the replay-holdout set for %s. "
                "Only untouched replay units are shown, so trajectory claims stay independent of development. "
                "<a href='/?subset=%s'>Back to fleet</a>" % (subset, subset)), 404
    if "cycle" in request.args:
        cycle = min(int(request.args["cycle"]), int(g.cycle.max()))
        defaulted = False
    else:
        fl0 = g[g.flag]
        cycle = int(fl0.iloc[0].cycle) if len(fl0) else int(g.cycle.max())
        defaulted = True
    cur = g[g.cycle <= cycle].iloc[-1]
    flagged = bool(cur.flag)
    if flagged:
        log_flag_once({"record_type": "flag", "subset": subset, "unit": unit,
                       "cycle": int(cur.cycle), "policy": policy,
                       "ts": datetime.datetime.now().isoformat(timespec="seconds")})
    state = {
        "subset": subset, "unit": unit, "cycle": cycle, "max_cycle": int(g.cycle.max()),
        "q50": round(float(cur.q50), 1), "q10": round(float(cur.q10), 1),
        "q90": round(float(cur.q90), 1), "true_rul": float(cur.true_rul),
        "q10_v2": None, "q90_v2": None,
        "flag": flagged, "flag_policy": policy_label, "policy_name": policy, "version": version,
        "exploratory": exploratory, "limited": bool(cur.cycle < 30),
        "envelope": bool(cur.envelope_flag),
        "prev": cycle - 1 if cycle > 1 else None,
        "next": cycle + 1 if cycle < g.cycle.max() else None,
        "show_truth": request.args.get("show_truth") == "1",
        "band_v2": request.args.get("band") == "v2",
        "play": request.args.get("play") == "1",
        "defaulted": defaulted,
    }
    cov = round(100 * load_results(subset)["replay"]["band"]["coverage_mean"])
    v2 = load_v2(subset)
    v2band = None
    if v2:
        if state["band_v2"]:
            from src import calibration as _cal
            calib = load_calibration(subset)
            lo, hi = _cal.apply_calibration([cur.q10], [cur.q50], [cur.q90], calib,
                                            v2["selected_variant"] == "stratified")
            state["q10_v2"], state["q90_v2"] = round(float(lo[0]), 1), round(float(hi[0]), 1)
        v2band = {"coverage": round(100 * v2["replay_band_v2"]["coverage_equal_unit"]),
                  "width": round(v2["replay_band_v2"]["width_equal_unit"]),
                  "v1_coverage": round(100 * v2["replay_band_v2"]["v1_coverage_equal_unit"]),
                  "v1_width": round(v2["replay_band_v2"]["v1_width_equal_unit"]),
                  "variant": v2["selected_variant"]}
    nav = ("&version=v2" if version == "v2" else "") + ("&band=v2" if state["band_v2"] else "")
    return render_template("unit.html", s=state, horizon=HORIZON_N, cov=cov, v2band=v2band,
                           version=version, nav=nav)


VERDICTS = {"reviewed_warrants_inspection", "reviewed_dismissed"}


@app.route("/verify", methods=["POST"])
def verify():
    import datetime
    try:
        entry = {"record_type": "human_verdict",
                 "subset": str(request.form["subset"]),
                 "unit": int(request.form["unit"]),
                 "cycle": int(request.form["cycle"]),
                 "verdict": str(request.form["verdict"]),
                 "ts": datetime.datetime.now().isoformat(timespec="seconds")}
    except (KeyError, ValueError):
        return jsonify({"error": "invalid fields: subset, unit (int), cycle (int), verdict required"}), 400
    if entry["verdict"] not in VERDICTS:
        return jsonify({"error": "verdict not allowed"}), 400
    version = str(request.form.get("version", "v1"))
    client_policy = str(request.form.get("policy", ""))
    if entry["subset"] in SUBSET_WHITELIST:
        # derive the policy server-side from version + subset; never trust the client
        policy, policy_label, _ = selected_policy(entry["subset"], version)
        if policy == "UNAVAILABLE":
            return jsonify({"error": f"v2 results unavailable for {entry['subset']}; resubmit under version=v1"}), 400
        if client_policy and client_policy != policy:
            return jsonify({"error": f"policy mismatch: view derived {policy}, form sent {client_policy}"}), 400
        g = replay_with_flags(entry["subset"], version)
        m = g[(g.unit == entry["unit"]) & (g.cycle == entry["cycle"])]
        if len(m) == 0:
            if len(g[g.unit == entry["unit"]]) == 0:
                return jsonify({"error": "unit not in replay set for subset"}), 400
            return jsonify({"error": "cycle does not exist for this unit"}), 400
        entry["policy"] = policy
        entry["flag_state_at_cycle"] = bool(m.iloc[0]["flag"])  # recorded even when quiet - review is allowed on unflagged cycles
    else:
        token = str(request.form.get("token", ""))
        if entry["subset"] != "import" or token not in IMPORTS or \
                len(IMPORTS[token]["feat"][IMPORTS[token]["feat"].unit == entry["unit"]]) == 0:
            return jsonify({"error": "subset or verdict not allowed"}), 400
        src = IMPORTS[token]
        policy, policy_label, _ = selected_policy(src["subset"], src.get("version", "v1"))
        if client_policy and client_policy != policy:
            return jsonify({"error": f"policy mismatch: import view derived {policy}, form sent {client_policy}"}), 400
        g = src["feat"]
        m = g[(g.unit == entry["unit"]) & (g.cycle == entry["cycle"])]
        if len(m) == 0:
            return jsonify({"error": "cycle does not exist for this imported unit"}), 400
        entry["policy"] = policy
        entry["flag_state_at_cycle"] = bool(m.iloc[0]["flag"]) if "flag" in m else bool(m.iloc[0].q50 <= HORIZON_N)
        entry["upload_sha256"] = src["sha256"]
        entry["model_subset"] = src["subset"]
        entry["token"] = token  # session routing only; sha256+subset+unit+cycle source the verdict after restart
    with open(AUDIT, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return jsonify({"logged": entry})


def _audit_entries():
    entries = []
    if os.path.exists(AUDIT):
        with open(AUDIT) as f:
            entries = [json.loads(l) for l in f if l.strip()]
    return entries


def log_flag_once(record):
    """Append a flag_shown record unless an identical one is already logged.
    Scope: flags are logged WHEN SHOWN to a reviewer (replay view) or produced
    at import. Replay back-history transitions never viewed are not individually
    logged; the frozen first-alert outcomes for ALL replay units are the
    scoreboard's warning metrics."""
    record["record_type"] = "flag_shown"
    record.setdefault("policy", "v1_q50")
    key = (record.get("subset"), record.get("unit"), record.get("cycle"),
           record.get("policy"), record.get("upload_sha256"))
    for e in _audit_entries():
        if e.get("record_type") == "flag_shown" and \
                (e.get("subset"), e.get("unit"), e.get("cycle"),
                 e.get("policy", "v1_q50"), e.get("upload_sha256")) == key:
            return
    with open(AUDIT, "a") as f:
        f.write(json.dumps(record) + "\n")


def ensure_frozen_reference_logged():
    """Append one frozen_first_alert_reference record per replay unit, deduped on
    (subset, unit, source artifact hash). These are RECONSTRUCTED references from
    the frozen prediction artifacts - not flags raised, not reviewer actions, and
    they carry no simulated historical timestamp."""
    existing = {(e.get("subset"), e.get("unit"), e.get("source_sha256"))
                for e in _audit_entries() if e.get("record_type") == "frozen_first_alert_reference"}
    now = datetime.datetime.now().isoformat(timespec="seconds")
    with open(AUDIT, "a") as f:
        for ds in subsets():
            src = os.path.join(RESULTS, f"replay_{ds}.pkl")
            sha = hashlib.sha256(open(src, "rb").read()).hexdigest()[:16]
            df = joblib.load(src)
            for u, g in df.groupby("unit"):
                if (ds, int(u), sha) in existing:
                    continue
                fa = _frozen_first_alert(ds, int(u), g.sort_values("cycle"))
                f.write(json.dumps({"record_type": "frozen_first_alert_reference",
                                    "subset": ds, "unit": int(u),
                                    "frozen_class": fa["class"],
                                    "first_flag_cycle": fa["first_flag_cycle"],
                                    "rul_at_first_flag": fa["rul_at_first_flag"],
                                    "source": f"results/replay_{ds}.pkl (frozen predictions)",
                                    "source_sha256": sha,
                                    "reconstructed_from_frozen_at": now}) + "\n")


def ensure_frozen_flag_cycle_reference_logged():
    """Append one frozen_flag_cycle_reference record per FLAGGED replay cycle
    (every cycle where the frozen model would have shown a flag), deduped on
    (subset, unit, cycle, source artifact hash). Reconstructed references from
    the frozen prediction artifacts - not raised alerts, no historical
    timestamp. Together with the per-unit first-alert reference this gives
    complete frozen flag-cycle reference coverage."""
    existing = {(e.get("subset"), e.get("unit"), e.get("cycle"), e.get("source_sha256"))
                for e in _audit_entries() if e.get("record_type") == "frozen_flag_cycle_reference"}
    now = datetime.datetime.now().isoformat(timespec="seconds")
    with open(AUDIT, "a") as f:
        for ds in subsets():
            src = os.path.join(RESULTS, f"replay_{ds}.pkl")
            sha = hashlib.sha256(open(src, "rb").read()).hexdigest()[:16]
            df = joblib.load(src)
            for r in df[df.q50 <= HORIZON_N].itertuples():
                key = (ds, int(r.unit), int(r.cycle), sha)
                if key in existing:
                    continue
                f.write(json.dumps({"record_type": "frozen_flag_cycle_reference",
                                    "subset": ds, "unit": int(r.unit), "cycle": int(r.cycle),
                                    "q50": round(float(r.q50), 1),
                                    "recorded_remaining_life": float(r.true_rul),
                                    "source": f"results/replay_{ds}.pkl (frozen predictions)",
                                    "source_sha256": sha,
                                    "reconstructed_from_frozen_at": now}) + "\n")
                existing.add(key)


def _frozen_first_alert(subset, unit, g=None, policy="v1_q50"):
    """Unit-level frozen first-alert outcome from the frozen replay predictions
    under the given policy (first flag; catch 0<RUL<=N, premature RUL>N,
    miss otherwise/never). v1_q50 reproduces the scoreboard's declared policy;
    other policies are exploratory post-hoc references and are labeled so."""
    if g is None:
        g = replay_with_flags(subset, "v2" if policy != "v1_q50" else "v1")
        g = g[g.unit == unit].sort_values("cycle")
        fl = g[g.flag]
    else:
        fl = g[g.q50 <= HORIZON_N] if policy == "v1_q50" else g[g.flag] if "flag" in g else g[g.q50 <= HORIZON_N]
    if len(fl):
        tr0 = float(fl.iloc[0].true_rul)
        cls = "catch" if 0 < tr0 <= HORIZON_N else ("miss" if tr0 <= 0 else "premature")
        return {"class": cls, "first_flag_cycle": int(fl.iloc[0].cycle), "rul_at_first_flag": tr0}
    return {"class": "miss", "first_flag_cycle": None, "rul_at_first_flag": None}


@app.route("/audit")
def audit():
    entries = _audit_entries()
    retro = []
    for e in entries:
        if e.get("record_type") != "human_verdict":
            continue
        row = dict(e)
        if e.get("subset") in SUBSET_WHITELIST:
            epol = e.get("policy", "v1_q50")
            g = replay_with_flags(e["subset"], "v2" if epol != "v1_q50" else "v1")
            g = g[g.unit == e["unit"]].sort_values("cycle")
            m = g[g.cycle == e["cycle"]]
            if len(m):
                pol_note = "" if epol == "v1_q50" else f" (under exploratory policy {epol})"
                row["cycle_local"] = ("recorded remaining life %.0f cycles; flag shown%s - this cycle only, not a policy result" % (float(m.iloc[0].true_rul), pol_note)) \
                    if bool(m.iloc[0]["flag"]) else \
                    ("recorded remaining life %.0f cycles; no flag shown%s - this cycle only, not a policy result" % (float(m.iloc[0].true_rul), pol_note))
                fa = _frozen_first_alert(e["subset"], e["unit"], g, policy=epol)
                pol_tag = "" if e.get("policy", "v1_q50") == "v1_q50" else " [exploratory policy %s - NOT the v1 reference]" % e["policy"]
                row["outcome"] = ("%s - first flag at cycle %d with recorded remaining life %.0f%s"
                                  % (fa["class"], fa["first_flag_cycle"], fa["rul_at_first_flag"], pol_tag)) if fa["first_flag_cycle"] is not None \
                    else "miss - never flagged in the frozen replay" + pol_tag
            else:
                row["cycle_local"] = "unit/cycle not in replay data"
                row["outcome"] = "unknown"
        else:
            row["cycle_local"] = "no cycle-local truth available"
            row["outcome"] = "no outcome available - imported unit, no ground truth"
        retro.append(row)
    # OPTIONAL reviewer-opinion agreement (NOT outcome scoring): agreement
    # between the reviewer's verdict and the model's flag state AT THE REVIEWED
    # CYCLE on simulator replay units. This is reviewer opinion vs recorded flag
    # state at one cycle - NOT first-alert policy accuracy, NOT machine-health
    # validation, and NOT the outcome-driven scoring required by the plan (that
    # lives in the decision-vs-recorded-truth section above). Imported units
    # have no ground truth and are excluded.
    scored = []
    for row in retro:
        if row.get("subset") not in SUBSET_WHITELIST:
            continue
        epol = row.get("policy", "v1_q50")
        g = replay_with_flags(row["subset"], "v2" if epol != "v1_q50" else "v1")
        m = g[(g.unit == row["unit"]) & (g.cycle == row["cycle"])]
        if not len(m):
            continue
        flag_shown = bool(m.iloc[0]["flag"])
        agrees = (row["verdict"] == "reviewed_warrants_inspection") == flag_shown
        scored.append({"ts": row["ts"], "subset": row["subset"], "unit": row["unit"],
                       "cycle": row["cycle"], "verdict": row["verdict"],
                       "flag_shown_at_cycle": flag_shown, "agrees": agrees})
    n_scored = len(scored)
    verdict_agreement = {
        "estimand": ("verdict agreement with the model's flag state at the reviewed cycle, "
                     "simulator replay units only - reviewer opinion vs recorded flag state "
                     "at one cycle; NOT first-alert policy accuracy, NOT machine-health "
                     "validation; imported units excluded (no ground truth)"),
        "n_verdicts_scored": n_scored,
        "n_agree": sum(1 for r in scored if r["agrees"]),
        "agreement_rate": (sum(1 for r in scored if r["agrees"]) / n_scored) if n_scored else None,
        "rows": scored,
    }
    n_warrant = sum(1 for e in entries if e.get("verdict") == "reviewed_warrants_inspection")
    n_dismiss = sum(1 for e in entries if e.get("verdict") == "reviewed_dismissed")
    behavior = {"warrants": n_warrant, "dismissed": n_dismiss, "total": n_warrant + n_dismiss}
    ensure_frozen_reference_logged()
    ensure_frozen_flag_cycle_reference_logged()
    # Retrospective model/decision evaluation vs recorded simulator truth.
    # Frozen, verdict-independent: computed from the replay prediction
    # artifacts and persisted as results/retrospective_decision_FD*.json.
    truth_by_subset = {}
    for ds in subsets():
        src = os.path.join(RESULTS, f"replay_{ds}.pkl")
        sha = hashlib.sha256(open(src, "rb").read()).hexdigest()[:16]
        out_p = os.path.join(RESULTS, f"retrospective_decision_{ds}.json")
        cached = json.load(open(out_p)) if os.path.exists(out_p) else None
        v2p = None
        v2 = load_v2(ds)
        if v2:
            v2p = v2["replay_warning_v2"]["policy"]
        if cached is None or cached.get("source_sha256") != sha or cached.get("v2_policy") != v2p:
            rdf = joblib.load(src)
            cached = _retro.decision_vs_truth(rdf, HORIZON_N, policy="v1_q50")
            cached["v1_policy"] = "v1_q50"
            cached["v2_policy"] = v2p
            if v2p and v2p != "v1_q50":
                cached["v2"] = _retro.decision_vs_truth(rdf, HORIZON_N, policy=v2p)
                cached["v2"]["post_hoc"] = ("v2 policy selected after v1 results were known; "
                                            "exploratory" + ("; candidate added to the set after "
                                            "run-1 replay numbers were seen" if ds == "FD004" else ""))
            else:
                cached["v2"] = None
            cached["source_sha256"] = sha
            cached["source"] = f"results/replay_{ds}.pkl (frozen predictions)"
            with open(out_p, "w") as f:
                json.dump(cached, f, indent=1)
        truth_by_subset[ds] = cached
    entries = _audit_entries()
    ref_types = {"frozen_first_alert_reference", "frozen_flag_cycle_reference"}
    events = []
    for e in entries:
        if e.get("record_type") in ref_types:
            continue
        row = dict(e)
        if e.get("record_type") in ("import_upload",) or e.get("upload_sha256") or e.get("subset") == "import":
            row["channel"] = "operational import (no ground truth)"
        else:
            row["channel"] = "simulator replay (frozen reference available)"
        events.append(row)
    frozen_cov = [e for e in entries if e.get("record_type") == "frozen_first_alert_reference"]
    frozen_cycles = [e for e in entries if e.get("record_type") == "frozen_flag_cycle_reference"]
    return render_template("audit.html", entries=events, retro=retro, behavior=behavior,
                           frozen_cov=frozen_cov, frozen_cycles=frozen_cycles,
                           verdict_agreement=verdict_agreement,
                           truth_by_subset=truth_by_subset, horizon=HORIZON_N)


def _quality_rows(subset, raw):
    res = load_results(subset)
    bundle = load_model(subset)
    pre = bundle["preprocessor"]
    const = set(res["constant_channels_excluded"])
    rows = []
    env_min = dict(zip(bundle["feature_cols"], pre.env_min_))
    env_max = dict(zip(bundle["feature_cols"], pre.env_max_))
    for ch in _data.CHANNELS:
        col_min = env_min.get(ch + "_mean")
        col_max = env_max.get(ch + "_mean")
        v = raw[ch]
        state = "excluded - constant on fit split (uninformative, not a fault)" if ch in const else "in use"
        rows.append({"ch": ch, "state": state, "n": int(v.size),
                     "min": round(float(v.min()), 3), "max": round(float(v.max()), 3),
                     "env": ("[" + format(col_min, ".2f") + ", " + format(col_max, ".2f") + "]") if col_min is not None else "-",
                     "outside": bool(col_min is not None and (v.mean() < col_min or v.mean() > col_max))})
    return rows


@app.route("/quality/<subset>/<int:unit>")
def quality(subset, unit):
    return render_template("quality.html", subset=subset, unit=unit,
                           rows=_quality_rows(subset, raw_unit(subset, unit)),
                           schema="schema-validated at import (monotone cycles, no nulls); gaps rejected with reasons, valid units processed")


@app.route("/import", methods=["GET", "POST"])
def import_view():
    import io as _io
    ctx = {"subsets": subsets(), "horizon": HORIZON_N}
    if request.method == "POST":
        subset = request.form.get("subset", "FD001")
        if subset not in SUBSET_WHITELIST:
            return jsonify({"error": "unknown subset"}), 400
        f = request.files.get("file")
        if not f or not f.filename:
            ctx["error"] = "no file provided"
            return render_template("import.html", **ctx)
        raw_bytes = f.read()
        upload_sha = hashlib.sha256(raw_bytes).hexdigest()
        raw = raw_bytes.decode("utf-8", "replace")
        if not raw.strip():
            ctx["error"] = "empty file"
            return render_template("import.html", **ctx)
        sep = r"\s+" if "," not in raw.splitlines()[0] else ","
        try:
            df = pd.read_csv(_io.StringIO(raw), sep=sep, header=None, engine="python")
            if df.shape[1] != len(_data.COLUMNS):
                df = pd.read_csv(_io.StringIO(raw), sep=sep, header=0, engine="python")
            if df.shape[1] != len(_data.COLUMNS):
                ctx["error"] = "expected %d columns (unit, cycle, 3 settings, 21 sensors), got %d" % (len(_data.COLUMNS), df.shape[1])
                return render_template("import.html", **ctx)
            df.columns = _data.COLUMNS
            df = df.apply(pd.to_numeric, errors="coerce")
        except Exception as e:
            ctx["error"] = "could not parse file: %s" % e
            return render_template("import.html", **ctx)
        clean, report = _data.validate_schema(df)
        ctx["report"] = report
        if len(clean) == 0:
            ctx["error"] = "no units passed schema validation"
            return render_template("import.html", **ctx)
        bundle = load_model(subset)
        feat = _feat.build_features(clean)
        pre, mdl = bundle["preprocessor"], bundle["model"]
        q10, q50, q90, diag = mdl.predict(pre.transform(feat))
        feat = feat.assign(q10=q10, q50=q50, q90=q90,
                           envelope_flag=pre.envelope_flags(feat))
        version = current_version()
        ipolicy, ipolicy_label, iexploratory = selected_policy(subset, version)
        feat["flag"] = _retro.policy_flags(feat, HORIZON_N, ipolicy)  # past-only: each cycle uses itself + earlier cycles
        latest = feat.sort_values("cycle").groupby("unit").tail(1)
        ctx["version"], ctx["policy_label"], ctx["exploratory"] = version, ipolicy_label, iexploratory
        ctx["subset"] = subset
        import uuid
        token = uuid.uuid4().hex[:12]
        for r in latest.itertuples():
            if r.flag:
                log_flag_once({"record_type": "flag", "subset": "import",
                               "upload_sha256": upload_sha, "model_subset": subset,
                               "unit": int(r.unit), "cycle": int(r.cycle), "policy": ipolicy,
                               "ts": datetime.datetime.now().isoformat(timespec="seconds")})
        IMPORTS[token] = {"subset": subset, "feat": feat, "raw": clean, "sha256": upload_sha,
                          "version": version}
        ctx["token"] = token
        with open(AUDIT, "a") as af:
            af.write(json.dumps({"record_type": "import_upload", "upload_sha256": upload_sha,
                                 "subset": subset, "policy": ipolicy, "version": version,
                                 "units": [int(u) for u in latest["unit"]],
                                 "cycles": {str(int(r.unit)): int(r.cycle) for r in latest.itertuples()},
                                 "ts": datetime.datetime.now().isoformat(timespec="seconds")}) + "\n")
        ctx["rows"] = [{"unit": int(r.unit), "n_cycles": int(feat[feat.unit == r.unit].shape[0]),
                        "cycle": int(r.cycle), "q50": round(float(r.q50), 1),
                        "q10": round(float(r.q10), 1), "q90": round(float(r.q90), 1),
                        "flag": bool(r.flag),
                        "persist_unconfirmable": bool(ipolicy == "persistence_3" and feat[feat.unit == r.unit].shape[0] < 3),
                        "limited": bool(feat[feat.unit == r.unit].shape[0] < _cfg.WINDOW),
                        "envelope": bool(r.envelope_flag)} for r in latest.itertuples()]
    return render_template("import.html", **ctx)


IMPORTS = {}


def _import_ctx(token, unit):
    e = IMPORTS.get(token)
    if e is None or len(e["feat"][e["feat"].unit == unit]) == 0:
        abort(404)
    return e


@app.route("/import/<token>/<int:unit>")
def import_unit(token, unit):
    e = _import_ctx(token, unit)
    feat, raw = e["feat"], e["raw"]
    g = feat[feat.unit == unit].sort_values("cycle")
    cur = g.iloc[-1]
    res = load_results(e["subset"])
    cov = round(100 * res["replay"]["band"]["coverage_mean"])
    s = {"subset": e["subset"], "unit": unit, "token": token, "cycle": int(cur.cycle),
         "q50": round(float(cur.q50), 1), "q10": round(float(cur.q10), 1),
         "q90": round(float(cur.q90), 1),
         "flag": bool(cur.q50 <= HORIZON_N),
         "limited": bool(len(g) < _cfg.WINDOW),
         "envelope": bool(cur.envelope_flag)}
    return render_template("import_unit.html", s=s, horizon=HORIZON_N, cov=cov)


@app.route("/chart_import/<kind>/<token>/<int:unit>.png")
def chart_import(kind, token, unit):
    e = _import_ctx(token, unit)
    feat, raw = e["feat"], e["raw"]
    g = feat[feat.unit == unit].sort_values("cycle")
    cycle = int(g.cycle.max())
    if kind == "rul":
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.fill_between(g.cycle, g.q10, g.q90, alpha=0.25, label="band: nominal 80%, measured coverage below nominal - see scoreboard")
        ax.plot(g.cycle, g.q50, label="predicted RUL (median)", color="C0")
        ax.axhline(HORIZON_N, color="gray", ls=":", lw=1)
        ax.set_xlabel("cycle"); ax.set_ylabel("remaining cycles")
        ax.legend(fontsize=7); ax.set_title(f"imported unit {unit} ({e['subset']} model) - uploaded history, no ground truth exists")
        return _fig_to_response(fig)
    if kind == "sensors":
        return _fig_to_response(_sensors_fig(g, raw[raw.unit == unit], cycle,
                                             "Context for the warning flag", e["subset"]))
    abort(404)


@app.route("/quality_import/<token>/<int:unit>")
def quality_import(token, unit):
    e = _import_ctx(token, unit)
    return render_template("quality.html", subset=e["subset"], unit=unit,
                           rows=_quality_rows(e["subset"], e["raw"][e["raw"].unit == unit]),
                           schema="schema-validated at import (monotone cycles, no nulls); gaps rejected with reasons, valid units processed")


def load_v2(subset):
    p = os.path.join(RESULTS, f"results_v2_{subset}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def load_calibration(subset):
    p = os.path.join(RESULTS, f"calibration_{subset}.json")
    return json.load(open(p)) if os.path.exists(p) else None


@app.route("/scoreboard")
def scoreboard():
    return render_template("scoreboard.html", results=[load_results(s) for s in subsets()],
                           v2={ds: load_v2(ds) for ds in subsets()})


NOTES = os.path.join(RESULTS, "review_notes.jsonl")


def _sensors_data_dict(df_unit, raw, cycle, subset):
    g = df_unit[df_unit.cycle <= cycle].sort_values("cycle")
    slope_cols = [c for c in df_unit.columns if c.endswith("_slope") and c.startswith("s")]
    last = g.iloc[-1]
    dev = {c: abs(float(last[c]) - float(g[c].median())) for c in slope_cols}
    top = sorted(dev, key=dev.get, reverse=True)[:4]
    raw_sub = raw[raw.cycle <= cycle].sort_values("cycle")
    ops = raw_sub[["op1", "op2", "op3"]].to_numpy(float)
    single = subset in SINGLE_REGIME_SUBSETS
    channels = []
    for col in top:
        ch = col.replace("_slope", "")
        vals = raw_sub[ch].to_numpy(float)
        cycles = [int(c) for c in raw_sub.cycle]
        if single:
            c_sum = np.cumsum(vals)
            base = np.concatenate([[vals[0]], c_sum[:-1] / np.arange(1, len(vals))]) if len(vals) > 1 else vals.copy()
            channels.append({
                "channel": ch,
                "cycles": cycles,
                "readings": [round(float(v), 2) for v in vals],
                "baseline": [round(float(b), 2) for b in base],
                "diff": [round(float(v - b), 2) for v, b in zip(vals, base)],
                "fallback": [False] * len(vals)
            })
        else:
            base, nm = matched_baseline(vals, ops)
            past = np.arange(1, len(vals) + 1)
            fb = nm == 0
            fb[1:] = nm[1:] == past[1:]
            channels.append({
                "channel": ch,
                "cycles": cycles,
                "readings": [round(float(v), 2) for v in vals],
                "baseline": [round(float(b), 2) for b in base],
                "diff": [round(float(v - b), 2) for v, b in zip(vals, base)],
                "fallback": [bool(f) for f in fb]
            })
    return {
        "subset": subset,
        "unit": int(last.unit),
        "cycle": cycle,
        "single_regime": single,
        "channels": channels,
        "method_note": ("Single operating regime for this subset: baseline is the plain mean of strictly earlier cycles."
                        if single else
                        "Deviation view: reading minus this unit's own past baseline matched on operating settings (strictly earlier cycles)."),
        "caveat": "Sensor channels selected as trend and context signals for this flag. Any channel ranking is a model association with the prediction, never evidence of physical cause in the machine."
    }


@app.context_processor
def inject_global_context():
    endpoint = request.endpoint or ""
    sub = (request.view_args or {}).get("subset") or request.args.get("subset", "FD001")
    if not isinstance(sub, str) or sub not in SUBSET_WHITELIST:
        sub = "FD001"
    ver = current_version()
    pol_name, pol_lbl, is_exp = selected_policy(sub, ver)
    return {
        "all_subsets": subsets(),
        "active_subset": sub,
        "active_version": ver,
        "active_policy_name": pol_name,
        "active_policy_label": pol_lbl,
        "active_exploratory": is_exp,
        "active_endpoint": endpoint,
        "HORIZON_N": HORIZON_N,
    }


@app.route("/api/fleet")
def api_fleet():
    subset = request.args.get("subset", "FD001")
    if subset not in SUBSET_WHITELIST:
        abort(404)
    try:
        cycle = int(request.args.get("cycle", 50))
    except (ValueError, TypeError):
        cycle = 50
    version = current_version()
    policy, policy_label, exploratory = selected_policy(subset, version)
    df = replay_with_flags(subset, version)
    cur = df[df.cycle <= cycle].sort_values("cycle").groupby("unit").tail(1)
    
    entries = _audit_entries()
    verdicts = [e for e in entries if e.get("record_type") == "human_verdict" and e.get("subset") == subset]
    
    rows = []
    for r in cur.itertuples():
        u = int(r.unit)
        c = int(r.cycle)
        exact = [e for e in verdicts if e.get("unit") == u and e.get("cycle") == c and e.get("policy", "v1_q50") == policy]
        earlier = [e for e in verdicts if e.get("unit") == u and e.get("policy", "v1_q50") == policy and e.get("cycle") != c]
        other_pol = [e for e in verdicts if e.get("unit") == u and e.get("policy", "v1_q50") != policy]
        
        rev_info = None
        if exact:
            rev_info = {"status": "reviewed_at_this_cycle", "verdict": exact[-1]["verdict"], "ts": exact[-1]["ts"], "cycle": c, "policy": policy}
        elif earlier:
            rev_info = {"status": "reviewed_earlier", "verdict": earlier[-1]["verdict"], "ts": earlier[-1]["ts"], "cycle": earlier[-1]["cycle"], "policy": policy}
        elif other_pol:
            rev_info = {"status": "reviewed_other_policy", "verdict": other_pol[-1]["verdict"], "ts": other_pol[-1]["ts"], "cycle": other_pol[-1]["cycle"], "policy": other_pol[-1]["policy"]}
        
        rows.append({
            "unit": u,
            "cycle": c,
            "q50": round(r.q50, 1),
            "q10": round(r.q10, 1),
            "q90": round(r.q90, 1),
            "band": f"[{r.q10:.0f}, {r.q90:.0f}]",
            "flag": bool(r.flag),
            "envelope": bool(r.envelope_flag),
            "limited": bool(r.cycle < 30),
            "review": rev_info
        })
    rows.sort(key=lambda r: r["q50"])
    cov = round(100 * load_results(subset)["replay"]["band"]["coverage_mean"])
    demo = None
    for u, g in df.groupby("unit"):
        f = g[g.flag]
        if len(f) and 0 < f.iloc[0].true_rul <= HORIZON_N:
            demo = {"unit": int(u), "first_flag_cycle": int(f.iloc[0].cycle),
                    "failure_cycle": int(g.cycle.max())}
            break
    return jsonify({
        "subset": subset,
        "cycle": cycle,
        "version": version,
        "policy": policy,
        "policy_label": policy_label,
        "exploratory": exploratory,
        "horizon": HORIZON_N,
        "cov": cov,
        "demo": demo,
        "rows": rows
    })


@app.route("/api/unit/<subset>/<int:unit>")
def api_unit(subset, unit):
    if subset not in SUBSET_WHITELIST:
        abort(404)
    version = current_version()
    policy, policy_label, exploratory = selected_policy(subset, version)
    df = replay_with_flags(subset, version)
    g = df[df.unit == unit].sort_values("cycle")
    if len(g) == 0:
        return jsonify({"error": f"unit {unit} not in replay-holdout set for {subset}"}), 404
    max_cycle = int(g.cycle.max())
    if "cycle" in request.args:
        try:
            cycle = min(int(request.args["cycle"]), max_cycle)
        except (ValueError, TypeError):
            cycle = max_cycle
    else:
        fl0 = g[g.flag]
        cycle = int(fl0.iloc[0].cycle) if len(fl0) else max_cycle
        
    g_shown = g[g.cycle <= cycle].sort_values("cycle")
    cur = g_shown.iloc[-1]
    flagged = bool(cur.flag)
    
    fl_all = g[g.flag]
    first_flag_cycle = int(fl_all.iloc[0].cycle) if len(fl_all) else None
    
    band_v2 = request.args.get("band") == "v2"
    show_truth = request.args.get("show_truth") == "1"
    
    v2 = load_v2(subset)
    v2band = None
    calibrated_lo, calibrated_hi = None, None
    if v2:
        v2band = {
            "coverage": round(100 * v2["replay_band_v2"]["coverage_equal_unit"]),
            "width": round(v2["replay_band_v2"]["width_equal_unit"]),
            "v1_coverage": round(100 * v2["replay_band_v2"]["v1_coverage_equal_unit"]),
            "v1_width": round(v2["replay_band_v2"]["v1_width_equal_unit"]),
            "variant": v2["selected_variant"]
        }
        if band_v2 and load_calibration(subset):
            from src import calibration as _cal
            calib = load_calibration(subset)
            calibrated_lo, calibrated_hi = _cal.apply_calibration(
                g_shown.q10.to_numpy(), g_shown.q50.to_numpy(), g_shown.q90.to_numpy(),
                calib, v2["selected_variant"] == "stratified"
            )
            
    cov = round(100 * load_results(subset)["replay"]["band"]["coverage_mean"])
    
    series = []
    for i, r in enumerate(g_shown.itertuples()):
        item = {
            "cycle": int(r.cycle),
            "q50": round(float(r.q50), 1),
            "q10": round(float(r.q10), 1),
            "q90": round(float(r.q90), 1),
            "flag": bool(r.flag)
        }
        if calibrated_lo is not None and calibrated_hi is not None:
            item["q10_v2"] = round(float(calibrated_lo[i]), 1)
            item["q90_v2"] = round(float(calibrated_hi[i]), 1)
        if show_truth:
            item["true_rul"] = float(r.true_rul)
        series.append(item)
        
    cur_q10_v2, cur_q90_v2 = None, None
    if calibrated_lo is not None and len(calibrated_lo):
        cur_q10_v2 = round(float(calibrated_lo[-1]), 1)
        cur_q90_v2 = round(float(calibrated_hi[-1]), 1)
        
    entries = _audit_entries()
    unit_verdicts = [
        {"ts": e.get("ts"), "cycle": e.get("cycle"), "policy": e.get("policy"), "verdict": e.get("verdict")}
        for e in entries if e.get("record_type") == "human_verdict" and e.get("subset") == subset and e.get("unit") == unit
    ]
    
    resp = {
        "subset": subset,
        "unit": unit,
        "cycle": cycle,
        "max_cycle": max_cycle,
        "first_flag_cycle": first_flag_cycle,
        "version": version,
        "policy": policy,
        "policy_label": policy_label,
        "exploratory": exploratory,
        "horizon": HORIZON_N,
        "cov": cov,
        "v2band": v2band,
        "band_v2": band_v2,
        "current": {
            "cycle": cycle,
            "q50": round(float(cur.q50), 1),
            "q10": round(float(cur.q10), 1),
            "q90": round(float(cur.q90), 1),
            "q10_v2": cur_q10_v2,
            "q90_v2": cur_q90_v2,
            "flag": flagged,
            "limited": bool(cur.cycle < 30),
            "envelope": bool(cur.envelope_flag)
        },
        "series": series,
        "past_verdicts": unit_verdicts
    }
    if show_truth:
        resp["retrospective"] = True
    return jsonify(resp)


@app.route("/api/unit/<subset>/<int:unit>/sensors")
def api_sensors(subset, unit):
    if subset not in SUBSET_WHITELIST:
        abort(404)
    df = load_replay(subset)
    if len(df[df.unit == unit]) == 0:
        abort(404)
    g_unit = df[df.unit == unit]
    max_c = int(g_unit.cycle.max())
    try:
        cycle = min(int(request.args.get("cycle", max_c)), max_c)
    except (ValueError, TypeError):
        cycle = max_c
    raw = raw_unit(subset, unit)
    data = _sensors_data_dict(g_unit, raw, cycle, subset)
    return jsonify(data)


@app.route("/api/display-flag", methods=["POST"])
def api_display_flag():
    try:
        payload = request.get_json(force=True, silent=True) or request.form
        subset = str(payload["subset"])
        unit = int(payload["unit"])
        cycle = int(payload["cycle"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "invalid fields: subset, unit (int), cycle (int) required"}), 400
    if subset not in SUBSET_WHITELIST:
        return jsonify({"error": "unknown subset"}), 400
    version = str(payload.get("version", "v1"))
    policy, _, _ = selected_policy(subset, version)
    if policy == "UNAVAILABLE":
        policy = "v1_q50"
    df = replay_with_flags(subset, version)
    m = df[(df.unit == unit) & (df.cycle == cycle)]
    if len(m) == 0:
        if len(df[df.unit == unit]) == 0:
            return jsonify({"error": "unit not in replay set"}), 404
        return jsonify({"error": "cycle does not exist for this unit"}), 400
    is_flagged = bool(m.iloc[0]["flag"])
    if is_flagged:
        rec = {
            "record_type": "flag",
            "subset": subset,
            "unit": unit,
            "cycle": cycle,
            "policy": policy,
            "ts": datetime.datetime.now().isoformat(timespec="seconds")
        }
        log_flag_once(rec)
        return jsonify({"logged": True, "flag": True, "policy": policy, "record": rec})
    return jsonify({"logged": False, "flag": False, "policy": policy})


@app.route("/api/review-status")
def api_review_status():
    subset = request.args.get("subset")
    policy = request.args.get("policy")
    entries = _audit_entries()
    verdicts = [e for e in entries if e.get("record_type") == "human_verdict"]
    if subset:
        if subset in SUBSET_WHITELIST:
            verdicts = [e for e in verdicts if e.get("subset") == subset]
        elif subset == "import":
            verdicts = [e for e in verdicts if e.get("subset") == "import" or e.get("upload_sha256")]
    if policy:
        verdicts = [e for e in verdicts if e.get("policy", "v1_q50") == policy]
    return jsonify({"verdicts": verdicts})


@app.route("/api/audit-summary")
def api_audit_summary():
    entries = _audit_entries()
    verdicts = [e for e in entries if e.get("record_type") == "human_verdict"]
    flags = [e for e in entries if e.get("record_type") == "flag_shown"]
    imports = [e for e in entries if e.get("record_type") == "import_upload"]
    
    unverdicted_flags = []
    for f in flags:
        f_sub = f.get("subset")
        f_unit = f.get("unit")
        f_cycle = f.get("cycle")
        f_pol = f.get("policy", "v1_q50")
        f_sha = f.get("upload_sha256")
        
        has_verdict = False
        for v in verdicts:
            v_pol = v.get("policy", "v1_q50")
            if v.get("unit") == f_unit and v.get("cycle") == f_cycle and v_pol == f_pol:
                if f_sha:
                    if v.get("upload_sha256") == f_sha and v.get("model_subset") == f.get("model_subset"):
                        has_verdict = True
                        break
                else:
                    if v.get("subset") == f_sub and not v.get("upload_sha256"):
                        has_verdict = True
                        break
        if not has_verdict:
            unverdicted_flags.append(f)
            
    return jsonify({
        "as_of": datetime.datetime.now().isoformat(timespec="seconds"),
        "limitation": "Generated from the local audit log. Not a work order. Not a maintenance record. Shows logged events only, not current fleet state. Unseen replay transitions are not logged.",
        "counts": {
            "flag_shown": len(flags),
            "human_verdict": len(verdicts),
            "import_upload": len(imports),
            "displayed_flags_unverdicted": len(unverdicted_flags)
        },
        "displayed_flags_unverdicted": unverdicted_flags,
        "verdicts": verdicts,
        "imports": imports,
        "flags": flags
    })


@app.route("/handover")
def handover():
    subset = request.args.get("subset")
    policy = request.args.get("policy")
    entries = _audit_entries()
    verdicts = [e for e in entries if e.get("record_type") == "human_verdict"]
    flags = [e for e in entries if e.get("record_type") == "flag_shown"]
    imports = [e for e in entries if e.get("record_type") == "import_upload"]
    
    if subset and subset in SUBSET_WHITELIST:
        flags = [e for e in flags if e.get("subset") == subset or e.get("model_subset") == subset]
        verdicts = [e for e in verdicts if e.get("subset") == subset or e.get("model_subset") == subset]
        imports = [e for e in imports if e.get("subset") == subset]
    if policy:
        flags = [e for e in flags if e.get("policy", "v1_q50") == policy]
        verdicts = [e for e in verdicts if e.get("policy", "v1_q50") == policy]

    unverdicted_flags = []
    for f in flags:
        f_sub = f.get("subset")
        f_unit = f.get("unit")
        f_cycle = f.get("cycle")
        f_pol = f.get("policy", "v1_q50")
        f_sha = f.get("upload_sha256")
        has_verdict = False
        for v in verdicts:
            v_pol = v.get("policy", "v1_q50")
            if v.get("unit") == f_unit and v.get("cycle") == f_cycle and v_pol == f_pol:
                if f_sha:
                    if v.get("upload_sha256") == f_sha and v.get("model_subset") == f.get("model_subset"):
                        has_verdict = True
                        break
                else:
                    if v.get("subset") == f_sub and not v.get("upload_sha256"):
                        has_verdict = True
                        break
        if not has_verdict:
            unverdicted_flags.append(f)
            
    as_of = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_operational = len(flags) + len(verdicts) + len(imports)
    return render_template("handover.html",
                           as_of=as_of,
                           subset=subset,
                           policy=policy,
                           subsets=subsets(),
                           flags=flags,
                           verdicts=verdicts,
                           imports=imports,
                           unverdicted_flags=unverdicted_flags,
                           total_operational=total_operational,
                           total_entries=total_operational)


@app.route("/api/note", methods=["POST"])
def api_add_note():
    try:
        data = request.get_json(force=True, silent=True) or request.form
        subset = str(data["subset"])
        unit = int(data["unit"])
        cycle = int(data["cycle"])
        policy = str(data.get("policy", "v1_q50"))
        text = str(data["text"]).strip()[:500]
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "invalid fields: subset, unit (int), cycle (int), text required"}), 400
    if not text:
        return jsonify({"error": "note text cannot be empty"}), 400
    entry = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "subset": subset,
        "unit": unit,
        "cycle": cycle,
        "policy": policy,
        "text": text,
        "label": "engineer note (free text, not part of the audit evidence)"
    }
    with open(NOTES, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return jsonify({"logged": entry})


@app.route("/api/notes")
def api_get_notes():
    subset = request.args.get("subset")
    unit = request.args.get("unit")
    notes = []
    if os.path.exists(NOTES):
        with open(NOTES, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    try:
                        n = json.loads(l)
                        if subset and n.get("subset") != subset:
                            continue
                        if unit and str(n.get("unit")) != str(unit):
                            continue
                        notes.append(n)
                    except Exception:
                        pass
    return jsonify({"notes": notes})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7777, debug=False)
