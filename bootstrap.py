import joblib
"""First-launch bootstrap: fetch C-MAPSS from NASA (no redistribution) and
build the inference caches the app serves, from the SAVED trained models.
Inference only - never refits, never rescores the official test.
Run: python3 bootstrap.py
"""
import io, json, os, pickle, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config, data, features

NASA_URL = ("https://phm-datasets.s3.amazonaws.com/NASA/"
            "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip")
NEEDED = [f"{sp}_{s}.txt" for s in config.SUBSETS for sp in ("train", "test")] + \
         [f"RUL_{s}.txt" for s in config.SUBSETS]


def ensure_data():
    missing = [f for f in NEEDED if not os.path.exists(os.path.join(config.DATA_DIR, f))]
    if not missing:
        print("data present"); return
    import urllib.request
    print("downloading C-MAPSS from NASA PCOE mirror...")
    with urllib.request.urlopen(NASA_URL, timeout=120) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    inner = [n for n in z.namelist() if n.endswith(".zip")]
    if inner:  # NASA ships a nested CMAPSSData.zip
        z = zipfile.ZipFile(io.BytesIO(z.read(inner[0])))
    os.makedirs(config.DATA_DIR, exist_ok=True)
    n_out = 0
    for n in z.namelist():
        base = os.path.basename(n)
        if base in NEEDED:
            with open(os.path.join(config.DATA_DIR, base), "wb") as f:
                f.write(z.read(n))
            n_out += 1
    if n_out != len(NEEDED):
        raise RuntimeError(f"expected {len(NEEDED)} files in NASA zip, extracted {n_out}")
    print("extracted", n_out, "files")


def build_caches(subset):
    rp = os.path.join(config.RESULTS_DIR, f"replay_{subset}.pkl")
    tp = os.path.join(config.RESULTS_DIR, f"test_endpoint_{subset}.pkl")
    if os.path.exists(rp) and os.path.exists(tp):
        print(subset, "caches present"); return
    with open(os.path.join(config.RESULTS_DIR, "splits.json")) as f:
        sp = json.load(f)["subsets"][subset]
    with open(os.path.join(config.RESULTS_DIR, f"model_{subset}.pkl"), "rb") as f:
        bundle = joblib.load(f)
    with open(os.path.join(config.RESULTS_DIR, f"results_{subset}.json")) as f:
        res = json.load(f)
    pre, mdl = bundle["preprocessor"], bundle["model"]
    train, _ = data.load_subset(subset, "train")
    train = data.true_rul_train(train)
    feat = features.build_features(train)
    feat = feat.merge(train[["unit", "cycle", "true_rul"]], on=["unit", "cycle"])
    replay_f = feat[feat.unit.isin(sp["replay"])]
    q10, q50, q90, _ = mdl.predict(pre.transform(replay_f))
    replay_f.assign(q10=q10, q50=q50, q90=q90,
                    envelope_flag=pre.envelope_flags(replay_f)).to_pickle(rp)
    test, _ = data.load_subset(subset, "test")
    tf = features.build_features(test)
    last = tf.sort_values("cycle").groupby("unit").tail(1)
    tq10, tq50, tq90, _ = mdl.predict(pre.transform(last))
    endpoints = data.load_test_endpoints(subset)[last["unit"].to_numpy() - 1]
    import numpy as np
    b1 = np.full(len(last), res["baselines"]["B1_constant"])
    b2 = np.maximum(0.0, res["baselines"]["B2_median_lifetime"] - last["cycle"].to_numpy())
    last.assign(q10=tq10, q50=tq50, q90=tq90, true_rul=endpoints, b1=b1, b2=b2).to_pickle(tp)
    print(subset, "caches built by inference from saved models")


if __name__ == "__main__":
    ensure_data()
    for s in config.SUBSETS:
        build_caches(s)
    print("bootstrap complete - start the app with: python3 app/server.py")
