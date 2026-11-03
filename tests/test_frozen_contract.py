"""Frozen-artifact contract: stored results must be internally consistent and
must match recomputation from the STORED prediction files. No model is
refitted and the official test is not re-scored - every number here is
re-derived from predictions the frozen pipeline already produced."""
import hashlib
import json
import os

import joblib
import numpy as np
import pytest

from conftest import ROOT, needs_data
from src import config, evaluate as ev

RESULTS = os.path.join(ROOT, "results")
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
EXPECTED_REPLAY_COUNTS = {"FD001": 15, "FD002": 39, "FD003": 15, "FD004": 38}


def res(subset):
    with open(os.path.join(RESULTS, f"results_{subset}.json")) as f:
        return json.load(f)


def replay(subset):
    return joblib.load(os.path.join(RESULTS, f"replay_{subset}.pkl"))


def endpoint(subset):
    return joblib.load(os.path.join(RESULTS, f"test_endpoint_{subset}.pkl"))


class TestSplits:
    def test_frozen_flags(self):
        sp = json.load(open(os.path.join(RESULTS, "splits.json")))
        assert sp["seed"] == config.SEED
        assert sp["fractions"] == list(config.SPLIT_FRACTIONS)
        assert sp["frozen_before_training"] is True

    @pytest.mark.parametrize("ds", SUBSETS)
    def test_disjoint_sizes(self, ds):
        sp = json.load(open(os.path.join(RESULTS, "splits.json")))["subsets"][ds]
        fit, tune, rep = set(sp["fit"]), set(sp["tune"]), set(sp["replay"])
        assert not (fit & tune) and not (fit & rep) and not (tune & rep)
        assert len(rep) == EXPECTED_REPLAY_COUNTS[ds]

    def test_total_replay_units_107(self):
        sp = json.load(open(os.path.join(RESULTS, "splits.json")))["subsets"]
        assert sum(len(sp[ds]["replay"]) for ds in SUBSETS) == 107

    @needs_data
    def test_splits_reproduce_from_seed(self):
        """Frozen unit lists must equal a fresh draw with the frozen seed over
        the actual train units, all four subsets drawn in one seeded sequence
        exactly as the pipeline did (proves no post-hoc split editing)."""
        from src import data, splits as splits_mod
        units = {}
        for ds in SUBSETS:
            train, _ = data.load_subset(ds, "train")
            units[ds] = sorted(train["unit"].unique())
        fresh = splits_mod.make_splits(units)
        frozen = json.load(open(os.path.join(RESULTS, "splits.json")))["subsets"]
        for ds in SUBSETS:
            assert fresh["subsets"][ds] == frozen[ds], ds


class TestReplayArtifacts:
    @pytest.mark.parametrize("ds", SUBSETS)
    def test_units_match_splits(self, ds):
        sp = json.load(open(os.path.join(RESULTS, "splits.json")))["subsets"][ds]
        assert sorted(replay(ds)["unit"].unique().tolist()) == sp["replay"]

    @pytest.mark.parametrize("ds", SUBSETS)
    def test_quantiles_ordered_nonnegative(self, ds):
        df = replay(ds)
        assert (df.q10 <= df.q50).all() and (df.q50 <= df.q90).all()
        assert (df.q10 >= 0).all() and (df.q50 >= 0).all() and (df.q90 >= 0).all()

    @pytest.mark.parametrize("ds", SUBSETS)
    def test_rmse_matches_stored(self, ds):
        df = replay(ds)
        assert ev.rmse(df.q50, df.true_rul) == pytest.approx(res(ds)["replay"]["rmse_all"], rel=1e-6)

    @pytest.mark.parametrize("ds", SUBSETS)
    def test_band_coverage_matches_stored(self, ds):
        df = replay(ds)
        cov, wid = ev.per_unit_coverage_width(df.unit.to_numpy(), df.true_rul.to_numpy(),
                                              df.q10.to_numpy(), df.q90.to_numpy())
        assert float(np.mean(list(cov.values()))) == pytest.approx(
            res(ds)["replay"]["band"]["coverage_mean"], rel=1e-6)
        assert float(np.mean(list(wid.values()))) == pytest.approx(
            res(ds)["replay"]["band"]["width_mean"], rel=1e-6)

    @pytest.mark.parametrize("ds", SUBSETS)
    def test_band_below_nominal_disclosed(self, ds):
        """The product's honesty claim: coverage below nominal 0.80 everywhere."""
        assert res(ds)["replay"]["band"]["coverage_mean"] < res(ds)["replay"]["band"]["nominal"]

    @pytest.mark.parametrize("ds", SUBSETS)
    def test_warning_policy_matches_stored(self, ds):
        out = ev.warning_episode_eval(replay(ds))
        w = res(ds)["replay"]["warning"]
        assert out["units"] == w["units"] == EXPECTED_REPLAY_COUNTS[ds]
        assert out["catch_fraction"] == pytest.approx(w["catch_fraction"], rel=1e-9)
        assert out["premature_unit_fraction"] == pytest.approx(w["premature_unit_fraction"], rel=1e-9)
        assert out["miss_fraction"] == pytest.approx(w["miss_fraction"], abs=1e-9)
        assert out["lead_time_cycles"]["median"] == pytest.approx(w["lead_time_cycles"]["median"], rel=1e-6)

    @pytest.mark.parametrize("ds", SUBSETS)
    def test_warning_counts_integer(self, ds):
        w = res(ds)["replay"]["warning"]
        n = w["units"]
        catch = round(w["catch_fraction"] * n)
        prem = round(w["premature_unit_fraction"] * n)
        miss = round(w["miss_fraction"] * n)
        assert catch + prem + miss == n


class TestEndpointArtifacts:
    @pytest.mark.parametrize("ds", SUBSETS)
    def test_endpoint_metrics_match_stored(self, ds):
        df = endpoint(ds)
        t = res(ds)["test"]
        assert ev.rmse(df.q50, df.true_rul) == pytest.approx(t["model_endpoint_rmse"], rel=1e-6)
        assert ev.mae(df.q50, df.true_rul) == pytest.approx(t["model_endpoint_mae"], rel=1e-6)
        assert ev.phm_score(df.q50, df.true_rul) == pytest.approx(t["model_phm_score"], rel=1e-6)
        assert ev.rmse(df.b1, df.true_rul) == pytest.approx(t["vs_B1"]["rmse_baseline"], rel=1e-6)
        assert ev.rmse(df.b2, df.true_rul) == pytest.approx(t["vs_B2"]["rmse_baseline"], rel=1e-6)
        cov = float(np.mean((df.true_rul >= df.q10) & (df.true_rul <= df.q90)))
        assert cov == pytest.approx(t["endpoint_coverage"], rel=1e-6)

    @pytest.mark.parametrize("ds", SUBSETS)
    def test_model_beats_both_baselines_ci(self, ds):
        t = res(ds)["test"]
        assert t["vs_B1"]["excludes_zero"] and t["vs_B2"]["excludes_zero"]
        assert t["model_endpoint_rmse"] < t["vs_B1"]["rmse_baseline"]
        assert t["model_endpoint_rmse"] < t["vs_B2"]["rmse_baseline"]
        assert res(ds)["verdicts"]["model_vs_baselines"] == "accepted"


class TestModelBundles:
    @pytest.mark.parametrize("ds", SUBSETS)
    def test_bundle_structure_and_features(self, ds):
        bundle = joblib.load(os.path.join(RESULTS, f"model_{ds}.pkl"))
        r = res(ds)
        for key in ("model", "preprocessor", "feature_cols", "constant_channels"):
            assert key in bundle
        assert len(bundle["feature_cols"]) == r["n_features"]
        assert bundle["constant_channels"] == r["constant_channels_excluded"]
        for ch in bundle["constant_channels"]:
            assert not any(c.startswith(ch + "_") for c in bundle["feature_cols"])
        assert set(bundle["model"].heads.keys()) == set(config.QUANTILES)

    @needs_data
    @pytest.mark.parametrize("ds", SUBSETS)
    def test_constant_channels_match_fit_data(self, ds):
        from src import data, features
        sp = json.load(open(os.path.join(RESULTS, "splits.json")))["subsets"][ds]
        train, _ = data.load_subset(ds, "train")
        fit_raw = train[train.unit.isin(sp["fit"])]
        assert features.find_constant_channels(fit_raw) == res(ds)["constant_channels_excluded"]


class TestConstants:
    @pytest.mark.parametrize("ds", SUBSETS)
    def test_results_constants_match_config(self, ds):
        c = res(ds)["constants"]
        assert c["window"] == config.WINDOW
        assert c["horizon_N"] == config.HORIZON_N
        assert c["quantiles"] == list(config.QUANTILES)
        assert c["split"] == list(config.SPLIT_FRACTIONS)
        assert c["seed"] == config.SEED


@needs_data
class TestDataHashes:
    def test_data_sha256_file(self):
        for line in open(os.path.join(ROOT, "DATA_SHA256.txt")):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sha, name = line.split()[0], line.split()[-1]
            path = os.path.join(ROOT, name) if os.sep in name else os.path.join(ROOT, "data_raw", name)
            if not os.path.exists(path):
                path = os.path.join(ROOT, name)
            actual = hashlib.sha256(open(path, "rb").read()).hexdigest()
            assert actual == sha, name


@needs_data
class TestBootstrapReproducibility:
    def test_replay_cache_regenerates_from_saved_model_fd001(self):
        """The shipped replay cache must equal fresh INFERENCE from the saved
        model (what bootstrap.py rebuilds on a user machine). Deterministic
        predict - no fitting, no scoring decisions."""
        import io as _io, contextlib
        from src import data, features
        ds = "FD001"
        sp = json.load(open(os.path.join(RESULTS, "splits.json")))["subsets"][ds]
        bundle = joblib.load(os.path.join(RESULTS, f"model_{ds}.pkl"))
        pre, mdl = bundle["preprocessor"], bundle["model"]
        train, _ = data.load_subset(ds, "train")
        train = data.true_rul_train(train)
        feat = features.build_features(train)
        feat = feat.merge(train[["unit", "cycle", "true_rul"]], on=["unit", "cycle"])
        rf = feat[feat.unit.isin(sp["replay"])].sort_values(["unit", "cycle"]).reset_index(drop=True)
        q10, q50, q90, _ = mdl.predict(pre.transform(rf))
        stored = replay(ds).sort_values(["unit", "cycle"]).reset_index(drop=True)
        assert (stored["unit"].to_numpy() == rf["unit"].to_numpy()).all()
        assert (stored["cycle"].to_numpy() == rf["cycle"].to_numpy()).all()
        np.testing.assert_allclose(stored["q50"].to_numpy(), q50, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(stored["q10"].to_numpy(), q10, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(stored["q90"].to_numpy(), q90, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(stored["true_rul"].to_numpy(), rf["true_rul"].to_numpy())
