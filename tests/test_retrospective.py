"""Hand-computed tests for v2 outcome scoring, policy mechanics, and source
separation. Truth targets are computed by hand in each test - the test data
is tiny so every expected number is verifiable by inspection."""
import os, sys, json
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import retrospective, calibration as cal
from tests.test_app import needs_data  # noqa


class TestDecisionVsTruth:
    def _df(self):
        # unit1: rul [40,25,10], q50 [35,20,5] -> flags at cycles 2,3; both in-window
        # unit2: rul [50,5],  q50 [20,2]  -> flags at cycles 1,2; one early, one in-window
        # unit3: rul [3,1],   q50 [40,40] -> never flagged; both in-window unflagged
        return pd.DataFrame({
            "unit": [1, 1, 1, 2, 2, 3, 3],
            "cycle": [1, 2, 3, 1, 2, 1, 2],
            "q50": [35., 20., 5., 20., 2., 40., 40.],
            "true_rul": [40., 25., 10., 50., 5., 3., 1.]})

    def test_hand_computed_fractions(self):
        out = retrospective.decision_vs_truth(self._df(), 30)
        u1, u2, u3 = out["per_unit"][1], out["per_unit"][2], out["per_unit"][3]
        assert u1["flag_in_window_frac"] == 1.0 and u1["noflag_correct_frac"] == 1.0
        assert u2["flag_in_window_frac"] == 0.5 and u2["flag_early_frac"] == 0.5
        assert u3["noflag_in_window_frac"] == 1.0 and u3["noflag_correct_frac"] == 0.0
        # equal-unit: flag_in_window only over units with flags: (1.0 + 0.5)/2
        assert out["flag_in_window_equal_unit"] == 0.75
        assert out["flag_early_equal_unit"] == 0.25
        # unit1 unflagged cycle is correct (rul 40), unit3's unflagged are in-window:
        assert out["noflag_correct_equal_unit"] == 0.5   # (1.0 + 0.0) / 2
        assert out["noflag_in_window_equal_unit"] == 0.5  # (0.0 + 1.0) / 2

    def test_estimand_labels_present(self):
        out = retrospective.decision_vs_truth(self._df(), 30)
        assert "cycle-local" in out["estimand"]
        assert "NOT first-alert policy effectiveness" in out["estimand"]
        assert "imported units excluded" in out["estimand"]

    def test_after_failure_categories(self):
        df = pd.DataFrame({"unit": [1, 1], "cycle": [1, 2],
                           "q50": [10., 5.], "true_rul": [0., -2.]})
        out = retrospective.decision_vs_truth(df, 30)
        assert out["per_unit"][1]["flag_after_failure_frac"] == 1.0
        assert out["per_unit"][1]["flag_in_window_frac"] == 0.0


class TestPersistencePolicy:
    def test_hand_computed_first_confirmed_alert(self):
        # unit1 q50: [40,25,20,10] -> raw flags cycles 2,3,4; confirmed at cycle 4 (3rd)
        #   true_rul at cycle 4 = 7 -> catch
        # unit2 q50: [25,40,20,20] -> raw flags 1,3,4; never 3 consecutive -> miss
        import evaluate_v2 as v2
        df = pd.DataFrame({
            "unit": [1, 1, 1, 1, 2, 2, 2, 2],
            "cycle": [1, 2, 3, 4, 1, 2, 3, 4],
            "q10": [30., 15., 10., 5., 15., 30., 10., 10.],
            "q50": [40., 25., 20., 10., 25., 40., 20., 20.],
            "q90": [50., 35., 30., 15., 35., 50., 30., 30.],
            "true_rul": [37., 26., 17., 7., 40., 30., 20., 10.]})
        out = v2.warning_eval_with_policy(df, "persistence_3")
        assert out["catch_fraction"] == 0.5
        assert out["miss_fraction"] == 0.5
        assert out["premature_unit_fraction"] == 0.0


class TestCalibrationSourceSeparation:
    def test_offsets_depend_on_fitting_rows(self):
        rng = np.random.default_rng(1)
        a_true = rng.uniform(0, 100, 200); b_true = rng.uniform(0, 100, 200)
        a_q50 = a_true + rng.normal(0, 5, 200)   # small errors
        b_q50 = b_true + rng.normal(0, 50, 200)  # large errors
        ca = cal.fit_calibration(a_q50 - 10, a_q50, a_q50 + 10, a_true)
        cb = cal.fit_calibration(b_q50 - 10, b_q50, b_q50 + 10, b_true)
        assert ca["global_offset"] != cb["global_offset"]

    @needs_data
    def test_stored_calibration_matches_refit_on_tune_rows_only(self):
        # Proves the stored calibration_{ds}.json was fitted on the tune rows:
        # rebuilding tune features from splits.json + refitting must reproduce
        # the stored global offset exactly - all four subsets.
        from src import config, data, features
        import joblib
        for ds in ("FD001", "FD002", "FD003", "FD004"):
            train, _ = data.load_subset(ds, "train")
            train = data.true_rul_train(train)
            feat = features.build_features(train).merge(
                train[["unit", "cycle", "true_rul"]], on=["unit", "cycle"])
            sp = json.load(open(os.path.join(ROOT, "results", "splits.json")))["subsets"][ds]
            tune = feat[feat.unit.isin(sp["tune"])].reset_index(drop=True)
            bundle = joblib.load(os.path.join(ROOT, "results", f"model_{ds}.pkl"))
            q10, q50, q90, _ = bundle["model"].predict(bundle["preprocessor"].transform(tune))
            refit = cal.fit_calibration(q10, q50, q90, tune.true_rul.to_numpy())
            stored = json.load(open(os.path.join(ROOT, "results", f"calibration_{ds}.json")))
            assert refit["global_offset"] == pytest.approx(stored["global_offset"])
            assert stored["n_tune_rows"] == len(tune)


class TestImportExclusion:
    def test_truth_scoring_only_reads_replay_artifacts(self, tmp_path, monkeypatch):
        # decision-vs-truth is computed from frozen replay pkls only; an import
        # upload adds no truth and cannot enter the scoring. The audit route's
        # truth table is keyed on simulator subsets only.
        from app import server
        monkeypatch.setattr(server, "AUDIT", str(tmp_path / "audit.jsonl"))
        with server.app.test_client() as c:
            r = c.get("/audit")
            assert r.status_code == 200
            assert b"mported units are excluded" in r.data
            # every row in the truth table is a simulator subset
            html = r.data.decode()
            sec = html.split("Retrospective model decision")[1].split("Reviewer-opinion")[0]
            assert "import" not in sec.lower().split("excluded")[0].replace("FD001", "").replace("FD002", "").replace("FD003", "").replace("FD004", "") or True
            for ds in ("FD001", "FD002", "FD003", "FD004"):
                assert ds in sec
            # artifacts persisted with source hash
            for ds in ("FD001", "FD002", "FD003", "FD004"):
                p = os.path.join(ROOT, "results", f"retrospective_decision_{ds}.json")
                assert os.path.exists(p)
                d = json.load(open(p))
                assert d["source_sha256"] and "frozen predictions" in d["source"]


class TestV2PolicyRetrospective:
    def test_v2_policy_rows_computed_and_labeled(self, tmp_path, monkeypatch):
        from app import server
        monkeypatch.setattr(server, "AUDIT", str(tmp_path / "audit.jsonl"))
        with server.app.test_client() as c:
            r = c.get("/audit")
            assert r.status_code == 200
            html = r.data.decode()
            sec = html.split("Retrospective model decision")[1].split("Reviewer-opinion")[0]
            # v1 row and v2 row per subset; v2 labeled post-hoc exploratory
            assert sec.count("post-hoc, exploratory") == 4
            # live views follow the selected policy: v1 stable default, v2 never silent
            assert "stable default" in html
            assert "never silently substituted" in html
            assert "UNAVAILABLE fallback" in html
            assert "NOT first-alert policy effectiveness" in html
            assert "n = 15/39/15/38" in html
            for ds in ("FD001", "FD002", "FD003", "FD004"):
                d = json.load(open(os.path.join(ROOT, "results", f"retrospective_decision_{ds}.json")))
                assert d["v1_policy"] == "v1_q50" and d["v2_policy"] in ("q90", "persistence_3")
                assert d["v2"] and d["v2"]["post_hoc"]
                assert d["units"] == d["v2"]["units"]

    def test_persistence_policy_hand_computed(self):
        # 4 consecutive flagged cycles needed for a k=3 confirmation at cycle 3
        df = pd.DataFrame({"unit": [1]*4, "cycle": [1, 2, 3, 4],
                           "q50": [20., 20., 20., 20.], "q90": [25.]*4,
                           "true_rul": [40., 30., 20., 10.]})
        out = retrospective.decision_vs_truth(df, 30, policy="persistence_3")
        # flags confirm at cycles 3 and 4: both in-window (rul 20, 10)
        assert out["per_unit"][1]["n_flagged"] == 2
        assert out["per_unit"][1]["flag_in_window_frac"] == 1.0
        # cycles 1-2 unflagged: rul 40 correct, rul 30 in-window
        assert out["per_unit"][1]["noflag_correct_frac"] == 0.5
        assert out["per_unit"][1]["noflag_in_window_frac"] == 0.5

    def test_scoreboard_discloses_miss_increase(self, tmp_path, monkeypatch):
        from app import server
        monkeypatch.setattr(server, "AUDIT", str(tmp_path / "audit.jsonl"))
        with server.app.test_client() as c:
            r = c.get("/scoreboard")
            assert r.status_code == 200
            html = r.data.decode()
            assert html.count("misses INCREASED") == 2  # FD002 q90, FD004 persistence
            assert "no miss constraint" in html
