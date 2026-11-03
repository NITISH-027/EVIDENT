"""Pure-function unit tests for src/: schema, splits, baselines, features,
model post-processing, evaluation contract."""
import numpy as np
import pandas as pd
import pytest

from src import config, data, features, splits as splits_mod, baselines as bl, model as mm, evaluate as ev


def _df(rows):
    return pd.DataFrame(rows, columns=data.COLUMNS)


def _valid_unit(unit, n, s1_base=100.0):
    return [[unit, c, 0.0, 0.0, 0.0] + [s1_base + c * 0.1 if i == 0 else 50.0 for i in range(21)]
            for c in range(1, n + 1)]


class TestSchema:
    def test_valid_unit_accepted(self):
        df = _df(_valid_unit(1, 5))
        clean, rep = data.validate_schema(df)
        assert rep["units_out"] == 1 and rep["units_rejected"] == {} and len(clean) == 5

    def test_gapped_cycles_rejected(self):
        rows = [r for r in _valid_unit(1, 5) if r[1] != 3]  # cycle 3 missing
        clean, rep = data.validate_schema(_df(rows))
        assert rep["units_out"] == 0 and "non-monotone" in rep["units_rejected"][1]

    def test_not_starting_at_1_rejected(self):
        rows = _valid_unit(1, 5)
        for r in rows:
            r[1] += 10
        clean, rep = data.validate_schema(_df(rows))
        assert rep["units_out"] == 0

    def test_single_cycle_rejected(self):
        clean, rep = data.validate_schema(_df(_valid_unit(1, 1)))
        assert rep["units_rejected"][1] == "fewer than 2 cycles"

    def test_null_rejected(self):
        rows = _valid_unit(1, 4)
        rows[2][5] = None
        df = _df(rows)
        clean, rep = data.validate_schema(df)
        assert rep["units_out"] == 0 and "null" in rep["units_rejected"][1]

    def test_mixed_units_partial_rejection(self):
        rows = _valid_unit(1, 5) + [r for r in _valid_unit(2, 4) if r[1] != 2]
        clean, rep = data.validate_schema(_df(rows))
        assert rep["units_out"] == 1 and 2 in rep["units_rejected"] and 1 not in rep["units_rejected"]


class TestTrueRul:
    def test_uncapped_lifetime_minus_cycle(self):
        df = data.true_rul_train(_df(_valid_unit(1, 4)))
        assert df.sort_values("cycle")["true_rul"].tolist() == [3, 2, 1, 0]


class TestSplits:
    def test_disjoint_cover_deterministic(self):
        units = {"FD001": range(1, 101), "FD002": range(1, 61)}
        s1 = splits_mod.make_splits(units)
        s2 = splits_mod.make_splits(units)
        assert s1 == s2  # same seed, same splits
        sp = s1["subsets"]["FD001"]
        fit, tune, replay = set(sp["fit"]), set(sp["tune"]), set(sp["replay"])
        assert not (fit & tune) and not (fit & replay) and not (tune & replay)
        assert fit | tune | replay == set(range(1, 101))
        assert (len(fit), len(tune), len(replay)) == (70, 15, 15)
        assert s1["seed"] == config.SEED and s1["frozen_before_training"] is True


class TestBaselines:
    def test_b1_b2_math(self):
        rows = _valid_unit(1, 3) + _valid_unit(2, 5)
        df = data.true_rul_train(_df(rows))
        b = bl.Baselines().fit(df)
        # true_rul rows: unit1 [2,1,0], unit2 [4,3,2,1,0] -> mean 13/8
        assert b.b1_constant == pytest.approx(13 / 8)
        assert b.b2_median_lifetime == pytest.approx(4.0)  # lifetimes 3,5 -> median 4
        b1, b2 = b.predict(np.array([0, 10]))
        assert b1.tolist() == [pytest.approx(13 / 8)] * 2
        assert b2.tolist() == [4.0, 0.0]  # clipped at 0


class TestWindowStats:
    def test_matches_naive(self):
        x = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 8.0, 7.0])
        m, s, l = features._window_stats_unit(x, 3)
        for i in range(len(x)):
            seg = x[max(0, i - 2):i + 1]
            assert m[i] == pytest.approx(seg.mean())
            assert s[i] == pytest.approx(seg.std(ddof=0))
            if len(seg) >= 2:
                t = np.arange(len(seg))
                slope = np.polyfit(t, seg, 1)[0]
                assert l[i] == pytest.approx(slope, rel=1e-6)
            else:
                assert l[i] == 0.0

    def test_single_point_zero_std_zero_slope(self):
        m, s, l = features._window_stats_unit(np.array([42.0]), 30)
        assert (m[0], s[0], l[0]) == (42.0, 0.0, 0.0)


class TestStrictlyPast:
    def test_features_ignore_future_rows(self):
        """Corrupting cycles after t must not change the features at t."""
        rows = _valid_unit(1, 12)
        df = _df(rows)
        f1 = features.build_features(df, w=5)
        poisoned = df.copy()
        mask = poisoned["cycle"] > 6
        poisoned.loc[mask, data.CHANNELS] = -99999.0
        f2 = features.build_features(poisoned, w=5)
        cols = [c for c in f1.columns if c not in ("unit", "cycle")]
        early1 = f1[f1.cycle <= 6][cols].to_numpy()
        early2 = f2[f2.cycle <= 6][cols].to_numpy()
        np.testing.assert_allclose(early1, early2, rtol=0, atol=0)


class TestPreprocessor:
    def test_fit_only_stats_and_envelope(self):
        cols = ["cycle", "s1_mean"]
        fit_df = pd.DataFrame({"cycle": [1, 2, 3], "s1_mean": [10.0, 20.0, 30.0]})
        other = pd.DataFrame({"cycle": [100], "s1_mean": [999.0]})
        pre = features.Preprocessor(cols).fit(fit_df)
        X = pre.transform(other)
        # uses fit stats, not other's
        assert X[0, 1] == pytest.approx((999.0 - 20.0) / fit_df["s1_mean"].std(ddof=0))
        assert bool(pre.envelope_flags(other)[0]) is True
        assert bool(pre.envelope_flags(fit_df)[0]) is False

    def test_zero_std_channel_safe(self):
        cols = ["a"]
        df = pd.DataFrame({"a": [7.0, 7.0]})
        pre = features.Preprocessor(cols).fit(df)
        assert np.isfinite(pre.transform(df)).all()


class TestConstantChannels:
    def test_detects_zero_variance(self):
        rows = _valid_unit(1, 4) + _valid_unit(2, 4)
        df = _df(rows)
        df["s5"] = 3.333
        const = features.find_constant_channels(df)
        assert "s5" in const and "s1" not in const


class TestModelPostprocess:
    def test_ordering_clipping_and_diagnostics(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 3))
        y = rng.normal(size=60) * 10 + 50
        m = mm.QuantileRULModel(dict(max_iter=10, learning_rate=0.3, max_leaf_nodes=7,
                                     min_samples_leaf=5, random_state=0)).fit(X, y)
        q10, q50, q90, diag = m.predict(X)
        assert np.all(q10 <= q50) and np.all(q50 <= q90)
        assert np.all(q10 >= 0) and np.all(q50 >= 0) and np.all(q90 >= 0)
        assert 0.0 <= diag["raw_crossing_row_rate"] <= 1.0
        assert 0.0 <= diag["raw_negative_row_rate"] <= 1.0

    def test_unit_weights(self):
        df = pd.DataFrame({"unit": [1, 1, 1, 2], "cycle": [1, 2, 3, 1]})
        w = mm.unit_weights(df)
        assert w[:3].sum() == pytest.approx(1.0) and w[3] == pytest.approx(1.0)


class TestEvaluate:
    def test_rmse_mae(self):
        assert ev.rmse([1, 2], [1, 4]) == pytest.approx(np.sqrt(2))
        assert ev.mae([1, 2], [1, 4]) == pytest.approx(1.0)

    def test_phm_asymmetry(self):
        # d = pred - true; d > 0 (over-estimating remaining life, i.e. a LATE
        # warning) is penalized more than d < 0 (early warning).
        late = ev.phm_score([20], [10])
        early = ev.phm_score([0], [10])
        assert late > early

    def test_pinball(self):
        assert ev.pinball([10], [8], 0.5) == pytest.approx(1.0)

    def test_banded_rmse(self):
        out = ev.banded_rmse([10, 100], [12, 110])
        assert out["0-30"]["n"] == 1 and out[">130"]["n"] == 0
        assert out["0-30"]["rmse"] == pytest.approx(2.0)
        assert out[">130"]["rmse"] is None

    def test_warning_policy_classes(self):
        N = config.HORIZON_N

        def mk(unit, true_ruls, flags):
            return pd.DataFrame({"unit": unit, "cycle": range(1, len(true_ruls) + 1),
                                 "true_rul": true_ruls,
                                 "q50": [N if f else N + 50 for f in flags]})
        # catch: first flag at 0<RUL<=N
        df = mk(1, [40, 25, 10, 0], [False, True, True, True])
        # premature: first flag at RUL>N, later cycles inside window never convert
        df = pd.concat([df, mk(2, [80, 20, 0], [True, True, True])])
        # miss: never flagged
        df = pd.concat([df, mk(3, [10, 0], [False, False])])
        # miss: first flag at RUL 0 (too late)
        df = pd.concat([df, mk(4, [10, 0], [False, True])])
        out = ev.warning_episode_eval(df)
        assert out["units"] == 4
        assert out["catch_fraction"] == pytest.approx(0.25)
        assert out["premature_unit_fraction"] == pytest.approx(0.25)
        assert out["miss_fraction"] == pytest.approx(0.5)
        assert out["lead_time_cycles"]["n"] == 1

    def test_unit_bootstrap_ci_small_sample(self):
        v, (lo, hi) = ev.unit_bootstrap_ci({1: 0.5})
        assert v == 0.5 and lo is None and hi is None

    def test_paired_bootstrap_direction(self):
        rng = np.random.default_rng(1)
        true = rng.uniform(0, 100, 200)
        good = true + rng.normal(0, 1, 200)
        bad = true + rng.normal(0, 30, 200)
        out = ev.paired_bootstrap_rmse_diff(good, bad, true, resamples=200)
        assert out["excludes_zero"] is True and out["diff_mean"] > 0
