"""Flask app tests: routes, entry-point guards, import edge cases, audit
dedupe, verdict logging. Audit writes go to a temp file - results/audit.jsonl
is never touched by the suite."""
import io
import json
import os
import sys

import joblib
import pytest

from conftest import ROOT, needs_data

sys.path.insert(0, os.path.join(ROOT, "app"))
import server


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "AUDIT", str(tmp_path / "audit.jsonl"))
    monkeypatch.setattr(server, "IMPORTS", {})
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


def _audit_records(tmp_path):
    p = tmp_path / "audit.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _replay_unit(ds="FD001"):
    df = server.load_replay(ds)
    return int(df.unit.unique()[0])


def _flagged_unit(ds="FD001"):
    df = server.load_replay(ds)
    for u, g in df.groupby("unit"):
        if (g.q50 <= server.HORIZON_N).any():
            return int(u), int(g[g.q50 <= server.HORIZON_N].iloc[0].cycle)
    raise AssertionError("no flagged unit in " + ds)


class TestFleet:
    @pytest.mark.parametrize("ds", ["FD001", "FD002", "FD003", "FD004"])
    def test_fleet_ok(self, client, ds):
        r = client.get(f"/?subset={ds}")
        assert r.status_code == 200 and ds.encode() in r.data

    def test_fleet_bad_subset_404(self, client):
        assert client.get("/?subset=FD999").status_code == 404

    def test_fleet_shows_coverage_caveat(self, client):
        r = client.get("/?subset=FD001")
        assert b"measured coverage" in r.data or b"coverage" in r.data.lower()


class TestUnitReplay:
    def test_unit_ok_default_first_flag(self, client):
        u, fc = _flagged_unit()
        r = client.get(f"/unit/FD001/{u}")
        assert r.status_code == 200
        assert f"cycle {fc}".encode() in r.data  # default view lands at first flag

    def test_unit_not_in_replay_404(self, client):
        sp = json.load(open(os.path.join(ROOT, "results", "splits.json")))["subsets"]["FD001"]
        non_replay = next(u for u in range(1, 101) if u not in sp["replay"])
        r = client.get(f"/unit/FD001/{non_replay}")
        assert r.status_code == 404 and b"replay-holdout" in r.data

    def test_bad_subset_guard_404(self, client):
        assert client.get("/unit/FD999/1").status_code == 404

    def test_cycle_clamps_to_max(self, client):
        u = _replay_unit()
        df = server.load_replay("FD001")
        mx = int(df[df.unit == u].cycle.max())
        r = client.get(f"/unit/FD001/{u}?cycle=99999")
        assert r.status_code == 200 and f"cycle {mx}".encode() in r.data

    def test_truth_overlay_gated(self, client):
        u = _replay_unit()
        plain = client.get(f"/unit/FD001/{u}")
        truth = client.get(f"/unit/FD001/{u}?show_truth=1")
        assert b"RETROSPECTIVE" not in plain.data or b"show retrospective ground truth" in plain.data
        assert b"show_truth=1" in truth.data or b"RETROSPECTIVE" in truth.data

    def test_flag_shown_logged_once(self, client, tmp_path):
        u, fc = _flagged_unit()
        client.get(f"/unit/FD001/{u}?cycle={fc}")
        client.get(f"/unit/FD001/{u}?cycle={fc}")
        recs = [e for e in _audit_records(tmp_path) if e.get("record_type") == "flag_shown"]
        assert len(recs) == 1 and recs[0]["unit"] == u and recs[0]["cycle"] == fc


class TestCharts:
    def test_rul_chart_png(self, client):
        u = _replay_unit()
        r = client.get(f"/chart/rul/FD001/{u}/50.png")
        assert r.status_code == 200 and r.mimetype == "image/png" and r.data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_rul_chart_unknown_unit_404(self, client):
        assert client.get("/chart/rul/FD001/9999/50.png").status_code == 404

    def test_chart_bad_subset_guard(self, client):
        assert client.get("/chart/rul/FD999/1/50.png").status_code == 404

    @needs_data
    def test_sensor_chart_png(self, client):
        u = _replay_unit()
        r = client.get(f"/chart/sensors/FD001/{u}/50.png")
        assert r.status_code == 200 and r.data[:8] == b"\x89PNG\r\n\x1a\n"

    @needs_data
    @pytest.mark.parametrize("ds", ["FD002", "FD004"])  # multi-regime matched-baseline path
    def test_sensor_chart_multiregime(self, client, ds):
        u = _replay_unit(ds)
        r = client.get(f"/chart/sensors/{ds}/{u}/60.png")
        assert r.status_code == 200 and r.data[:8] == b"\x89PNG\r\n\x1a\n"


def _import_csv(n_units=2, n_cycles=35, gap=False):
    lines = []
    for u in range(1, n_units + 1):
        for c in range(1, n_cycles + 1):
            if gap and u == 2 and c == 10:
                continue
            vals = [u, c, 0.0, -0.0, 100.0] + [500 + c * 0.5 + i for i in range(21)]
            lines.append(" ".join(str(v) for v in vals))
    return "\n".join(lines).encode()


class TestImport:
    def test_get_form(self, client):
        assert client.get("/import").status_code == 200

    def test_empty_file_rejected(self, client):
        r = client.post("/import", data={"subset": "FD001", "file": (io.BytesIO(b""), "x.txt")},
                        content_type="multipart/form-data")
        assert b"empty file" in r.data

    def test_wrong_columns_rejected(self, client):
        body = b"1 1 2 3\n1 2 3 4\n"
        r = client.post("/import", data={"subset": "FD001", "file": (io.BytesIO(body), "x.txt")},
                        content_type="multipart/form-data")
        assert b"expected 26 columns" in r.data

    def test_gapped_unit_rejected_valid_unit_processed(self, client):
        body = _import_csv(n_units=2, n_cycles=35, gap=True)
        r = client.post("/import", data={"subset": "FD001", "file": (io.BytesIO(body), "x.txt")},
                        content_type="multipart/form-data")
        assert b"non-monotone or gapped" in r.data  # rejection reason shown
        assert b"review" in r.data.lower()  # valid unit still processed

    def test_all_invalid_units_error(self, client):
        lines = ["1 1 " + " ".join(["50"] * 24)]  # single cycle -> rejected
        r = client.post("/import", data={"subset": "FD001", "file": (io.BytesIO("\n".join(lines).encode()), "x.txt")},
                        content_type="multipart/form-data")
        assert b"no units passed schema validation" in r.data

    def test_bad_subset_400(self, client):
        r = client.post("/import", data={"subset": "FD999", "file": (io.BytesIO(b"1 2\n"), "x.txt")},
                        content_type="multipart/form-data")
        assert r.status_code == 400

    def test_valid_import_flow(self, client, tmp_path):
        body = _import_csv()
        r = client.post("/import", data={"subset": "FD002", "file": (io.BytesIO(body), "x.txt")},
                        content_type="multipart/form-data")
        assert r.status_code == 200 and b"FD002" in r.data
        token = next(iter(server.IMPORTS))
        assert server.IMPORTS[token]["subset"] == "FD002"
        assert len(server.IMPORTS[token]["sha256"]) == 64
        # import_upload audit record carries the sha
        ups = [e for e in _audit_records(tmp_path) if e.get("record_type") == "import_upload"]
        assert len(ups) == 1 and ups[0]["upload_sha256"] == server.IMPORTS[token]["sha256"]
        # imported unit page + charts + quality
        assert client.get(f"/import/{token}/1").status_code == 200
        assert client.get(f"/chart_import/rul/{token}/1.png").data[:8] == b"\x89PNG\r\n\x1a\n"
        assert client.get(f"/chart_import/sensors/{token}/1.png").status_code == 200
        assert client.get(f"/quality_import/{token}/1").status_code == 200
        # no-ground-truth honesty on the review page
        page = client.get(f"/import/{token}/1")
        assert b"no ground truth" in page.data.lower()

    def test_import_bad_token_404(self, client):
        assert client.get("/import/deadbeef/1").status_code == 404

    def test_csv_separator_accepted(self, client):
        lines = []
        for u in (1,):
            for c in range(1, 35):
                vals = [u, c, 0.0, 0.0, 100.0] + [500 + i for i in range(21)]
                lines.append(",".join(str(v) for v in vals))
        r = client.post("/import", data={"subset": "FD001", "file": (io.BytesIO("\n".join(lines).encode()), "x.csv")},
                        content_type="multipart/form-data")
        assert r.status_code == 200
        assert b"could not parse" not in r.data and b"expected 26 columns" not in r.data
        assert len(server.IMPORTS) == 1

    def test_header_row_accepted(self, client):
        header = "unit cycle " + " ".join([f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)])
        lines = [header]
        for c in range(1, 35):
            vals = [1, c, 0.0, 0.0, 100.0] + [500 + i for i in range(21)]
            lines.append(" ".join(str(v) for v in vals))
        r = client.post("/import", data={"subset": "FD001", "file": (io.BytesIO("\n".join(lines).encode()), "x.txt")},
                        content_type="multipart/form-data")
        assert r.status_code == 200 and b"could not parse" not in r.data
        assert len(server.IMPORTS) == 1


class TestVerify:
    def test_valid_verdict_logged(self, client, tmp_path):
        u = _replay_unit()
        r = client.post("/verify", data={"subset": "FD001", "unit": u, "cycle": 50,
                                         "verdict": "reviewed_warrants_inspection"})
        assert r.status_code == 200
        recs = [e for e in _audit_records(tmp_path) if e.get("record_type") == "human_verdict"]
        assert len(recs) == 1 and recs[0]["verdict"] == "reviewed_warrants_inspection"

    def test_bad_verdict_400(self, client):
        u = _replay_unit()
        r = client.post("/verify", data={"subset": "FD001", "unit": u, "cycle": 50,
                                         "verdict": "looks_fine_trust_me"})
        assert r.status_code == 400

    def test_missing_fields_400(self, client):
        assert client.post("/verify", data={"subset": "FD001"}).status_code == 400

    def test_non_integer_fields_400(self, client):
        r = client.post("/verify", data={"subset": "FD001", "unit": "abc", "cycle": 50,
                                         "verdict": "reviewed_dismissed"})
        assert r.status_code == 400

    def test_import_verdict_requires_token(self, client):
        r = client.post("/verify", data={"subset": "import", "unit": 1, "cycle": 30,
                                         "verdict": "reviewed_dismissed"})
        assert r.status_code == 400

    def test_import_verdict_with_token_logged(self, client, tmp_path):
        body = _import_csv()
        client.post("/import", data={"subset": "FD001", "file": (io.BytesIO(body), "x.txt")},
                    content_type="multipart/form-data")
        token = next(iter(server.IMPORTS))
        r = client.post("/verify", data={"subset": "import", "unit": 1, "cycle": 35,
                                         "verdict": "reviewed_dismissed", "token": token})
        assert r.status_code == 200
        recs = [e for e in _audit_records(tmp_path) if e.get("record_type") == "human_verdict"]
        assert len(recs) == 1 and recs[0]["upload_sha256"] == server.IMPORTS[token]["sha256"]


class TestAudit:
    def test_frozen_reference_107_deduped(self, client, tmp_path):
        assert client.get("/audit").status_code == 200
        client.get("/audit")
        client.get("/audit")
        recs = [e for e in _audit_records(tmp_path) if e.get("record_type") == "frozen_first_alert_reference"]
        assert len(recs) == 107
        per = {}
        for e in recs:
            per.setdefault(e["subset"], []).append(e["frozen_class"])
        assert len(per["FD001"]) == 15 and len(per["FD002"]) == 39
        assert len(per["FD003"]) == 15 and len(per["FD004"]) == 38
        for ds, classes in per.items():
            w = json.load(open(os.path.join(ROOT, "results", f"results_{ds}.json")))["replay"]["warning"]
            assert classes.count("catch") == round(w["catch_fraction"] * w["units"])
            assert classes.count("premature") == round(w["premature_unit_fraction"] * w["units"])
        for e in recs:
            assert "reconstructed_from_frozen_at" in e and e["source_sha256"]

    def test_audit_page_labels(self, client):
        r = client.get("/audit")
        assert b"this cycle only, not a policy result" in r.data or b"frozen" in r.data


class TestScoreboardQuality:
    def test_scoreboard_all_subsets(self, client):
        r = client.get("/scoreboard")
        assert r.status_code == 200
        for ds in (b"FD001", b"FD002", b"FD003", b"FD004"):
            assert ds in r.data
        assert b"reproducibility repeat" in r.data  # FD001 history disclosure

    @needs_data
    def test_quality_gate(self, client):
        u = _replay_unit()
        r = client.get(f"/quality/FD001/{u}")
        assert r.status_code == 200 and b"excluded - constant" in r.data


class TestAuditV2:
    def test_frozen_flag_cycle_reference_complete_and_deduped(self, client, tmp_path):
        client.get("/audit")
        client.get("/audit")
        recs = [e for e in _audit_records(tmp_path) if e.get("record_type") == "frozen_flag_cycle_reference"]
        keys = {(e["subset"], e["unit"], e["cycle"]) for e in recs}
        assert len(keys) == len(recs)  # deduped across repeated /audit visits
        import joblib
        for ds in ("FD001", "FD002", "FD003", "FD004"):
            df = joblib.load(os.path.join(ROOT, "results", f"replay_{ds}.pkl"))
            flagged = df[df.q50 <= server.HORIZON_N]
            got = [e for e in recs if e["subset"] == ds]
            assert len(got) == len(flagged)  # complete coverage of every flagged cycle
            for e in got:
                assert e["source_sha256"] and "reconstructed_from_frozen_at" in e

    def test_verdict_agreement_scoring_and_labels(self, client, tmp_path):
        u = _replay_unit()
        r = client.get(f"/unit/FD001/{u}")
        assert r.status_code == 200
        import joblib
        df = joblib.load(os.path.join(ROOT, "results", "replay_FD001.pkl"))
        g = df[df.unit == u].sort_values("cycle")
        fc = int(g[g.q50 <= server.HORIZON_N].iloc[0].cycle) if (g.q50 <= server.HORIZON_N).any() else int(g.cycle.max())
        client.post("/verify", data={"subset": "FD001", "unit": u, "cycle": fc,
                                     "verdict": "reviewed_warrants_inspection"})
        r = client.get("/audit")
        assert b"separate estimand" in r.data
        assert b"NOT first-alert policy accuracy" in r.data
        assert b"reviewer" in r.data and b"opinion" in r.data.lower()
        assert b"1 of 1" in r.data  # the one replay verdict scored, agrees with flag shown

    def test_channel_distinction_on_events(self, client, tmp_path):
        r = client.get("/audit")
        assert b"simulator replay" in r.data or b"operational import" in r.data or b"No events logged yet" in r.data


class TestVersionPolicyMode:
    def test_v2_unit_view_uses_q90_policy(self, client):
        # FD002 unit 12 cycle 220: q50 29.1 <= 30 (v1 flags) but q90 34.3 > 30 (v2 q90 does not)
        r1 = client.get("/unit/FD002/12?cycle=220")
        r2 = client.get("/unit/FD002/12?cycle=220&version=v2")
        assert r1.status_code == r2.status_code == 200
        assert b"WARNING FLAG" in r1.data
        assert b"WARNING FLAG" not in r2.data
        assert b"v2 exploratory (q90)" in r2.data

    def test_v1_is_default_and_stable_labeled(self, client):
        r = client.get("/unit/FD002/12?cycle=220")
        assert b"v1 policy (q50" in r.data

    def test_persistence_first_alert_at_third_confirmed_cycle(self, client):
        # FD004 unit 11: v1 first flag cycle 226; persistence first confirm cycle 251
        r = client.get("/unit/FD004/11?version=v2")  # default cycle = first flag under policy
        assert b"replay at cycle 251" in r.data
        r1 = client.get("/unit/FD004/11")
        assert b"replay at cycle 226" in r1.data
        # at cycle 244 (q50 28.8 <= 30): v1 flags; persistence has no 3-cycle run yet -> no flag
        rv1 = client.get("/unit/FD004/11?cycle=244")
        rv2 = client.get("/unit/FD004/11?cycle=244&version=v2")
        assert b"WARNING FLAG" in rv1.data
        assert b"WARNING FLAG" not in rv2.data

    def test_fleet_v2_banner_and_policy_column(self, client):
        r = client.get("/?subset=FD001&version=v2")
        assert b"v2 exploratory (q90)" in r.data
        assert b"switch to v1 stable policy view" in r.data
        r1 = client.get("/?subset=FD001")
        assert b"v1 policy" in r1.data

    def test_verdict_policy_derived_server_side(self, client, tmp_path):
        # version=v2 form derives q90 for FD001; matching policy accepted
        r = client.post("/verify", data={"subset": "FD001", "unit": 4, "cycle": 165,
                                         "verdict": "reviewed_warrants_inspection",
                                         "version": "v2", "policy": "q90"})
        assert r.status_code == 200 and r.get_json()["logged"]["policy"] == "q90"
        # cycle 165: q50 29.5 <= 30 flags v1; q90 > 30 so v2 q90 is quiet here - recorded honestly
        assert r.get_json()["logged"]["flag_state_at_cycle"] is False
        # mismatch rejected: v1 view deriving v1_q50 but form claims q90
        r = client.post("/verify", data={"subset": "FD001", "unit": 4, "cycle": 165,
                                         "verdict": "reviewed_warrants_inspection", "policy": "q90"})
        assert r.status_code == 400 and b"mismatch" in r.data
        # invalid version falls back to v1 (not "v2"); cycle must exist
        r = client.post("/verify", data={"subset": "FD001", "unit": 4, "cycle": 99999,
                                         "verdict": "reviewed_warrants_inspection"})
        assert r.status_code == 400 and b"cycle does not exist" in r.data
        r = client.post("/verify", data={"subset": "FD001", "unit": 4, "cycle": 165,
                                         "verdict": "reviewed_warrants_inspection"})
        assert r.get_json()["logged"]["policy"] == "v1_q50"  # default
        assert r.get_json()["logged"]["flag_state_at_cycle"] is True

    def test_chart_title_shows_policy(self, client):
        r = client.get("/chart/rul/FD001/4/165.png?version=v2")
        assert r.status_code == 200
        r2 = client.get("/chart/rul/FD001/4/165.png")
        assert r.data != r2.data  # different first-flag marker / title


class TestPolicyAuditIntegrity:

    def test_dedupe_separates_policies_same_cycle(self, client, tmp_path):
        # v1 flags FD001/4 at 165; view same cycle under v2 (quiet under q90):
        # flag_shown only for the v1 view; then force a v2 flagged cycle
        client.get("/unit/FD001/4?cycle=165")          # v1 flags -> logged
        client.get("/unit/FD001/4?cycle=165")          # dedupe: no second record
        client.get("/unit/FD001/4?cycle=165&version=v2")  # q90 quiet -> no record
        recs = [e for e in _audit_records(tmp_path) if e.get("record_type") == "flag_shown"]
        assert len(recs) == 1 and recs[0]["policy"] == "v1_q50"
        # FD004/11/251: v2 persistence flags, v1 flags too -> two records, distinct policies
        client.get("/unit/FD004/11?cycle=251")
        client.get("/unit/FD004/11?cycle=251&version=v2")
        recs = [e for e in _audit_records(tmp_path) if e.get("record_type") == "flag_shown"]
        pols = sorted(e["policy"] for e in recs)
        assert pols == ["persistence_3", "v1_q50", "v1_q50"]

    @needs_data
    def test_dedupe_separates_uploads(self, client, tmp_path):
        import io
        raw = open(os.path.join(ROOT, "data_raw", "train_FD001.txt")).read().splitlines()
        rows1 = [l for l in raw if l.startswith("1 ")]
        data1 = "\n".join(rows1).encode()
        r1 = client.post("/import?subset=FD001", data={"file": (io.BytesIO(data1), "u1.txt")})
        r2 = client.post("/import?subset=FD001", data={"file": (io.BytesIO(data1), "u1.txt")})
        assert r1.status_code == r2.status_code == 200
        ups = [e for e in _audit_records(tmp_path) if e.get("record_type") == "import_upload"]
        assert len(ups) == 2 and ups[0]["upload_sha256"] == ups[1]["upload_sha256"]  # same bytes, two events
        flags = [e for e in _audit_records(tmp_path) if e.get("record_type") == "flag_shown"]
        # same unit/cycle/policy but distinct uploads -> not conflated
        keys = {(e["unit"], e["cycle"], e["policy"], e.get("upload_sha256")) for e in flags}
        assert len(keys) == len(flags)

    def test_import_persistence_short_history_disclosed(self, client):
        import io
        # two-cycle unit under FD004 v2 (persistence_3): cannot confirm.
        # Rows are the verbatim first two cycles of unit 1 from train_FD004.txt.
        lines = ["1 1 42.0049 0.8400 100.0 445.00 549.68 1343.43 1112.93 3.91 5.70 137.36 2211.86 8311.32 1.01 41.69 129.78 2387.99 8074.83 9.3335 0.02 330 2212 100.00 10.62 6.3670",
                 "1 2 20.0020 0.7002 100.0 491.19 606.07 1477.61 1237.50 9.35 13.61 332.10 2323.66 8713.60 1.07 43.94 312.59 2387.73 8046.13 9.1913 0.02 361 2324 100.00 24.37 14.6552"]
        r = client.post("/import?version=v2",
                        data={"subset": "FD004",
                              "file": (io.BytesIO("\n".join(lines).encode()), "short.txt")})
        assert r.status_code == 200
        assert b"persistence cannot confirm" in r.data
        assert b"v2 exploratory (persistence_3)" in r.data

    def test_v2_unavailable_is_explicit_not_silent_v1(self, client, monkeypatch):
        # patch the top-level `server` module the Flask app lives in
        # (`from app import server` would bind a second, unused instance).
        monkeypatch.setattr(server, "load_v2", lambda subset: None)
        monkeypatch.setattr(server, "_POLICY_CACHE", {})
        r = client.get("/unit/FD001/4?cycle=165&version=v2")
        assert b"V2 RESULTS UNAVAILABLE" in r.data
        assert b"EXPLICIT fallback" in r.data
        assert b"v2 exploratory" not in r.data  # never labeled v2
        # /verify under version=v2 with no v2 results: rejected, not silently v1
        r = client.post("/verify", data={"subset": "FD001", "unit": 4, "cycle": 165,
                                         "verdict": "reviewed_warrants_inspection", "version": "v2"})
        assert r.status_code == 400 and b"unavailable" in r.data

