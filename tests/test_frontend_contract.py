"""Frontend contract tests: new JSON endpoints, equality with server logic,
audit immutability on GETs, explicit display flag logging, handover rendering,
and static asset checks."""
import json
import os
import re
import sys
import pytest

from conftest import ROOT, needs_data

sys.path.insert(0, os.path.join(ROOT, "app"))
import server


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "AUDIT", str(tmp_path / "audit.jsonl"))
    monkeypatch.setattr(server, "NOTES", str(tmp_path / "review_notes.jsonl"))
    monkeypatch.setattr(server, "IMPORTS", {})
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


def _audit_records(tmp_path):
    p = tmp_path / "audit.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


class TestRequiredStrings:
    """Verifies all hard requirement strings from Section 2E are present."""

    def test_strings_in_templates(self):
        tmpl_dir = os.path.join(ROOT, "app", "templates")
        unit_html = open(os.path.join(tmpl_dir, "unit.html"), encoding="utf-8").read()
        base_html = open(os.path.join(tmpl_dir, "base.html"), encoding="utf-8").read()
        import_unit_html = open(os.path.join(tmpl_dir, "import_unit.html"), encoding="utf-8").read()
        scoreboard_html = open(os.path.join(tmpl_dir, "scoreboard.html"), encoding="utf-8").read()
        quality_html = open(os.path.join(tmpl_dir, "quality.html"), encoding="utf-8").read()
        audit_html = open(os.path.join(tmpl_dir, "audit.html"), encoding="utf-8").read()

        assert "not a failure probability" in unit_html
        assert "Never commands any machine" in base_html
        assert "measured coverage" in unit_html
        assert "RETROSPECTIVE" in unit_html or "retrospective" in unit_html
        assert "no ground truth" in import_unit_html.lower()
        assert "opinion" in unit_html.lower() or "opinion" in audit_html.lower()
        assert "reproducibility repeat" in scoreboard_html
        assert "no miss constraint" in scoreboard_html
        assert "excluded - constant" in quality_html
        assert "mported units are excluded" in audit_html
        assert "stable default" in audit_html
        assert "never silently substituted" in audit_html
        assert "UNAVAILABLE fallback" in audit_html
        assert "NOT first-alert policy effectiveness" in audit_html
        assert "n = 15/39/15/38" in audit_html
        assert "frozen" in audit_html

    def test_audit_cycle_local_label_in_server(self):
        src = open(os.path.join(ROOT, "app", "server.py"), encoding="utf-8").read()
        assert "this cycle only, not a policy result" in src


class TestApiFleetEndpoint:
    @pytest.mark.parametrize("ds", ["FD001", "FD002", "FD003", "FD004"])
    def test_fleet_equality_with_server_logic(self, client, ds):
        r = client.get(f"/api/fleet?subset={ds}&cycle=50")
        assert r.status_code == 200
        data = r.get_json()
        assert data["subset"] == ds
        assert data["cycle"] == 50
        assert data["horizon"] == 30

        # Compare directly against server df
        df = server.replay_with_flags(ds, "v1")
        cur = df[df.cycle <= 50].sort_values("cycle").groupby("unit").tail(1)
        expected_rows = [{
            "unit": int(row.unit), "cycle": int(row.cycle), "q50": round(row.q50, 1),
            "flag": bool(row.flag)
        } for row in cur.itertuples()]
        expected_rows.sort(key=lambda x: x["q50"])

        assert len(data["rows"]) == len(expected_rows)
        for actual, exp in zip(data["rows"], expected_rows):
            assert actual["unit"] == exp["unit"]
            assert actual["cycle"] == exp["cycle"]
            assert actual["q50"] == exp["q50"]
            assert actual["flag"] == exp["flag"]

    def test_fleet_invalid_subset(self, client):
        assert client.get("/api/fleet?subset=FD999").status_code == 404


class TestApiUnitEndpoint:
    @pytest.mark.parametrize("ds,unit", [("FD001", 4), ("FD002", 12), ("FD003", 5), ("FD004", 11)])
    def test_unit_equality_and_truth_gating(self, client, ds, unit):
        # Plain call without show_truth
        r = client.get(f"/api/unit/{ds}/{unit}")
        assert r.status_code == 200
        data = r.get_json()
        assert data["subset"] == ds
        assert data["unit"] == unit
        assert "retrospective" not in data
        assert len(data["series"]) > 0
        for s in data["series"]:
            assert "true_rul" not in s

        # With show_truth=1
        r_truth = client.get(f"/api/unit/{ds}/{unit}?show_truth=1")
        assert r_truth.status_code == 200
        data_truth = r_truth.get_json()
        assert data_truth.get("retrospective") is True
        for s in data_truth["series"]:
            assert "true_rul" in s

        # Check exact equality on current cycle metrics
        cur = data["current"]
        df = server.replay_with_flags(ds, "v1")
        g = df[df.unit == unit].sort_values("cycle")
        row = g[g.cycle == cur["cycle"]].iloc[-1]
        assert cur["q50"] == round(float(row.q50), 1)
        assert cur["q10"] == round(float(row.q10), 1)
        assert cur["q90"] == round(float(row.q90), 1)
        assert cur["flag"] == bool(row.flag)

    def test_unit_bad_subset_and_unit(self, client):
        assert client.get("/api/unit/FD999/1").status_code == 404
        assert client.get("/api/unit/FD001/99999").status_code == 404


class TestAuditImmutabilityOnGet:
    def test_get_endpoints_do_not_write_files(self, client, tmp_path):
        audit_file = tmp_path / "audit.jsonl"
        notes_file = tmp_path / "review_notes.jsonl"

        # Touch empty
        audit_file.write_text("")
        notes_file.write_text("")

        # Issue multiple GET calls
        client.get("/api/fleet?subset=FD001&cycle=50")
        client.get("/api/unit/FD001/1?cycle=50")
        client.get("/api/unit/FD001/1/sensors?cycle=50")
        client.get("/api/review-status?subset=FD001")
        client.get("/api/audit-summary")
        client.get("/api/notes?subset=FD001")

        assert audit_file.read_text() == ""
        assert notes_file.read_text() == ""


class TestDisplayFlagEndpoint:
    def test_display_flag_writes_deduped_record(self, client, tmp_path):
        # Unit 4 cycle 165 is flagged under v1
        df = server.replay_with_flags("FD001", "v1")
        g = df[(df.unit == 4) & (df.cycle == 165)]
        assert bool(g.iloc[0].flag) is True

        r1 = client.post("/api/display-flag", json={"subset": "FD001", "unit": 4, "cycle": 165, "version": "v1"})
        assert r1.status_code == 200
        res1 = r1.get_json()
        assert res1["logged"] is True
        assert res1["flag"] is True

        # Second post dedupes
        r2 = client.post("/api/display-flag", json={"subset": "FD001", "unit": 4, "cycle": 165, "version": "v1"})
        assert r2.status_code == 200

        recs = [e for e in _audit_records(tmp_path) if e.get("record_type") == "flag_shown"]
        assert len(recs) == 1
        assert recs[0]["subset"] == "FD001"
        assert recs[0]["unit"] == 4
        assert recs[0]["cycle"] == 165
        assert recs[0]["policy"] == "v1_q50"

    def test_display_flag_unflagged_writes_nothing(self, client, tmp_path):
        # Unit 4 cycle 1 is unflagged
        r = client.post("/api/display-flag", json={"subset": "FD001", "unit": 4, "cycle": 1, "version": "v1"})
        assert r.status_code == 200
        res = r.get_json()
        assert res["logged"] is False
        assert res["flag"] is False
        assert len(_audit_records(tmp_path)) == 0

    def test_display_flag_invalid_input(self, client):
        assert client.post("/api/display-flag", json={"subset": "FD999", "unit": 1, "cycle": 1}).status_code == 400
        assert client.post("/api/display-flag", json={"subset": "FD001", "unit": "abc", "cycle": 1}).status_code == 400


class TestHandoverRoute:
    def test_handover_with_empty_log(self, client):
        r = client.get("/handover")
        assert r.status_code == 200
        assert b"Generated from the local audit log" in r.data
        assert b"Not a work order" in r.data
        assert b"unseen replay transitions" in r.data.lower()


class TestNotesEndpoint:
    def test_add_and_get_note(self, client):
        r = client.post("/api/note", json={
            "subset": "FD001", "unit": 4, "cycle": 165, "policy": "v1_q50",
            "text": "Checked sensor 11 slope, deviation warrants inspection."
        })
        assert r.status_code == 200
        logged = r.get_json()["logged"]
        assert logged["text"] == "Checked sensor 11 slope, deviation warrants inspection."
        assert "engineer note" in logged["label"]

        # Retrieve notes
        r_get = client.get("/api/notes?subset=FD001&unit=4")
        assert r_get.status_code == 200
        notes = r_get.get_json()["notes"]
        assert len(notes) == 1
        assert notes[0]["text"] == "Checked sensor 11 slope, deviation warrants inspection."


class TestStaticAssets:
    @pytest.mark.parametrize("asset", [
        "/static/css/tokens.css",
        "/static/css/layout.css",
        "/static/css/components.css",
        "/static/css/print.css",
        "/static/js/prefs.js",
        "/static/js/chart.js",
        "/static/js/queue.js",
        "/static/js/evidence.js",
        "/static/js/verdict.js",
        "/static/js/shortcuts.js",
    ])
    def test_static_asset_loads_and_has_no_external_urls(self, client, asset):
        r = client.get(asset)
        assert r.status_code == 200
        content = r.data.decode("utf-8")
        # Ensure no external network references (http:// or https://)
        assert not re.search(r"https?://", content)
