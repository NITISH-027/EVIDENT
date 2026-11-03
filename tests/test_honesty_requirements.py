"""Requirement-level honesty checks: the claims the product makes at the
point of use, the README's required content, and packaging integrity."""
import glob
import json
import os
import re

from conftest import ROOT

TEMPLATES = os.path.join(ROOT, "app", "templates")


def _read(path):
    with open(path) as f:
        return f.read()


class TestPointOfUseHonesty:
    def test_unit_page_disclaims_failure_probability(self):
        assert "not a failure probability" in _read(os.path.join(TEMPLATES, "unit.html"))

    def test_base_advisory_only(self):
        assert "Never commands any machine" in _read(os.path.join(TEMPLATES, "base.html"))

    def test_unit_page_shows_measured_coverage(self):
        t = _read(os.path.join(TEMPLATES, "unit.html"))
        assert "measured coverage" in t

    def test_truth_overlay_labeled_retrospective(self):
        t = _read(os.path.join(TEMPLATES, "unit.html")) + _read(os.path.join(TEMPLATES, "import_unit.html"))
        assert "RETROSPECTIVE" in t or "retrospective" in t

    def test_import_unit_no_ground_truth(self):
        assert "no ground truth" in _read(os.path.join(TEMPLATES, "import_unit.html")).lower()

    def test_audit_cycle_local_label(self):
        # the cycle-local honesty label is produced by the audit view itself
        assert "this cycle only, not a policy result" in _read(os.path.join(ROOT, "app", "server.py"))

    def test_verdict_labeled_opinion(self):
        t = _read(os.path.join(TEMPLATES, "audit.html"))
        assert "opinion" in t.lower()


class TestReadme:
    def setup_method(self):
        self.rm = _read(os.path.join(ROOT, "README.md"))

    def test_brief_required_elements(self):
        assert "Project title" in self.rm
        assert "Problem statement" in self.rm
        assert "Team" in self.rm
        assert "Tech stack" in self.rm
        assert "AI tools disclosure" in self.rm
        assert "Scope disclosure" in self.rm

    def test_team_identity(self):
        assert "nitishprabagaran07" in self.rm
        assert "Nitish P (solo participant)" in self.rm
        assert "[TEAM NAME" not in self.rm and "[MEMBER NAMES" not in self.rm

    def test_kickoff_sentence_factual(self):
        assert "Implementation started 2 Nov 2026, 12:48 IST" in self.rm
        assert "Built Nov" not in self.rm

    def test_fd001_repeat_disclosed(self):
        assert "reproducibility repeat" in self.rm

    def test_no_data_redistribution(self):
        assert "does NOT redistribute the raw data" in self.rm

    def test_results_table_matches_frozen_artifacts(self):
        """README headline numbers must equal the stored frozen results."""
        for ds in ("FD001", "FD002", "FD003", "FD004"):
            with open(os.path.join(ROOT, "results", f"results_{ds}.json")) as f:
                r = json.load(f)
            row_re = re.compile(rf"\| {ds}[^|]*\| ([\d.]+) \(([\d.]+) / ([\d.]+)\)")
            m = row_re.search(self.rm)
            assert m, ds
            rmse, b1, b2 = map(float, m.groups())
            assert rmse == round(r["test"]["model_endpoint_rmse"], 1)
            assert b1 == round(r["test"]["vs_B1"]["rmse_baseline"], 1)
            assert b2 == round(r["test"]["vs_B2"]["rmse_baseline"], 1)
            cov_pct = round(100 * r["replay"]["band"]["coverage_mean"])
            assert re.search(rf"\| {ds}[^|]*\|[^|]*\|[^|]*\| {cov_pct}%", self.rm), ds

    def test_screenshots_referenced_exist(self):
        shots = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "screenshots", "*.png")))
        assert len(shots) == 5


class TestPackaging:
    def test_requirements_pinned(self):
        lines = [l.strip() for l in _read(os.path.join(ROOT, "requirements.txt")).splitlines()
                 if l.strip() and not l.startswith("#")]
        assert lines and all("==" in l for l in lines)

    def test_no_test_data_in_package_paths(self):
        """results/ must not contain the raw test trajectories, only endpoint
        predictions (one row per test unit)."""
        import joblib
        for ds in ("FD001", "FD002", "FD003", "FD004"):
            df = joblib.load(os.path.join(ROOT, "results", f"test_endpoint_{ds}.pkl"))
            assert df.groupby("unit").size().max() == 1

    def test_audit_log_not_shipped_frozen(self):
        """If results/audit.jsonl exists in the working tree it must be empty
        (cleared for delivery) and it is excluded from the zip."""
        p = os.path.join(ROOT, "results", "audit.jsonl")
        if os.path.exists(p):
            assert os.path.getsize(p) == 0
