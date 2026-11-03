"""EVIDENT test suite. Reads frozen artifacts only - never retrains, never
re-scores the official test, never tunes. Recomputation tests re-derive
metrics from STORED predictions to verify the stored numbers are consistent;
that is analysis of frozen outputs, not new scoring."""
import os, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "data_raw")
NEEDED = [f"{sp}_{s}.txt" for s in ("FD001", "FD002", "FD003", "FD004") for sp in ("train", "test")] + \
         [f"RUL_{s}.txt" for s in ("FD001", "FD002", "FD003", "FD004")]
DATA_AVAILABLE = all(os.path.exists(os.path.join(DATA_DIR, f)) for f in NEEDED)

needs_data = pytest.mark.skipif(not DATA_AVAILABLE,
                                reason="data_raw not present (run bootstrap.py first)")
