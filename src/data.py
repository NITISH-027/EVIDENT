"""Data loading and schema validation for C-MAPSS.

Schema per row: unit_id, cycle, 3 operating settings, 21 sensor channels.
Units failing schema validation are REJECTED with reasons and counts reported -
never silently imputed.
"""
import os
import numpy as np
import pandas as pd
from . import config

COLUMNS = (["unit", "cycle"] +
           [f"op{i}" for i in range(1, config.N_SETTINGS + 1)] +
           [f"s{i}" for i in range(1, config.N_SENSORS + 1)])
CHANNELS = COLUMNS[2:]


def load_raw(subset, split):
    """split in {'train','test'}. Returns raw DataFrame (unvalidated)."""
    path = os.path.join(config.DATA_DIR, f"{split}_{subset}.txt")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    return df


def validate_schema(df):
    """Reject units failing schema rules. Returns (clean_df, report dict).

    Rules: correct column count/dtypes (enforced at parse), monotone increasing
    cycles per unit starting at 1 with no gaps, no nulls, at least 2 cycles.
    """
    rejected = {}
    if df["unit"].isna().any():
        # e.g. a header row coerced to NaN: no unit id to key on
        rejected["(rows without a numeric unit id)"] = "non-numeric or missing unit id"
    if df.isnull().any().any():
        bad = df[df.isnull().any(axis=1)]["unit"].dropna().unique()
        for u in bad:
            rejected[int(u)] = "null values in row(s)"
    for u, g in df.groupby("unit"):
        if int(u) in rejected:
            continue
        cyc = g["cycle"].to_numpy()
        if len(cyc) < 2:
            rejected[int(u)] = "fewer than 2 cycles"
        elif cyc[0] != 1 or not np.all(np.diff(cyc) == 1):
            rejected[int(u)] = "non-monotone or gapped cycle index"
    rejected_ids = [k for k in rejected if isinstance(k, int)]
    clean = df[df["unit"].notna() & ~df["unit"].isin(rejected_ids)].copy()
    report = {
        "rows_in": int(len(df)), "units_in": int(df["unit"].nunique()),
        "units_rejected": {k: v for k, v in sorted(rejected.items())},
        "units_out": int(clean["unit"].nunique()), "rows_out": int(len(clean)),
    }
    return clean, report


def load_subset(subset, split):
    df = load_raw(subset, split)
    return validate_schema(df)


def load_test_endpoints(subset):
    """Published endpoint RUL per test unit (scoring only, never fitting)."""
    path = os.path.join(config.DATA_DIR, f"RUL_{subset}.txt")
    return np.loadtxt(path).astype(float)


def true_rul_train(df):
    """Uncapped true RUL for training trajectories: lifetime - cycle.
    All C-MAPSS training units run to failure; no censoring."""
    lifetimes = df.groupby("unit")["cycle"].max().rename("lifetime")
    out = df.merge(lifetimes, on="unit")
    out["true_rul"] = out["lifetime"] - out["cycle"]
    return out
