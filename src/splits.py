"""Whole-unit 70/15/15 fit/tune/replay-holdout splits.
Unit ID lists and the seed are frozen in results/splits.json BEFORE training.
"""
import json
import os
import numpy as np
from . import config


def make_splits(units_by_subset):
    rng = np.random.default_rng(config.SEED)
    splits = {"seed": config.SEED, "fractions": config.SPLIT_FRACTIONS,
              "frozen_before_training": True, "subsets": {}}
    for subset, units in units_by_subset.items():
        units = np.array(sorted(units))
        rng.shuffle(units)
        n = len(units)
        n_fit = int(round(n * config.SPLIT_FRACTIONS[0]))
        n_tune = int(round(n * config.SPLIT_FRACTIONS[1]))
        splits["subsets"][subset] = {
            "fit": sorted(units[:n_fit].tolist()),
            "tune": sorted(units[n_fit:n_fit + n_tune].tolist()),
            "replay": sorted(units[n_fit + n_tune:].tolist()),
        }
    return splits


def save_splits(splits, path=None):
    path = path or os.path.join(config.RESULTS_DIR, "splits.json")
    with open(path, "w") as f:
        json.dump(splits, f, indent=1)
    return path
