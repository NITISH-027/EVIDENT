"""EVIDENT (working name) - frozen protocol constants.
These are the proposed engineering defaults from the approved protocol,
fixed in writing here BEFORE any training. Do not edit after training starts;
any change requires a full re-evaluation from scratch.
"""
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_raw")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
PRIMARY_SUBSET = "FD001"

WINDOW = 30            # trailing feature window, cycles
HORIZON_N = 30         # warning horizon: flag when predicted RUL <= N
QUANTILES = (0.1, 0.5, 0.9)   # q0.5 = point estimate; [q0.1, q0.9] = 80% band
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)   # fit / tune / replay-holdout, whole-unit
SEED = 20261102
BOOTSTRAP_RESAMPLES = 2000
CI_LEVEL = 0.95
RUL_BANDS = [(0, 30), (31, 75), (76, 130), (131, None)]  # error profile bands; last is >130

N_SETTINGS = 3
N_SENSORS = 21
