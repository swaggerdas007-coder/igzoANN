"""Data loading, preprocessing and train/val/test splitting for the a-GIZO TFT
drain-current dataset, following the static ANN modeling methodology of
Bahubalindruni et al., "a-GIZO TFT neural modeling, circuit simulation and
validation", Solid-State Electronics 105 (2015) 30-36.

Inputs:  VG, VD, W, L   (gate voltage, drain voltage, channel width, channel length)
Target:  ID             (drain current)

The raw current spans ~12 decades (leakage ~1e-14 A to on-state ~1e-2 A) and the
off-state samples carry sign noise from the simulator's numerical floor (not a
real bipolar current, since VD >= 0 always). Directly regressing raw ID would be
dominated by the on-state and would blow up any relative-error metric near zero.
Instead we regress log10(|ID|); the dataset already ships this as `log_ID`.

Each physical operating point (VG, VD, W, L) is repeated several times in the
csv with different noise realizations. Splits are therefore made by grouping
on the unique operating point, not by row, so no point leaks its repeats
across train/val/test.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = ["VG", "VD", "W", "L"]
FEATURE_BOUNDS = {
    "VG": (-5.0, 5.0),
    "VD": (0.0, 5.0),
    "W": (5.0, 160.0),
    "L": (5.0, 20.0),
}


@dataclass
class Split:
    X: np.ndarray
    log_id: np.ndarray
    id_raw: np.ndarray
    df: pd.DataFrame


def scale_features(df: pd.DataFrame) -> np.ndarray:
    cols = []
    for name in FEATURES:
        lo, hi = FEATURE_BOUNDS[name]
        cols.append(((df[name].to_numpy() - lo) / (hi - lo)).astype(np.float32))
    return np.stack(cols, axis=1)


def load_and_split(csv_path: str, seed: int = 42, train_frac=0.7, val_frac=0.15):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates().reset_index(drop=True)

    group_key = df[FEATURES].apply(lambda r: tuple(r), axis=1)
    df["_group"] = group_key.astype("category").cat.codes

    rng = np.random.default_rng(seed)
    unique_groups = df["_group"].unique()
    rng.shuffle(unique_groups)

    n = len(unique_groups)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train:n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val:])

    def make_split(groups):
        sub = df[df["_group"].isin(groups)].reset_index(drop=True)
        X = scale_features(sub)
        log_id = sub["log_ID"].to_numpy(dtype=np.float32)
        id_raw = sub["ID"].to_numpy(dtype=np.float64)
        return Split(X=X, log_id=log_id, id_raw=id_raw, df=sub)

    return make_split(train_groups), make_split(val_groups), make_split(test_groups)
