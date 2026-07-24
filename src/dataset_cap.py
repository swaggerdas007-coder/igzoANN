"""Data loading, preprocessing and train/val/test splitting for the CGD/CGS
capacitance ANNs, mirroring src/dataset.py's conventions but for
data_cv_cleaned/merged_cv_dataset.csv.

Inputs:  VG, W, L        (no VDS -- the dataset is a fixed-VDS=0 slice,
                           see data_cv_cleaned/README.md)
Targets: CGD, CGS         (farads; regressed directly, standardized to zero
                           mean/unit variance -- capacitance here spans
                           only ~2 decades and is always positive/smooth,
                           unlike ID, so no log transform is needed)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = ["VG", "W", "L"]
FEATURE_BOUNDS = {
    "VG": (-3.0, 5.0),
    "W": (20.0, 160.0),
    "L": (15.0, 20.0),
}


@dataclass
class Split:
    X: np.ndarray
    y: dict  # {"CGD": array, "CGS": array}
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
        y = {t: sub[t].to_numpy(dtype=np.float64) for t in ("CGD", "CGS")}
        return Split(X=X, y=y, df=sub)

    return make_split(train_groups), make_split(val_groups), make_split(test_groups)
