"""Data loading, preprocessing and train/val/test splitting for the a-GIZO TFT
parasitic-capacitance datasets (C_GD and C_GS), following the ANN equivalent-
circuit methodology of Bahubalindruni et al., "InGaZnO TFT behavioral model for
IC design", Analog Integr Circ Sig Process (2016) 87:73-80.

Inputs:  VG, VD, W, L   (same 4-input interface as the I_D ANN in dataset.py,
                         so all three ANNs are pin-compatible for the EC)
Target:  C              (capacitance, in pF)

Unlike the drain current, the parasitic capacitance is smooth, strictly
positive and spans well under two decades (~0.1-6.3 pF), so we regress the
capacitance directly (in pF) rather than its logarithm.

Baseline note: the current C_GD/C_GS measurements are all at VD = 0 V, so VD is
constant across the dataset (the network simply learns to ignore it until
VD-resolved data is added). Geometry coverage is only 4 devices, so a random
point split reports interpolation accuracy within the measured geometries, not
extrapolation to unseen (W, L).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Reuse the exact input feature bounds of the I_D model so the scaling (and the
# Verilog-A EC that consumes all three ANNs) is identical across elements.
from src.dataset import FEATURES, FEATURE_BOUNDS, scale_features


@dataclass
class CapSplit:
    X: np.ndarray
    c_pf: np.ndarray      # target capacitance, pF
    df: pd.DataFrame


def load_and_split(csv_path: str, cap_type: str, seed: int = 42,
                   train_frac=0.7, val_frac=0.15):
    """Load one capacitance component ('cgd' or 'cgs') and split by unique
    operating point (VG, VD, W, L) so repeated measurements of the same point
    never straddle the split."""
    df = pd.read_csv(csv_path)
    df = df[df["cap_type"] == cap_type].drop_duplicates().reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows for cap_type={cap_type!r} in {csv_path}")

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
        c_pf = sub["C_pF"].to_numpy(dtype=np.float32)
        return CapSplit(X=X, c_pf=c_pf, df=sub)

    return make_split(train_groups), make_split(val_groups), make_split(test_groups)
