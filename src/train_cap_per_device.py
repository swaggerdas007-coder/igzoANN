"""Train per-device CGD/CGS ANNs: one tiny 1-input (VG only) MLP per (W, L)
geometry, instead of a single 3-input (VG, W, L) network.

Why: outputs_cap/plots/cgd_cgs_ann_fit.png (from the combined VG,W,L model,
see src/train_cap.py) shows the combined model visibly misplacing each
device's turn-on knee -- with only 4 (W, L) points to interpolate a 2D
threshold-location surface from, the network cannot learn where each
device's transition sits. This is the same failure mode the existing ID
pipeline already solved for L with tft_ann_static_per_L.va: "each
subnetwork only ever had to fit its own geometry's data -- no
cross-geometry interpolation, no geometry-imbalance in the training loss."
We do the same here, one network per (W, L) point (all 4 are measured
points, so there is no interpolation claim being made or needed).

Unlike the ID/combined-capacitance models, these per-device networks are
NOT meant to generalize to unseen (VG, W, L) combinations across geometry
-- with exactly one measured device per (W, L) point, no train/val/test
split across geometry is even possible. Their job is purely to be a smooth,
differentiable stand-in for that one device's own measured VG sweep (41
points), for use inside the Verilog-A capacitive-current model. So each
network is fit to ALL of its device's points (no held-out split), with
several random restarts to avoid a bad local minimum on the sharp turn-on
knee, and judged by fit quality (R^2, RMSE over all points), not
generalization error.

Usage:
    python -m src.train_cap_per_device
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.model import TFTNet

SEED = 42
N_HIDDEN = 8
N_RESTARTS = 8
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data_cv_cleaned", "merged_cv_dataset.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs_cap", "per_device")

VG_LO, VG_HI = -3.0, 5.0


def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


def fit_once(X, y, seed):
    torch.manual_seed(seed)
    model = TFTNet(n_inputs=1, n_hidden=N_HIDDEN, n_outputs=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100)
    loss_fn = nn.MSELoss()

    best_loss, best_state, no_improve = float("inf"), None, 0
    for epoch in range(1, 4001):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        if loss.item() < best_loss - 1e-9:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 400:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_loss


def train_one(vg, c, seed):
    vg_s = (vg - VG_LO) / (VG_HI - VG_LO)
    c_mean, c_std = c.mean(), c.std()
    X = to_tensor(vg_s).unsqueeze(1)
    y = to_tensor((c - c_mean) / c_std).unsqueeze(1)

    best_model, best_loss = None, float("inf")
    for r in range(N_RESTARTS):
        model, loss = fit_once(X, y, seed=seed * 100 + r)
        if loss < best_loss:
            best_loss, best_model = loss, model

    with torch.no_grad():
        c_pred = best_model(X).squeeze(1).numpy() * c_std + c_mean
    rmse = float(np.sqrt(np.mean((c_pred - c) ** 2)))
    mare = float(np.mean(np.abs(c - c_pred) / np.maximum(np.abs(c), 1e-15)) * 100)
    ss_res = np.sum((c - c_pred) ** 2)
    ss_tot = np.sum((c - c.mean()) ** 2)
    r2_all = float(1 - ss_res / ss_tot)

    sd = best_model.state_dict()
    weights = {
        "architecture": {"n_inputs": 1, "n_hidden": N_HIDDEN, "n_outputs": 1, "input_order": ["VG"]},
        "input_scaling_minmax": {"VG": [VG_LO, VG_HI]},
        "wh": sd["hidden.weight"].numpy().tolist(),
        "bh": sd["hidden.bias"].numpy().tolist(),
        "wo": sd["output.weight"].numpy().tolist(),
        "bo": sd["output.bias"].numpy().tolist(),
        "target_transform": {"mean_F": float(c_mean), "std_F": float(c_std)},
    }
    metrics = {
        "n_hidden": N_HIDDEN, "n_points": len(vg), "n_restarts": N_RESTARTS,
        "fit_rmse_F": rmse, "fit_MARE_percent": mare, "r2_all_points": r2_all,
    }
    return weights, metrics, best_model


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    devices = sorted(df[["W", "L"]].drop_duplicates().itertuples(index=False, name=None))

    all_metrics = {}
    for dev_i, (W, L) in enumerate(devices):
        sub = df[(df.W == W) & (df.L == L)].sort_values("VG").reset_index(drop=True)
        vg = sub["VG"].to_numpy(dtype=np.float64)
        for tgt_i, target in enumerate(("CGD", "CGS")):
            c = sub[target].to_numpy(dtype=np.float64)
            weights, metrics, model = train_one(vg, c, seed=SEED + 10 * dev_i + tgt_i)
            key = f"{target.lower()}_W{W}_L{L}"
            with open(os.path.join(OUT_DIR, f"weights_{key}.json"), "w") as f:
                json.dump(weights, f, indent=2)
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"model_{key}.pt"))
            all_metrics[key] = metrics
            print(f"{key}: r2_all={metrics['r2_all_points']:.4f} "
                  f"fit_MARE={metrics['fit_MARE_percent']:.1f}% fit_rmse={metrics['fit_rmse_F']:.3e} F")

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(os.path.join(OUT_DIR, "devices.json"), "w") as f:
        json.dump([[int(w_), int(l_)] for w_, l_ in devices], f)
    print(f"Saved {len(devices)*2} per-device models to {OUT_DIR}")


if __name__ == "__main__":
    main()
