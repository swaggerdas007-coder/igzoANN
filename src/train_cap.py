"""Train a parasitic-capacitance ANN (C_GD or C_GS) for the a-GIZO TFT
equivalent-circuit model of Bahubalindruni et al. (2016).

Mirrors src/train.py (same TFTNet MLP: n_hidden tanh neurons, linear output,
min-max scaled (VG, VD, W, L) inputs) but regresses capacitance in pF instead
of log10|ID|. One ANN is trained per capacitance component; the paper joins the
I_D, C_GD and C_GS ANNs as an equivalent circuit in Verilog-A.

Usage:
    python -m src.train_cap --cap cgd
    python -m src.train_cap --cap cgs
    python -m src.train_cap --cap both      # (default) train and save both
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from src.cap_dataset import load_and_split
from src.model import TFTNet

SEED = 42
N_HIDDEN = 32
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data_cleaned_cv",
                         "merged_cap_dataset.csv")
OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "outputs")


def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


def train_one(cap_type: str, n_hidden: int = N_HIDDEN, seed: int = SEED,
              max_epochs: int = 2500, verbose: bool = True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = os.path.join(OUT_ROOT, f"cap_{cap_type}")
    os.makedirs(out_dir, exist_ok=True)

    train, val, test = load_and_split(DATA_PATH, cap_type=cap_type, seed=seed)
    if verbose:
        print(f"[{cap_type}] train={len(train.df)} val={len(val.df)} "
              f"test={len(test.df)} (grouped by unique operating point)")

    # Standardize the target (capacitance in pF) using train-set stats only.
    y_mean = float(train.c_pf.mean())
    y_std = float(train.c_pf.std())

    def target(split):
        return (split.c_pf - y_mean) / y_std

    Xtr, ytr = to_tensor(train.X), to_tensor(target(train))
    Xva, yva = to_tensor(val.X), to_tensor(target(val))
    Xte = to_tensor(test.X)

    model = TFTNet(n_inputs=4, n_hidden=n_hidden, n_outputs=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    # Cosine annealing over the fixed budget: smooth LR decay that does NOT
    # depend on the (tiny, noisy) validation loss, so the model converges fully
    # on the training curves instead of being frozen by a premature plateau.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-4)
    loss_fn = nn.MSELoss()

    batch_size = 32
    n_train = Xtr.shape[0]

    # The C-V curves are smooth and low-noise, but only 4 geometries were
    # measured, so the 25-point validation set is too small to be a reliable
    # early-stopping signal (best-val models are badly under-trained and predict
    # near-linear ramps instead of the sigmoidal floor->rise->plateau shape).
    # We therefore train a fixed epoch budget with plateau LR annealing and keep
    # the final model; validation loss is still tracked, only for the scheduler
    # and reporting.
    val_loss = float("nan")
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx], ytr[idx].unsqueeze(1)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n_train

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), yva.unsqueeze(1)).item()

        if verbose and (epoch % 200 == 0 or epoch == 1):
            print(f"[{cap_type}] epoch {epoch:4d}  train_mse={epoch_loss:.5f}  val_mse={val_loss:.5f}")

    model.eval()
    final_val_mse = float(val_loss)

    # ---- Evaluation on held-out test set (capacitance in pF) ----
    with torch.no_grad():
        c_pred = model(Xte).squeeze(1).numpy() * y_std + y_mean
    c_true = test.c_pf

    rmse = float(np.sqrt(np.mean((c_pred - c_true) ** 2)))
    mae = float(np.mean(np.abs(c_pred - c_true)))
    ss_res = float(np.sum((c_true - c_pred) ** 2))
    ss_tot = float(np.sum((c_true - c_true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    # Mean absolute relative error (Eq. 7 in the paper). The full MARE is
    # dominated by the tiny (~0.1 pF) sub-threshold/off-state points where a
    # small absolute error is a huge relative one, so also report it on the
    # on-state (accumulation) region where the paper evaluates accuracy.
    mare_full = float(np.mean(np.abs(c_true - c_pred) / np.clip(np.abs(c_true), 1e-3, None)) * 100)
    on_mask = c_true >= 0.5  # pF; above the off-state floor
    mare_on = (float(np.mean(np.abs(c_true[on_mask] - c_pred[on_mask]) / c_true[on_mask]) * 100)
               if on_mask.sum() else float("nan"))

    metrics = {
        "cap_type": cap_type,
        "n_hidden": n_hidden,
        "hidden_activation": "tanh",
        "output_activation": "linear",
        "target": "capacitance_pF",
        "train_rows": len(train.df),
        "val_rows": len(val.df),
        "test_rows": len(test.df),
        "geometries": sorted({(int(w), int(l)) for w, l in zip(test.df.W, test.df.L)}),
        "target_standardization": {"y_mean_pF": y_mean, "y_std_pF": y_std},
        "final_val_mse_standardized": final_val_mse,
        "test_rmse_pF": rmse,
        "test_mae_pF": mae,
        "test_r2": r2,
        "test_MARE_percent_full": mare_full,
        "test_MARE_percent_onstate_C_ge_0p5pF": mare_on,
        "note": "Baseline: all C measurements are at VD=0V, so the model learns "
                "C(VG,W,L) only and does not yet capture C-VDS dependence. Only 4 "
                "geometries measured; random point split reports interpolation "
                "accuracy within measured geometries.",
    }
    if verbose:
        print(json.dumps({k: metrics[k] for k in
                          ("cap_type", "test_rmse_pF", "test_mae_pF", "test_r2",
                           "test_MARE_percent_full", "test_MARE_percent_onstate_C_ge_0p5pF")},
                         indent=2))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), os.path.join(out_dir, "model_weights.pt"))

    # Export explicit wh, bh, wo, bo + pre/post-processing (paper's notation),
    # matching src/train.py's weights.json so the cap ANNs can be ported to the
    # Verilog-A equivalent circuit alongside the I_D ANN.
    sd = model.state_dict()
    weights = {
        "element": cap_type.upper(),
        "architecture": {"n_inputs": 4, "n_hidden": n_hidden, "n_outputs": 1,
                         "input_order": ["VG", "VD", "W", "L"]},
        "input_scaling_minmax": {
            "VG": [-5.0, 5.0], "VD": [0.0, 5.0], "W": [5.0, 160.0], "L": [5.0, 20.0],
        },
        "wh": sd["hidden.weight"].numpy().tolist(),
        "bh": sd["hidden.bias"].numpy().tolist(),
        "wo": sd["output.weight"].numpy().tolist(),
        "bo": sd["output.bias"].numpy().tolist(),
        "target_transform": {
            "definition": "network_output = (C_pF - y_mean) / y_std",
            "y_mean_pF": y_mean,
            "y_std_pF": y_std,
            "recover_C_pF": "C_pF = network_output * y_std + y_mean",
            "recover_C_F": "C_F = C_pF * 1e-12",
        },
    }
    with open(os.path.join(out_dir, "weights.json"), "w") as f:
        json.dump(weights, f, indent=2)

    np.savez(os.path.join(out_dir, "test_predictions.npz"),
             X=test.X, c_true=c_true, c_pred=c_pred)

    if verbose:
        print(f"[{cap_type}] saved outputs to {out_dir}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", choices=["cgd", "cgs", "both"], default="both")
    args = ap.parse_args()

    caps = ["cgd", "cgs"] if args.cap == "both" else [args.cap]
    all_metrics = {}
    for cap in caps:
        all_metrics[cap] = train_one(cap)
        print()

    if len(caps) > 1:
        with open(os.path.join(OUT_ROOT, "cap_metrics_summary.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)
        print("Summary:")
        for cap, m in all_metrics.items():
            print(f"  {cap}: R2={m['test_r2']:.4f}  RMSE={m['test_rmse_pF']:.4f} pF  "
                  f"MARE(on-state)={m['test_MARE_percent_onstate_C_ge_0p5pF']:.2f}%")


if __name__ == "__main__":
    main()
