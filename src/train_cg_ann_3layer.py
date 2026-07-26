"""Train three-hidden-layer ANNs (10 + 5 + 5 tanh neurons) for the a-IGZO TFT
parasitic capacitances C_GD and C_GS, saving everything under trained_Cg_ANN_3layer/.

A separate ANN is trained per component, using a deeper 4 -> 10 -> 5 -> 5 -> 1 MLP
(src/model.py::TFTNet3) to better capture fine features on small-geometry devices
(W=20-40 µm) where the 2-layer variant overshoots and wiggles.

Outputs (per component <cap> in {cgd, cgs}):
    trained_Cg_ANN_3layer/<cap>/model_weights.pt      PyTorch state dict
    trained_Cg_ANN_3layer/<cap>/weights.json          w1,b1,w2,b2,w3,b3,wo,bo + scaling
    trained_Cg_ANN_3layer/<cap>/metrics.json          all test metrics
    trained_Cg_ANN_3layer/<cap>/test_predictions.npz
    trained_Cg_ANN_3layer/<cap>/plots/scatter_C.png
    trained_Cg_ANN_3layer/<cap>/plots/C_vg_curves.png
    trained_Cg_ANN_3layer/metrics_summary.json

Run:
    python -m src.train_cg_ann_3layer
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.dataset import FEATURE_BOUNDS
from src.cap_dataset import load_and_split
from src.model import TFTNet3

SEED = 42
N_HIDDEN1 = 10
N_HIDDEN2 = 5
N_HIDDEN3 = 5
MAX_EPOCHS = 2500
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data_cleaned_cv",
                         "merged_cap_dataset.csv")
OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "trained_Cg_ANN_3layer")

BG = "#f7f7f5"
FG = "#1f1f1f"
MEASURED = "#4c72b0"
MODEL = "#c44e52"


def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


def style_axes(ax):
    ax.set_facecolor(BG)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, alpha=0.25)


def scale_point(vg, vd, w, l):
    def s(name, val):
        lo, hi = FEATURE_BOUNDS[name]
        return (val - lo) / (hi - lo)
    return np.array([s("VG", vg), s("VD", vd), s("W", w), s("L", l)], dtype=np.float32)


def predict_c(model, y_mean, y_std, vg, vd, w, l):
    X = np.stack([scale_point(a, b, c, d) for a, b, c, d in zip(vg, vd, w, l)])
    with torch.no_grad():
        pred = model(to_tensor(X)).squeeze(1).numpy()
    return pred * y_std + y_mean


def metrics_block(c_true, c_pred):
    rmse = float(np.sqrt(np.mean((c_pred - c_true) ** 2)))
    mae = float(np.mean(np.abs(c_pred - c_true)))
    ss_res = float(np.sum((c_true - c_pred) ** 2))
    ss_tot = float(np.sum((c_true - c_true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mare_full = float(np.mean(np.abs(c_true - c_pred) / np.clip(np.abs(c_true), 1e-3, None)) * 100)
    on = c_true >= 0.5
    mare_on = (float(np.mean(np.abs(c_true[on] - c_pred[on]) / c_true[on]) * 100)
               if on.sum() else float("nan"))
    return dict(rmse_pF=rmse, mae_pF=mae, r2=r2,
                MARE_percent_full=mare_full,
                MARE_percent_onstate_C_ge_0p5pF=mare_on)


def scatter_plot(cap, c_true, c_pred, m, path):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    fig.patch.set_facecolor(BG)
    ax.scatter(c_true, c_pred, s=18, alpha=0.6, color=MEASURED, edgecolors="none")
    hi = float(max(c_true.max(), c_pred.max())) * 1.05
    ax.plot([0, hi], [0, hi], color=FG, lw=1, ls="--", alpha=0.6, label="y = x")
    ax.set_xlim(0, hi)
    ax.set_ylim(min(0, float(c_pred.min()) * 1.05), hi)
    ax.set_xlabel(r"measured $C_{%s}$ (pF)" % cap[1:].upper())
    ax.set_ylabel(r"model $C_{%s}$ (pF)" % cap[1:].upper())
    ax.set_title(f"{cap.upper()} 3-layer ANN test set  "
                 f"(R$^2$={m['test']['r2']:.3f}, RMSE={m['test']['rmse_pF']:.2f} pF)")
    style_axes(ax)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cv_curves(cap, model, y_mean, y_std, path):
    df = pd.read_csv(DATA_PATH)
    df = df[df.cap_type == cap]
    geoms = sorted(set(zip(df.W, df.L)))
    ncol = len(geoms)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.2), dpi=140)
    fig.patch.set_facecolor(BG)
    if ncol == 1:
        axes = [axes]
    for ax, (w, l) in zip(axes, geoms):
        sub = df[(df.W == w) & (df.L == l)].sort_values("VG")
        vd0 = float(sub.VD.iloc[0])
        ax.scatter(sub.VG, sub.C_pF, s=16, color=MEASURED, alpha=0.7, label="measured")
        vg_fine = np.linspace(sub.VG.min(), sub.VG.max(), 200)
        c_fine = predict_c(model, y_mean, y_std, vg_fine, np.full_like(vg_fine, vd0),
                           np.full_like(vg_fine, w), np.full_like(vg_fine, l))
        ax.plot(vg_fine, c_fine, color=MODEL, lw=2, label="ANN")
        ax.set_xlabel("VG (V)")
        ax.set_ylabel(r"$C_{%s}$ (pF)" % cap[1:].upper())
        ax.set_title(f"W={w:g}, L={l:g} um (VD={vd0:g}V)")
        style_axes(ax)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{cap.upper()} 3-layer ANN vs measurement (dots=measured, line=model)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def train_one(cap_type, verbose=True):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    out_dir = os.path.join(OUT_ROOT, cap_type)
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    train, val, test = load_and_split(DATA_PATH, cap_type=cap_type, seed=SEED)
    if verbose:
        print(f"[{cap_type}] train={len(train.df)} val={len(val.df)} test={len(test.df)}")

    y_mean = float(train.c_pf.mean())
    y_std = float(train.c_pf.std())

    def tgt(split):
        return (split.c_pf - y_mean) / y_std

    Xtr, ytr = to_tensor(train.X), to_tensor(tgt(train))
    Xva, yva = to_tensor(val.X), to_tensor(tgt(val))
    Xte = to_tensor(test.X)

    model = TFTNet3(n_inputs=4, n_hidden1=N_HIDDEN1, n_hidden2=N_HIDDEN2,
                    n_hidden3=N_HIDDEN3, n_outputs=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-4)
    loss_fn = nn.MSELoss()

    batch_size = 32
    n_train = Xtr.shape[0]
    val_loss = float("nan")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx].unsqueeze(1)).backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), yva.unsqueeze(1)).item()
        if verbose and (epoch % 500 == 0 or epoch == 1):
            with torch.no_grad():
                tr_loss = loss_fn(model(Xtr), ytr.unsqueeze(1)).item()
            print(f"[{cap_type}] epoch {epoch:4d}  train_mse={tr_loss:.5f}  val_mse={val_loss:.5f}")

    model.eval()

    def eval_split(split):
        with torch.no_grad():
            pred = model(to_tensor(split.X)).squeeze(1).numpy() * y_std + y_mean
        return metrics_block(split.c_pf, pred), pred

    train_m, _ = eval_split(train)
    val_m, _ = eval_split(val)
    test_m, c_pred = eval_split(test)

    metrics = {
        "cap_type": cap_type,
        "architecture": {
            "layers": "4 -> 10 (tanh) -> 5 (tanh) -> 5 (tanh) -> 1 (linear)",
            "n_inputs": 4, "n_hidden1": N_HIDDEN1, "n_hidden2": N_HIDDEN2,
            "n_hidden3": N_HIDDEN3, "n_outputs": 1,
            "input_order": ["VG", "VD", "W", "L"],
        },
        "target": "capacitance_pF",
        "rows": {"train": len(train.df), "val": len(val.df), "test": len(test.df)},
        "geometries": sorted({(int(w), int(l)) for w, l in zip(test.df.W, test.df.L)}),
        "target_standardization": {"y_mean_pF": y_mean, "y_std_pF": y_std},
        "final_val_mse_standardized": float(val_loss),
        "train": train_m,
        "val": val_m,
        "test": test_m,
        "note": "Baseline: all C measurements are at VD=0V, so the model learns "
                "C(VG,W,L) only and does not capture C-VDS dependence. Only 4 "
                "geometries; random point split reports interpolation accuracy.",
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), os.path.join(out_dir, "model_weights.pt"))

    sd = model.state_dict()
    weights = {
        "element": cap_type.upper(),
        "architecture": metrics["architecture"],
        "input_scaling_minmax": {
            "VG": [-5.0, 5.0], "VD": [0.0, 5.0], "W": [5.0, 160.0], "L": [5.0, 20.0],
        },
        "w1": sd["hidden1.weight"].numpy().tolist(),   # [10, 4]
        "b1": sd["hidden1.bias"].numpy().tolist(),     # [10]
        "w2": sd["hidden2.weight"].numpy().tolist(),   # [5, 10]
        "b2": sd["hidden2.bias"].numpy().tolist(),     # [5]
        "w3": sd["hidden3.weight"].numpy().tolist(),   # [5, 5]
        "b3": sd["hidden3.bias"].numpy().tolist(),     # [5]
        "wo": sd["output.weight"].numpy().tolist(),    # [1, 5]
        "bo": sd["output.bias"].numpy().tolist(),      # [1]
        "target_transform": {
            "definition": "network_output = (C_pF - y_mean) / y_std",
            "y_mean_pF": y_mean, "y_std_pF": y_std,
            "recover_C_pF": "C_pF = network_output * y_std + y_mean",
            "recover_C_F": "C_F = C_pF * 1e-12",
        },
    }
    with open(os.path.join(out_dir, "weights.json"), "w") as f:
        json.dump(weights, f, indent=2)

    np.savez(os.path.join(out_dir, "test_predictions.npz"),
             X=test.X, c_true=test.c_pf, c_pred=c_pred)

    scatter_plot(cap_type, test.c_pf, c_pred, metrics, os.path.join(plot_dir, "scatter_C.png"))
    cv_curves(cap_type, model, y_mean, y_std, os.path.join(plot_dir, "C_vg_curves.png"))

    if verbose:
        print(f"[{cap_type}] test: R2={test_m['r2']:.4f}  RMSE={test_m['rmse_pF']:.4f} pF  "
              f"MAE={test_m['mae_pF']:.4f} pF  MARE_on={test_m['MARE_percent_onstate_C_ge_0p5pF']:.2f}%")
        print(f"[{cap_type}] saved to {out_dir}")
    return metrics


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    summary = {}
    for cap in ("cgd", "cgs"):
        summary[cap] = train_one(cap)
        print()
    with open(os.path.join(OUT_ROOT, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("=== Summary (3-layer 10+5+5 ANNs) ===")
    for cap, m in summary.items():
        t = m["test"]
        print(f"  {cap.upper()}: test R2={t['r2']:.4f}  RMSE={t['rmse_pF']:.4f} pF  "
              f"MAE={t['mae_pF']:.4f} pF  MARE_on={t['MARE_percent_onstate_C_ge_0p5pF']:.2f}%")


if __name__ == "__main__":
    main()
