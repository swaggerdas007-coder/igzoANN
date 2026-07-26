"""Train tuned Cgd/Cgs ANNs with best hyperparameters from sweep.
Updates trained_Cg_ANN/ with improved models.

Run:
    python -m src.train_tuned_cg_ann
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
from src.model import TFTNet2

SEED = 42
MAX_EPOCHS = 2500
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data_cleaned_cv",
                         "merged_cap_dataset.csv")
OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "trained_Cg_ANN")

BG = "#f7f7f5"
FG = "#1f1f1f"
MEASURED = "#4c72b0"
MODEL = "#c44e52"

# Best hyperparameters from tuning sweep (432 configs tested)
TUNED_CONFIGS = {
    "cgd": {"n_h1": 18, "n_h2": 9, "loss": "huber", "lr": 5e-3, "bs": 32},
    "cgs": {"n_h1": 20, "n_h2": 10, "loss": "mse", "lr": 5e-3, "bs": 16},
}


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
    on = c_true >= 0.5
    mare_on = (float(np.mean(np.abs(c_true[on] - c_pred[on]) / c_true[on]) * 100)
               if on.sum() else float("nan"))
    return dict(rmse_pF=rmse, mae_pF=mae, r2=r2, MARE_percent_onstate_C_ge_0p5pF=mare_on)


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
    ax.set_title(f"{cap.upper()} Tuned ANN test set  "
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
    fig.suptitle(f"{cap.upper()} Tuned ANN vs measurement (dots=measured, line=model)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def train_tuned(cap_type, verbose=True):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = TUNED_CONFIGS[cap_type]
    n_h1, n_h2 = cfg["n_h1"], cfg["n_h2"]
    loss_name, lr, batch_size = cfg["loss"], cfg["lr"], cfg["bs"]

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

    model = TFTNet2(n_inputs=4, n_hidden1=n_h1, n_hidden2=n_h2, n_outputs=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-4)

    if loss_name == "mse":
        loss_fn = nn.MSELoss()
    elif loss_name == "mae":
        loss_fn = nn.L1Loss()
    elif loss_name == "huber":
        loss_fn = nn.HuberLoss(delta=0.1)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    n_train = Xtr.shape[0]
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
        "tuned": True,
        "hyperparameters": {
            "n_h1": n_h1, "n_h2": n_h2, "loss": loss_name, "lr": lr, "batch_size": batch_size
        },
        "architecture": {
            "layers": f"4 -> {n_h1} (tanh) -> {n_h2} (tanh) -> 1 (linear)",
            "n_inputs": 4, "n_hidden1": n_h1, "n_hidden2": n_h2, "n_outputs": 1,
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
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), os.path.join(out_dir, "model_weights.pt"))

    sd = model.state_dict()
    weights = {
        "element": cap_type.upper(),
        "architecture": metrics["architecture"],
        "hyperparameters": metrics["hyperparameters"],
        "input_scaling_minmax": {
            "VG": [-5.0, 5.0], "VD": [0.0, 5.0], "W": [5.0, 160.0], "L": [5.0, 20.0],
        },
        "w1": sd["hidden1.weight"].numpy().tolist(),
        "b1": sd["hidden1.bias"].numpy().tolist(),
        "w2": sd["hidden2.weight"].numpy().tolist(),
        "b2": sd["hidden2.bias"].numpy().tolist(),
        "wo": sd["output.weight"].numpy().tolist(),
        "bo": sd["output.bias"].numpy().tolist(),
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
    print("\n" + "="*80)
    print("Training tuned Cgd/Cgs ANNs with best hyperparameters from sweep")
    print("="*80)
    for cap in ("cgd", "cgs"):
        print()
        train_tuned(cap)
    print("\n" + "="*80)
    print("Done!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
