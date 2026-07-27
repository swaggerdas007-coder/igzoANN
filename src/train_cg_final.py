"""Train the final Cgd/Cgs ANNs and write trained_Cg_ANN/.

Architecture and optimiser settings come from the 432-config hyperparameter
sweep (2 hidden layers, <=20 neurons in layer 1, <=10 in layer 2, tanh hidden
activations and a linear output so the network stays directly portable to
Verilog-A).  The loss/target variant comes from src/cg_loss_experiments.py --
see cg_experiments/variant_summary.json for the comparison that selected it.

Run:
    python -m src.train_cg_final
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
from src.cg_loss_experiments import (MAX_EPOCHS, TUNED, area_um2, make_loss,
                                     metrics_block, per_geometry, sample_weights)

SEED = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data_cleaned_cv",
                         "merged_cap_dataset.csv")
OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "trained_Cg_ANN")
EXP_SUMMARY = os.path.join(os.path.dirname(__file__), "..", "cg_experiments",
                           "variant_summary.json")

BG = "#f7f7f5"
FG = "#1f1f1f"
MEASURED = "#4c72b0"
MODEL = "#c44e52"

# Winning loss/target variant per component (see cg_experiments/).
VARIANT = {"cgd": "areanorm", "cgs": "areanorm"}


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


def predict_curve(model, y_mean, y_std, use_area, vg, vd, w, l):
    X = np.stack([scale_point(a, b, c, d) for a, b, c, d in zip(vg, vd, w, l)])
    with torch.no_grad():
        out = model(to_tensor(X)).squeeze(1).numpy()
    c = out * y_std + y_mean
    if use_area:
        c = c * (np.asarray(w) * np.asarray(l)) / 1e3
    return c


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
    ax.set_title(f"{cap.upper()} ANN test set  "
                 f"(R$^2$={m['test']['r2']:.3f}, RMSE={m['test']['rmse_pF']:.2f} pF)")
    style_axes(ax)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cv_curves(cap, model, y_mean, y_std, use_area, path):
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
        c_fine = predict_curve(model, y_mean, y_std, use_area, vg_fine,
                               np.full_like(vg_fine, vd0),
                               np.full_like(vg_fine, w), np.full_like(vg_fine, l))
        ax.plot(vg_fine, c_fine, color=MODEL, lw=2, label="ANN")
        ax.set_xlabel("VG (V)")
        ax.set_ylabel(r"$C_{%s}$ (pF)" % cap[1:].upper())
        ax.set_title(f"W={w:g}, L={l:g} um (VD={vd0:g}V)")
        style_axes(ax)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{cap.upper()} ANN vs measurement (dots=measured, line=model)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def train_final(cap_type, verbose=True):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = TUNED[cap_type]
    variant = VARIANT[cap_type]
    use_area = variant.startswith("areanorm")
    use_weights = variant in ("weighted", "areanorm_w")

    out_dir = os.path.join(OUT_ROOT, cap_type)
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    train, val, test = load_and_split(DATA_PATH, cap_type=cap_type, seed=SEED)
    if verbose:
        print(f"[{cap_type}] variant={variant}  arch={cfg['n_h1']}->{cfg['n_h2']}  "
              f"loss={cfg['loss']} lr={cfg['lr']:.0e} bs={cfg['bs']}")
        print(f"[{cap_type}] train={len(train.df)} val={len(val.df)} test={len(test.df)}")

    def raw_target(split):
        if use_area:
            return split.c_pf / area_um2(split) * 1e3
        return split.c_pf

    y_mean = float(raw_target(train).mean())
    y_std = float(raw_target(train).std())

    def tgt(split):
        return (raw_target(split) - y_mean) / y_std

    Xtr, ytr = to_tensor(train.X), to_tensor(tgt(train))
    Xva, yva = to_tensor(val.X), to_tensor(tgt(val))
    wtr = to_tensor(sample_weights(train) if use_weights
                    else np.ones(len(train.c_pf), dtype=np.float32))

    model = TFTNet2(n_inputs=4, n_hidden1=cfg["n_h1"], n_hidden2=cfg["n_h2"], n_outputs=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-4)
    loss_fn = make_loss(cfg["loss"])

    n_train = Xtr.shape[0]
    bs = cfg["bs"]
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, bs):
            idx = perm[i:i + bs]
            optimizer.zero_grad()
            per_sample = loss_fn(model(Xtr[idx]), ytr[idx].unsqueeze(1)).squeeze(1)
            (per_sample * wtr[idx]).mean().backward()
            optimizer.step()
        scheduler.step()
        if verbose and (epoch % 500 == 0 or epoch == 1):
            model.eval()
            with torch.no_grad():
                tr = loss_fn(model(Xtr), ytr.unsqueeze(1)).mean().item()
                va = loss_fn(model(Xva), yva.unsqueeze(1)).mean().item()
            print(f"[{cap_type}] epoch {epoch:4d}  train={tr:.6f}  val={va:.6f}")

    model.eval()

    def predict(split):
        with torch.no_grad():
            out = model(to_tensor(split.X)).squeeze(1).numpy()
        c = out * y_std + y_mean
        if use_area:
            c = c * area_um2(split) / 1e3
        return c

    p_train, p_val, p_test = predict(train), predict(val), predict(test)
    train_m = metrics_block(train.c_pf, p_train)
    val_m = metrics_block(val.c_pf, p_val)
    test_m = metrics_block(test.c_pf, p_test)

    class _All:
        pass
    allsplit = _All()
    allsplit.c_pf = np.concatenate([train.c_pf, val.c_pf, test.c_pf])
    allsplit.df = pd.concat([train.df, val.df, test.df], ignore_index=True)
    geo = per_geometry(allsplit, np.concatenate([p_train, p_val, p_test]))

    metrics = {
        "cap_type": cap_type,
        "variant": variant,
        "hyperparameters": {
            "n_h1": cfg["n_h1"], "n_h2": cfg["n_h2"], "loss": cfg["loss"],
            "lr": cfg["lr"], "batch_size": cfg["bs"], "epochs": MAX_EPOCHS,
        },
        "architecture": {
            "layers": f"4 -> {cfg['n_h1']} (tanh) -> {cfg['n_h2']} (tanh) -> 1 (linear)",
            "n_inputs": 4, "n_hidden1": cfg["n_h1"], "n_hidden2": cfg["n_h2"],
            "n_outputs": 1, "input_order": ["VG", "VD", "W", "L"],
        },
        "target": ("C_pF / (W_um * L_um) * 1e3" if use_area else "C_pF"),
        "area_normalised": use_area,
        "rows": {"train": len(train.df), "val": len(val.df), "test": len(test.df)},
        "geometries": sorted({(int(w), int(l)) for w, l in zip(test.df.W, test.df.L)}),
        "target_standardization": {"y_mean": y_mean, "y_std": y_std},
        "train": train_m,
        "val": val_m,
        "test": test_m,
        "per_geometry_all_points": geo,
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), os.path.join(out_dir, "model_weights.pt"))

    sd = model.state_dict()
    recover = ("C_pF = (network_output * y_std + y_mean) * W_um * L_um / 1e3"
               if use_area else "C_pF = network_output * y_std + y_mean")
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
            "definition": ("network_output = (C_pF/(W_um*L_um)*1e3 - y_mean) / y_std"
                           if use_area else
                           "network_output = (C_pF - y_mean) / y_std"),
            "area_normalised": use_area,
            "y_mean": y_mean,
            "y_std": y_std,
            "recover_C_pF": recover,
            "recover_C_F": "C_F = C_pF * 1e-12",
        },
    }
    with open(os.path.join(out_dir, "weights.json"), "w") as f:
        json.dump(weights, f, indent=2)

    np.savez(os.path.join(out_dir, "test_predictions.npz"),
             X=test.X, c_true=test.c_pf, c_pred=p_test)

    scatter_plot(cap_type, test.c_pf, p_test, metrics,
                 os.path.join(plot_dir, "scatter_C.png"))
    cv_curves(cap_type, model, y_mean, y_std, use_area,
              os.path.join(plot_dir, "C_vg_curves.png"))

    if verbose:
        print(f"[{cap_type}] test: R2={test_m['r2']:.4f}  "
              f"RMSE={test_m['rmse_pF']:.4f} pF  MAE={test_m['mae_pF']:.4f} pF  "
              f"MARE_on={test_m['MARE_percent_onstate']:.2f}%")
        for g, v in geo.items():
            print(f"    {g:12s} NRMSE={v['nrmse_percent']:5.2f}%  "
                  f"RMSE={v['rmse_pF']:.4f} pF")
        print(f"[{cap_type}] saved to {out_dir}")
    return metrics


def main():
    summary = {}
    for cap in ("cgd", "cgs"):
        print()
        summary[cap] = train_final(cap)
    with open(os.path.join(OUT_ROOT, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_ROOT}/metrics_summary.json")


if __name__ == "__main__":
    main()
