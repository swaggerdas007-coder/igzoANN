"""Loss-weighting / target-normalisation experiments for the Cgd/Cgs ANNs.

Motivation
----------
The tuned 2-layer models (18->9 for CGD, 20->10 for CGS) fit the two W=160
devices almost exactly but overshoot the turn-on knee of the small-geometry
devices (W=20, W=40).  The cause is not depth or input scaling (both were
tested and neither helped): an unweighted loss on absolute capacitance is
dominated by the W=160 curves, which are ~8x larger in magnitude, so the
small devices contribute almost nothing to the gradient.

Measured device capacitances are very close to proportional to the gate area:

    W=20  L=20 -> ~0.80 pF   C/(W*L) = 2.0e-3
    W=40  L=20 -> ~1.60 pF   C/(W*L) = 2.0e-3
    W=160 L=15 -> ~4.70 pF   C/(W*L) = 1.96e-3
    W=160 L=20 -> ~6.30 pF   C/(W*L) = 1.97e-3

so dividing the target by W*L puts all four curves on the same scale and
removes the imbalance at the source.

Variants compared (all keep the Verilog-A-safe topology: tanh hidden layers,
linear output, min-max [0,1] input scaling):

    baseline  unweighted loss on C [pF]                        (current model)
    weighted  per-geometry weighted loss on C [pF]; each curve
              contributes equally regardless of its magnitude
    areanorm  unweighted loss on the area-normalised target
              C / (W*L) [pF/um^2], rescaled by W*L at inference
    areanorm_w  area-normalised target + per-geometry weighting

Every variant is run over several seeds and scored both globally (test set)
and per geometry (all points, which is what the C-V curve plots show), because
the 24-point test split is too small to rank per-geometry behaviour on its own.

Run:
    python -m src.cg_loss_experiments
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

from src.cap_dataset import load_and_split
from src.model import TFTNet2

MAX_EPOCHS = 2500
SEEDS = (42, 7, 1234)
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data_cleaned_cv",
                         "merged_cap_dataset.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "cg_experiments")

# Winning hyperparameters from the 432-config sweep (2 layers, <=20/<=10 neurons)
TUNED = {
    "cgd": {"n_h1": 18, "n_h2": 9, "loss": "huber", "lr": 5e-3, "bs": 32},
    "cgs": {"n_h1": 20, "n_h2": 10, "loss": "mse", "lr": 5e-3, "bs": 16},
}

VARIANTS = ("baseline", "weighted", "areanorm", "areanorm_w")


def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


def make_loss(name):
    if name == "mse":
        return nn.MSELoss(reduction="none")
    if name == "mae":
        return nn.L1Loss(reduction="none")
    if name == "huber":
        return nn.HuberLoss(delta=0.1, reduction="none")
    raise ValueError(name)


def geometry_keys(split):
    """Integer code per (W, L) geometry, aligned with split rows."""
    wl = list(zip(split.df.W.to_numpy(), split.df.L.to_numpy()))
    uniq = sorted(set(wl))
    index = {g: i for i, g in enumerate(uniq)}
    return np.array([index[g] for g in wl], dtype=np.int64), uniq


def sample_weights(split):
    """Weight each sample so that every (W, L) curve contributes equally to the
    loss: w_i proportional to 1 / (mean |C| of that curve)^2, mean-normalised."""
    codes, uniq = geometry_keys(split)
    scale = np.ones(len(uniq), dtype=np.float64)
    for i in range(len(uniq)):
        m = np.abs(split.c_pf[codes == i]).mean()
        scale[i] = max(m, 1e-6)
    w = (scale.mean() / scale[codes]) ** 2
    return (w / w.mean()).astype(np.float32)


def area_um2(split):
    return (split.df.W.to_numpy() * split.df.L.to_numpy()).astype(np.float32)


def metrics_block(c_true, c_pred):
    rmse = float(np.sqrt(np.mean((c_pred - c_true) ** 2)))
    mae = float(np.mean(np.abs(c_pred - c_true)))
    ss_res = float(np.sum((c_true - c_pred) ** 2))
    ss_tot = float(np.sum((c_true - c_true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    on = c_true >= 0.5
    mare = (float(np.mean(np.abs(c_true[on] - c_pred[on]) / c_true[on]) * 100)
            if on.sum() else float("nan"))
    return dict(rmse_pF=rmse, mae_pF=mae, r2=r2, MARE_percent_onstate=mare)


def per_geometry(split, c_pred):
    """RMSE per geometry, plus RMSE normalised by that geometry's C range so
    small and large devices are directly comparable."""
    codes, uniq = geometry_keys(split)
    out = {}
    for i, (w, l) in enumerate(uniq):
        m = codes == i
        t, p = split.c_pf[m], c_pred[m]
        rng = float(t.max() - t.min())
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        out[f"W{int(w)}_L{int(l)}"] = {
            "rmse_pF": rmse,
            "nrmse_percent": float(rmse / rng * 100) if rng > 0 else float("nan"),
            "n": int(m.sum()),
        }
    return out


def train_variant(cap_type, variant, seed, cfg):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train, val, test = load_and_split(DATA_PATH, cap_type=cap_type, seed=42)

    use_area = variant.startswith("areanorm")
    use_weights = variant in ("weighted", "areanorm_w")

    def raw_target(split):
        if use_area:
            # pF/um^2 is ~2e-3; scale by 1e3 to keep the target O(1) before
            # standardisation so the optimiser sees well-conditioned gradients
            return split.c_pf / area_um2(split) * 1e3
        return split.c_pf

    y_mean = float(raw_target(train).mean())
    y_std = float(raw_target(train).std())

    def tgt(split):
        return (raw_target(split) - y_mean) / y_std

    Xtr, ytr = to_tensor(train.X), to_tensor(tgt(train))
    wtr = to_tensor(sample_weights(train) if use_weights
                    else np.ones(len(train.c_pf), dtype=np.float32))

    model = TFTNet2(n_inputs=4, n_hidden1=cfg["n_h1"], n_hidden2=cfg["n_h2"], n_outputs=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-4)
    loss_fn = make_loss(cfg["loss"])

    n_train = Xtr.shape[0]
    bs = cfg["bs"]
    for _ in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, bs):
            idx = perm[i:i + bs]
            optimizer.zero_grad()
            per_sample = loss_fn(model(Xtr[idx]), ytr[idx].unsqueeze(1)).squeeze(1)
            (per_sample * wtr[idx]).mean().backward()
            optimizer.step()
        scheduler.step()

    model.eval()

    def predict(split):
        with torch.no_grad():
            out = model(to_tensor(split.X)).squeeze(1).numpy()
        c = out * y_std + y_mean
        if use_area:
            c = c * area_um2(split) / 1e3
        return c

    preds = {name: predict(s) for name, s in (("train", train), ("val", val), ("test", test))}

    # per-geometry scoring over every measured point (what the C-V plots show)
    all_c = np.concatenate([train.c_pf, val.c_pf, test.c_pf])
    all_p = np.concatenate([preds["train"], preds["val"], preds["test"]])

    class _All:
        pass
    allsplit = _All()
    allsplit.c_pf = all_c
    import pandas as pd
    allsplit.df = pd.concat([train.df, val.df, test.df], ignore_index=True)

    geo = per_geometry(allsplit, all_p)
    nrmse_vals = [g["nrmse_percent"] for g in geo.values()]

    return {
        "cap_type": cap_type,
        "variant": variant,
        "seed": seed,
        "test": metrics_block(test.c_pf, preds["test"]),
        "val": metrics_block(val.c_pf, preds["val"]),
        "train": metrics_block(train.c_pf, preds["train"]),
        "per_geometry_all_points": geo,
        "mean_nrmse_percent": float(np.mean(nrmse_vals)),
        "worst_nrmse_percent": float(np.max(nrmse_vals)),
        "min_pred_pF": float(min(p.min() for p in preds.values())),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []

    for cap_type in ("cgd", "cgs"):
        cfg = TUNED[cap_type]
        print(f"\n{'='*78}\n{cap_type.upper()}  (arch {cfg['n_h1']}->{cfg['n_h2']}, "
              f"loss={cfg['loss']}, lr={cfg['lr']:.0e}, bs={cfg['bs']})\n{'='*78}")
        for variant in VARIANTS:
            runs = [train_variant(cap_type, variant, s, cfg) for s in SEEDS]
            results.extend(runs)
            r2 = np.mean([r["test"]["r2"] for r in runs])
            rmse = np.mean([r["test"]["rmse_pF"] for r in runs])
            mean_n = np.mean([r["mean_nrmse_percent"] for r in runs])
            worst_n = np.mean([r["worst_nrmse_percent"] for r in runs])
            minp = np.mean([r["min_pred_pF"] for r in runs])
            print(f"  {variant:11s} test R2={r2:.4f}  RMSE={rmse:.4f} pF   "
                  f"per-geom NRMSE mean={mean_n:5.2f}%  worst={worst_n:5.2f}%   "
                  f"min_pred={minp:+.3f} pF")

    with open(os.path.join(OUT_DIR, "variant_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- summary table + decision ----
    print(f"\n{'='*78}\nSUMMARY (mean over {len(SEEDS)} seeds)\n{'='*78}")
    summary = {}
    for cap_type in ("cgd", "cgs"):
        summary[cap_type] = {}
        print(f"\n{cap_type.upper()}")
        print(f"  {'variant':12s} {'test R2':>9s} {'test RMSE':>10s} "
              f"{'geomNRMSE':>10s} {'worstNRMSE':>11s}")
        for variant in VARIANTS:
            runs = [r for r in results if r["cap_type"] == cap_type and r["variant"] == variant]
            entry = {
                "test_r2": float(np.mean([r["test"]["r2"] for r in runs])),
                "test_rmse_pF": float(np.mean([r["test"]["rmse_pF"] for r in runs])),
                "mean_nrmse_percent": float(np.mean([r["mean_nrmse_percent"] for r in runs])),
                "worst_nrmse_percent": float(np.mean([r["worst_nrmse_percent"] for r in runs])),
                "per_geometry": {
                    k: float(np.mean([r["per_geometry_all_points"][k]["nrmse_percent"]
                                      for r in runs]))
                    for k in runs[0]["per_geometry_all_points"]
                },
            }
            summary[cap_type][variant] = entry
            print(f"  {variant:12s} {entry['test_r2']:9.4f} {entry['test_rmse_pF']:10.4f} "
                  f"{entry['mean_nrmse_percent']:9.2f}% {entry['worst_nrmse_percent']:10.2f}%")
        print("  per-geometry NRMSE (%, all points):")
        geoms = list(summary[cap_type][VARIANTS[0]]["per_geometry"].keys())
        print("    " + " ".join(f"{g:>12s}" for g in ["variant"] + geoms))
        for variant in VARIANTS:
            row = summary[cap_type][variant]["per_geometry"]
            print("    " + f"{variant:>12s} " + " ".join(f"{row[g]:11.2f}%" for g in geoms))

    with open(os.path.join(OUT_DIR, "variant_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR}/variant_summary.json")


if __name__ == "__main__":
    main()
