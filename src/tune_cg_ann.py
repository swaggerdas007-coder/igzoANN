"""Hyperparameter tuning for 2-layer Cgd/Cgs ANNs.
Explores architectures (n_h1 ≤ 20, n_h2 ≤ 10), loss functions, learning rates,
batch sizes. Reports best configuration per component.

Run:
    python -m src.tune_cg_ann
"""
import json
import os
from itertools import product

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
OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "tune_results")

# Hyperparameter grids
ARCHITECTURES = [
    (10, 5),   # baseline shallow
    (10, 8),
    (12, 6),
    (15, 7),
    (15, 10),  # max second layer
    (18, 9),
    (20, 5),   # max first layer
    (20, 10),  # max both
]

LOSS_FUNCTIONS = ["mse", "mae", "huber"]

LEARNING_RATES = [1e-3, 2e-3, 5e-3]

BATCH_SIZES = [16, 32, 64]


def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


def metrics_block(c_true, c_pred):
    rmse = float(np.sqrt(np.mean((c_pred - c_true) ** 2)))
    mae = float(np.mean(np.abs(c_pred - c_true)))
    ss_res = float(np.sum((c_true - c_pred) ** 2))
    ss_tot = float(np.sum((c_true - c_true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    on = c_true >= 0.5
    mare_on = (float(np.mean(np.abs(c_true[on] - c_pred[on]) / c_true[on]) * 100)
               if on.sum() else float("nan"))
    return dict(rmse_pF=rmse, mae_pF=mae, r2=r2, MARE_percent_onstate=mare_on)


def train_one_config(cap_type, n_h1, n_h2, loss_name, lr, batch_size, verbose=False):
    """Train a single config and return test metrics."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train, val, test = load_and_split(DATA_PATH, cap_type=cap_type, seed=SEED)

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

    batch_size_val = batch_size
    n_train = Xtr.shape[0]
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size_val):
            idx = perm[i:i + batch_size_val]
            optimizer.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx].unsqueeze(1)).backward()
            optimizer.step()
        scheduler.step()

    model.eval()

    def eval_split(split):
        with torch.no_grad():
            pred = model(to_tensor(split.X)).squeeze(1).numpy() * y_std + y_mean
        return metrics_block(split.c_pf, pred), pred

    _, c_pred_test = eval_split(test)
    test_m, _ = eval_split(test)

    return test_m, c_pred_test


def tune_component(cap_type, verbose=True):
    """Tune all hyperparameters for one component, return best config."""
    if verbose:
        print(f"\n{'='*80}")
        print(f"Tuning {cap_type.upper()}")
        print(f"{'='*80}")

    results = []
    config_id = 0
    total_configs = (len(ARCHITECTURES) * len(LOSS_FUNCTIONS) *
                     len(LEARNING_RATES) * len(BATCH_SIZES))

    for (n_h1, n_h2), loss_name, lr, bs in product(
            ARCHITECTURES, LOSS_FUNCTIONS, LEARNING_RATES, BATCH_SIZES):
        config_id += 1
        try:
            test_m, _ = train_one_config(cap_type, n_h1, n_h2, loss_name, lr, bs)
            r2 = test_m["r2"]
            rmse = test_m["rmse_pF"]
            results.append({
                "config_id": config_id,
                "total": total_configs,
                "n_h1": n_h1,
                "n_h2": n_h2,
                "loss": loss_name,
                "lr": lr,
                "batch_size": bs,
                "r2": r2,
                "rmse_pF": rmse,
                "mae_pF": test_m["mae_pF"],
                "mare_on": test_m["MARE_percent_onstate"],
            })
            if verbose and config_id % 10 == 0:
                print(f"[{config_id}/{total_configs}] n_h1={n_h1} n_h2={n_h2} "
                      f"loss={loss_name} lr={lr:.1e} bs={bs}: R²={r2:.4f} RMSE={rmse:.4f}")
        except Exception as e:
            if verbose:
                print(f"[{config_id}/{total_configs}] FAILED: {e}")
            continue

    df_results = pd.DataFrame(results)
    if df_results.empty:
        print(f"No successful configs for {cap_type}")
        return None

    best_idx = df_results["r2"].idxmax()
    best = df_results.iloc[best_idx]

    if verbose:
        print(f"\n{'='*80}")
        print(f"TOP 10 CONFIGS (by R²) for {cap_type.upper()}:")
        print(f"{'='*80}")
        top10 = df_results.nlargest(10, "r2")
        for idx, row in top10.iterrows():
            print(f"  R²={row['r2']:.4f}  RMSE={row['rmse_pF']:.4f}  "
                  f"n_h1={int(row['n_h1'])} n_h2={int(row['n_h2'])} "
                  f"loss={row['loss']} lr={row['lr']:.1e} bs={int(row['batch_size'])}")

        print(f"\nBEST CONFIG:")
        print(f"  n_h1={int(best['n_h1'])} n_h2={int(best['n_h2'])}")
        print(f"  loss={best['loss']} lr={best['lr']:.1e} batch_size={int(best['batch_size'])}")
        print(f"  R²={best['r2']:.4f}  RMSE={best['rmse_pF']:.4f} pF  "
              f"MAE={best['mae_pF']:.4f} pF  MARE_on={best['mare_on']:.2f}%")

    return best, df_results


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    summary = {}

    for cap_type in ("cgd", "cgs"):
        result = tune_component(cap_type, verbose=True)
        if result:
            best, df = result
            summary[cap_type] = best.to_dict()
            df.to_csv(os.path.join(OUT_ROOT, f"{cap_type}_sweep.csv"), index=False)

    with open(os.path.join(OUT_ROOT, "best_configs.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("SUMMARY: BEST CONFIGS")
    print(f"{'='*80}")
    for cap, cfg in summary.items():
        print(f"\n{cap.upper()}:")
        print(f"  Architecture: {int(cfg['n_h1'])} → {int(cfg['n_h2'])}")
        print(f"  Loss: {cfg['loss']}, LR: {cfg['lr']:.1e}, Batch: {int(cfg['batch_size'])}")
        print(f"  R²={cfg['r2']:.4f}  RMSE={cfg['rmse_pF']:.4f} pF  "
              f"MAE={cfg['mae_pF']:.4f} pF")


if __name__ == "__main__":
    main()
