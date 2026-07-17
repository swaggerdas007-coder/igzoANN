"""Hyperparameter sweep over hidden-layer size (12-32 neurons) for the a-GIZO
TFT ANN (4 inputs -> N tanh hidden -> linear output, Eq. 1-2 of the paper).

For each hidden size, trains 3 random seeds (same fixed train/val/test split
throughout -- only the weight init and minibatch order vary) so the sweep
plots can show mean +/- std across seeds rather than a single noisy run per
config. Writes one row per run to outputs/sweep_results.csv (append-as-you-go,
so a partial run is still usable), plus full per-epoch loss curves for a
handful of representative hidden sizes to outputs/sweep_curves.json for the
convergence-comparison plot.

Usage:
    python -m src.hparam_sweep
"""
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from src.dataset import load_and_split
from src.model import TFTNet

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data_cleaned", "merged_ann_dataset.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
RESULTS_CSV = os.path.join(OUT_DIR, "sweep_results.csv")
CURVES_JSON = os.path.join(OUT_DIR, "sweep_curves.json")

DATA_SEED = 42            # fixed split for every run in the sweep
HIDDEN_SIZES = list(range(12, 33))   # 12..32 inclusive
SEEDS = [0, 1, 2]
LR = 2e-3
BATCH_SIZE = 256
MAX_EPOCHS = 300
PATIENCE = 40
CURVE_HIDDEN_SIZES = {12, 16, 22, 27, 32}  # representative sizes to keep full curves for

FIELDS = ["n_hidden", "seed", "n_params", "epochs_run", "best_epoch", "best_val_mse",
          "test_rmse_log10ID", "test_mae_log10ID", "test_r2_log10ID", "train_seconds"]


def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


def train_one(n_hidden, seed, Xtr, ytr, Xva, yva, Xte, log_id_test, y_mean, y_std, keep_curve):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TFTNet(n_inputs=4, n_hidden=n_hidden, n_outputs=1)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=25)
    loss_fn = nn.MSELoss()

    n_train = Xtr.shape[0]
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    val_curve = []

    t0 = time.time()
    epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = Xtr[idx], ytr[idx].unsqueeze(1)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), yva.unsqueeze(1)).item()
        scheduler.step(val_loss)
        if keep_curve:
            val_curve.append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            break

    train_seconds = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_std = model(Xte).squeeze(1).numpy()
    log_id_pred = pred_std * y_std + y_mean

    rmse_log = float(np.sqrt(np.mean((log_id_pred - log_id_test) ** 2)))
    mae_log = float(np.mean(np.abs(log_id_pred - log_id_test)))
    ss_res = np.sum((log_id_test - log_id_pred) ** 2)
    ss_tot = np.sum((log_id_test - log_id_test.mean()) ** 2)
    r2_log = float(1 - ss_res / ss_tot)

    row = {
        "n_hidden": n_hidden, "seed": seed, "n_params": n_params,
        "epochs_run": epoch, "best_epoch": best_epoch, "best_val_mse": best_val,
        "test_rmse_log10ID": rmse_log, "test_mae_log10ID": mae_log, "test_r2_log10ID": r2_log,
        "train_seconds": train_seconds,
    }
    return row, val_curve


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train, val, test = load_and_split(DATA_PATH, seed=DATA_SEED)
    y_mean = float(train.log_id.mean())
    y_std = float(train.log_id.std())

    Xtr = to_tensor(train.X)
    ytr = to_tensor((train.log_id - y_mean) / y_std)
    Xva = to_tensor(val.X)
    yva = to_tensor((val.log_id - y_mean) / y_std)
    Xte = to_tensor(test.X)
    log_id_test = test.log_id

    print(f"train={len(train.df)} val={len(val.df)} test={len(test.df)}  "
          f"sweeping n_hidden={HIDDEN_SIZES} x seeds={SEEDS}")

    write_header = not os.path.exists(RESULTS_CSV)
    curves = {}
    if os.path.exists(CURVES_JSON):
        with open(CURVES_JSON) as f:
            curves = json.load(f)

    done = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV) as f:
            for r in csv.DictReader(f):
                done.add((int(r["n_hidden"]), int(r["seed"])))

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        for n_hidden in HIDDEN_SIZES:
            for seed in SEEDS:
                if (n_hidden, seed) in done:
                    continue
                keep_curve = (n_hidden in CURVE_HIDDEN_SIZES and seed == 0)
                row, val_curve = train_one(n_hidden, seed, Xtr, ytr, Xva, yva, Xte,
                                            log_id_test, y_mean, y_std, keep_curve)
                writer.writerow(row)
                f.flush()
                if keep_curve:
                    curves[str(n_hidden)] = val_curve
                    with open(CURVES_JSON, "w") as cf:
                        json.dump(curves, cf)
                print(f"n_hidden={n_hidden:2d} seed={seed}  "
                      f"test_r2={row['test_r2_log10ID']:.4f}  "
                      f"test_rmse={row['test_rmse_log10ID']:.4f}  "
                      f"best_epoch={row['best_epoch']:3d}  "
                      f"({row['train_seconds']:.1f}s)")

    print("Sweep complete ->", RESULTS_CSV, CURVES_JSON)


if __name__ == "__main__":
    main()
