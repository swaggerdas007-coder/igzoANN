"""Train the MLP (22 hidden neurons, tanh hidden / linear output) on the
a-GIZO TFT dataset to model ID = f(VG, VD, W, L), following the ANN
methodology of Bahubalindruni et al. (2015).

Usage:
    python -m src.train
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

from src.dataset import load_and_split
from src.model import TFTNet

SEED = 42
N_HIDDEN = 22
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data_cleaned_2", "merged_ann_dataset.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

torch.manual_seed(SEED)
np.random.seed(SEED)


def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train, val, test = load_and_split(DATA_PATH, seed=SEED)
    print(f"train={len(train.df)}  val={len(val.df)}  test={len(test.df)}  "
          f"(rows, grouped by unique operating point)")

    # Standardize the regression target (log10|ID|) using train-set stats only.
    y_mean = float(train.log_id.mean())
    y_std = float(train.log_id.std())

    def target(split):
        return (split.log_id - y_mean) / y_std

    Xtr, ytr = to_tensor(train.X), to_tensor(target(train))
    Xva, yva = to_tensor(val.X), to_tensor(target(val))
    Xte, yte = to_tensor(test.X), to_tensor(target(test))

    model = TFTNet(n_inputs=4, n_hidden=N_HIDDEN, n_outputs=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=25)
    loss_fn = nn.MSELoss()

    batch_size = 256
    n_train = Xtr.shape[0]
    max_epochs = 600
    patience = 50
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx], ytr[idx].unsqueeze(1)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n_train

        model.eval()
        with torch.no_grad():
            val_pred = model(Xva)
            val_loss = loss_fn(val_pred, yva.unsqueeze(1)).item()
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  train_mse={epoch_loss:.5f}  val_mse={val_loss:.5f}")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best val_mse={best_val:.5f})")
            break

    model.load_state_dict(best_state)
    model.eval()

    # ---- Evaluation on held-out test set ----
    with torch.no_grad():
        test_pred_std = model(Xte).squeeze(1).numpy()
    log_id_pred = test_pred_std * y_std + y_mean
    id_pred = 10.0 ** log_id_pred  # magnitude model -> nonnegative current estimate
    log_id_true = test.log_id

    rmse_log = float(np.sqrt(np.mean((log_id_pred - log_id_true) ** 2)))
    mae_log = float(np.mean(np.abs(log_id_pred - log_id_true)))
    ss_res = np.sum((log_id_true - log_id_pred) ** 2)
    ss_tot = np.sum((log_id_true - log_id_true.mean()) ** 2)
    r2_log = float(1 - ss_res / ss_tot)

    # Noise ceiling: the dataset repeats many (VG,VD,W,L) operating points with
    # different noise realizations of ID (median ~5 repeats/point, up to 24).
    # No function of (VG,VD,W,L) can do better than predicting each point's true
    # conditional mean, so SS_within (residual around the group mean) is the best
    # any model could achieve -- this bounds the achievable R^2 on this data.
    group_mean = test.df.groupby(["VG", "VD", "W", "L"])["log_ID"].transform("mean").to_numpy()
    ss_within = np.sum((log_id_true - group_mean) ** 2)
    r2_ceiling = float(1 - ss_within / ss_tot)

    # MARE (Eq. 7 in the paper) stratified by current magnitude. Sub-uA currents
    # sit inside the dataset's noise floor (see r2_ceiling above), so a plain
    # overall MARE is dominated by relative error on near-zero noisy currents;
    # report it per decade band instead, matching where the paper evaluates
    # accuracy (on-state currents, uA-mA range).
    bins = [1e-9, 1e-6, 1e-3, 1e0]
    mare_by_band = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (np.abs(test.id_raw) >= lo) & (np.abs(test.id_raw) < hi)
        if mask.sum() == 0:
            continue
        mare = float(np.mean(np.abs(np.abs(test.id_raw[mask]) - id_pred[mask]) / np.abs(test.id_raw[mask])) * 100)
        mare_by_band[f"{lo:.0e}_to_{hi:.0e}_A"] = {"MARE_percent": mare, "n_rows": int(mask.sum())}

    metrics = {
        "n_hidden": N_HIDDEN,
        "hidden_activation": "tanh",
        "output_activation": "linear",
        "train_rows": len(train.df),
        "val_rows": len(val.df),
        "test_rows": len(test.df),
        "target_standardization": {"y_mean_log10_absID": y_mean, "y_std_log10_absID": y_std},
        "test_rmse_log10ID": rmse_log,
        "test_mae_log10ID": mae_log,
        "test_r2_log10ID": r2_log,
        "test_r2_ceiling_from_measurement_noise": r2_ceiling,
        "note": "r2_ceiling is the max R^2 any (VG,VD,W,L)-only model could reach, "
                "given that repeated noise realizations of the same operating point "
                "differ by up to several decades in this dataset.",
        "test_MARE_percent_by_current_band": mare_by_band,
    }
    print(json.dumps(metrics, indent=2))

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "model_weights.pt"))

    # Export explicit wh, bh, wo, bo (paper's notation) + pre/post-processing
    # parameters, so the model can be ported (e.g. to Verilog-A as in the paper).
    sd = model.state_dict()
    weights = {
        "architecture": {"n_inputs": 4, "n_hidden": N_HIDDEN, "n_outputs": 1,
                          "input_order": ["VG", "VD", "W", "L"]},
        "input_scaling_minmax": {
            "VG": [-5.0, 5.0], "VD": [0.0, 5.0], "W": [5.0, 160.0], "L": [5.0, 20.0],
        },
        "wh": sd["hidden.weight"].numpy().tolist(),   # [n_hidden, 4]
        "bh": sd["hidden.bias"].numpy().tolist(),     # [n_hidden]
        "wo": sd["output.weight"].numpy().tolist(),   # [1, n_hidden]
        "bo": sd["output.bias"].numpy().tolist(),     # [1]
        "target_transform": {
            "definition": "network_output = (log10(|ID|) - y_mean) / y_std",
            "y_mean_log10_absID": y_mean,
            "y_std_log10_absID": y_std,
            "recover_ID": "ID = 10 ** (network_output * y_std + y_mean)",
        },
    }
    with open(os.path.join(OUT_DIR, "weights.json"), "w") as f:
        json.dump(weights, f, indent=2)

    np.savez(os.path.join(OUT_DIR, "test_predictions.npz"),
              X=test.X, id_true=test.id_raw, id_pred=id_pred,
              log_id_true=log_id_true, log_id_pred=log_id_pred)

    print("Saved outputs/model_weights.pt, outputs/weights.json, outputs/metrics.json, "
          "outputs/test_predictions.npz")


if __name__ == "__main__":
    main()
