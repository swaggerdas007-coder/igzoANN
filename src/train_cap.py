"""Train two small MLPs (tanh hidden / linear output) to model
CGD = f(VG, W, L) and CGS = f(VG, W, L), following the same "separate ANN
per element" approach as the paper (Bahubalindruni et al. 2016, Sec. 3) and
the same architecture family as src/train.py's ID model -- but see
data_cv_cleaned/README.md: this is a VDS=0 slice, not the paper's full
VDS-resolved CGD/CGS surface, and only 4 (W, L) geometries are covered
(vs. 19 for the ID model), so a much smaller hidden layer is used to avoid
overfitting the tiny dataset (164 rows total).

Usage:
    python -m src.train_cap
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

from src.dataset_cap import FEATURE_BOUNDS, FEATURES, load_and_split
from src.model import TFTNet

SEED = 42
N_HIDDEN = 10
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data_cv_cleaned", "merged_cv_dataset.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs_cap")

torch.manual_seed(SEED)
np.random.seed(SEED)


def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


def train_one(target_name, train, val, test):
    y_mean = float(train.y[target_name].mean())
    y_std = float(train.y[target_name].std())

    def target(split):
        return (split.y[target_name] - y_mean) / y_std

    Xtr, ytr = to_tensor(train.X), to_tensor(target(train))
    Xva, yva = to_tensor(val.X), to_tensor(target(val))
    Xte = to_tensor(test.X)

    model = TFTNet(n_inputs=len(FEATURES), n_hidden=N_HIDDEN, n_outputs=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=60)
    loss_fn = nn.MSELoss()

    batch_size = 32
    n_train = Xtr.shape[0]
    max_epochs = 2000
    patience = 200
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx], ytr[idx].unsqueeze(1)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), yva.unsqueeze(1)).item()
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 200 == 0 or epoch == 1:
            print(f"[{target_name}] epoch {epoch:4d}  val_mse={val_loss:.6f}  best={best_val:.6f}")
        if epochs_no_improve >= patience:
            print(f"[{target_name}] early stop at epoch {epoch} (best val_mse={best_val:.6f})")
            break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_pred_std = model(Xte).squeeze(1).numpy()
    c_pred = test_pred_std * y_std + y_mean
    c_true = test.y[target_name]

    rmse = float(np.sqrt(np.mean((c_pred - c_true) ** 2)))
    mae = float(np.mean(np.abs(c_pred - c_true)))
    ss_res = np.sum((c_true - c_pred) ** 2)
    ss_tot = np.sum((c_true - c_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mare = float(np.mean(np.abs(c_true - c_pred) / np.abs(c_true)) * 100)

    metrics = {
        "target": target_name,
        "n_hidden": N_HIDDEN,
        "hidden_activation": "tanh",
        "output_activation": "linear",
        "train_rows": len(train.df), "val_rows": len(val.df), "test_rows": len(test.df),
        "target_standardization_farads": {"mean": y_mean, "std": y_std},
        "test_rmse_F": rmse, "test_mae_F": mae, "test_r2": r2, "test_MARE_percent": mare,
    }

    sd = model.state_dict()
    weights = {
        "architecture": {"n_inputs": len(FEATURES), "n_hidden": N_HIDDEN, "n_outputs": 1,
                          "input_order": FEATURES},
        "input_scaling_minmax": {name: list(FEATURE_BOUNDS[name]) for name in FEATURES},
        "wh": sd["hidden.weight"].numpy().tolist(),
        "bh": sd["hidden.bias"].numpy().tolist(),
        "wo": sd["output.weight"].numpy().tolist(),
        "bo": sd["output.bias"].numpy().tolist(),
        "target_transform": {
            "definition": f"network_output = ({target_name}[F] - mean) / std",
            "mean_F": y_mean,
            "std_F": y_std,
            "recover": f"{target_name}[F] = network_output * std + mean",
        },
    }

    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"model_weights_{target_name.lower()}.pt"))
    with open(os.path.join(OUT_DIR, f"weights_{target_name.lower()}.json"), "w") as f:
        json.dump(weights, f, indent=2)

    return metrics, (test.X, c_true, c_pred)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train, val, test = load_and_split(DATA_PATH, seed=SEED)
    print(f"train={len(train.df)}  val={len(val.df)}  test={len(test.df)}  (rows)")

    all_metrics = {}
    all_preds = {}
    for target_name in ("CGD", "CGS"):
        metrics, preds = train_one(target_name, train, val, test)
        print(json.dumps(metrics, indent=2))
        all_metrics[target_name] = metrics
        all_preds[target_name] = preds

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    np.savez(os.path.join(OUT_DIR, "test_predictions.npz"),
              X=test.X,
              cgd_true=all_preds["CGD"][1], cgd_pred=all_preds["CGD"][2],
              cgs_true=all_preds["CGS"][1], cgs_pred=all_preds["CGS"][2])

    print(f"Saved {OUT_DIR}/weights_cgd.json, weights_cgs.json, metrics.json, "
          f"model_weights_{{cgd,cgs}}.pt, test_predictions.npz")


if __name__ == "__main__":
    main()
