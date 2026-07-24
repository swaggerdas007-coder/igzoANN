"""2-hidden-layer variant: 4 inputs -> 20 tanh -> 20 tanh -> 1 linear.

Same training recipe as the delivered 1-layer model (ann_train.py): Adam,
monotonicity penalty (dID/dVD>=0 everywhere, dID/dVG>=0 on-region) via
autograd collocation, and L-balanced loss weighting. Trained on
cleaned_output_meas/merged_output_dataset.csv.

Usage:
    python ann_train_deep.py
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "cleaned_output_meas", "merged_output_dataset.csv")
OUT_DIR = os.path.join(HERE, "deep")

SEED = 42
HIDDEN_SIZES = [20, 20]
MONO_LAMBDA = 100.0
N_COLLOC = 512
VG_ON_THRESH = 0.6
LR = 2e-3
BATCH_SIZE = 256
MAX_EPOCHS = 2500
PATIENCE = 200

FEATURES = ["VG", "VD", "W", "L"]
FEATURE_BOUNDS = {"VG": (-5.0, 5.0), "VD": (0.0, 5.0), "W": (5.0, 160.0), "L": (5.0, 20.0)}


class TFTNetDeep(nn.Module):
    def __init__(self, hidden_sizes=HIDDEN_SIZES):
        super().__init__()
        self.h1 = nn.Linear(4, hidden_sizes[0])
        self.h2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.out = nn.Linear(hidden_sizes[1], 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.h1(x))
        x = self.tanh(self.h2(x))
        return self.out(x)


def scale_features(df):
    cols = []
    for name in FEATURES:
        lo, hi = FEATURE_BOUNDS[name]
        cols.append(((df[name].to_numpy() - lo) / (hi - lo)).astype(np.float32))
    return np.stack(cols, axis=1)


def l_balance_weights(df):
    counts = df["L"].map(df["L"].value_counts())
    n_l = df["L"].nunique()
    return ((len(df) / n_l) / counts).to_numpy(dtype=np.float32)


def load_and_split(seed=SEED, train_frac=0.7, val_frac=0.15):
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_train = int(round(len(df) * train_frac))
    n_val = int(round(len(df) * val_frac))
    splits = {}
    for name, part in [("train", idx[:n_train]), ("val", idx[n_train:n_train + n_val]),
                        ("test", idx[n_train + n_val:])]:
        sub = df.iloc[part].reset_index(drop=True)
        splits[name] = {"df": sub, "X": scale_features(sub),
                         "log_id": sub["log_ID"].to_numpy(dtype=np.float32),
                         "id_raw": sub["ID"].to_numpy(dtype=np.float64),
                         "l_weight": l_balance_weights(sub)}
    return splits["train"], splits["val"], splits["test"]


def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


def monotonicity_penalty(model, n_points, gen):
    x = torch.rand(n_points, 4, generator=gen)
    x.requires_grad_(True)
    y = model(x).sum()
    (grad,) = torch.autograd.grad(y, x, create_graph=True)
    g_vg, g_vd = grad[:, 0], grad[:, 1]
    pen_vd = torch.relu(-g_vd).pow(2).mean()
    on_mask = (x[:, 0] > VG_ON_THRESH).detach().float()
    pen_vg = (torch.relu(-g_vg).pow(2) * on_mask).sum() / on_mask.sum().clamp(min=1.0)
    return pen_vd + pen_vg


def gds_audit(model, y_mean, y_std):
    vg = torch.linspace(0.5, 1.0, 12)
    vd = torch.linspace(0.002, 0.998, 100)
    w_ = torch.tensor([0.0, 1 / 31., 3 / 31., 7 / 31., 15 / 31., 1.0])
    l_ = torch.tensor([0.0, 1 / 3., 2 / 3., 1.0])
    G, D, W, L = torch.meshgrid(vg, vd, w_, l_, indexing="ij")
    x = torch.stack([G, D, W, L], dim=-1).reshape(-1, 4)
    x.requires_grad_(True)
    y = model(x)
    (grad,) = torch.autograd.grad(y.sum(), x)
    log_id = (y.squeeze(1) * y_std + y_mean).detach()
    id_lin = 10.0 ** log_id
    gds = id_lin * np.log(10.0) * y_std * grad[:, 1].detach() / (FEATURE_BOUNDS["VD"][1] - FEATURE_BOUNDS["VD"][0])
    return float((gds < 0).float().mean()), float(gds.min())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train_s, val_s, test_s = load_and_split()
    y_mean = float(train_s["log_id"].mean())
    y_std = float(train_s["log_id"].std())

    Xtr = to_tensor(train_s["X"]); ytr = to_tensor((train_s["log_id"] - y_mean) / y_std)
    Wtr = to_tensor(train_s["l_weight"])
    Xva = to_tensor(val_s["X"]); yva = to_tensor((val_s["log_id"] - y_mean) / y_std)
    Wva = to_tensor(val_s["l_weight"])
    Xte = to_tensor(test_s["X"]); log_id_test = test_s["log_id"]

    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)

    model = TFTNetDeep(HIDDEN_SIZES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=60)

    def weighted_mse(pred, target, weight):
        return (weight.unsqueeze(1) * (pred - target) ** 2).mean()

    n_train = Xtr.shape[0]
    best_score = float("inf"); best_state = None; best_epoch = 0; no_improve = 0
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train, generator=gen)
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = weighted_mse(model(Xtr[idx]), ytr[idx].unsqueeze(1), Wtr[idx])
            loss = loss + MONO_LAMBDA * monotonicity_penalty(model, N_COLLOC, gen)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mse = weighted_mse(model(Xva), yva.unsqueeze(1), Wva).item()
        val_pen = monotonicity_penalty(model, 4096, torch.Generator().manual_seed(7)).item()
        score = val_mse + MONO_LAMBDA * val_pen
        scheduler.step(score)

        if score < best_score - 1e-6:
            best_score, best_state, best_epoch, no_improve = score, {k: v.clone() for k, v in model.state_dict().items()}, epoch, 0
        else:
            no_improve += 1
        if epoch % 100 == 0 or epoch == 1:
            print(f"epoch {epoch:4d} val_mse={val_mse:.5f} val_pen={val_pen:.5f} best@{best_epoch}", flush=True)
        if no_improve >= PATIENCE:
            print(f"early stop at {epoch} (best {best_epoch})", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_std = model(Xte).squeeze(1).numpy()
    log_id_pred = pred_std * y_std + y_mean
    id_pred = 10.0 ** log_id_pred

    rmse = float(np.sqrt(np.mean((log_id_pred - log_id_test) ** 2)))
    mae = float(np.mean(np.abs(log_id_pred - log_id_test)))
    ss_res = np.sum((log_id_test - log_id_pred) ** 2)
    ss_tot = np.sum((log_id_test - log_id_test.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    frac_neg, worst_gds = gds_audit(model, y_mean, y_std)

    rmse_by_l = {}
    for l_val in sorted(test_s["df"]["L"].unique()):
        mask = (test_s["df"]["L"] == l_val).to_numpy()
        rmse_by_l[int(l_val)] = float(np.sqrt(np.mean((log_id_pred[mask] - log_id_test[mask]) ** 2)))

    result = {
        "hidden_sizes": HIDDEN_SIZES, "mono_lambda": MONO_LAMBDA, "seed": SEED, "balance_l": True,
        "best_epoch": best_epoch, "train_seconds": time.time() - t0,
        "train_rows": len(train_s["df"]), "val_rows": len(val_s["df"]), "test_rows": len(test_s["df"]),
        "test_rmse_log10ID": rmse, "test_mae_log10ID": mae, "test_r2_log10ID": r2,
        "gds_negative_fraction_on_region": frac_neg, "worst_gds_S": worst_gds,
        "test_rmse_log10ID_by_L": rmse_by_l,
        "target_standardization": {"y_mean_log10_absID": y_mean, "y_std_log10_absID": y_std},
    }
    print(json.dumps(result, indent=2))

    sd = model.state_dict()
    weights = {
        "architecture": {"n_inputs": 4, "hidden_sizes": HIDDEN_SIZES, "n_outputs": 1,
                          "input_order": FEATURES},
        "input_scaling_minmax": {k: list(v) for k, v in FEATURE_BOUNDS.items()},
        "layers": [
            {"w": sd["h1.weight"].numpy().tolist(), "b": sd["h1.bias"].numpy().tolist()},
            {"w": sd["h2.weight"].numpy().tolist(), "b": sd["h2.bias"].numpy().tolist()},
        ],
        "output": {"w": sd["out.weight"].numpy().tolist(), "b": sd["out.bias"].numpy().tolist()},
        "target_transform": {
            "definition": "network_output = (log10(|ID|) - y_mean) / y_std",
            "y_mean_log10_absID": y_mean, "y_std_log10_absID": y_std,
            "recover_ID": "ID = 10 ** (network_output * y_std + y_mean)",
        },
    }
    with open(os.path.join(OUT_DIR, "weights.json"), "w") as f:
        json.dump(weights, f, indent=2)
    torch.save(sd, os.path.join(OUT_DIR, "model_weights.pt"))
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    np.savez(os.path.join(OUT_DIR, "test_predictions.npz"), id_true=test_s["id_raw"], id_pred=id_pred)
    print("Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
