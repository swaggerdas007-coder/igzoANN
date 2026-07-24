"""Per-length ANN ensemble: trains one small MLP per measured channel
length L (5, 10, 15, 20 um), each learning ID = f(VG, VD, W) -- i.e. the
same architecture as ann_train.py but with L dropped as an input, since a
single Verilog-A instance always has one fixed L (a device parameter, not
something swept at runtime).

Motivation: the single 4-input (VG,VD,W,L) model's worst-fitting geometry
was consistently L=10, because the measured (W,L) grid is unbalanced (L=5
has 6 measured widths, L=10-20 have 4-5) and a shared network has to
interpolate/extrapolate the L dependence from that uneven, incomplete
grid. Training one 3-input network per L removes L-interpolation from the
problem entirely -- each subnetwork only ever has to fit its own W range,
using only its own geometry's data, with no cross-L interference.

Merging for Verilog-A: L is a `parameter real` fixed per instance, so the
merged module picks the nearest trained L's weight set at `initial_step`
(compile-time-equivalent selection, zero runtime cost) via distance-based
matching against {5, 10, 15, 20} um -- see export_per_L_verilog_a() and
scripts/export_verilog_a_per_L.py.

Usage:
    python ann_train_per_L.py
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
OUT_DIR = os.path.join(HERE, "per_L")

SEED = 42
N_HIDDEN = 22                 # same size as the unified model, for a fair comparison
MONO_LAMBDA = 100.0
N_COLLOC = 512
LR = 2e-3
BATCH_SIZE = 128               # smaller batch: each per-L dataset is ~1/4 the size
MAX_EPOCHS = 2500
PATIENCE = 250
ALL_L = [5, 10, 15, 20]

FEATURES = ["VG", "VD", "W"]
FEATURE_BOUNDS = {
    "VG": (-5.0, 5.0),
    "VD": (0.0, 5.0),
    "W": (5.0, 160.0),
}


class TFTNet3(nn.Module):
    """Same Eq.(1)-(2) architecture as the unified model's TFTNet, but with
    3 inputs (VG, VD, W) since L is fixed per network."""
    def __init__(self, n_hidden=N_HIDDEN):
        super().__init__()
        self.hidden = nn.Linear(3, n_hidden)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(n_hidden, 1)

    def forward(self, x):
        return self.output(self.tanh(self.hidden(x)))


def scale_features(df):
    cols = []
    for name in FEATURES:
        lo, hi = FEATURE_BOUNDS[name]
        cols.append(((df[name].to_numpy() - lo) / (hi - lo)).astype(np.float32))
    return np.stack(cols, axis=1)


def load_and_split_for_L(df_all, l_value, seed=SEED, train_frac=0.7, val_frac=0.15):
    df = df_all[df_all["L"] == l_value].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_train = int(round(len(df) * train_frac))
    n_val = int(round(len(df) * val_frac))
    splits = {}
    for name, part in [("train", idx[:n_train]), ("val", idx[n_train:n_train + n_val]),
                        ("test", idx[n_train + n_val:])]:
        sub = df.iloc[part].reset_index(drop=True)
        splits[name] = {
            "df": sub, "X": scale_features(sub),
            "log_id": sub["log_ID"].to_numpy(dtype=np.float32),
            "id_raw": sub["ID"].to_numpy(dtype=np.float64),
        }
    return splits["train"], splits["val"], splits["test"]


def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


def monotonicity_penalty(model, n_points, gen, vg_on_thresh=0.6):
    x = torch.rand(n_points, 3, generator=gen)
    x.requires_grad_(True)
    y = model(x).sum()
    (grad,) = torch.autograd.grad(y, x, create_graph=True)
    g_vg, g_vd = grad[:, 0], grad[:, 1]
    pen_vd = torch.relu(-g_vd).pow(2).mean()
    on_mask = (x[:, 0] > vg_on_thresh).detach().float()
    pen_vg = (torch.relu(-g_vg).pow(2) * on_mask).sum() / on_mask.sum().clamp(min=1.0)
    return pen_vd + pen_vg


def gds_audit(model, y_mean, y_std, w_lo, w_hi):
    vg = torch.linspace(0.5, 1.0, 12)
    vd = torch.linspace(0.002, 0.998, 100)
    w_ = torch.linspace(0.0, 1.0, 6)
    G, D, W = torch.meshgrid(vg, vd, w_, indexing="ij")
    x = torch.stack([G, D, W], dim=-1).reshape(-1, 3)
    x.requires_grad_(True)
    y = model(x)
    (grad,) = torch.autograd.grad(y.sum(), x)
    log_id = (y.squeeze(1) * y_std + y_mean).detach()
    id_lin = 10.0 ** log_id
    gds = id_lin * np.log(10.0) * y_std * grad[:, 1].detach() / (FEATURE_BOUNDS["VD"][1] - FEATURE_BOUNDS["VD"][0])
    neg = gds < 0
    return float(neg.float().mean()), float(gds.min())


def train_one_L(df_all, l_value, n_hidden=N_HIDDEN, lam=MONO_LAMBDA, seed=SEED, verbose=True):
    train_s, val_s, test_s = load_and_split_for_L(df_all, l_value, seed=seed)
    y_mean = float(train_s["log_id"].mean())
    y_std = float(train_s["log_id"].std())

    Xtr = to_tensor(train_s["X"]); ytr = to_tensor((train_s["log_id"] - y_mean) / y_std)
    Xva = to_tensor(val_s["X"]); yva = to_tensor((val_s["log_id"] - y_mean) / y_std)
    Xte = to_tensor(test_s["X"]); log_id_test = test_s["log_id"]

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    model = TFTNet3(n_hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=60)

    n_train = Xtr.shape[0]
    best_score = float("inf"); best_state = None; best_epoch = 0; no_improve = 0
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train, generator=gen)
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = ((model(Xtr[idx]) - ytr[idx].unsqueeze(1)) ** 2).mean()
            if lam > 0:
                loss = loss + lam * monotonicity_penalty(model, N_COLLOC, gen)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mse = ((model(Xva) - yva.unsqueeze(1)) ** 2).mean().item()
        val_pen = monotonicity_penalty(model, 4096, torch.Generator().manual_seed(7)).item() if lam > 0 else 0.0
        score = val_mse + lam * val_pen
        scheduler.step(score)

        if score < best_score - 1e-6:
            best_score, best_state, best_epoch, no_improve = score, {k: v.clone() for k, v in model.state_dict().items()}, epoch, 0
        else:
            no_improve += 1
        if verbose and (epoch % 200 == 0 or epoch == 1):
            print(f"  [L={l_value}] epoch {epoch:4d} val_mse={val_mse:.5f} val_pen={val_pen:.5f} best@{best_epoch}", flush=True)
        if no_improve >= PATIENCE:
            print(f"  [L={l_value}] early stop at {epoch} (best {best_epoch})", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_std = model(Xte).squeeze(1).numpy()
    log_id_pred = pred_std * y_std + y_mean
    rmse = float(np.sqrt(np.mean((log_id_pred - log_id_test) ** 2)))
    mae = float(np.mean(np.abs(log_id_pred - log_id_test)))
    ss_res = np.sum((log_id_test - log_id_pred) ** 2)
    ss_tot = np.sum((log_id_test - log_id_test.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    frac_neg, worst_gds = gds_audit(model, y_mean, y_std, *FEATURE_BOUNDS["W"])

    result = {
        "L": l_value, "n_hidden": n_hidden, "mono_lambda": lam, "seed": seed,
        "best_epoch": best_epoch, "train_seconds": time.time() - t0,
        "train_rows": len(train_s["df"]), "val_rows": len(val_s["df"]), "test_rows": len(test_s["df"]),
        "test_rmse_log10ID": rmse, "test_mae_log10ID": mae, "test_r2_log10ID": r2,
        "gds_negative_fraction_on_region": frac_neg, "worst_gds_S": worst_gds,
        "target_standardization": {"y_mean_log10_absID": y_mean, "y_std_log10_absID": y_std},
    }
    return model, result


def export_one(model, result, l_value, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sd = model.state_dict()
    y_mean = result["target_standardization"]["y_mean_log10_absID"]
    y_std = result["target_standardization"]["y_std_log10_absID"]
    weights = {
        "architecture": {"n_inputs": 3, "n_hidden": result["n_hidden"], "n_outputs": 1,
                          "input_order": FEATURES, "fixed_L_um": l_value},
        "input_scaling_minmax": {k: list(v) for k, v in FEATURE_BOUNDS.items()},
        "wh": sd["hidden.weight"].numpy().tolist(),
        "bh": sd["hidden.bias"].numpy().tolist(),
        "wo": sd["output.weight"].numpy().tolist(),
        "bo": sd["output.bias"].numpy().tolist(),
        "target_transform": {
            "definition": "network_output = (log10(|ID|) - y_mean) / y_std",
            "y_mean_log10_absID": y_mean, "y_std_log10_absID": y_std,
            "recover_ID": "ID = 10 ** (network_output * y_std + y_mean)",
        },
    }
    with open(os.path.join(out_dir, f"weights_L{l_value}.json"), "w") as f:
        json.dump(weights, f, indent=2)
    torch.save(sd, os.path.join(out_dir, f"model_L{l_value}.pt"))
    with open(os.path.join(out_dir, f"metrics_L{l_value}.json"), "w") as f:
        json.dump(result, f, indent=2)


def main():
    df_all = pd.read_csv(DATA_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = []
    for l_value in ALL_L:
        print(f"\n=== training L={l_value}um ===", flush=True)
        model, result = train_one_L(df_all, l_value)
        export_one(model, result, l_value, OUT_DIR)
        all_results.append(result)
        print(f"[L={l_value}] DONE r2={result['test_r2_log10ID']:.4f} rmse={result['test_rmse_log10ID']:.4f} "
              f"gds_neg={result['gds_negative_fraction_on_region']*100:.2f}% worst={result['worst_gds_S']:.2e}S "
              f"({result['train_seconds']:.0f}s)", flush=True)

    with open(os.path.join(OUT_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n=== SUMMARY ===")
    for r in all_results:
        print(f"L={r['L']:3d}  r2={r['test_r2_log10ID']:.4f}  rmse={r['test_rmse_log10ID']:.4f}  "
              f"gds_neg={r['gds_negative_fraction_on_region']*100:.2f}%  worst={r['worst_gds_S']:.2e}S")


if __name__ == "__main__":
    main()
