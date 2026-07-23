"""Self-contained ANN training script for the a-GIZO TFT drain-current
model, trained on cleaned_output_meas/merged_output_dataset.csv (the
cleanest single-device-per-geometry output/Id-Vd sweep dataset).

Architecture follows Eq. (1)-(2) of Bahubalindruni et al., "a-GIZO TFT
neural modeling, circuit simulation and validation", Solid-State
Electronics 105 (2015) 30-36:

    yh = tanh(x . wh + bh)      hidden layer
    y  = yh . wo + bo           output layer (LINEAR, no activation)

Inputs: VG, VD, W, L, min-max scaled to [0,1].
Target: log10(|ID|), standardized to zero mean / unit variance (ID spans
~11 decades from leakage floor to on-state, so raw-current regression
would only fit the top decade).

Training also enforces monotonicity (dID/dVD >= 0 everywhere, dID/dVG >= 0
in the on-region) via an autograd penalty at random collocation points.
This directly targets negative output conductance (gds < 0), which is
what previously destabilized Cadence Spectre transient simulation of a
circuit built from this model -- a plain R^2-optimized fit can look
accurate on held-out data while still being locally non-monotonic enough
to break a SPICE Newton-Raphson solve.

Usage:
    python ann_train.py
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
OUT_DIR = HERE

SEED = 42
# Winner of a 4-way comparison (n_hidden in {22,32} x lambda in {0,100}, see
# git history): 22 neurons + lambda=100 matched the 32-neuron model's accuracy
# (R^2 0.973 vs 0.974) with an 8x smaller worst-case negative-gds violation
# and less than half the parameters -- strictly better for circuit simulation.
N_HIDDEN = 22
MONO_LAMBDA = 100.0          # squared-hinge weight on non-monotonic derivatives
N_COLLOC = 512                # collocation points per training step for the penalty
VG_ON_THRESH = 0.6            # scaled VG above which dID/dVG >= 0 is enforced (VG > 1V)
LR = 2e-3
BATCH_SIZE = 256
MAX_EPOCHS = 2500
PATIENCE = 250

FEATURES = ["VG", "VD", "W", "L"]
FEATURE_BOUNDS = {
    "VG": (-5.0, 5.0),
    "VD": (0.0, 5.0),
    "W": (5.0, 160.0),
    "L": (5.0, 20.0),
}


class TFTNet(nn.Module):
    def __init__(self, n_inputs=4, n_hidden=N_HIDDEN, n_outputs=1):
        super().__init__()
        self.hidden = nn.Linear(n_inputs, n_hidden)   # wh, bh
        self.tanh = nn.Tanh()
        self.output = nn.Linear(n_hidden, n_outputs)  # wo, bo (linear)

    def forward(self, x):
        return self.output(self.tanh(self.hidden(x)))


def scale_features(df):
    cols = []
    for name in FEATURES:
        lo, hi = FEATURE_BOUNDS[name]
        cols.append(((df[name].to_numpy() - lo) / (hi - lo)).astype(np.float32))
    return np.stack(cols, axis=1)


def load_and_split(csv_path, seed=SEED, train_frac=0.7, val_frac=0.15):
    df = pd.read_csv(csv_path).drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_train = int(round(len(df) * train_frac))
    n_val = int(round(len(df) * val_frac))
    splits = {}
    for name, part in [("train", idx[:n_train]), ("val", idx[n_train:n_train + n_val]),
                        ("test", idx[n_train + n_val:])]:
        sub = df.iloc[part].reset_index(drop=True)
        splits[name] = {
            "df": sub,
            "X": scale_features(sub),
            "log_id": sub["log_ID"].to_numpy(dtype=np.float32),
            "id_raw": sub["ID"].to_numpy(dtype=np.float64),
        }
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
    """Fraction of a dense on-region (VG>=0V) grid with predicted gds < 0,
    and the worst violation in siemens -- the direct proxy for whether this
    model will behave in a SPICE transient solve."""
    vg = torch.linspace(0.5, 1.0, 12)   # VG in [0, 5] V
    vd = torch.linspace(0.002, 0.998, 100)
    w_ = torch.tensor([0.0, 1/31., 3/31., 7/31., 15/31., 1.0])
    l_ = torch.tensor([0.0, 1/3., 2/3., 1.0])
    G, D, W, L = torch.meshgrid(vg, vd, w_, l_, indexing="ij")
    x = torch.stack([G, D, W, L], dim=-1).reshape(-1, 4)
    x.requires_grad_(True)
    y = model(x)
    (grad,) = torch.autograd.grad(y.sum(), x)
    log_id = (y.squeeze(1) * y_std + y_mean).detach()
    id_lin = 10.0 ** log_id
    gds = id_lin * np.log(10.0) * y_std * grad[:, 1].detach() / (FEATURE_BOUNDS["VD"][1] - FEATURE_BOUNDS["VD"][0])
    neg = gds < 0
    return float(neg.float().mean()), float(gds.min())


def train(n_hidden=N_HIDDEN, lam=MONO_LAMBDA, seed=SEED, verbose=True):
    train_s, val_s, test_s = load_and_split(DATA_PATH, seed=seed)
    y_mean = float(train_s["log_id"].mean())
    y_std = float(train_s["log_id"].std())

    Xtr = to_tensor(train_s["X"])
    ytr = to_tensor((train_s["log_id"] - y_mean) / y_std)
    Xva = to_tensor(val_s["X"])
    yva = to_tensor((val_s["log_id"] - y_mean) / y_std)
    Xte = to_tensor(test_s["X"])
    log_id_test = test_s["log_id"]

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    model = TFTNet(4, n_hidden, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=60)
    loss_fn = nn.MSELoss()

    n_train = Xtr.shape[0]
    best_score = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train, generator=gen)
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx].unsqueeze(1))
            if lam > 0:
                loss = loss + lam * monotonicity_penalty(model, N_COLLOC, gen)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mse = loss_fn(model(Xva), yva.unsqueeze(1)).item()
        val_pen = monotonicity_penalty(model, 4096, torch.Generator().manual_seed(7)).item() if lam > 0 else 0.0
        score = val_mse + lam * val_pen
        scheduler.step(score)

        if score < best_score - 1e-6:
            best_score, best_state, best_epoch, no_improve = score, {k: v.clone() for k, v in model.state_dict().items()}, epoch, 0
        else:
            no_improve += 1

        if verbose and (epoch % 100 == 0 or epoch == 1):
            print(f"epoch {epoch:4d}  val_mse={val_mse:.5f}  val_pen={val_pen:.5f}  best@{best_epoch}", flush=True)
        if no_improve >= PATIENCE:
            print(f"early stop at epoch {epoch} (best epoch {best_epoch})", flush=True)
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

    bins = [1e-9, 1e-6, 1e-3, 1e0]
    mare_by_band = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (np.abs(test_s["id_raw"]) >= lo) & (np.abs(test_s["id_raw"]) < hi)
        if mask.sum() == 0:
            continue
        mare = float(np.mean(np.abs(np.abs(test_s["id_raw"][mask]) - id_pred[mask]) / np.abs(test_s["id_raw"][mask])) * 100)
        mare_by_band[f"{lo:.0e}_to_{hi:.0e}_A"] = {"MARE_percent": mare, "n_rows": int(mask.sum())}

    result = {
        "n_hidden": n_hidden, "mono_lambda": lam, "seed": seed,
        "best_epoch": best_epoch, "train_seconds": time.time() - t0,
        "train_rows": len(train_s["df"]), "val_rows": len(val_s["df"]), "test_rows": len(test_s["df"]),
        "test_rmse_log10ID": rmse, "test_mae_log10ID": mae, "test_r2_log10ID": r2,
        "gds_negative_fraction_on_region": frac_neg, "worst_gds_S": worst_gds,
        "test_MARE_percent_by_current_band": mare_by_band,
        "target_standardization": {"y_mean_log10_absID": y_mean, "y_std_log10_absID": y_std},
    }
    return model, result, (Xte, test_s)


def export(model, result, out_dir=OUT_DIR):
    sd = model.state_dict()
    y_mean = result["target_standardization"]["y_mean_log10_absID"]
    y_std = result["target_standardization"]["y_std_log10_absID"]

    weights = {
        "architecture": {"n_inputs": 4, "n_hidden": result["n_hidden"], "n_outputs": 1,
                          "input_order": FEATURES},
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
    with open(os.path.join(out_dir, "weights.json"), "w") as f:
        json.dump(weights, f, indent=2)
    torch.save(sd, os.path.join(out_dir, "model_weights.pt"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    wh, bh, wo, bo = weights["wh"], weights["bh"], weights["wo"][0], weights["bo"][0]
    lines = [
        "a-GIZO TFT ANN model -- trained weights and biases",
        "=" * 55, "",
        f"Architecture: 4 inputs -> {result['n_hidden']} hidden neurons (tanh) -> 1 output (linear)",
        "Input order: VG, VD, W, L", "",
        "Trained on: cleaned_output_meas/merged_output_dataset.csv "
        "(one cleanest device per geometry, output/Id-Vd sweeps only)",
        f"  train rows = {result['train_rows']}, val rows = {result['val_rows']}, test rows = {result['test_rows']}",
        f"Training: Adam lr={LR}, batch={BATCH_SIZE}, monotonicity lambda={result['mono_lambda']}, "
        f"best epoch {result['best_epoch']}",
        f"Test RMSE (log10|ID|) = {result['test_rmse_log10ID']:.6f} decades",
        f"Test MAE  (log10|ID|) = {result['test_mae_log10ID']:.6f} decades",
        f"Test R^2  (log10|ID|) = {result['test_r2_log10ID']:.6f}",
        f"gds audit: negative on {result['gds_negative_fraction_on_region']*100:.1f}% of on-region grid, "
        f"worst {result['worst_gds_S']:.2e} S",
        "",
        "Input scaling (min-max to [0,1]): x_scaled = (x - lo) / (hi - lo)",
    ]
    for k, (lo, hi) in FEATURE_BOUNDS.items():
        lines.append(f"  {k}: lo={lo}, hi={hi}")
    lines += [
        "",
        "Output transform: network_output = (log10(|ID|) - y_mean) / y_std",
        f"  y_mean = {y_mean:.8f}",
        f"  y_std  = {y_std:.8f}",
        "  recover ID: ID = 10 ** (network_output * y_std + y_mean)",
        "",
        "-" * 55, "Hidden layer weights wh[neuron][input]  (input order: VG, VD, W, L)", "-" * 55,
    ]
    for i, row in enumerate(wh):
        lines.append(f"neuron {i:2d}:  " + "  ".join(f"{v: .6f}" for v in row))
    lines += ["", "-" * 55, "Hidden layer biases bh[neuron]", "-" * 55]
    for i, v in enumerate(bh):
        lines.append(f"neuron {i:2d}:  {v: .6f}")
    lines += ["", "-" * 55, "Output layer weights wo[neuron]", "-" * 55]
    for i, v in enumerate(wo):
        lines.append(f"neuron {i:2d}:  {v: .6f}")
    lines += ["", "-" * 55, "Output layer bias bo", "-" * 55, f"bo:  {bo: .6f}", ""]

    with open(os.path.join(out_dir, "weights_and_biases.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    model, result, _ = train()
    print(json.dumps(result, indent=2))
    export(model, result)
    print("Saved model_weights.pt, weights.json, weights_and_biases.txt, metrics.json to", OUT_DIR)


if __name__ == "__main__":
    main()
