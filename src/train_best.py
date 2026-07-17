"""Extended training runs to find the best-performing model on
data_cleaned_2/merged_ann_dataset.csv.

Two levers over src/train.py, which had a binding 600-epoch cap:
  * much longer budget (up to 2500 epochs, patience 250), and
  * an optional monotonicity penalty enforcing dID/dVD >= 0 everywhere
    (gds >= 0 -- negative output conductance is what destabilizes SPICE
    Newton iterations) and dID/dVG >= 0 in the on-region (vg_s > 0.6,
    i.e. VG > 1V; the measured off-region has a real non-monotonic
    leakage valley that must NOT be penalized away).

The penalty is computed by autograd on collocation points drawn uniformly
from the scaled input cube each step, so monotonicity is encouraged across
the whole (VG, VD, W, L) continuum, not just at training samples. Since
ID = 10**(y*y_std + y_mean) with y_std > 0, monotonicity of the raw network
output y is equivalent to monotonicity of ID.

Each config trains and reports test metrics plus a gds-violation audit;
results go to outputs/best_runs/. Pick the winner with --finalize <name>,
which copies its artifacts into outputs/ (the canonical location the
Verilog-A exporter reads from).

Usage:
    python -m src.train_best            # run all configs
    python -m src.train_best --finalize <config_name>
"""
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from src.dataset import load_and_split
from src.model import TFTNet

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data_cleaned_2", "merged_ann_dataset.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
RUNS_DIR = os.path.join(OUT_DIR, "best_runs")

SEED = 42
MAX_EPOCHS = 2500
PATIENCE = 250
BATCH_SIZE = 256
LR = 2e-3
N_COLLOC = 512          # collocation points per step for the penalty
VG_ON_THRESH = 0.6      # scaled VG above which gm >= 0 is enforced (VG > 1V)

CONFIGS = [
    {"name": "h22_plain", "n_hidden": 22, "lam": 0.0},
    {"name": "h22_mono",  "n_hidden": 22, "lam": 0.5},
    {"name": "h32_mono",  "n_hidden": 32, "lam": 0.5},
    # batch 2: lambda=0.5 proved far too weak (penalty ~0.002 against a data
    # MSE of ~0.06 barely moved the gds audit); step it up hard
    {"name": "h22_mono_l10",  "n_hidden": 22, "lam": 10.0},
    {"name": "h22_mono_l100", "n_hidden": 22, "lam": 100.0},
    {"name": "h32_mono_l10",  "n_hidden": 32, "lam": 10.0},
    {"name": "h32_mono_l100", "n_hidden": 32, "lam": 100.0},
]


def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


def monotonicity_penalty(model, n_points, gen):
    """Mean squared hinge on negative dID/dVD (everywhere) and negative
    dID/dVG (on-region only), evaluated at random collocation points."""
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
    """Fraction of a dense (VG, VD, W, L) grid with gds < 0, and the worst
    negative gds in siemens, computed from the model's analytic gradient."""
    vg = torch.linspace(0.5, 1.0, 12)        # VG in [0, 5] V (on-region and above)
    vd = torch.linspace(0.002, 0.998, 100)
    w_ = torch.tensor([0.0, 1/31., 3/31., 7/31., 15/31., 1.0])   # the 6 measured widths
    l_ = torch.tensor([0.0, 1/3., 2/3., 1.0])
    G, D, W, L = torch.meshgrid(vg, vd, w_, l_, indexing="ij")
    x = torch.stack([G, D, W, L], dim=-1).reshape(-1, 4)
    x.requires_grad_(True)
    y = model(x)
    (grad,) = torch.autograd.grad(y.sum(), x)
    log_id = (y.squeeze(1) * y_std + y_mean).detach()
    id_lin = 10.0 ** log_id
    # gds = dID/dVD = ID * ln10 * y_std * dy/d(vd_s) / (vd_hi - vd_lo)
    gds = id_lin * np.log(10.0) * y_std * grad[:, 1].detach() / 5.0
    neg = gds < 0
    return float(neg.float().mean()), float(gds.min())


def train_config(cfg, splits):
    train, val, test = splits
    y_mean = float(train.log_id.mean())
    y_std = float(train.log_id.std())

    Xtr = to_tensor(train.X)
    ytr = to_tensor((train.log_id - y_mean) / y_std)
    Xva = to_tensor(val.X)
    yva = to_tensor((val.log_id - y_mean) / y_std)
    Xte = to_tensor(test.X)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    gen = torch.Generator().manual_seed(SEED)

    model = TFTNet(n_inputs=4, n_hidden=cfg["n_hidden"], n_outputs=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=60)
    loss_fn = nn.MSELoss()
    lam = cfg["lam"]

    n_train = Xtr.shape[0]
    best_score = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
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
        if lam > 0:
            val_pen = monotonicity_penalty(model, 4096, torch.Generator().manual_seed(7)).item()
        else:
            val_pen = 0.0
        score = val_mse + lam * val_pen
        scheduler.step(score)

        if score < best_score - 1e-6:
            best_score = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 100 == 0 or epoch == 1:
            print(f"[{cfg['name']}] epoch {epoch:4d}  val_mse={val_mse:.5f}  "
                  f"val_pen={val_pen:.5f}  best@{best_epoch}", flush=True)
        if epochs_no_improve >= PATIENCE:
            print(f"[{cfg['name']}] early stop at {epoch} (best epoch {best_epoch})", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_std = model(Xte).squeeze(1).numpy()
    log_id_pred = pred_std * y_std + y_mean
    log_id_true = test.log_id
    rmse = float(np.sqrt(np.mean((log_id_pred - log_id_true) ** 2)))
    mae = float(np.mean(np.abs(log_id_pred - log_id_true)))
    ss_res = np.sum((log_id_true - log_id_pred) ** 2)
    ss_tot = np.sum((log_id_true - log_id_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    frac_neg, worst_gds = gds_audit(model, y_mean, y_std)

    result = {
        "config": cfg, "best_epoch": best_epoch, "epochs_no_improve_stop": PATIENCE,
        "val_score": best_score,
        "test_rmse_log10ID": rmse, "test_mae_log10ID": mae, "test_r2_log10ID": r2,
        "gds_negative_fraction_on_region": frac_neg, "worst_gds_S": worst_gds,
        "train_seconds": time.time() - t0,
        "target_standardization": {"y_mean_log10_absID": y_mean, "y_std_log10_absID": y_std},
    }

    run_dir = os.path.join(RUNS_DIR, cfg["name"])
    os.makedirs(run_dir, exist_ok=True)
    torch.save(best_state, os.path.join(run_dir, "model_weights.pt"))
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{cfg['name']}] DONE  r2={r2:.4f} rmse={rmse:.4f} "
          f"gds_neg={frac_neg*100:.1f}% worst={worst_gds:.2e}S  "
          f"({result['train_seconds']:.0f}s)", flush=True)
    return result


def finalize(name):
    """Promote a run's artifacts to outputs/ in the exact format train.py
    writes, so evaluate.py and export_verilog_a.py work unchanged."""
    run_dir = os.path.join(RUNS_DIR, name)
    with open(os.path.join(run_dir, "result.json")) as f:
        res = json.load(f)
    n_hidden = res["config"]["n_hidden"]
    y_mean = res["target_standardization"]["y_mean_log10_absID"]
    y_std = res["target_standardization"]["y_std_log10_absID"]

    sd = torch.load(os.path.join(run_dir, "model_weights.pt"))
    shutil.copy(os.path.join(run_dir, "model_weights.pt"),
                os.path.join(OUT_DIR, "model_weights.pt"))

    weights = {
        "architecture": {"n_inputs": 4, "n_hidden": n_hidden, "n_outputs": 1,
                          "input_order": ["VG", "VD", "W", "L"]},
        "input_scaling_minmax": {
            "VG": [-5.0, 5.0], "VD": [0.0, 5.0], "W": [5.0, 160.0], "L": [5.0, 20.0],
        },
        "wh": sd["hidden.weight"].numpy().tolist(),
        "bh": sd["hidden.bias"].numpy().tolist(),
        "wo": sd["output.weight"].numpy().tolist(),
        "bo": sd["output.bias"].numpy().tolist(),
        "target_transform": {
            "definition": "network_output = (log10(|ID|) - y_mean) / y_std",
            "y_mean_log10_absID": y_mean,
            "y_std_log10_absID": y_std,
            "recover_ID": "ID = 10 ** (network_output * y_std + y_mean)",
        },
    }
    with open(os.path.join(OUT_DIR, "weights.json"), "w") as f:
        json.dump(weights, f, indent=2)

    # regenerate test predictions + metrics.json for the promoted model
    train, val, test = load_and_split(DATA_PATH, seed=SEED)
    model = TFTNet(4, n_hidden, 1)
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        pred_std = model(to_tensor(test.X)).squeeze(1).numpy()
    log_id_pred = pred_std * y_std + y_mean
    id_pred = 10.0 ** log_id_pred
    np.savez(os.path.join(OUT_DIR, "test_predictions.npz"),
              X=test.X, id_true=test.id_raw, id_pred=id_pred,
              log_id_true=test.log_id, log_id_pred=log_id_pred)

    metrics = {
        "n_hidden": n_hidden,
        "hidden_activation": "tanh",
        "output_activation": "linear",
        "training": f"src/train_best.py config '{name}' "
                    f"(monotonicity lambda={res['config']['lam']}, best epoch {res['best_epoch']})",
        "train_rows": len(train.df), "val_rows": len(val.df), "test_rows": len(test.df),
        "target_standardization": res["target_standardization"],
        "test_rmse_log10ID": res["test_rmse_log10ID"],
        "test_mae_log10ID": res["test_mae_log10ID"],
        "test_r2_log10ID": res["test_r2_log10ID"],
        "gds_negative_fraction_on_region": res["gds_negative_fraction_on_region"],
        "worst_gds_S": res["worst_gds_S"],
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Promoted '{name}' -> outputs/ (n_hidden={n_hidden})")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--finalize":
        finalize(sys.argv[2])
        return
    os.makedirs(RUNS_DIR, exist_ok=True)
    splits = load_and_split(DATA_PATH, seed=SEED)
    print(f"train={len(splits[0].df)} val={len(splits[1].df)} test={len(splits[2].df)}", flush=True)
    results = []
    for cfg in CONFIGS:
        done = os.path.join(RUNS_DIR, cfg["name"], "result.json")
        if os.path.exists(done):
            with open(done) as f:
                results.append(json.load(f))
            continue
        results.append(train_config(cfg, splits))
    print("\n=== summary ===")
    for r in results:
        print(f"{r['config']['name']:12s} r2={r['test_r2_log10ID']:.4f} "
              f"rmse={r['test_rmse_log10ID']:.4f} gds_neg={r['gds_negative_fraction_on_region']*100:5.1f}% "
              f"worst_gds={r['worst_gds_S']:.2e}S best_epoch={r['best_epoch']}")


if __name__ == "__main__":
    main()
