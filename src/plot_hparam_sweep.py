"""Plots for the hidden-size hyperparameter sweep (src/hparam_sweep.py).

Reads outputs/sweep_results.csv (n_hidden x seed grid) and outputs/sweep_curves.json
(per-epoch validation loss for a handful of representative hidden sizes), and
produces:

  outputs/plots/sweep_summary.png      -- test R^2, test RMSE, and the
                                           accuracy/complexity tradeoff vs
                                           hidden-layer size, mean +/- std
                                           across seeds
  outputs/plots/sweep_convergence.png  -- validation-loss training curves for
                                           representative hidden sizes

Palette: sequential blue ramp + categorical slot 1 (blue) / slot 6 (orange)
from the project's validated default palette (dataviz skill).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
RESULTS_CSV = os.path.join(OUT_DIR, "sweep_results.csv")
CURVES_JSON = os.path.join(OUT_DIR, "sweep_curves.json")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

BG = "#f7f7f5"
FG = "#1f1f1f"
MUTED = "#6b6a66"
BLUE = "#2a78d6"      # categorical slot 1
ORANGE = "#eb6834"    # categorical slot 6 (highlight)
GREEN = "#008300"     # categorical slot 2
SEQ_BLUE = ["#b7d3f6", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]  # steps 150,300,450,600,700

CHOSEN_N_HIDDEN = 22  # the "final" architecture delivered earlier


def style_axes(ax):
    ax.set_facecolor(BG)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=FG)
    ax.grid(True, alpha=0.2, color=MUTED)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def band_plot(ax, df, metric, color, better="higher"):
    g = df.groupby("n_hidden")[metric].agg(["mean", "std", "count"])
    g["std"] = g["std"].fillna(0.0)
    x = g.index.to_numpy()
    mean = g["mean"].to_numpy()
    std = g["std"].to_numpy()

    ax.scatter(df["n_hidden"], df[metric], s=14, color=color, alpha=0.25, edgecolors="none",
               zorder=2, label="_nolegend_")
    ax.plot(x, mean, color=color, lw=2, zorder=4, label="mean across seeds")
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, zorder=3,
                     label="±1 std (3 seeds)")

    best_idx = np.argmax(mean) if better == "higher" else np.argmin(mean)
    best_n = x[best_idx]
    ax.scatter([best_n], [mean[best_idx]], marker="*", s=220, color=ORANGE, zorder=6,
               edgecolors=FG, linewidths=0.6, label=f"best: n_hidden={best_n}")

    if CHOSEN_N_HIDDEN in x:
        chosen_val = mean[list(x).index(CHOSEN_N_HIDDEN)]
        ax.scatter([CHOSEN_N_HIDDEN], [chosen_val], marker="D", s=70, color=FG, zorder=6,
                   label=f"delivered model (n_hidden={CHOSEN_N_HIDDEN})")
    return g


def summary_plot(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=150)
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    band_plot(ax, df, "test_r2_log10ID", BLUE, better="higher")
    ax.set_xlabel("hidden neurons")
    ax.set_ylabel(r"test $R^2$ (on $\log_{10}|I_D|$)")
    ax.set_title(r"Test $R^2$ vs. hidden-layer size")
    style_axes(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    ax = axes[1]
    band_plot(ax, df, "test_rmse_log10ID", GREEN, better="lower")
    ax.set_xlabel("hidden neurons")
    ax.set_ylabel(r"test RMSE (decades of $|I_D|$)")
    ax.set_title("Test RMSE vs. hidden-layer size")
    style_axes(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    ax = axes[2]
    g = df.groupby("n_hidden").agg(r2_mean=("test_r2_log10ID", "mean"),
                                     n_params=("n_params", "first"))
    ax.plot(g["n_params"], g["r2_mean"], color=BLUE, lw=1.5, alpha=0.5, zorder=2)
    sc = ax.scatter(g["n_params"], g["r2_mean"], c=g.index, cmap="Blues", s=60,
                     vmin=g.index.min() - 4, vmax=g.index.max(), zorder=4,
                     edgecolors=FG, linewidths=0.4)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("hidden neurons", color=FG)
    cbar.ax.yaxis.set_tick_params(color=FG)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=FG)
    ax.set_xlabel("model parameters (weights + biases)")
    ax.set_ylabel(r"mean test $R^2$")
    ax.set_title("Accuracy vs. model complexity")
    style_axes(ax)

    fig.suptitle("Hidden-layer size sweep: 12-32 neurons, 3 seeds each, "
                  "fixed train/val/test split", color=FG, fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "sweep_summary.png"), bbox_inches="tight")
    plt.close(fig)


def convergence_plot(curves):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    fig.patch.set_facecolor(BG)

    sizes = sorted(int(k) for k in curves.keys())
    colors = SEQ_BLUE[:len(sizes)] if len(sizes) <= len(SEQ_BLUE) else \
        plt.cm.Blues(np.linspace(0.35, 0.9, len(sizes)))

    for n_hidden, color in zip(sizes, colors):
        curve = curves[str(n_hidden)]
        ax.plot(range(1, len(curve) + 1), curve, color=color, lw=2,
                 label=f"n_hidden={n_hidden}" + ("  (delivered)" if n_hidden == CHOSEN_N_HIDDEN else ""))

    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MSE (standardized " + r"$\log_{10}|I_D|$" + ")")
    ax.set_title("Validation-loss convergence by hidden-layer size")
    # zoom past the epoch-1 initialization spike so the converged region
    # (where the hidden-size differences actually live) is legible
    all_after_epoch5 = np.concatenate([curves[str(n)][5:] for n in sizes])
    ax.set_ylim(all_after_epoch5.min() * 0.95, np.percentile(all_after_epoch5, 99) * 1.05)
    style_axes(ax)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "sweep_convergence.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    df = pd.read_csv(RESULTS_CSV)
    summary_plot(df)

    with open(CURVES_JSON) as f:
        curves = json.load(f)
    convergence_plot(curves)

    g = df.groupby("n_hidden").agg(
        r2_mean=("test_r2_log10ID", "mean"), r2_std=("test_r2_log10ID", "std"),
        rmse_mean=("test_rmse_log10ID", "mean"), rmse_std=("test_rmse_log10ID", "std"),
        n_params=("n_params", "first"),
    ).reset_index()
    best_row = g.loc[g["r2_mean"].idxmax()]
    chosen_row = g.loc[g["n_hidden"] == CHOSEN_N_HIDDEN].iloc[0]
    print(f"Best by mean test R^2: n_hidden={int(best_row.n_hidden)}  "
          f"R^2={best_row.r2_mean:.4f}+/-{best_row.r2_std:.4f}  "
          f"RMSE={best_row.rmse_mean:.4f}+/-{best_row.rmse_std:.4f}  "
          f"params={int(best_row.n_params)}")
    print(f"Delivered n_hidden={CHOSEN_N_HIDDEN}:  "
          f"R^2={chosen_row.r2_mean:.4f}+/-{chosen_row.r2_std:.4f}  "
          f"RMSE={chosen_row.rmse_mean:.4f}+/-{chosen_row.rmse_std:.4f}  "
          f"params={int(chosen_row.n_params)}")
    g.to_csv(os.path.join(OUT_DIR, "sweep_summary_by_hidden_size.csv"), index=False)
    print("Saved plots to", PLOT_DIR)
    print("Saved outputs/sweep_summary_by_hidden_size.csv")


if __name__ == "__main__":
    main()
