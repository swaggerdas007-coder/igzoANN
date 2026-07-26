"""Hyperparameter sweep over both hidden-layer widths of the 2-layer ANN
(trained_ANN/ann_train_deep.py), reasoned rather than a blind grid:

  (12, 12)  -- capacity floor: is 20-20 more than necessary?
  (16, 16)  -- modest reduction
  (20, 20)  -- current delivered baseline
  (24, 24)  -- modest increase
  (32, 32)  -- larger increase: does accuracy keep improving, or plateau
              (as the single-hidden-layer 12-32 sweep did)?
  (32, 16)  -- funnel (wide -> narrow): first layer forms a rich basis
              from the 4 raw inputs, second layer recombines/compresses --
              the conventional deep-learning shape
  (16, 32)  -- reverse funnel (narrow -> wide): tests whether that
              convention actually holds here, rather than assuming it
  (24, 12)  -- moderate funnel at lower total capacity than 32,16

Every config trained with the exact recipe as the delivered model:
monotonicity penalty (lambda=100) + L-balanced loss, same data/seed. Each
result includes a Cadence-style gds audit, since that's the metric that
actually matters for circuit simulation, not just held-out R^2. Results
saved incrementally to trained_ANN/deep_sweep/results.json (resumable).

Usage:
    python ann_train_deep_sweep.py
"""
import json
import os

from ann_train_deep import export, train

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(HERE, "deep_sweep")
RESULTS_PATH = os.path.join(SWEEP_DIR, "results.json")

CONFIGS = [
    (12, 12), (16, 16), (20, 20), (24, 24), (32, 32), (32, 16), (16, 32), (24, 12),
]


def main():
    os.makedirs(SWEEP_DIR, exist_ok=True)
    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)

    for h1, h2 in CONFIGS:
        key = f"{h1}_{h2}"
        if key in results:
            print(f"skip {key} (already done)")
            continue
        print(f"\n=== training ({h1},{h2}) ===", flush=True)
        model, result = train([h1, h2], verbose=False)
        export(model, result, out_dir=os.path.join(SWEEP_DIR, key))
        results[key] = result
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"({h1},{h2}) DONE r2={result['test_r2_log10ID']:.4f} "
              f"rmse={result['test_rmse_log10ID']:.4f} "
              f"gds_neg={result['gds_negative_fraction_on_region']*100:.2f}% "
              f"worst={result['worst_gds_S']:.2e}S "
              f"params={result['n_params']} ({result['train_seconds']:.0f}s)", flush=True)

    print("\n=== SWEEP SUMMARY ===")
    for key, r in results.items():
        print(f"{key:8s} r2={r['test_r2_log10ID']:.4f}  rmse={r['test_rmse_log10ID']:.4f}  "
              f"gds_neg={r['gds_negative_fraction_on_region']*100:5.2f}%  "
              f"worst={r['worst_gds_S']:.2e}S  params={r['n_params']:4d}")


if __name__ == "__main__":
    main()
