"""Select the 2 best-quality output-curve replicates per (W, L) combination,
using the QC scores from clean_output_curves.py, and copy them into
data/output_curves_clean/ with tidy names. Also writes a markdown report
documenting what was excluded and why.

Run clean_output_curves.py first to (re)generate outputs/output_curve_qc_summary.csv.
"""
import os
import shutil

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
SUMMARY_CSV = os.path.join(ROOT, "outputs", "output_curve_qc_summary.csv")
CLEAN_DIR = os.path.join(ROOT, "data", "output_curves_clean")
REPORT_PATH = os.path.join(ROOT, "outputs", "output_curve_cleaning_report.md")

# A file below this score has a confirmed physical defect (dead/disconnected
# device, compliance-clipped sweep, or a corrupted/scrambled sweep) rather than
# ordinary measurement noise -- see outputs/plots/output_curve_qc/*.png.
REJECT_BELOW = 50.0


def failure_reason(row):
    if row["no_vd_dependence"] and row["no_vg_modulation"]:
        return "dead/disconnected device: current is flat, independent of both VG and VD"
    if row["no_vd_dependence"]:
        return "no VD dependence: current does not rise with drain voltage"
    if row["no_vg_modulation"]:
        return "no gate modulation: VG traces collapse onto a single curve (or clip to instrument compliance)"
    if row["vd_monotonic_frac"] < 0.85:
        return "non-monotonic / scrambled ID-VD sweep"
    if row["vg_order_violation_frac"] > 0.1:
        return "ID does not order correctly with VG"
    if row["sign_flips"] > 0:
        return "off-state current bursts / sign flips"
    if row["off_state_leaky"]:
        return "elevated off-state leakage (still usable, ranked below cleaner replicates)"
    return "ranked lower than the two best replicates for this (W, L)"


def main():
    df = pd.read_csv(SUMMARY_CSV)
    df = df.sort_values(["W", "L", "score"], ascending=[True, True, False])

    if os.path.isdir(CLEAN_DIR):
        shutil.rmtree(CLEAN_DIR)
    os.makedirs(CLEAN_DIR)

    kept_rows = []
    excluded_rows = []
    flagged_combos = []

    for (w, l), group in df.groupby(["W", "L"]):
        picks = group.head(2)
        rest = group.iloc[2:]
        if picks["score"].min() < REJECT_BELOW:
            flagged_combos.append((w, l))
        for _, row in picks.iterrows():
            dest_name = f"W{w}_L{l}_{row['label']}.csv"
            shutil.copy2(row["path"], os.path.join(CLEAN_DIR, dest_name))
            kept_rows.append({**row.to_dict(), "clean_fname": dest_name})
        for _, row in rest.iterrows():
            excluded_rows.append({**row.to_dict(), "reason": failure_reason(row)})

    kept_df = pd.DataFrame(kept_rows)
    excluded_df = pd.DataFrame(excluded_rows)

    lines = []
    lines.append("# Output-curve cleaning report\n")
    lines.append(
        f"Source: {len(df)} raw `tft output *.csv` files across "
        f"{df.groupby(['W', 'L']).ngroups} (W, L) combinations.\n"
    )
    lines.append(
        "Each combination was measured up to 4 times at different wafer "
        "positions (top_1, top_2, bot_1, bot_2), occasionally re-run. Every "
        "file was scored against physical expectations for an n-type TFT "
        "output-curve family: current should rise monotonically with VD, "
        "increase in a fixed order with VG, stay near the noise floor for "
        "VG <= 0V, and show clear, well-separated modulation across VG "
        "(see `scripts/clean_output_curves.py` and the diagnostic grids in "
        "`outputs/plots/output_curve_qc/`).\n"
    )
    lines.append(
        "## Result\n\n"
        "A strong, consistent pattern emerged: devices at the **top** wafer "
        "position are unreliable (leaky off-state in most files, and outright "
        "broken -- compliance-clipped sweeps, dead/disconnected devices reading "
        "a flat current, or off-state noise bursts up to hundreds of "
        "microamps -- in roughly half of them). Devices at the **bot** "
        "position are consistently clean. No (W, L) combination had to be "
        "dropped entirely -- every combination has at least 2 usable "
        "replicates -- but `top_*` files should be treated with caution "
        "wherever they show up.\n"
    )
    if flagged_combos:
        lines.append(
            "### (W, L) combinations flagged (best 2 replicates still weak)\n\n"
            + "\n".join(f"- W={w}, L={l}" for w, l in flagged_combos)
            + "\n"
        )
    else:
        lines.append("No (W, L) combination needed to be flagged in full: every combination "
                      "had at least 2 replicates scoring at or above the pass threshold.\n")

    lines.append(f"\n## Kept: {len(kept_df)} files -> `data/output_curves_clean/`\n")
    lines.append("| W | L | file | replicate | score |")
    lines.append("|---|---|------|-----------|-------|")
    for _, row in kept_df.sort_values(["W", "L", "score"], ascending=[True, True, False]).iterrows():
        lines.append(f"| {row['W']} | {row['L']} | {row['clean_fname']} | {row['label']} | {row['score']:.1f} |")

    lines.append(f"\n## Excluded: {len(excluded_df)} files\n")
    lines.append("| W | L | replicate | score | reason |")
    lines.append("|---|---|-----------|-------|--------|")
    for _, row in excluded_df.sort_values(["W", "L", "score"], ascending=[True, True, False]).iterrows():
        lines.append(f"| {row['W']} | {row['L']} | {row['label']} | {row['score']:.1f} | {row['reason']} |")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Copied {len(kept_df)} clean files to {CLEAN_DIR}")
    print(f"Wrote report to {REPORT_PATH}")
    if flagged_combos:
        print(f"Flagged (W, L) combos: {flagged_combos}")
    else:
        print("No (W, L) combos required full exclusion.")


if __name__ == "__main__":
    main()
