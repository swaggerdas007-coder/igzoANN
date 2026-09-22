"""Bench-test verilogA/ntft_full.va against the measured data, and probe it for
the behaviours an analog designer relies on (monotonicity, saturation, a
sensible gm/gds, charge conservation).

Run:  python scripts/validate_va_model.py
Writes outputs/va_test/{accuracy.csv, physics_checks.csv, model_check.png}
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.va_model import TFTModel                      # noqa: E402
from src.dataset import load_and_split                 # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "va_test")
os.makedirs(OUT, exist_ok=True)


def decade_stats(meas, pred):
    """Error in decades of current -- the metric that matters for a log10 model."""
    m = np.abs(meas) > 1e-14
    e = np.log10(np.abs(pred[m])) - np.log10(np.abs(meas[m]))
    return dict(n=int(m.sum()), rmse_dec=float(np.sqrt((e ** 2).mean())),
                mae_dec=float(np.abs(e).mean()), p90_dec=float(np.percentile(np.abs(e), 90)),
                max_dec=float(np.abs(e).max()))


def main():
    model = TFTModel()
    rows = []

    # ---------- 1. ID accuracy on the held-out test split ----------
    csv = os.path.join(REPO, "data_cleaned", "merged_ann_dataset.csv")
    tr, va, te = load_and_split(csv, seed=42)
    for name, sp in (("train", tr), ("val", va), ("test", te)):
        d = sp.df
        pred = model.id_(d.VG.to_numpy(), d.VD.to_numpy(),
                         d.W.to_numpy() * 1e-6, d.L.to_numpy() * 1e-6)
        st = decade_stats(d.ID.to_numpy(), pred)
        rows.append(dict(split=name, region="all", **st))
        # on-state only: what a circuit actually biases at
        on = (d.ID.abs() > 1e-9).to_numpy()
        st = decade_stats(d.ID.to_numpy()[on], pred[on])
        rows.append(dict(split=name, region="ID>1nA", **st))
        sat = on & (d.VD.to_numpy() >= 3.0)
        st = decade_stats(d.ID.to_numpy()[sat], pred[sat])
        rows.append(dict(split=name, region="ID>1nA & VD>=3V", **st))
    acc = pd.DataFrame(rows)
    acc.to_csv(os.path.join(OUT, "accuracy.csv"), index=False)
    print("=== ID accuracy (errors in decades of current) ===")
    print(acc.to_string(index=False))

    # how much of that error is irreducible measurement spread?
    full = pd.read_csv(csv)
    on_full = full[full.ID.abs() > 1e-9]
    spread = on_full.groupby(["VG", "VD", "W", "L"]).log_ID.std().dropna()
    print(f"\nmeasurement replicate spread, on-state points: median std "
          f"{spread.median():.3f} dec, mean {spread.mean():.3f} dec "
          f"(irreducible floor for the model's {acc[(acc.split=='test')&(acc.region=='ID>1nA')].rmse_dec.iloc[0]:.3f} dec test RMSE)")

    # per-geometry on-state error, saturation sweep only
    d = pd.read_csv(csv)
    d = d[(d.sweep == "saturation") & (d.ID.abs() > 1e-9)]
    pred = model.id_(d.VG.to_numpy(), d.VD.to_numpy(), d.W.to_numpy() * 1e-6, d.L.to_numpy() * 1e-6)
    d = d.assign(err=np.log10(np.abs(pred)) - np.log10(d.ID.abs()))
    g = d.groupby(["W", "L"]).err.agg(rmse_dec=lambda x: np.sqrt((x ** 2).mean()),
                                      bias_dec="mean", n="size").reset_index()
    g.to_csv(os.path.join(OUT, "accuracy_by_geometry.csv"), index=False)
    print("\n=== on-state saturation error per geometry ===")
    print(g.to_string(index=False))

    # ---------- 2. capacitance accuracy ----------
    cv = pd.read_csv(os.path.join(REPO, "data_cv_cleaned", "merged_cv_dataset.csv"))
    op = model.op(cv.VG.to_numpy(), cv.VDS.to_numpy(), cv.W.to_numpy() * 1e-6, cv.L.to_numpy() * 1e-6)
    cap_rows = []
    for tag, pred in (("CGD", op["cgd"]), ("CGS", op["cgs"])):
        meas = cv[tag].to_numpy()
        cap_rows.append(dict(cap=tag, n=len(meas),
                             rmse_fF=float(np.sqrt(((pred - meas) ** 2).mean()) * 1e15),
                             mean_meas_fF=float(meas.mean() * 1e15),
                             rel_rmse_pct=float(np.sqrt(((pred - meas) ** 2).mean()) / meas.mean() * 100)))
    cap = pd.DataFrame(cap_rows)
    cap.to_csv(os.path.join(OUT, "accuracy_cap.csv"), index=False)
    print("\n=== C-V accuracy (all rows are VDS=0 -- the model's only cap training bias) ===")
    print(cap.to_string(index=False))

    # ---------- 3. physics / robustness probes ----------
    checks = []

    def chk(name, ok, detail):
        checks.append(dict(check=name, pass_=bool(ok), detail=detail))

    W, L = 160e-6, 20e-6
    vg = np.linspace(-5, 5, 401)
    i_sat = model.id_(vg, 5.0, W, L)
    bad = np.diff(i_sat) <= 0
    on = i_sat[:-1] > 1e-9
    chk("ID monotonic in VG over the on-region (W160/L20, VD=5)",
        not np.any(bad & on),
        f"{int((bad & on).sum())} non-monotonic steps in the on-region; "
        f"{int(bad.sum())} overall, all at ID ~ "
        f"{np.median(i_sat[:-1][bad]):.1e} A -- i.e. in the pA leakage floor "
        "where the measurement is noise anyway")

    vd = np.linspace(0, 5, 201)
    i_out = model.id_(3.0, vd, W, L)
    chk("ID monotonic in VD (VG=3)", np.all(np.diff(i_out) >= 0),
        f"{int((np.diff(i_out) < 0).sum())} non-monotonic steps of 200")

    i0 = float(np.ravel(model.id_(3.0, 0.0, W, L))[0])
    chk("ID(VDS=0) == 0", i0 < 1e-12,
        f"ID(VG=3,VD=0) = {i0*1e6:.3f} uA -- a log10 output cannot reach zero")

    op5 = model.op(np.array([2.0, 3.0, 4.0]), 4.0, W, L)
    gain = op5["gm"] / op5["gds"]
    chk("intrinsic gain gm/gds > 10 somewhere in VG=2..4 @VD=4",
        np.any(gain > 10), "gm/gds = " + ", ".join(f"{x:.1f}" for x in gain))

    # output-conductance slope beyond the training box
    g_at5 = float(np.ravel(model.op(3.0, 5.0, W, L)["gds"])[0])
    g_over = float(np.ravel(model.op(3.0, 6.0, W, L)["gds"])[0])
    chk("gds nonzero above VD=5V (extrapolation)", g_over > 0,
        f"gds(VD=5)={g_at5*1e6:.3f} uS, gds(VD=6)={g_over*1e6:.3f} uS "
        "-- VD is clamped at 5 V, so the model is flat (ro = inf) above it")

    # geometry scaling: ID should be ~linear in W at fixed L
    ws = np.array([20e-6, 40e-6, 80e-6, 160e-6])
    iw = model.id_(3.0, 4.0, ws, 20e-6)
    ratio = iw / iw[0] / (ws / ws[0])
    chk("ID scales ~linearly with W (L=20um, VG=3, VD=4)",
        np.all((ratio > 0.5) & (ratio < 2.0)),
        "ID/(W-normalised) = " + ", ".join(f"{x:.2f}" for x in ratio))

    ls = np.array([5e-6, 10e-6, 15e-6, 20e-6])
    il = model.id_(3.0, 4.0, 160e-6, ls)
    chk("ID decreases with L (W=160um)", np.all(np.diff(il) < 0),
        "ID[L=5,10,15,20um] = " + ", ".join(f"{x*1e6:.1f}uA" for x in il))

    # charge conservation: the .va writes I <+ ddt(C(V)*V), which is NOT
    # the same as I <+ C(V)*dV/dt unless C is bias independent.
    v = np.linspace(-3, 5, 401)
    c = model.op(v, 0.0, W, L)["cgs"]
    q = c * v
    cap_eff = np.gradient(q, v)          # what the simulator actually sees
    err = np.abs(cap_eff - c) / np.maximum(c, 1e-18)
    chk("ddt(C*V) formulation: dQ/dV == C", err.max() < 0.05,
        f"max |dQ/dV - C|/C = {err.max()*100:.0f}% (peaks at the turn-on knee)")

    cgd0 = float(np.ravel(model.op(0.0, 0.0, W, L)["cgd"])[0])
    cgd5 = float(np.ravel(model.op(0.0, 5.0, W, L)["cgd"])[0])
    # NOT a feature: every C-V row was taken at VDS=0, so any VDS slope here
    # is invented by the network, not learnt.
    chk("CGD/CGS free of spurious VDS dependence", abs(cgd5 - cgd0) / cgd0 < 0.05,
        f"CGD(VD=0)={cgd0*1e12:.2f}pF vs CGD(VD=5)={cgd5*1e12:.2f}pF "
        f"= {abs(cgd5-cgd0)/cgd0*100:.0f}% swing conjured from VDS-free training data")

    # negative output conductance anywhere useful? (the repo's own older model
    # logged gds<0 on 4.6% of the on-region -- outputs/metrics.json)
    vgg, vdd = np.meshgrid(np.linspace(0, 5, 61), np.linspace(0.2, 5, 61))
    o = model.op(vgg.ravel(), vdd.ravel(), W, L)
    on = o["id"] > 1e-9
    fneg_g = float((o["gds"][on] < 0).mean())
    fneg_m = float((o["gm"][on] < 0).mean())
    chk("gds >= 0 over the on-region (W160/L20)", fneg_g < 0.01,
        f"{fneg_g*100:.1f}% of on-region points have gds<0 "
        f"(min {o['gds'][on].min()*1e6:.2f} uS)")
    chk("gm >= 0 over the on-region (W160/L20)", fneg_m < 0.01,
        f"{fneg_m*100:.1f}% of on-region points have gm<0")

    # analytic gm/gds vs finite difference -- validates the .va chain rule
    h = 1e-4
    base = model.op(np.array([1.0, 2.0, 3.0]), 3.0, W, L)
    fd_gm = (model.id_(np.array([1.0, 2.0, 3.0]) + h, 3.0, W, L)
             - model.id_(np.array([1.0, 2.0, 3.0]) - h, 3.0, W, L)) / (2 * h)
    fd_gds = (model.id_(np.array([1.0, 2.0, 3.0]), 3.0 + h, W, L)
              - model.id_(np.array([1.0, 2.0, 3.0]), 3.0 - h, W, L)) / (2 * h)
    e_gm = np.abs(base["gm"] - fd_gm) / np.abs(fd_gm)
    e_gds = np.abs(base["gds"] - fd_gds) / np.abs(fd_gds)
    chk("analytic gm/gds match finite difference", max(e_gm.max(), e_gds.max()) < 1e-4,
        f"max rel. error gm {e_gm.max():.2e}, gds {e_gds.max():.2e}")

    ch = pd.DataFrame(checks).rename(columns={"pass_": "pass"})
    ch.to_csv(os.path.join(OUT, "physics_checks.csv"), index=False)
    print("\n=== physics / robustness probes ===")
    for r in checks:
        print(f"[{'PASS' if r['pass_'] else 'FAIL'}] {r['check']}\n        {r['detail']}")

    # ---------- 4. plots ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    meas = pd.read_csv(csv)

    for k, (Wg, Lg) in enumerate([(160, 20), (40, 20), (20, 5)]):
        s = meas[(meas.W == Wg) & (meas.L == Lg) & (meas.sweep == "saturation")]
        s = s.groupby("VG").ID.median()
        ax[0, k].semilogy(s.index, s.abs(), "k.", ms=3, label="measured VD=5")
        vgx = np.linspace(-5, 5, 201)
        ax[0, k].semilogy(vgx, model.id_(vgx, 5.0, Wg * 1e-6, Lg * 1e-6), "r-", label="ANN model")
        ax[0, k].set_title(f"transfer W={Wg} L={Lg} um")
        ax[0, k].set_xlabel("VGS [V]"); ax[0, k].set_ylabel("ID [A]"); ax[0, k].legend(fontsize=8)
        ax[0, k].grid(alpha=.3)

        o = meas[(meas.W == Wg) & (meas.L == Lg) & (meas.sweep == "output")]
        for vgv in sorted(o.VG.unique())[::max(1, len(o.VG.unique()) // 5)]:
            oo = o[o.VG == vgv].groupby("VD").ID.median()
            ax[1, k].plot(oo.index, oo.values * 1e6, ".", ms=3)
            vdx = np.linspace(0, 5, 101)
            ax[1, k].plot(vdx, model.id_(vgv, vdx, Wg * 1e-6, Lg * 1e-6) * 1e6, "-",
                          label=f"VG={vgv:g}")
        ax[1, k].set_title(f"output W={Wg} L={Lg} um"); ax[1, k].set_xlabel("VDS [V]")
        ax[1, k].set_ylabel("ID [uA]"); ax[1, k].legend(fontsize=7); ax[1, k].grid(alpha=.3)

    fig.suptitle("ntft_full.va vs measured (dots = median over device replicates)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "model_check.png"), dpi=130)
    print(f"\nwrote {OUT}/model_check.png")


if __name__ == "__main__":
    main()
