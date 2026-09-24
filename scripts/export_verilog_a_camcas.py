"""Export the CAMCAS-style semi-empirical DC model (scripts/fit_camcas_model.py)
to verilogA/tft_camcas_model.va, as an alternative to this repo's ANN-based
models (tft_ann_static.va etc.) -- both target the same (VG, VD, W, L) -> ID
problem, but this one is Carolina de Almeida's thesis "Development of IGZO
TFTs compact models" (2025) unified analytical expression (Sec 2.3.4,
Appendix F) instead of a neural network.

Geometry handling: NEAREST-MATCH per-device parameters, not the smooth
(L, W) polynomial surface scripts/fit_camcas_model.py also fits. See
outputs_camcas/README.md for why -- in short, on this dataset (one device
per (W, L) geometry, each a genuinely different physical die site) most of
CAMCAS's 10 regime parameters don't have a clean geometric trend once
per-device measurement noise is accounted for (confirmed by leave-one-out
cross-validated ridge regression picking heavy-to-total shrinkage for most
of them), so a smooth surface fit reproduces even its own training
geometries very poorly. The thesis's own Dataset 2 hit the same wall
(Sec. 3.3.2). This mirrors the nearest-match pattern this repo's own
tft_ann_full_model.va already uses for CGD/CGS for the identical reason
(export_verilog_a_full.py) -- pick the closest of the geometries whose
per-device fit actually converged (see FILTER in fit_camcas_model.py /
plot_camcas_fit.py's oracle_params) once at initial_step, zero runtime cost.

Usage:
    python scripts/export_verilog_a_camcas.py
"""
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "outputs_camcas")
VA_PATH = os.path.join(REPO, "verilogA", "tft_camcas_model.va")

PARAM_NAMES = ["Von_lin", "Von_sat", "alpha_lin", "alpha_sat",
               "kappa_lin", "kappa_sat", "G0_lin", "G0_sat",
               "Ioff_lin", "Ioff_sat"]


def fmt(x):
    return f"{x:.6e}"


def load_valid_devices():
    """Geometries whose per-device NLS fit converged to a physically sane
    optimum in both regimes (same filter as plot_camcas_fit.py's
    oracle_params: G0 didn't run away to its search bound)."""
    extracted = pd.read_csv(os.path.join(OUT_DIR, "extracted_params.csv"))
    dl_rsd = pd.read_csv(os.path.join(OUT_DIR, "dl_rsd_per_width.csv")).set_index("W")

    devices = []
    for _, row in extracted.iterrows():
        if not (1e-10 < row["G0_lin"] < 1) or not (1e-10 < row["G0_sat"] < 1):
            continue
        if row["W"] not in dl_rsd.index:
            continue
        params = {name: row[name] for name in PARAM_NAMES}
        params["DeltaL"] = dl_rsd.loc[row["W"], "DeltaL"]
        params["RSD"] = dl_rsd.loc[row["W"], "RSD_ohm"]
        devices.append((row["W"], row["L"], params))
    return devices


def device_branch(idx, W, L, params):
    lines = [f"\tif (dev_sel == {idx}) begin // W={W:.0f} L={L:.0f} um"]
    for name in PARAM_NAMES:
        lines.append(f"\t\t{name} = {fmt(params[name])};")
    lines.append(f"\t\tDeltaL = {fmt(params['DeltaL'])};")
    lines.append(f"\t\tRSD = {fmt(params['RSD'])};")
    lines.append("\tend")
    return "\n".join(lines)


def nearest_select_block(devices):
    w_lo = min(d[0] for d in devices)
    w_hi = max(d[0] for d in devices)
    l_lo = min(d[1] for d in devices)
    l_hi = max(d[1] for d in devices)
    lines = ["\t// nearest-match device selection (device parameters w, l are fixed per instance)"]
    lines.append("\tbest_dist = 1.0e30;")
    for idx, (W, L, _params) in enumerate(devices):
        lines.append(
            f"\tdist = pow((w - {W:.0f}.0e-6)/({w_hi:.0f}.0e-6-{w_lo:.0f}.0e-6), 2) "
            f"+ pow((l - {L:.0f}.0e-6)/({l_hi:.0f}.0e-6-{l_lo:.0f}.0e-6), 2);\n"
            f"\tif (dist < best_dist) begin best_dist = dist; dev_sel = {idx}; end"
        )
    return "\n".join(lines)


def main():
    devices = load_valid_devices()
    print(f"{len(devices)} geometries with a converged per-device CAMCAS fit: "
          f"{[(w, l) for w, l, _ in devices]}")

    w_lo = min(d[0] for d in devices)
    w_hi = max(d[0] for d in devices)
    l_lo = min(d[1] for d in devices)
    l_hi = max(d[1] for d in devices)
    device_list_str = ", ".join(f"({w:.0f},{l:.0f})" for w, l, _ in devices)

    branches = "\n".join(device_branch(i, W, L, p) for i, (W, L, p) in enumerate(devices))
    select_block = nearest_select_block(devices)

    va = f"""\
// VerilogA for a-IGZO TFT CAMCAS model -- semi-empirical unified DC compact
// model, adapted from Carolina de Almeida's thesis "Development of IGZO
// thin-film transistors (TFTs) compact models" (2025), Sec. 2.3.4 /
// Appendix F, itself based on the CAMCAS (Cambridge Compact Analytical
// Semiconductor) unified TFT model. This is an ALTERNATIVE to this repo's
// ANN-based models (tft_ann_static.va etc.) -- same (VG, VD, W, L) -> ID
// problem, a physics-flavored analytical expression instead of a neural
// network:
//
//   IDS = G0 * (W/L') * exp(kappa*(VGS-Von)^alpha) * VDS' + Ioff
//
// fit separately in the linear (VDS' = VDS - 2*Rc*IDS) and saturation
// (VDS' = VGS-Von, Vds-independent) regimes, blended by the harmonic-mean
// smoothing function Ids_total = (Ids_lin^-m + Ids_sat^-m)^(-1/m) so a
// single expression covers both regimes and the triode-to-saturation
// transition at any Vds.
//
// GEOMETRY HANDLING: nearest-match per-device parameters -- {len(devices)}
// measured geometries, {device_list_str} um -- selected once at
// initial_step by (w,l) distance (device parameters w, l are fixed per
// instance, so this is zero runtime cost), NOT a continuous (L,W)
// polynomial surface. See scripts/export_verilog_a_camcas.py's docstring
// and outputs_camcas/README.md: on this dataset most of CAMCAS's 10 regime
// parameters don't have a clean geometric trend once real device-to-device
// variation (each (W,L) here is a different physical die site, not a
// repeat measurement) is accounted for, so a smooth surface reproduces even
// its own training geometries very poorly -- confirmed numerically
// (scripts/plot_camcas_fit.py). Per-device, the same functional form
// tracks the measured transfer curves to a median ~15-45% relative error
// depending on geometry and regime (worst near turn-on, matching the
// thesis's own reported weakness there) -- see
// outputs_camcas/camcas_relative_error.csv.
//
// Auto-generated by scripts/export_verilog_a_camcas.py from
// outputs_camcas/extracted_params.csv, outputs_camcas/dl_rsd_per_width.csv.
// Do not hand-edit the per-device parameter tables -- regenerate instead.

`include "constants.h"
`include "disciplines.h"

module tft_camcas_model(D, G, S);
inout G, D, S;
electrical D, G, S;

parameter real w = 80e-6 from [{w_lo:.0f}e-6:{w_hi:.0f}e-6];  // channel width, um range [{w_lo:.0f},{w_hi:.0f}]
parameter real l = 15e-6 from [{l_lo:.0f}e-6:{l_hi:.0f}e-6];  // channel length, um range [{l_lo:.0f},{l_hi:.0f}]
parameter real m_smooth = 2.2;  // harmonic-mean smoothing exponent (thesis Eq. 2.35 default)

real Von_lin, Von_sat, alpha_lin, alpha_sat, kappa_lin, kappa_sat;
real G0_lin, G0_sat, Ioff_lin, Ioff_sat, DeltaL, RSD;
real dist, best_dist;
integer dev_sel;

real Wm, Lm, Lp, von_eff, Vgs, Vds;
real x, expo_lin, expo_sat, Gc, ids_lin, ids_sat, ids_total;

analog begin
	@(initial_step or initial_step("static")) begin
	Wm = w*1.0e6;  // this model's parameters were all fit in micrometers
	Lm = l*1.0e6;

{select_block}

{branches}

	end // end of initialization

	Vgs = V(G,S);
	Vds = V(D,S);
	Lp = Lm - DeltaL;
	if (Lp < 0.1) Lp = 0.1;
	if (Von_lin > Von_sat)
		von_eff = Von_lin;
	else
		von_eff = Von_sat;

	if (Vgs <= von_eff) begin
		// off-state: constant-current floor from whichever regime applies
		if (Vds < (Vgs - von_eff))
			ids_total = Ioff_lin;
		else
			ids_total = Ioff_sat;
	end else begin
		x = Vgs - von_eff;

		expo_lin = kappa_lin * pow(x, alpha_lin);
		if (expo_lin > 700.0) expo_lin = 700.0;
		if (expo_lin < -700.0) expo_lin = -700.0;
		Gc = G0_lin * (Wm/Lp) * exp(expo_lin);
		// linear branch: ACTUAL Vds (not a fixed reference) so the harmonic
		// mean below can hand off to the Vds-independent saturation branch
		// as Vds grows -- get this wrong and the model can never saturate.
		ids_lin = (Gc*Vds + Ioff_lin) / (1 + RSD*Gc);
		if (ids_lin < 1e-15) ids_lin = 1e-15;

		expo_sat = kappa_sat * pow(x, alpha_sat);
		if (expo_sat > 700.0) expo_sat = 700.0;
		if (expo_sat < -700.0) expo_sat = -700.0;
		ids_sat = G0_sat * (Wm/Lp) * exp(expo_sat) * x + Ioff_sat;
		if (ids_sat < 1e-15) ids_sat = 1e-15;

		ids_total = pow(pow(ids_lin, -m_smooth) + pow(ids_sat, -m_smooth), -1.0/m_smooth);
	end

	I(D,S) <+ ids_total;
end

endmodule
"""

    os.makedirs(os.path.dirname(VA_PATH), exist_ok=True)
    with open(VA_PATH, "w") as f:
        f.write(va)
    print(f"Wrote {VA_PATH}")


if __name__ == "__main__":
    main()
