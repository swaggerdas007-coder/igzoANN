"""Export the full behavioral model -- ID + CGD + CGS, one ANN each, matching
the paper's equivalent-circuit approach (Bahubalindruni et al. 2016, Fig. 1b:
"a separate ANN for each element of the EC") -- to a single Verilog-A module.

Combines:
  * outputs/weights.json                     (ID ANN, 4 inputs: VG, VD, W, L)
  * outputs_cap/per_device/weights_{cgd,cgs}_W{W}_L{L}.json
    (8 per-device, 1-input (VG only) ANNs -- see src/train_cap_per_device.py
    for why per-device instead of a combined (VG,W,L) capacitance network:
    with only 4 measured (W,L) points, a combined network cannot learn
    where each device's turn-on knee sits and visibly misplaces it --
    outputs_cap/plots/cgd_cgs_ann_fit.png. This mirrors the existing
    tft_ann_static_per_L.va pattern: nearest-match device selection once at
    initial_step, zero runtime cost versus a unified network.)

into verilogA/tft_ann_full_model.va.

CAVEAT (also written into the generated file's header): the CGD/CGS ANNs
have no VDS dependence (measured at a fixed VDS = 0 bias only -- see
data_cv_cleaned/README.md), and only support the 4 measured (W, L) points
by nearest-match, not continuous geometry interpolation like the ID model.
The paper's own CGSi/CGDi model is a function of both VGS and VDS (its
Fig. 5/6) and captures the saturation-region asymmetry (CGDi -> 0); this
module does NOT capture that.

CGD and CGS are added as bias-dependent (VG only) capacitors between G-D
and G-S, driven as charge sources (I <+ ddt(C*V)).

Usage:
    python scripts/export_verilog_a_full.py
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ID_WEIGHTS_PATH = os.path.join(REPO, "outputs", "weights.json")
PER_DEVICE_DIR = os.path.join(REPO, "outputs_cap", "per_device")
VA_PATH = os.path.join(REPO, "verilogA", "tft_ann_full_model.va")


def fmt(x):
    return f"{x:.6f}"


def fmt_sci(x):
    """Fixed-point fmt() truncates sub-1e-6 magnitudes (e.g. farads, ~1e-12)
    to 0.000000; use scientific notation for those instead."""
    return f"{x:.6e}"


def emit_flat_assignments(name, values, per_line, indent="\t\t"):
    lines = []
    for start in range(0, len(values), per_line):
        chunk = values[start:start + per_line]
        parts = [f"{name}[{start + j}] = {fmt(v)};" for j, v in enumerate(chunk)]
        lines.append(indent + " \t ".join(parts))
    return "\n".join(lines)


def load_device_weights(prefix, W, L):
    with open(os.path.join(PER_DEVICE_DIR, f"weights_{prefix}_W{W}_L{L}.json")) as f:
        return json.load(f)


def device_branch(prefix, W, L, w):
    nnhl = w["architecture"]["n_hidden"]
    wh = w["wh"]      # [nnhl][1]
    bh = w["bh"]      # [nnhl]
    wo = w["wo"][0]   # [nnhl]
    bo = w["bo"][0]
    c_mean = w["target_transform"]["mean_F"]
    c_std = w["target_transform"]["std_F"]

    hw_flat = [wh[i][0] for i in range(nnhl)]
    hw_block = emit_flat_assignments(f"{prefix}_hlayer_w", hw_flat, per_line=nnhl)
    hb_block = "\n".join(f"\t\t{prefix}_hlayer_b[{i}] = {fmt(bh[i])};" for i in range(nnhl))
    ow_block = "\n".join(f"\t\t{prefix}_olayer_w[{i}] = {fmt(wo[i])};" for i in range(nnhl))

    return f"""\
	if ({prefix}_dev_sel == {DEVICE_INDEX[(W, L)]}) begin // W={W} L={L} um
{hw_block}
{hb_block}
{ow_block}
		{prefix}_olayer_b = {fmt(bo)};
		{prefix}_c_mean = {fmt_sci(c_mean)};
		{prefix}_c_std  = {fmt_sci(c_std)};
	end"""


def nearest_device_select_block(prefix, devices):
    """Emit the once-at-elaboration nearest-(w,l)-match selection, mirroring
    tft_ann_static_per_L.va's l_bucket pattern but in 2D (W, L)."""
    w_lo = min(d[0] for d in devices)
    w_hi = max(d[0] for d in devices)
    l_lo = min(d[1] for d in devices)
    l_hi = max(d[1] for d in devices)
    lines = [f"\t// nearest-match device selection for {prefix} (device parameters w, l are fixed per instance)"]
    lines.append(f"\t{prefix}_best_dist = 1.0e30;")
    for (W, L) in devices:
        idx = DEVICE_INDEX[(W, L)]
        lines.append(
            f"\t{prefix}_dist = pow((w - {W}.0e-6)/({w_hi}.0e-6-{w_lo}.0e-6), 2) "
            f"+ pow((l - {L}.0e-6)/({l_hi}.0e-6-{l_lo}.0e-6), 2);\n"
            f"\tif ({prefix}_dist < {prefix}_best_dist) begin "
            f"{prefix}_best_dist = {prefix}_dist; {prefix}_dev_sel = {idx}; end"
        )
    return "\n".join(lines)


DEVICE_INDEX = {}


def main():
    global DEVICE_INDEX

    with open(ID_WEIGHTS_PATH) as f:
        idw = json.load(f)
    with open(os.path.join(PER_DEVICE_DIR, "devices.json")) as f:
        devices = [tuple(d) for d in json.load(f)]
    DEVICE_INDEX = {dev: i for i, dev in enumerate(devices)}

    ni = idw["architecture"]["n_inputs"]
    nnhl = idw["architecture"]["n_hidden"]
    assert idw["architecture"]["input_order"] == ["VG", "VD", "W", "L"]

    wh = idw["wh"]
    bh = idw["bh"]
    wo = idw["wo"][0]
    bo = idw["bo"][0]

    vg_lo, vg_hi = idw["input_scaling_minmax"]["VG"]
    vd_lo, vd_hi = idw["input_scaling_minmax"]["VD"]
    w_lo, w_hi = idw["input_scaling_minmax"]["W"]
    l_lo, l_hi = idw["input_scaling_minmax"]["L"]
    y_mean = idw["target_transform"]["y_mean_log10_absID"]
    y_std = idw["target_transform"]["y_std_log10_absID"]

    id_hw_flat = [wh[i][k] for i in range(nnhl) for k in range(ni)]
    id_hlayer_w_block = emit_flat_assignments("hlayer_w", id_hw_flat, per_line=ni, indent="\t")
    id_hlayer_b_block = "\n".join(f"\thlayer_b[{i}] = {fmt(bh[i])};" for i in range(nnhl))
    id_olayer_w_block = "\n".join(f"\tolayer_w[{i}] = {fmt(wo[i])};" for i in range(nnhl))

    cap_nnhl = load_device_weights("cgd", *devices[0])["architecture"]["n_hidden"]
    cap_vg_lo, cap_vg_hi = load_device_weights("cgd", *devices[0])["input_scaling_minmax"]["VG"]

    cgd_branches = "\n".join(
        device_branch("cgd", W, L, load_device_weights("cgd", W, L)) for (W, L) in devices)
    cgs_branches = "\n".join(
        device_branch("cgs", W, L, load_device_weights("cgs", W, L)) for (W, L) in devices)
    cgd_select = nearest_device_select_block("cgd", devices)
    cgs_select = nearest_device_select_block("cgs", devices)

    device_list_str = ", ".join(f"({w_},{l_})" for w_, l_ in devices)

    va = f"""\
// VerilogA for a-IGZO TFT ANN FULL model -- ID + CGD + CGS, three separate
// ANNs joined per the equivalent circuit of Bahubalindruni et al., "InGaZnO
// TFT behavioral model for IC design", Analog Integr Circ Sig Process
// (2016) 87:73-80, Fig. 1(b) ("Each electrical element in the EC is
// modeled with an ANN... these ANNs are connected together as per the EC").
//
// ID block: MLP, {nnhl} tanh hidden neurons -> linear output, on
// (VG, VD, W, L) -> log10(|ID|), architecture and pre/post-processing
// identical to tft_ann_static.va (see that file / export_verilog_a.py for
// the full rationale).
//
// CGD, CGS blocks: PER-DEVICE MLPs, {cap_nnhl} tanh hidden neurons -> linear
// output, on VG ALONE -> capacitance in farads. One subnetwork per measured
// (W, L) geometry -- {device_list_str} um -- selected by nearest match at
// initial_step (device parameters w, l are fixed per instance, so this is
// zero runtime cost), the same pattern tft_ann_static_per_L.va uses for L.
// A combined (VG, W, L) capacitance network was tried first
// (src/train_cap.py) but visibly misplaced each device's turn-on knee with
// only 4 geometry points to interpolate from -- see
// outputs_cap/plots/cgd_cgs_ann_fit.png vs. cgd_cgs_ann_fit_per_device.png.
//
// *** CAVEAT *** CGD and CGS have NO VDS DEPENDENCE. The underlying C-V
// measurements (data_cv_cleaned/) were taken at a single fixed VDS = 0 V
// bias -- the paper's own CGSi/CGDi model additionally depends on VDS and
// captures the saturation-region asymmetry (CGDi -> 0, CGSi -> 2/3 CCH,
// paper Fig. 5/6); this module cannot reproduce that. CGD(VG) and CGS(VG)
// here are each applied as a bias-dependent (in VG only) capacitor, driven
// as a charge source I <+ ddt(C*V) for a well-posed capacitive branch
// current. Also, geometry support is only the 4 measured (W,L) points
// (nearest-match, not interpolated) vs. the ID model's continuous (W,L)
// interpolation over 19 geometries. See data_cv_cleaned/README.md.
//
// Auto-generated by scripts/export_verilog_a_full.py from outputs/weights.json,
// outputs_cap/per_device/weights_{{cgd,cgs}}_W{{W}}_L{{L}}.json.
// Do not hand-edit the weight/bias tables -- regenerate instead.

`include "constants.h"
`include "disciplines.h"

module tft_ann_full_model(D, G, S);
inout G, D, S;
electrical D, G, S;

parameter integer NI = {ni};      // ID inputs: VG, VD, W, L
parameter integer NNHL = {nnhl};    // ID hidden neurons
parameter integer NO = 1;
parameter integer CAP_NNHL = {cap_nnhl};   // CGD/CGS per-device hidden neurons

// device geometry (SI units, matching Spectre convention)
parameter real w = 40e-6 from [{w_lo:.0f}e-6:{w_hi:.0f}e-6];  // channel width, ID training range [{w_lo:.0f},{w_hi:.0f}] um
parameter real l = 20e-6 from [{l_lo:.0f}e-6:{l_hi:.0f}e-6];  // channel length, ID training range [{l_lo:.0f},{l_hi:.0f}] um
// NOTE: CGD/CGS only support the {len(devices)} measured (W,L) points above by
// nearest match -- {device_list_str} um -- regardless of w/l elsewhere in range.

// ---- ID block ----
real hlayer_w[0:(NI*NNHL)-1];
real hlayer_b[0:NNHL-1];
real hlayer_y[0:NNHL-1];
real hlayer_v[0:NNHL-1];
real olayer_w[0:NNHL-1];
real olayer_b;
real vg_s, vd_s, width_s, length_s;
real log_id, id, olayer_v;
real gm, gds, temp, temp1;
real vg_lo, vg_hi, vd_lo, vd_hi, w_lo, w_hi, l_lo, l_hi;
real y_mean, y_std;

// ---- CGD / CGS blocks (per-device, VG-only) ----
real cgd_hlayer_w[0:CAP_NNHL-1];
real cgd_hlayer_b[0:CAP_NNHL-1];
real cgd_hlayer_y[0:CAP_NNHL-1];
real cgd_olayer_w[0:CAP_NNHL-1];
real cgd_olayer_b;
real cgd_c_mean, cgd_c_std;
real cgd_dist, cgd_best_dist;
integer cgd_dev_sel;

real cgs_hlayer_w[0:CAP_NNHL-1];
real cgs_hlayer_b[0:CAP_NNHL-1];
real cgs_hlayer_y[0:CAP_NNHL-1];
real cgs_olayer_w[0:CAP_NNHL-1];
real cgs_olayer_b;
real cgs_c_mean, cgs_c_std;
real cgs_dist, cgs_best_dist;
integer cgs_dev_sel;

real cap_vg_lo, cap_vg_hi;
real vgc_s;
real cgd, cgs, cgd_out, cgs_out;
real qgd, qgs;
integer i, ii, jj;

analog begin
	@(initial_step or initial_step("static")) begin

	vg_lo = {fmt(vg_lo)}; vg_hi = {fmt(vg_hi)};
	vd_lo = {fmt(vd_lo)}; vd_hi = {fmt(vd_hi)};
	w_lo  = {w_lo:.1f}e-6; w_hi  = {w_hi:.1f}e-6;
	l_lo  = {l_lo:.1f}e-6; l_hi  = {l_hi:.1f}e-6;
	cap_vg_lo = {fmt(cap_vg_lo)}; cap_vg_hi = {fmt(cap_vg_hi)};

	y_mean = {fmt(y_mean)};
	y_std  = {fmt(y_std)};

	// ID hidden layer weights: hlayer_w[NI*i + k], k in {{0:VG, 1:VD, 2:W, 3:L}}
{id_hlayer_w_block}

	// ID hidden layer bias
{id_hlayer_b_block}

	// ID output layer weights
{id_olayer_w_block}

	// ID output layer bias
	olayer_b = {fmt(bo)};

{cgd_select}
{cgd_branches}

{cgs_select}
{cgs_branches}

	end // end of initialization

	// ==== ID: preprocessing (min-max to [0,1], clamped to training range) ====
	vg_s = (V(G,S) - vg_lo) / (vg_hi - vg_lo);
	if (vg_s < 0) vg_s = 0;
	if (vg_s > 1) vg_s = 1;

	vd_s = (V(D,S) - vd_lo) / (vd_hi - vd_lo);
	if (vd_s < 0) vd_s = 0;
	if (vd_s > 1) vd_s = 1;

	width_s = (w - w_lo) / (w_hi - w_lo);
	if (width_s < 0) width_s = 0;
	if (width_s > 1) width_s = 1;

	length_s = (l - l_lo) / (l_hi - l_lo);
	if (length_s < 0) length_s = 0;
	if (length_s > 1) length_s = 1;

	temp  = 0;
	temp1 = 0;
	for (i = 0; i < NNHL; i = i + 1) begin
		hlayer_v[i] = vg_s*hlayer_w[NI*i]   + vd_s*hlayer_w[NI*i+1]
		            + width_s*hlayer_w[NI*i+2] + length_s*hlayer_w[NI*i+3]
		            + hlayer_b[i];
		hlayer_y[i] = tanh(hlayer_v[i]);
		temp  = temp  + olayer_w[i]*hlayer_w[NI*i]   * (1 - hlayer_y[i]*hlayer_y[i]);
		temp1 = temp1 + olayer_w[i]*hlayer_w[NI*i+1] * (1 - hlayer_y[i]*hlayer_y[i]);
	end

	olayer_v = 0;
	for (jj = 0; jj < NNHL; jj = jj + 1) begin
		olayer_v = olayer_v + hlayer_y[jj]*olayer_w[jj];
	end
	olayer_v = olayer_v + olayer_b;

	log_id = olayer_v*y_std + y_mean;
	id = pow(10.0, log_id);

	gm  = id * `M_LN10 * y_std * temp  / (vg_hi - vg_lo);
	gds = id * `M_LN10 * y_std * temp1 / (vd_hi - vd_lo);

	I(D,S) <+ id;

	// ==== CGD: MLP forward pass on VG -> capacitance [F] (device-selected weights) ====
	vgc_s = (V(G,S) - cap_vg_lo) / (cap_vg_hi - cap_vg_lo);
	if (vgc_s < 0) vgc_s = 0;
	if (vgc_s > 1) vgc_s = 1;
	cgd_out = 0;
	for (i = 0; i < CAP_NNHL; i = i + 1) begin
		cgd_hlayer_y[i] = tanh(vgc_s*cgd_hlayer_w[i] + cgd_hlayer_b[i]);
		cgd_out = cgd_out + cgd_hlayer_y[i]*cgd_olayer_w[i];
	end
	cgd_out = cgd_out + cgd_olayer_b;
	cgd = cgd_out*cgd_c_std + cgd_c_mean;
	if (cgd < 0) cgd = 0;   // capacitance floor: reject small extrapolation excursions below 0

	// ==== CGS: MLP forward pass on VG -> capacitance [F] (device-selected weights) ====
	cgs_out = 0;
	for (i = 0; i < CAP_NNHL; i = i + 1) begin
		cgs_hlayer_y[i] = tanh(vgc_s*cgs_hlayer_w[i] + cgs_hlayer_b[i]);
		cgs_out = cgs_out + cgs_hlayer_y[i]*cgs_olayer_w[i];
	end
	cgs_out = cgs_out + cgs_olayer_b;
	cgs = cgs_out*cgs_c_std + cgs_c_mean;
	if (cgs < 0) cgs = 0;   // capacitance floor: reject small extrapolation excursions below 0

	// Charge-based capacitive branch currents (well-posed for the simulator's
	// charge-conservation checks, unlike a raw I <+ C*ddt(V) linear-cap form)
	qgd = cgd * V(G,D);
	qgs = cgs * V(G,S);
	I(G,D) <+ ddt(qgd);
	I(G,S) <+ ddt(qgs);

	end

endmodule
"""

    os.makedirs(os.path.dirname(VA_PATH), exist_ok=True)
    with open(VA_PATH, "w") as f:
        f.write(va)
    print(f"Wrote {VA_PATH}")


if __name__ == "__main__":
    main()
