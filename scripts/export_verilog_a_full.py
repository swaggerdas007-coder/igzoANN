"""Export the full behavioural TFT model -- ID + CGD + CGS, one ANN per
equivalent-circuit element -- to a single Verilog-A module.

Follows Bahubalindruni et al., "InGaZnO TFT behavioral model for IC design",
Analog Integr Circ Sig Process (2016) 87:73-80, Fig. 1(b): each element of the
equivalent circuit is modelled by its own ANN, and the ANNs are then wired
together as the EC dictates -- ID as a current source across D-S, CGD as a
bias-dependent capacitor across G-D, CGS across G-S.

Sources:
  * trained_ANN/deep/weights.json      ID   ANN, 4 -> 32 -> 16 -> 1
  * trained_Cg_ANN/cgd/weights.json    CGD  ANN, 4 -> NH1 -> NH2 -> 1
  * trained_Cg_ANN/cgs/weights.json    CGS  ANN, 4 -> NH1 -> NH2 -> 1

All three networks share the identical 4-input interface (VG, VD, W, L), the
identical min-max [0,1] input scaling, and the identical topology (tanh hidden
layers, linear output) required by Eqs. (1)-(2) of the paper, so they are
pin-compatible and the merge is a straight wiring exercise.

Usage:
    python scripts/export_verilog_a_full.py
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ID_WEIGHTS = os.path.join(REPO, "trained_ANN", "deep", "weights.json")
CGD_WEIGHTS = os.path.join(REPO, "trained_Cg_ANN", "cgd", "weights.json")
CGS_WEIGHTS = os.path.join(REPO, "trained_Cg_ANN", "cgs", "weights.json")
VA_PATH = os.path.join(REPO, "verilogA", "tft_ann_full_model.va")

FEATURES = ["VG", "VD", "W", "L"]


def fmt(x):
    return f"{x:.6f}"


def emit_rows(name, mat, indent="\t"):
    """Emit a flattened row-major matrix as `name[stride*i + k] = ...;` lines,
    one source line per row (matching the existing hand-reviewed .va style)."""
    stride = len(mat[0])
    lines = []
    for i, row in enumerate(mat):
        parts = [f"{name}[{stride * i + k}] = {fmt(v)};" for k, v in enumerate(row)]
        lines.append(indent + " \t ".join(parts))
    return "\n".join(lines)


def emit_vec(name, vec, indent="\t"):
    return "\n".join(f"{indent}{name}[{i}] = {fmt(v)};" for i, v in enumerate(vec))


def load_id():
    with open(ID_WEIGHTS) as f:
        w = json.load(f)
    assert w["architecture"]["input_order"] == FEATURES, w["architecture"]["input_order"]
    return {
        "w1": w["layers"][0]["w"], "b1": w["layers"][0]["b"],
        "w2": w["layers"][1]["w"], "b2": w["layers"][1]["b"],
        "wo": w["output"]["w"][0], "bo": w["output"]["b"][0],
        "scaling": w["input_scaling_minmax"],
        "y_mean": w["target_transform"]["y_mean_log10_absID"],
        "y_std": w["target_transform"]["y_std_log10_absID"],
    }


def load_cap(path):
    with open(path) as f:
        w = json.load(f)
    assert w["architecture"]["input_order"] == FEATURES, w["architecture"]["input_order"]
    tt = w["target_transform"]
    return {
        "w1": w["w1"], "b1": w["b1"], "w2": w["w2"], "b2": w["b2"],
        "wo": w["wo"][0], "bo": w["bo"][0],
        "scaling": w["input_scaling_minmax"],
        "y_mean": tt["y_mean"], "y_std": tt["y_std"],
        # area_normalised: the network regresses C/(W*L) in pF/um^2 * 1e3, so
        # the prediction must be multiplied back by the gate area
        "area_normalised": bool(tt.get("area_normalised", False)),
        "hp": w.get("hyperparameters", {}),
    }


def cap_recover_block(prefix, cap):
    """Verilog-A lines turning the linear network output into a capacitance
    in farads, honouring the target transform actually used in training."""
    if cap["area_normalised"]:
        return (
            f"\t{prefix} = ({prefix}_out*{prefix}_y_std + {prefix}_y_mean);"
            f"   // C/(W*L) in pF/um^2 * 1e3\n"
            f"\t{prefix} = {prefix} * (w*1.0e6) * (l*1.0e6) * 1.0e-3;"
            f"   // -> pF (re-apply gate area)\n"
            f"\t{prefix} = {prefix} * 1.0e-12;   // pF -> F"
        )
    return (
        f"\t{prefix} = ({prefix}_out*{prefix}_y_std + {prefix}_y_mean);   // pF\n"
        f"\t{prefix} = {prefix} * 1.0e-12;   // pF -> F"
    )


def main():
    idw = load_id()
    cgd = load_cap(CGD_WEIGHTS)
    cgs = load_cap(CGS_WEIGHTS)

    # every ANN must agree on the input scaling, otherwise the EC merge is unsound
    for name, net in (("CGD", cgd), ("CGS", cgs)):
        assert net["scaling"] == idw["scaling"], (
            f"{name} input scaling {net['scaling']} != ID scaling {idw['scaling']}")

    sc = idw["scaling"]
    vg_lo, vg_hi = sc["VG"]
    vd_lo, vd_hi = sc["VD"]
    w_lo, w_hi = sc["W"]
    l_lo, l_hi = sc["L"]

    id_nh1, id_nh2 = len(idw["b1"]), len(idw["b2"])
    cgd_nh1, cgd_nh2 = len(cgd["b1"]), len(cgd["b2"])
    cgs_nh1, cgs_nh2 = len(cgs["b1"]), len(cgs["b2"])

    def hp_str(net):
        hp = net["hp"]
        if not hp:
            return "n/a"
        return (f"loss={hp.get('loss')}, lr={hp.get('lr')}, "
                f"batch={hp.get('batch_size')}")

    cgd_target = ("C/(W*L) [pF/um^2]" if cgd["area_normalised"] else "C [pF]")
    cgs_target = ("C/(W*L) [pF/um^2]" if cgs["area_normalised"] else "C [pF]")

    va = f"""\
// VerilogA for a-IGZO TFT ANN FULL behavioural model -- ID + CGD + CGS.
//
// Three separate ANNs, one per element of the equivalent circuit, joined as
// the EC prescribes -- Bahubalindruni et al., "InGaZnO TFT behavioral model
// for IC design", Analog Integr Circ Sig Process (2016) 87:73-80, Fig. 1(b):
// "Each electrical element in the EC is modeled with an ANN ... these ANNs
// are connected together as per the EC".
//
//     ID  : current source across D-S      -- 4 -> {id_nh1} -> {id_nh2} -> 1
//     CGD : capacitor across G-D           -- 4 -> {cgd_nh1} -> {cgd_nh2} -> 1
//     CGS : capacitor across G-S           -- 4 -> {cgs_nh1} -> {cgs_nh2} -> 1
//
// All three networks use the paper's Eqs. (1)-(2) form -- tanh hidden layers,
// linear output layer -- and the identical 4-input interface (VG, VD, W, L)
// with the identical min-max [0,1] input scaling, so they are pin-compatible
// and every element supports continuous (W, L) geometry scaling.
//
// ID   target: log10(|ID|), standardised. Source trained_ANN/deep/weights.json
//              (2-hidden-layer variant of tft_ann_static.va, trained with the
//              monotonicity penalty + L-balanced loss; see
//              scripts/export_verilog_a_deep.py).
// CGD  target: {cgd_target}, standardised. {hp_str(cgd)}
// CGS  target: {cgs_target}, standardised. {hp_str(cgs)}
//              Sources trained_Cg_ANN/{{cgd,cgs}}/weights.json.
//
// *** CAVEAT -- CGD/CGS HAVE NO REAL VDS DEPENDENCE ***
// VD is present as an input for pin-compatibility with the ID network, but
// every C-V measurement behind these two ANNs was taken at a single VDS = 0 V
// bias (data_cleaned_cv/), so the networks cannot have learnt any VDS
// dependence -- they are effectively C(VG, W, L). The paper's own CGSi/CGDi
// model does depend on VDS and reproduces the saturation-region asymmetry
// (CGDi -> 0, CGSi -> 2/3 CCH, its Figs. 5-6); this module does NOT. Adding
// that requires C-V data swept over VDS.
// Geometry coverage of the capacitance ANNs is also thinner than the ID
// network's: 4 measured (W, L) points -- (20,20), (40,20), (160,15), (160,20)
// um -- versus 19 for ID, so CGD/CGS extrapolate outside that hull.
//
// Auto-generated by scripts/export_verilog_a_full.py.
// Do not hand-edit the weight/bias tables -- regenerate instead.

`include "constants.h"
`include "disciplines.h"

module tft_ann_full_model(D, G, S);
inout G, D, S;
electrical D, G, S;

parameter integer NI = 4;          // shared inputs: VG, VD, W, L
parameter integer ID_NH1  = {id_nh1};   // ID  hidden layer 1
parameter integer ID_NH2  = {id_nh2};   // ID  hidden layer 2
parameter integer CGD_NH1 = {cgd_nh1};   // CGD hidden layer 1
parameter integer CGD_NH2 = {cgd_nh2};    // CGD hidden layer 2
parameter integer CGS_NH1 = {cgs_nh1};   // CGS hidden layer 1
parameter integer CGS_NH2 = {cgs_nh2};   // CGS hidden layer 2

// device geometry (SI units, Spectre convention)
parameter real w = 40e-6 from [{w_lo:.0f}e-6:{w_hi:.0f}e-6];  // channel width,  training range [{w_lo:.0f},{w_hi:.0f}] um
parameter real l = 20e-6 from [{l_lo:.0f}e-6:{l_hi:.0f}e-6];  // channel length, training range [{l_lo:.0f},{l_hi:.0f}] um

// ---- ID network ----
real id_w1[0:(NI*ID_NH1)-1];      real id_b1[0:ID_NH1-1];
real id_v1[0:ID_NH1-1];           real id_y1[0:ID_NH1-1];
real id_w2[0:(ID_NH1*ID_NH2)-1];  real id_b2[0:ID_NH2-1];
real id_v2[0:ID_NH2-1];           real id_y2[0:ID_NH2-1];
real id_wo[0:ID_NH2-1];           real id_bo;
real id_y_mean, id_y_std;
real log_id, id, id_out;
real gm, gds, dvg, dvd, s_vg, s_vd;

// ---- CGD network ----
real cgd_w1[0:(NI*CGD_NH1)-1];      real cgd_b1[0:CGD_NH1-1];
real cgd_y1[0:CGD_NH1-1];
real cgd_w2[0:(CGD_NH1*CGD_NH2)-1]; real cgd_b2[0:CGD_NH2-1];
real cgd_y2[0:CGD_NH2-1];           real cgd_acc;
real cgd_wo[0:CGD_NH2-1];           real cgd_bo;
real cgd_y_mean, cgd_y_std;
real cgd, cgd_out, qgd;

// ---- CGS network ----
real cgs_w1[0:(NI*CGS_NH1)-1];      real cgs_b1[0:CGS_NH1-1];
real cgs_y1[0:CGS_NH1-1];
real cgs_w2[0:(CGS_NH1*CGS_NH2)-1]; real cgs_b2[0:CGS_NH2-1];
real cgs_y2[0:CGS_NH2-1];           real cgs_acc;
real cgs_wo[0:CGS_NH2-1];           real cgs_bo;
real cgs_y_mean, cgs_y_std;
real cgs, cgs_out, qgs;

// ---- shared preprocessing ----
real vg_lo, vg_hi, vd_lo, vd_hi, w_lo, w_hi, l_lo, l_hi;
real vg_s, vd_s, width_s, length_s;
integer i, j;

analog begin
	@(initial_step or initial_step("static")) begin

	vg_lo = {fmt(vg_lo)}; vg_hi = {fmt(vg_hi)};
	vd_lo = {fmt(vd_lo)}; vd_hi = {fmt(vd_hi)};
	w_lo  = {w_lo:.1f}e-6; w_hi  = {w_hi:.1f}e-6;
	l_lo  = {l_lo:.1f}e-6; l_hi  = {l_hi:.1f}e-6;

	// ================= ID network =================
	id_y_mean = {fmt(idw['y_mean'])};
	id_y_std  = {fmt(idw['y_std'])};

	// hidden layer 1 weights: id_w1[NI*i + k], k in {{0:VG, 1:VD, 2:W, 3:L}}
{emit_rows("id_w1", idw["w1"])}
	// hidden layer 1 bias
{emit_vec("id_b1", idw["b1"])}

	// hidden layer 2 weights: id_w2[ID_NH1*i + j]
{emit_rows("id_w2", idw["w2"])}
	// hidden layer 2 bias
{emit_vec("id_b2", idw["b2"])}

	// output layer (linear)
{emit_vec("id_wo", idw["wo"])}
	id_bo = {fmt(idw["bo"])};

	// ================= CGD network =================
	cgd_y_mean = {fmt(cgd['y_mean'])};
	cgd_y_std  = {fmt(cgd['y_std'])};

	// hidden layer 1 weights: cgd_w1[NI*i + k]
{emit_rows("cgd_w1", cgd["w1"])}
	// hidden layer 1 bias
{emit_vec("cgd_b1", cgd["b1"])}

	// hidden layer 2 weights: cgd_w2[CGD_NH1*i + j]
{emit_rows("cgd_w2", cgd["w2"])}
	// hidden layer 2 bias
{emit_vec("cgd_b2", cgd["b2"])}

	// output layer (linear)
{emit_vec("cgd_wo", cgd["wo"])}
	cgd_bo = {fmt(cgd["bo"])};

	// ================= CGS network =================
	cgs_y_mean = {fmt(cgs['y_mean'])};
	cgs_y_std  = {fmt(cgs['y_std'])};

	// hidden layer 1 weights: cgs_w1[NI*i + k]
{emit_rows("cgs_w1", cgs["w1"])}
	// hidden layer 1 bias
{emit_vec("cgs_b1", cgs["b1"])}

	// hidden layer 2 weights: cgs_w2[CGS_NH1*i + j]
{emit_rows("cgs_w2", cgs["w2"])}
	// hidden layer 2 bias
{emit_vec("cgs_b2", cgs["b2"])}

	// output layer (linear)
{emit_vec("cgs_wo", cgs["wo"])}
	cgs_bo = {fmt(cgs["bo"])};

	end // end of initialization

	// ==== shared preprocessing: min-max to [0,1], clamped to training range ====
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

	// ================= ID: forward pass =================
	for (i = 0; i < ID_NH1; i = i + 1) begin
		id_v1[i] = vg_s*id_w1[NI*i] + vd_s*id_w1[NI*i+1]
		         + width_s*id_w1[NI*i+2] + length_s*id_w1[NI*i+3] + id_b1[i];
		id_y1[i] = tanh(id_v1[i]);
	end

	for (i = 0; i < ID_NH2; i = i + 1) begin
		id_v2[i] = 0;
		for (j = 0; j < ID_NH1; j = j + 1) begin
			id_v2[i] = id_v2[i] + id_y1[j]*id_w2[ID_NH1*i+j];
		end
		id_v2[i] = id_v2[i] + id_b2[i];
		id_y2[i] = tanh(id_v2[i]);
	end

	id_out = 0;
	for (i = 0; i < ID_NH2; i = i + 1) begin
		id_out = id_out + id_y2[i]*id_wo[i];
	end
	id_out = id_out + id_bo;

	log_id = id_out*id_y_std + id_y_mean;   // linear output, then de-standardise
	id = pow(10.0, log_id);

	// analytic small-signal conductances (chain rule through both tanh layers)
	dvg = 0;
	dvd = 0;
	for (i = 0; i < ID_NH2; i = i + 1) begin
		s_vg = 0;
		s_vd = 0;
		for (j = 0; j < ID_NH1; j = j + 1) begin
			s_vg = s_vg + id_w2[ID_NH1*i+j]*(1 - id_y1[j]*id_y1[j])*id_w1[NI*j];
			s_vd = s_vd + id_w2[ID_NH1*i+j]*(1 - id_y1[j]*id_y1[j])*id_w1[NI*j+1];
		end
		dvg = dvg + id_wo[i]*(1 - id_y2[i]*id_y2[i])*s_vg;
		dvd = dvd + id_wo[i]*(1 - id_y2[i]*id_y2[i])*s_vd;
	end
	gm  = id * `M_LN10 * id_y_std * dvg / (vg_hi - vg_lo);
	gds = id * `M_LN10 * id_y_std * dvd / (vd_hi - vd_lo);

	// ================= CGD: forward pass =================
	for (i = 0; i < CGD_NH1; i = i + 1) begin
		cgd_y1[i] = tanh(vg_s*cgd_w1[NI*i] + vd_s*cgd_w1[NI*i+1]
		                 + width_s*cgd_w1[NI*i+2] + length_s*cgd_w1[NI*i+3] + cgd_b1[i]);
	end
	for (i = 0; i < CGD_NH2; i = i + 1) begin
		cgd_acc = 0;
		for (j = 0; j < CGD_NH1; j = j + 1) begin
			cgd_acc = cgd_acc + cgd_y1[j]*cgd_w2[CGD_NH1*i+j];
		end
		cgd_y2[i] = tanh(cgd_acc + cgd_b2[i]);
	end
	cgd_out = 0;
	for (i = 0; i < CGD_NH2; i = i + 1) begin
		cgd_out = cgd_out + cgd_y2[i]*cgd_wo[i];
	end
	cgd_out = cgd_out + cgd_bo;
{cap_recover_block("cgd", cgd)}
	if (cgd < 0) cgd = 0;   // floor: reject small sub-zero extrapolation excursions

	// ================= CGS: forward pass =================
	for (i = 0; i < CGS_NH1; i = i + 1) begin
		cgs_y1[i] = tanh(vg_s*cgs_w1[NI*i] + vd_s*cgs_w1[NI*i+1]
		                 + width_s*cgs_w1[NI*i+2] + length_s*cgs_w1[NI*i+3] + cgs_b1[i]);
	end
	for (i = 0; i < CGS_NH2; i = i + 1) begin
		cgs_acc = 0;
		for (j = 0; j < CGS_NH1; j = j + 1) begin
			cgs_acc = cgs_acc + cgs_y1[j]*cgs_w2[CGS_NH1*i+j];
		end
		cgs_y2[i] = tanh(cgs_acc + cgs_b2[i]);
	end
	cgs_out = 0;
	for (i = 0; i < CGS_NH2; i = i + 1) begin
		cgs_out = cgs_out + cgs_y2[i]*cgs_wo[i];
	end
	cgs_out = cgs_out + cgs_bo;
{cap_recover_block("cgs", cgs)}
	if (cgs < 0) cgs = 0;   // floor: reject small sub-zero extrapolation excursions

	// ================= equivalent circuit =================
	// ID as a current source across D-S; CGD, CGS as bias-dependent capacitors
	// driven as charge sources so the branch currents stay charge-conserving.
	qgd = cgd * V(G,D);
	qgs = cgs * V(G,S);

	I(D,S) <+ id;
	I(G,D) <+ ddt(qgd);
	I(G,S) <+ ddt(qgs);

	end

endmodule
"""

    os.makedirs(os.path.dirname(VA_PATH), exist_ok=True)
    with open(VA_PATH, "w") as f:
        f.write(va)
    print(f"Wrote {VA_PATH}")
    print(f"  ID : 4 -> {id_nh1} -> {id_nh2} -> 1   (log10|ID|)")
    print(f"  CGD: 4 -> {cgd_nh1} -> {cgd_nh2} -> 1   "
          f"({'area-normalised' if cgd['area_normalised'] else 'direct'} C)")
    print(f"  CGS: 4 -> {cgs_nh1} -> {cgs_nh2} -> 1   "
          f"({'area-normalised' if cgs['area_normalised'] else 'direct'} C)")


if __name__ == "__main__":
    main()
