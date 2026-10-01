"""Step 7 -- PROPOSED Verilog-A change for the AC/charge model. Does NOT
modify verilogA/tft_camcas_thesis.va: writes the change as a unified diff
(outputs_camcas_ac/proposed_ac_patch.diff), applies it to a scratch copy and
verifies that copy:
  1. compiles with OpenVAF (verilogae),
  2. DC current Ids_total is bit-identical to the unpatched model at every
     measured DC bias point of all 19 devices,
  3. the Verilog-A charges equal the Python reference model (acmodel.py).
"""
import difflib
import json
import os
import sys
import tempfile

import numpy as np

import acmodel as M
from cvdata import OUT, REPO

VA = os.path.join(REPO, "verilogA", "tft_camcas_thesis.va")


def patched(src, P):
    def rep(old, new):
        nonlocal src
        assert src.count(old) == 1, old[:60]
        src = src.replace(old, new)

    rep("""parameter real eps_ox = 3.45e-11;
parameter real tox = 0.1e-6;
""", f"""// AC / charge model, extracted from measured C-V (scripts/camcas_ac/,
// outputs_camcas_ac/README.md). Per unit gate area W*L:
parameter real Cch_area   = {P['Cch_area']:.6e}; // F/m^2 channel capacitance (VG -> inf)
parameter real Cov_d_area = {P['Cov_d_area']:.6e}; // F/m^2 gate-drain overlap
parameter real Cov_s_area = {P['Cov_s_area']:.6e}; // F/m^2 gate-source overlap
// channel-charge turn-on (V): sharp part (width = DC SS/ln10) + broad part
parameter real dV_C  = {P['dV_C']:.6g};  // charge turn-on relative to the DC Von_eff
parameter real sC1   = {P['s1']:.6g};
parameter real betaC = {P['beta']:.6g};
parameter real sC2   = {P['s2']:.6g};
parameter real dC2   = {P['d2']:.6g};
""")
    rep("""real Covs, Covd, CCH;
real Qovs, Qovd, Qch;
real Lov, Lovch;
""", """real Aox, Cch, Covs, Covd, V0c, vs_c, vd_c, Qch, Qdch, Qg, Qd;
""")
    rep("""analog begin
""", """analog function real softplus;
	input x;
	real x;
	begin
		if (x > 40.0)
			softplus = x;
		else if (x < -40.0)
			softplus = exp(x);
		else
			softplus = ln(1.0 + exp(x));
	end
endfunction

// effective overdrive (V) = integral of the normalized channel capacitance
analog function real veff_c;
	input v, s1, s2, beta, d2;
	real v, s1, s2, beta, d2;
	begin
		veff_c = (1.0 - beta) * s1 * softplus(v / s1) + beta * s2 * softplus((v - d2) / s2);
	end
endfunction

analog begin
""")
    rep("""	Lov = L*10e-5;
	Lovch = 0.8*L*1e-6;

	Covs = eps_ox * W * 1e-5 * Lov / tox;
	Covd = eps_ox * W * 1e-5 * Lov / tox;
	CCH = eps_ox * W * 1e-5 * Lovch / tox;

	Qovs = Covs * V(g,s);
	Qovd = Covd * V(g,d);
	Qch = CCH * V(g,s);

""", "")
    rep("""	I(d,s) <+ Ids_total;
	I(g,s) <+ ddt(Qovs + Qch);
	I(g,d) <+ ddt(Qovd);
""", """	// charge model: W, L in um -> gate area in m^2
	Aox = W * L * 1e-12;
	Cch = Cch_area * Aox;
	Covs = Cov_s_area * Aox;
	Covd = Cov_d_area * Aox;
	V0c = Von_eff + dV_C;
	vs_c = veff_c(Vgs - V0c, sC1, sC2, betaC, dC2) + 1e-12;
	vd_c = veff_c(Vgd - V0c, sC1, sC2, betaC, dC2) + 1e-12;
	// Ward-Dutton long-channel charge, symmetric in source/drain
	Qch = (2.0/3.0) * Cch * (vs_c*vs_c + vs_c*vd_c + vd_c*vd_c) / (vs_c + vd_c);
	Qdch = -Cch * (6.0*vd_c*vd_c*vd_c + 12.0*vd_c*vd_c*vs_c + 8.0*vd_c*vs_c*vs_c
	       + 4.0*vs_c*vs_c*vs_c) / (15.0 * (vs_c + vd_c) * (vs_c + vd_c));
	Qg = Qch + Covs * Vgs + Covd * Vgd;
	Qd = Qdch - Covd * Vgd;          // Qs = -Qg - Qd (charge conserving)

	I(d,s) <+ Ids_total;
	I(g,s) <+ ddt(Qg);
	I(d,s) <+ ddt(Qd);
""")
    rep("""// The charge block is copied verbatim from F.1 and was not re-extracted.""",
        """// The charge block of F.1 is replaced by a charge model extracted from
// measured C-V (outputs_camcas_ac/README.md).""")
    return src


def compile_va(src):
    import verilogae
    with tempfile.NamedTemporaryFile("w", suffix=".va", delete=False) as t:
        t.write(src)
    return verilogae.load(t.name)


def main():
    P = json.load(open(os.path.join(OUT, "ac_model_params_final.json")))
    orig = open(VA).read()
    new = patched(orig, P)
    diff = "".join(difflib.unified_diff(orig.splitlines(True), new.splitlines(True),
                                        "a/verilogA/tft_camcas_thesis.va",
                                        "b/verilogA/tft_camcas_thesis.va"))
    with open(os.path.join(OUT, "proposed_ac_patch.diff"), "w") as f:
        f.write(diff)
    print(f"wrote proposed_ac_patch.diff ({diff.count(chr(10))} lines)")

    # verification on scratch copies -- the real .va is not touched
    vo = compile_va(orig.replace("real Ids_total;", "(*retrieve*) real Ids_total;"))
    vn_src = new.replace("real Ids_total;", "(*retrieve*) real Ids_total;").replace(
        "real Aox, Cch, Covs, Covd, V0c, vs_c, vd_c, Qch, Qdch, Qg, Qd;",
        "real Aox, Cch, Covs, Covd, V0c, vs_c, vd_c, Qch, Qdch;\n(*retrieve*) real Qg;\n(*retrieve*) real Qd;")
    vn = compile_va(vn_src)
    m = json.load(open(os.path.join(REPO, "outputs_camcas_thesis",
                                    "step3_coefficients_thesis_g.json")))["m"]
    sys.path.insert(0, os.path.join(REPO, "scripts", "camcas_thesis"))
    import step5_validate as dcv
    pts = dcv.measured()
    worst = 0.0
    for (w, l), g in pts.groupby(["W", "L"]):
        kw = dict(temperature=300.0, voltages={"br_gs": g.VG.to_numpy(), "br_ds": g.VD.to_numpy()},
                  W=float(w), L=float(l), m=m)
        a = vo.functions["Ids_total"].eval(**kw)
        b = vn.functions["Ids_total"].eval(**kw)
        worst = max(worst, float(np.max(np.abs(a - b))))
    print(f"DC Ids_total, patched vs original, all 19 devices: max |diff| = {worst:.3e} A")

    from common import scaled_params
    with open(os.path.join(REPO, "outputs_camcas_thesis", "step3_coefficients_thesis_g.json")) as f:
        coef = json.load(f)
    acp = dict(Cch_area=P["Cch_area"], Cov_d_area=P["Cov_d_area"], Cov_s_area=P["Cov_s_area"],
               dV_C=P["dV_C"], sC1=P["s1"], betaC=P["beta"], sC2=P["s2"], dC2=P["d2"])
    rng = np.random.default_rng(1)
    vgs, vds = rng.uniform(-3, 5, 400), rng.uniform(-5, 5, 400)
    errs = []
    for w, l in ((20, 20), (160, 15), (80, 5)):
        p = scaled_params(coef, w, l)
        von = max(p["Von_lin"], p["Von_sat"])
        f = vn.functions["Qg"]
        volt = {k: v for k, v in {"br_gs": vgs, "br_ds": vds, "br_gd": vgs - vds}.items()
                if k in f.voltages}
        kw = dict(temperature=300.0, voltages=volt, W=float(w), L=float(l), m=m, **acp)
        qg_va = f.eval(**kw)
        qd_va = vn.functions["Qd"].eval(**kw)
        qg, qd, _ = M.charges(vgs, vds, w, l, dict(P, V0=von + P["dV_C"]))
        errs.append(np.max(np.abs(qg_va - qg) / np.max(np.abs(qg))))
        errs.append(np.max(np.abs(qd_va - qd) / np.max(np.abs(qg))))
    print(f"charges Verilog-A vs Python reference: max rel diff = {max(errs):.2e}")


if __name__ == "__main__":
    main()
