"""Unipolar (n-only) a-IGZO inverter topologies and their figures of merit.

a-IGZO has no p-type device, so a CMOS inverter is impossible and the NOT gate
has to be ratioed. Four standard unipolar styles are built here, following
Huang et al., "Pseudo-CMOS: A Design Style for Low-Cost and Robust Flexible
Electronics", IEEE TED 58(1), 2011:

  pE   zero-VGS load. Load gate tied to its own source (the output), VGS = 0.
       Cheapest (2T), but the load is the device's own off-current, so the
       pull-up is ~nA and the gate is unusably slow.
  pD   diode (saturated) load. Load gate tied to its drain at VDD. Fast pull-up
       but VOH = VDD - Vth and the pull-down fights a load that is always on,
       so VOL is set purely by the W ratio.
  pC   pseudo-CMOS. A weak ratioed first stage (M1 load, M2 driver) makes the
       inverse signal on an internal node X, and X drives the gate of the
       output pull-up M3. The pull-up is therefore actively switched *off* while
       M4 pulls down, which is what breaks the VOL/ratio tradeoff of pD and is
       the whole point of the style.
  pCb  pseudo-CMOS, bootstrapped. A capacitor from OUT back to X. M3 is a
       source follower, so as OUT rises its VGS collapses and the last few
       hundred mV take forever; coupling the rising output into its own gate
       holds VGS up and lets X be driven *above* VDD, so OUT reaches the rail.
       Note M3's own CGS already sits between X and OUT, so pC is partly
       self-bootstrapping and pCb only adds to an effect already there.
  pCz  pseudo-CMOS with a zero-VGS first-stage load (M1 gate on X instead of
       VDD). Cuts the first stage's static current by ~3 decades, but then
       nothing can charge X quickly -- only interesting together with the
       bootstrap cap, which is what pCzb is.
  pCzb pCz + bootstrap cap.

Rails: single supply, drivers' sources at 0 V. A negative rail for the first
stage (which is how this style is usually drawn) is actively harmful with these
devices and was measured to be so: at VDD = 3 V, moving only M2's source to
-1 V collapses VOH from 2.31 V to 1 mV, because the input low level is 0 V and
a -1 V source means M2 sits at VGS = +1 V and is *on* when it should be off.
The measured devices turn on at VG = -0.25 V (see the README), so the driver
source has to be at the input's low level, i.e. 0 V.

Only the ID network is exercised outside its trained box by this topology (X
goes above VDD by design); `Net.box_report` flags it. All four builders take the
same `Sizing`, so the comparison is at equal device count and equal geometry.

Geometry note: every device here is L = 5 um. That is the only length at which
the ANN's W scaling is monotonic (at L = 10 um the model puts W80 *below* W40;
see outputs/va_test/README.md, per-geometry errors), and it is also the fastest
row. W is restricted to the measured grid {5,10,20,40,80,160} um.
"""
from dataclasses import dataclass, replace

import numpy as np

from .logic_sim import Net, eval_batch, model

KINDS = ("pE", "pD", "pC", "pCb", "pCz", "pCzb")
W_GRID = (5e-6, 10e-6, 20e-6, 40e-6, 80e-6, 160e-6)
L_LOGIC = 5e-6


@dataclass(frozen=True)
class Sizing:
    """Device widths [m]; L is common. w1/w2 are the first stage, w3/w4 the output."""
    w1: float = 20e-6      # pC/pCb: X-node load     | pE/pD: the load device
    w2: float = 160e-6     # pC/pCb: X-node driver
    w3: float = 160e-6     # output pull-up (gate on X)
    w4: float = 160e-6     # output pull-down (gate on IN)  | pE/pD: the driver
    l: float = L_LOGIC
    cboot: float = 0.0     # F, OUT -> X bootstrap cap (pCb only)
    # The stage-1 load M1 is the only device whose length is worth varying: it
    # sets the static current, which is ~80% of this ring's power, and because
    # only one width is ever used for it the unreliable W-scaling at L > 5 um
    # (see the module docstring) never comes into play. None means "use l".
    l1: float = None

    def lload(self):
        return self.l if self.l1 is None else self.l1

    def tag(self):
        return (f"w1{self.w1*1e6:g}_w2{self.w2*1e6:g}_w3{self.w3*1e6:g}"
                f"_w4{self.w4*1e6:g}_L{self.l*1e6:g}_L1{self.lload()*1e6:g}"
                f"_cb{self.cboot*1e12:g}p")


def add_inverter(net, pfx, s, vin, vout, vdd="vdd", vss="gnd", kind="pCb"):
    """Add one inverter. Returns the internal node name (or None for 2T styles)."""
    L = s.l
    if kind == "pE":
        net.tft(f"{pfx}ML", vdd, vout, vout, s.w1, L)      # gate = source: VGS = 0
        net.tft(f"{pfx}MD", vout, vin, vss, s.w4, L)
        return None
    if kind == "pD":
        net.tft(f"{pfx}ML", vdd, vdd, vout, s.w1, L)       # gate = drain = VDD
        net.tft(f"{pfx}MD", vout, vin, vss, s.w4, L)
        return None
    x = f"{pfx}x"
    gate1 = x if kind in ("pCz", "pCzb") else vdd          # zero-VGS vs saturated load
    net.tft(f"{pfx}M1", vdd, gate1, x, s.w1, s.lload())    # load on X
    net.tft(f"{pfx}M2", x, vin, vss, s.w2, L)              # driver of stage 1
    net.tft(f"{pfx}M3", vdd, x, vout, s.w3, L)             # output pull-up
    net.tft(f"{pfx}M4", vout, vin, vss, s.w4, L)           # output pull-down
    if kind in ("pCb", "pCzb") and s.cboot > 0:
        net.cap(vout, x, s.cboot)
    return x


def input_cap(s, kind, vg):
    """Gate capacitance one inverter presents at its input, at gate bias `vg`.

    Used for the fanout-1 load and for the delay proxy. The input drives M2's
    and M4's gates (or just the driver's, for pE/pD).
    """
    m = model()
    tot = 0.0
    ws = [s.w4] if kind in ("pE", "pD") else [s.w2, s.w4]
    for w in ws:
        o = m.op(np.array([vg]), np.array([0.5]), w, s.l)
        tot += float(o["cgd"][0] + o["cgs"][0])
    return tot


def node_caps(s, kind, vdd):
    """Rough C at the X and OUT nodes, at mid-swing bias. For the delay proxy."""
    m = model()

    def c(w, vg, vd, ll=None):
        o = m.op(np.array([vg]), np.array([vd]), w, ll or s.l)
        return float(o["cgd"][0]), float(o["cgs"][0])

    if kind in ("pE", "pD"):
        cx = 0.0
    else:
        _, c1s = c(s.w1, vdd * 0.5, vdd * 0.5, s.lload())   # M1 cgs sits on X
        c2d, _ = c(s.w2, vdd * 0.5, vdd * 0.5)      # M2 cgd sits on X
        _, c3s = c(s.w3, vdd * 0.5, vdd * 0.5)      # M3 cgs: X <-> OUT (intrinsic boot)
        cx = c1s + c2d + c3s + s.cboot
    c3d, _ = c(s.w3, vdd * 0.5, vdd * 0.5)
    c4d, _ = c(s.w4, vdd * 0.5, vdd * 0.5)
    cout = c3d + c4d + s.cboot + input_cap(s, kind, vdd * 0.5)
    return cx, cout


def delay_proxy(s, kind, vdd):
    """Cheap tpd estimate: C*dV/I at mid-swing. Only used to rank the DC screen."""
    m = model()
    cx, cout = node_caps(s, kind, vdd)
    half = 0.5 * vdd

    def i(w, vgs, vds, ll=None):
        return max(float(np.ravel(eval_batch(np.array([vgs]), np.array([vds]),
                                             w, ll or s.l)["id"])[0]), 1e-15)

    i_pd = i(s.w4, vdd, half)                      # pull-down, gate at VDD
    if kind in ("pE", "pD"):
        i_pu = i(s.w1, 0.0 if kind == "pE" else half, half)
        return half * cout * (1.0 / i_pd + 1.0 / i_pu) / 2.0
    i_pu = i(s.w3, half, half)                     # source follower at mid-swing
    i_x = i(s.w1, 0.0 if kind in ("pCz", "pCzb") else half, half, s.lload())
    return half * (cout * (1.0 / i_pd + 1.0 / i_pu) / 2.0 + cx / i_x)


# ---------------- noise margin ----------------
def butterfly_snm(vin, vout):
    """Static noise margin of a *cascade* of these inverters.

    The unity-gain VIL/VIH criterion degenerates for these gates: the VTC's
    steepest point sits at VIN ~ 0, so no VIL exists inside [0, VDD] and NML
    comes out zero or undefined. Two cascaded inverters are the same structure
    as a cross-coupled pair, so the butterfly construction still gives a
    meaningful number: superimpose the VTC with its mirror about VOUT = VIN and
    measure the eye. Reported as the side of the largest axis-aligned square
    that fits in the smaller lobe, in the 45-degree-rotated frame (Seevinck);
    the mirror curve is just v -> -v there, so the gap at a given u is 2|v|.
    """
    ok = np.isfinite(vout)
    vi, vo = np.asarray(vin)[ok], np.asarray(vout)[ok]
    if len(vi) < 10:
        return np.nan
    r2 = np.sqrt(2.0)
    u = (vi + vo) / r2
    v = (vo - vi) / r2
    order = np.argsort(u)
    u, v = u[order], v[order]
    out = []
    for lobe in (v > 0, v < 0):
        if lobe.sum() < 3:
            return 0.0
        uu, gap = u[lobe], 2.0 * np.abs(v[lobe])
        # largest s with an interval of length >= s where gap >= s
        best = 0.0
        for s in np.linspace(0.0, gap.max(), 160)[1:]:
            m = gap >= s
            if not m.any():
                break
            # longest run of consecutive True
            idx = np.where(m)[0]
            splits = np.where(np.diff(idx) != 1)[0]
            runs = np.split(idx, splits + 1)
            if max(uu[r[-1]] - uu[r[0]] for r in runs) >= s:
                best = s
        out.append(best)
    return float(min(out))


# ---------------- DC characterisation ----------------
def vtc(s, kind, vdd, vss=0.0, npts=161, cload=0.0):
    """DC transfer curve with continuation in VIN. Returns a metrics dict."""
    net = Net()
    net.node("vdd")
    net.node("in")
    net.node("out")
    vssn = "gnd"
    if vss != 0.0:
        vssn = "vss"
        net.node(vssn)
    x = add_inverter(net, "", s, "in", "out", "vdd", vssn, kind)
    if cload:
        net.cap("out", "gnd", cload)
    net.force("vdd", vdd)
    if vss != 0.0:
        net.force(vssn, vss)
    net.force("in", vss)
    net.build()

    vin = np.linspace(vss, vdd, npts)
    vout = np.full(npts, np.nan)
    vx = np.full(npts, np.nan)
    istat = np.full(npts, np.nan)
    oob = np.zeros(npts, dtype=bool)
    guess = np.zeros(net.n)
    guess[net._idx["out"]] = vdd
    if x:
        guess[net._idx[x]] = vdd
    iv = net._idx["vdd"]
    for k, vi in enumerate(vin):
        net.force("in", vi)
        v, ok = net.solve_dc(guess=guess)
        if not ok:
            continue
        guess = v
        vout[k] = v[net._idx["out"]]
        if x:
            vx[k] = v[net._idx[x]]
        d = net.devices(v)
        # supply current = sum of device currents flowing out of the vdd node
        istat[k] = float(np.sum(d["id"][net.nd == iv]) - np.sum(d["id"][net.ns == iv]))
        oob[k] = bool(np.any(d["oob"]))
    return _vtc_metrics(vin, vout, vx, istat, oob, vdd, vss, s, kind)


def _vtc_metrics(vin, vout, vx, istat, oob, vdd, vss, s, kind):
    good = np.isfinite(vout)
    out = dict(kind=kind, vdd=vdd, vss=vss, vin=vin, vout=vout, vx=vx,
               istat=istat, oob=oob, converged=float(good.mean()),
               w1=s.w1, w2=s.w2, w3=s.w3, w4=s.w4, l=s.l, cboot=s.cboot)
    if good.sum() < 10:
        out.update(gain=np.nan, voh=np.nan, vol=np.nan, swing=np.nan, vm=np.nan,
                   nmh=np.nan, nml=np.nan, nm=np.nan, p_static=np.nan,
                   i_hi=np.nan, i_lo=np.nan, oob_frac=1.0, valid=False)
        return out
    vi, vo = vin[good], vout[good]
    g = np.gradient(vo, vi)
    out["gain"] = float(np.max(np.abs(g)))
    out["voh"] = float(vo[0])
    out["vol"] = float(vo[-1])
    out["swing"] = out["voh"] - out["vol"]
    # trip point
    d = vo - vi
    sgn = np.where(np.diff(np.sign(d)) != 0)[0]
    out["vm"] = float(np.interp(0.0, [d[sgn[0] + 1], d[sgn[0]]],
                                [vi[sgn[0] + 1], vi[sgn[0]]])) if len(sgn) else np.nan
    # unity-gain points -> noise margins
    k = int(np.argmax(np.abs(g)))
    lo = np.where(np.abs(g[:k + 1]) <= 1.0)[0]
    hi = np.where(np.abs(g[k:]) <= 1.0)[0]
    vil = float(vi[lo[-1]]) if len(lo) else float(vi[0])
    vih = float(vi[k + hi[0]]) if len(hi) else float(vi[-1])
    out["vil"], out["vih"] = vil, vih
    # Loop gain at the trip point, which is the number that decides whether a
    # ring made of these gates oscillates -- max |dVout/dVin| does not. The
    # zero-VGS variant peaks at |A| = 220 at VIN = 4.5 mV and is only -0.53 at
    # its own VM = 0.238 V, so its symmetric ring point is stable and the ring
    # latches despite the headline gain.
    out["gain_vm"] = (float(abs(np.interp(out["vm"], vi, g)))
                      if np.isfinite(out["vm"]) else np.nan)
    out["nmh"] = out["voh"] - vih
    out["nml"] = vil - out["vol"]
    out["nm"] = min(out["nmh"], out["nml"])
    ii = istat[good]
    out["i_lo"] = float(ii[0])        # IN low  (output high)
    out["i_hi"] = float(ii[-1])       # IN high (output low)
    out["p_static"] = float(vdd * 0.5 * (abs(ii[0]) + abs(ii[-1])))
    out["oob_frac"] = float(oob[good].mean())
    out["snm"] = butterfly_snm(vi, vo)
    out["tpd_proxy"] = delay_proxy(s, kind, vdd - vss)
    out["pdp_proxy"] = out["tpd_proxy"] * out["p_static"]
    sw = vdd - vss
    # "works as a cascadable gate": full-ish swing, regenerative (|A| > 1 with
    # margin), levels that actually reach the rails, and a non-zero butterfly
    # eye. The unity-gain NM is reported but not used as a gate -- see
    # butterfly_snm().
    out["valid"] = bool(out["swing"] > 0.6 * sw and out["gain_vm"] > 1.0
                        and out["vol"] < 0.1 * sw and out["voh"] > 0.6 * sw
                        and out["snm"] > 0.05 * sw and out["oob_frac"] < 0.02
                        and out["converged"] > 0.95)
    return out


# ---------------- chains and rings ----------------
def build_chain(s, kind, nstage, vdd, vss=0.0, cload=0.0, ring=False):
    """N inverters in series ('s0'..'s{N-1}'); ring=True closes the loop."""
    net = Net()
    net.node("vdd")
    vssn = "gnd"
    if vss != 0.0:
        vssn = "vss"
        net.node(vssn)
    outs = [f"o{i}" for i in range(nstage)]
    ins = (outs[-1:] + outs[:-1]) if ring else (["in"] + outs[:-1])
    for nm in ([] if ring else ["in"]) + outs:
        net.node(nm)
    xs = []
    for i in range(nstage):
        xs.append(add_inverter(net, f"s{i}_", s, ins[i], outs[i], "vdd", vssn, kind))
    if cload:
        for o in outs:
            net.cap(o, "gnd", cload)
    net.force("vdd", vdd)
    if vss != 0.0:
        net.force(vssn, vss)
    if not ring:
        net.force("in", vss)
    net.build()
    return net, outs, xs


# ---------------- fast exact DC screen ----------------
# In DC the two stages are decoupled: M3 and M4 load node X only through their
# gates, which draw no DC current, so vx(vin) depends on (w1, w2) alone and
# vout(vin) on (w3, w4) given vx. Each node is then a single unknown, so the
# whole VIN sweep can be Newton-solved as one vectorised batch -- ~30x faster
# than calling solve_dc() per bias point, and the same equations.
# `scripts/pseudo_cmos_screen.py` asserts it against the full engine.
def _nr1d(f_and_df, lo, hi, iters=60, reltol=1e-9):
    """Vectorised 1-unknown solve, bisection-safeguarded Newton.

    Both node equations are monotonically increasing in their unknown (the
    pull-down's current rises with the node, the pull-up's falls), so the sign
    of the residual brackets the root and bisection can never be defeated --
    plain Newton on an 8-decade exponential I-V can be, and was.
    """
    lo = np.array(lo, dtype=float)
    hi = np.array(hi, dtype=float)
    flo = f_and_df(lo)[0]
    fhi = f_and_df(hi)[0]
    # if the root is outside the bracket, the clamped end is the answer
    same = np.sign(flo) == np.sign(fhi)
    x = 0.5 * (lo + hi)
    for _ in range(iters):
        f, df = f_and_df(x)
        scale = np.maximum(np.abs(flo), np.abs(fhi))
        if np.max(np.abs(f) / np.where(scale > 0, scale, 1.0)) < reltol:
            break
        pos = f > 0
        hi = np.where(pos, x, hi)
        lo = np.where(pos, lo, x)
        step = -f / np.where(np.abs(df) < 1e-20, 1e-20, df)
        xn = x + step
        bad = ~np.isfinite(xn) | (xn <= lo) | (xn >= hi)
        x = np.where(bad, 0.5 * (lo + hi), xn)
    return np.where(same, np.where(np.abs(flo) < np.abs(fhi), lo, hi), x)


def _i(vgs, vds, w, l):
    d = eval_batch(vgs, vds, w, l, caps=False)
    return d["id"], d["didvg"], d["didvd"]


def screen_vtc(w1, w2, w3, w4, kind, vdd, vin, l=L_LOGIC, l1=None):
    """Exact DC VTC of a pC/pCz inverter, vectorised over the VIN sweep."""
    zero_load = kind in ("pCz", "pCzb")
    n = len(vin)
    ll = l1 or l

    def fx(x):
        g1 = np.zeros(n) if zero_load else (vdd - x)
        i1, dg1, dd1 = _i(g1, vdd - x, w1, ll)
        i2, _, dd2 = _i(vin, x, w2, l)
        f = i2 - i1
        df = dd2 + (dd1 if zero_load else dg1 + dd1)
        return f, df

    x = _nr1d(fx, np.full(n, -1.0), np.full(n, 2.5 * vdd))

    def fo(o):
        i3, dg3, dd3 = _i(x - o, vdd - o, w3, l)
        i4, _, dd4 = _i(vin, o, w4, l)
        return i4 - i3, dd4 + dg3 + dd3

    o = _nr1d(fo, np.full(n, -1.0), np.full(n, 1.5 * vdd))
    g1 = np.zeros(n) if zero_load else (vdd - x)
    i1 = _i(g1, vdd - x, w1, ll)[0]
    i3 = _i(x - o, vdd - o, w3, l)[0]
    return x, o, i1 + i3


def screen_metrics(w1, w2, w3, w4, kind, vdd, npts=401, l=L_LOGIC, l1=None):
    vin = np.linspace(0.0, vdd, npts)
    vx, vo, ivdd = screen_vtc(w1, w2, w3, w4, kind, vdd, vin, l, l1)
    s = Sizing(w1=w1, w2=w2, w3=w3, w4=w4, l=l, l1=l1)
    return _vtc_metrics(vin, vo, vx, ivdd, np.zeros(npts, dtype=bool), vdd, 0.0, s, kind)
