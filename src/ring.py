"""N-stage ring oscillator of pseudo-CMOS a-IGZO inverters: build, run, measure."""
import numpy as np

from .logic_sim import period_from_crossings
from .pseudo_cmos import Sizing, build_chain, node_caps


def intrinsic_cx(s, kind, vdd):
    """Capacitance already sitting on node X, with no explicit bootstrap cap."""
    return node_caps(replace_cboot(s, 0.0), kind, vdd)[0]


def replace_cboot(s, c):
    return Sizing(w1=s.w1, w2=s.w2, w3=s.w3, w4=s.w4, l=s.l, cboot=c)


def build_ring(s, kind, nstage, vdd, cload=0.0, init="alternating", cap_scale=1.0):
    """Close N inverters into a ring and pick an initial condition.

    `init` matters for long rings. "alternating" sets neighbouring outputs to
    opposite rails, which is the *highest* spatial mode the ring supports and can
    leave a many-stage ring running several waves at once, at k x the
    fundamental. "perturb" instead starts every node at the ring's symmetric DC
    solution (all stages at their own trip point) and nudges one node by 1 mV,
    so the fastest-growing -- i.e. the fundamental -- mode is what develops.
    """
    net, outs, xs = build_chain(s, kind, nstage, vdd, cload=cload, ring=True)
    net.cap_scale = cap_scale
    v0 = np.zeros(net.n)
    v0[net._idx["vdd"]] = vdd
    if init == "perturb":
        g = np.full(net.n, vdd * 0.3)
        g[net._idx["vdd"]] = vdd
        v, ok = net.solve_dc(guess=g)
        if ok:
            v0 = v.copy()
            v0[net._idx[outs[0]]] += 1e-3
            return net, outs, xs, v0
    for i, o in enumerate(outs):
        lv = vdd if (i % 2 == 0) else 0.0
        v0[net._idx[o]] = lv
        if xs[i]:
            v0[net._idx[xs[i]]] = lv        # X tracks its own stage's output
    return net, outs, xs, v0


def run_ring(s, kind, nstage, vdd, t_end, cload=0.0, cap_mode="incremental",
             dvmax=0.04, dt_max=None, settle=0.45, max_steps=200000,
             init="alternating", cap_scale=1.0):
    """Transient a ring and measure frequency, swing, power.

    `settle` is the fraction of the window discarded before measuring, so the
    startup transient and any amplitude build-up are not counted.
    """
    net, outs, xs, v0 = build_ring(s, kind, nstage, vdd, cload, init, cap_scale)
    # dt_max has to stay small compared with the *period*, not the window. The
    # windows here hold ~9 periods, so t_end/400 leaves only ~44 steps per period
    # and backward Euler's numerical damping then competes with the oscillation
    # itself -- harmless for a ring in full swing, but it suppresses marginal
    # ones outright and so fakes a supply limit. t_end/2000 is ~220 steps per
    # period; the dt refinement in ring_osc5.py puts the residual error at 0.1%.
    t, V = net.transient(v0, t_end, dvmax=dvmax, cap_mode=cap_mode,
                         dt_max=dt_max or t_end / 2000.0, max_steps=max_steps)
    st = dict(net.stats)
    res = dict(kind=kind, nstage=nstage, vdd=vdd, cboot=s.cboot, cload=cload,
               w1=s.w1 * 1e6, w2=s.w2 * 1e6, w3=s.w3 * 1e6, w4=s.w4 * 1e6,
               cap_mode=cap_mode, init=init, cap_scale=cap_scale,
               dvmax=dvmax, steps=st["steps"], rejects=st["rejects"],
               oob_frac=st["oob_steps"] / max(st["steps"], 1),
               complete=st["complete"], t_end=t_end)
    if len(t) < 50 or not st["complete"]:
        res.update(freq=np.nan, period=np.nan, swing=np.nan, oscillates=False,
                   reason="transient did not finish")
        return res, (t, V, net, outs, xs)

    t0 = settle * t[-1]
    o0 = V[:, net._idx[outs[0]]]
    pm = period_from_crossings(t, o0, t_start=t0)
    m = t >= t0
    vmin, vmax = float(o0[m].min()), float(o0[m].max())
    # a ring that has latched shows a flat node; require a real, repeated swing
    if pm is None or pm["n_periods"] < 2 or (vmax - vmin) < 0.15 * vdd:
        res.update(freq=np.nan, period=np.nan, swing=vmax - vmin,
                   vmin=vmin, vmax=vmax, oscillates=False,
                   reason="latched / no sustained swing")
        return res, (t, V, net, outs, xs)

    # supply current, time-averaged over whole periods
    iv = net._idx["vdd"]
    nper = int(pm["n_periods"])
    tw = min(nper * pm["period"], t[-1] - t0)
    sel = (t >= t[-1] - tw)
    isup = np.empty(int(sel.sum()))
    for k, idx in enumerate(np.where(sel)[0]):
        d = net.devices(V[idx])
        isup[k] = float(np.sum(d["id"][net.nd == iv]) - np.sum(d["id"][net.ns == iv]))
    tt = t[sel]
    i_avg = float(np.trapezoid(isup, tt) / (tt[-1] - tt[0]))

    # Which spatial mode is running. Stage i+1 inverts stage i and lags it by
    # one stage delay, so rise-to-rise = T/2 + k*T/(2N) and the odd integer k is
    # the mode index: k = 1 is the fundamental, k = 3, 5, ... are the
    # multi-wave modes a long ring can also sustain.
    mode = np.nan
    if nstage >= 3:
        from .logic_sim import crossings
        lv = 0.5 * (vmin + vmax)
        c0 = crossings(t[m], V[m, net._idx[outs[0]]], lv, rising=True)
        c1 = crossings(t[m], V[m, net._idx[outs[1]]], lv, rising=True)
        if len(c0) and len(c1):
            dly = (c1[np.argmin(np.abs(c1 - c0[0]))] - c0[0]) % pm["period"]
            mode = float(2 * nstage * ((dly / pm["period"] - 0.5) % 1.0))

    xnode = V[:, net._idx[xs[0]]] if xs[0] else None
    res.update(freq=pm["freq"], period=pm["period"], jitter=pm["jitter"],
               n_periods=pm["n_periods"], swing=vmax - vmin, vmin=vmin, vmax=vmax,
               swing_pct=100.0 * (vmax - vmin) / vdd,
               i_supply=i_avg, power=vdd * i_avg,
               e_per_cycle=vdd * i_avg / pm["freq"],
               e_per_transition=vdd * i_avg / (pm["freq"] * 2 * nstage),
               tpd=pm["period"] / (2.0 * nstage),
               fo1_tpd_ns=pm["period"] / (2.0 * nstage) * 1e9,
               x_max=float(xnode[m].max()) if xnode is not None else np.nan,
               x_min=float(xnode[m].min()) if xnode is not None else np.nan,
               boost=float(xnode[m].max() - vdd) if xnode is not None else np.nan,
               mode_index=mode, oscillates=True, reason="")
    return res, (t, V, net, outs, xs)


def run_ring_auto(s, kind, nstage, vdd, t_end=60e-6, tries=4, min_periods=3, **kw):
    """run_ring, stretching the window until enough periods are captured."""
    last = None
    for _ in range(tries):
        r, extra = run_ring(s, kind, nstage, vdd, t_end, **kw)
        last = (r, extra)
        if r["oscillates"] and r.get("n_periods", 0) >= min_periods:
            return r, extra
        if r["oscillates"] and r.get("n_periods", 0) >= 2:
            return r, extra
        if not r["complete"]:
            break
        t_end *= 5.0
    return last
