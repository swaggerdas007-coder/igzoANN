"""Vectorised transient simulator for circuits of verilogA/ntft_full.va TFTs.

ngspice can run the model through the behavioural transliteration in
src/ngspice_tft.py, but each device then costs ~10 ms per Newton iteration
(the symbolic derivative of a 4-32-16-1 net is enormous), which is hours for
a 100+ TFT counter. This engine evaluates the *same* equations -- weights
parsed out of the .va by src/va_model.py, the .va's [0,1] input clamps,
ID = 10^out, C = max(out,0)*W*L*1e-15, and the .va's Q = C(V)*V charges with
I = dQ/dt -- for all devices at once in numpy, with exact analytic Jacobians
(including the V*dC/dV term that the ddt(C*V) form produces).

Two capacitor forms are available (Circuit.tran(form=...)):

  'qcv'  the .va exactly as written: I(G,D) <+ ddt(cgd*V(G,D)). Its
         incremental capacitance is C + V*dC/dV, which goes strongly
         *negative* when a gate crosses threshold with V(G,D) ~ -4 V (any
         logic input or pull-up whose drain is high): -5.9 pF against a
         physical 0.4 pF on an 80/5 um device. The node equations then have
         no solution through the knee and the transient stalls.
  'cdv'  I(G,D) <+ cgd*ddt(V(G,D)) -- same nets, same C values, but the
         incremental capacitance is exactly C >= 0 (verilogA/ntft_full_cdv.va).

Integration is variable-step BDF2 (Gear-2, what ngspice uses with
method=gear), first order after breakpoints. The step is set by the largest
per-step node-voltage change (target dv_step) and never crosses a source
breakpoint. Every unknown node has a g_shunt (default 1e-13 S, i.e. ngspice
rshunt=1e13) to ground so that a stack node isolated by two off devices still
has a defined voltage; at 5 V that is 0.5 pA, well below the devices' own
leakage.

Nodes are strings; '0' is ground. Sources are fixed nodes whose voltage is a
function of time.
"""
import numpy as np

from .va_model import TFTModel

_M = None


def _model():
    global _M
    if _M is None:
        _M = TFTModel()
    return _M


def _net_eval(n, x):
    """Forward pass + d(out)/d(x0), d(out)/d(x1). x: (N,4) scaled inputs."""
    y1 = np.tanh(x @ n["W1"].T + n["b1"])                 # (N,h1)
    y2 = np.tanh(y1 @ n["W2"].T + n["b2"])                # (N,h2)
    out = (y2 @ n["wo"] + n["bo"]) * n["std"] + n["mean"]
    # d out / d x_k for k = 0 (VG) and 1 (VD)
    g2 = (1 - y2 ** 2) * n["wo"]                          # (N,h2)
    g1 = (g2 @ n["W2"]) * (1 - y1 ** 2)                   # (N,h1)
    dx = g1 @ n["W1"][:, :2]                              # (N,2)
    return out, dx[:, 0] * n["std"], dx[:, 1] * n["std"]


def device_eval(vgs, vds, ws, ls, area, symmetric=False):
    """All TFTs at once. Returns id, gm, gds, cgd, cgs and dC/dVGS, dC/dVDS.

    symmetric=False is the .va as written: VDS < 0 is clamped to 0, so the
    device keeps driving its VDS = 0 current D->S and never conducts in
    reverse. symmetric=True swaps D and S for VDS < 0 (ID = -F(VGD, VSD)) --
    a sensitivity check only; the capacitances are left as the .va has them.
    """
    if symmetric:
        rev = vds < 0
        r = device_eval(np.where(rev, vgs - vds, vgs), np.abs(vds), ws, ls, area)
        rf = device_eval(vgs, vds, ws, ls, area)
        fg, fd = r["gm"], r["gds"]
        rf["id"] = np.where(rev, -r["id"], r["id"])
        rf["gm"] = np.where(rev, -fg, fg)
        rf["gds"] = np.where(rev, fg + fd, fd)
        return rf
    m = _model()
    rg = m.vg_hi - m.vg_lo
    rd = m.vd_hi - m.vd_lo
    xg = (vgs - m.vg_lo) / rg
    xd = (vds - m.vd_lo) / rd
    in_g = (xg > 0) & (xg < 1)
    in_d = (xd > 0) & (xd < 1)
    x = np.stack([np.clip(xg, 0, 1), np.clip(xd, 0, 1), ws, ls], axis=1)

    o, dg, dd = _net_eval(m.nets["id"], x)
    idd = np.power(10.0, o)
    k = idd * np.log(10.0)
    gm = np.where(in_g, k * dg / rg, 0.0)
    gds = np.where(in_d, k * dd / rd, 0.0)
    res = dict(id=idd, gm=gm, gds=gds)
    for tag in ("cgd", "cgs"):
        o, dg, dd = _net_eval(m.nets[tag], x)
        pos = o > 0
        res[tag] = np.where(pos, o, 0.0) * area
        res[tag + "_g"] = np.where(pos & in_g, dg / rg, 0.0) * area
        res[tag + "_d"] = np.where(pos & in_d, dd / rd, 0.0) * area
    return res


class Circuit:
    def __init__(self):
        self.nodes = {"0": 0}
        self.tft = []        # (name, d, g, s, w, l)
        self.caps = []       # (name, a, b, C)
        self.sources = {}    # node -> (fn(t)->V, breakpoints)

    def node(self, n):
        if n not in self.nodes:
            self.nodes[n] = len(self.nodes)
        return self.nodes[n]

    def add_tft(self, name, d, g, s, w, l):
        self.tft.append((name, self.node(d), self.node(g), self.node(s), w, l))

    def add_cap(self, name, a, b, c):
        self.caps.append((name, self.node(a), self.node(b), c))

    def add_source(self, node, fn, breakpoints=()):
        self.sources[self.node(node)] = (fn, sorted(breakpoints))

    # ------------------------------------------------------------------
    def _compile(self):
        m = _model()
        N = len(self.nodes)
        self.N = N
        fixed = [0] + list(self.sources)
        self.fixed = np.array(fixed)
        self.unk = np.array([k for k in range(N) if k not in set(fixed)])
        T = self.tft
        self.td = np.array([t[1] for t in T])
        self.tg = np.array([t[2] for t in T])
        self.ts = np.array([t[3] for t in T])
        w = np.array([t[4] for t in T])
        l = np.array([t[5] for t in T])
        self.ws = np.clip((w - m.w_lo) / (m.w_hi - m.w_lo), 0, 1)
        self.ls = np.clip((l - m.l_lo) / (m.l_hi - m.l_lo), 0, 1)
        self.area = (w * 1e6) * (l * 1e6) * 1e-3 * 1e-12
        self.ca = np.array([c[1] for c in self.caps], dtype=int)
        self.cb = np.array([c[2] for c in self.caps], dtype=int)
        self.cc = np.array([c[3] for c in self.caps])
        self.names = {v: k for k, v in self.nodes.items()}

    def _source_v(self, v, t):
        for k, (fn, _) in self.sources.items():
            v[k] = fn(t)

    def _eval(self, v):
        """DC currents I(v), conductance Jacobian G, and capacitor branches.

        Branches: arrays a, b (nodes), C (F), and dC/dv at nodes (g, d, s)
        of the owning TFT (zero for the fixed capacitors).
        """
        N = self.N
        d, g, s = self.td, self.tg, self.ts
        vgs = v[g] - v[s]
        vds = v[d] - v[s]
        r = device_eval(vgs, vds, self.ws, self.ls, self.area, self.symmetric)
        I = np.zeros(N)
        np.add.at(I, d, r["id"])
        np.add.at(I, s, -r["id"])
        G = np.zeros(N * N)
        gm, gds = r["gm"], r["gds"]
        for node, sg in ((d, 1.0), (s, -1.0)):
            np.add.at(G, node * N + g, sg * gm)
            np.add.at(G, node * N + d, sg * gds)
            np.add.at(G, node * N + s, -sg * (gm + gds))
        # C depends on (vgs, vds): dC/dvg = C_g, dC/dvd = C_d, dC/dvs = -(C_g + C_d)
        z = np.zeros(len(self.cc))
        zi = np.zeros(len(self.cc), dtype=int)
        br = dict(
            a=np.concatenate([g, g, self.ca]),
            b=np.concatenate([d, s, self.cb]),
            C=np.concatenate([r["cgd"], r["cgs"], self.cc]),
            cg=np.concatenate([r["cgd_g"], r["cgs_g"], z]),
            cd=np.concatenate([r["cgd_d"], r["cgs_d"], z]),
            ng=np.concatenate([g, g, zi]),
            nd=np.concatenate([d, d, zi]),
            ns=np.concatenate([s, s, zi]),
        )
        return I, G.reshape(N, N), br

    def _cap(self, v, br, coef, hist):
        """Capacitor currents and Jacobian for the BDF formula `coef`.

        form 'qcv' -- the .va as written: I = d/dt[C(V) * V], history = branch Q.
        form 'cdv' -- I = C(V) * dV/dt, history = branch V. Incremental
                      capacitance is exactly C >= 0.
        """
        N = self.N
        a, b, C = br["a"], br["b"], br["C"]
        vab = v[a] - v[b]
        a0 = coef[0]
        if self.form == "qcv":
            x = C * vab
            i = a0 * x + sum(c * h for c, h in zip(coef[1:], hist))
            k = vab                     # d i / dC
            kc = a0 * C                 # d i / dvab (explicit)
            kk = a0                     # scale on dC/dv contributions
        else:
            dvdt = a0 * vab + sum(c * h for c, h in zip(coef[1:], hist))
            i = C * dvdt
            k = dvdt
            kc = a0 * C
            kk = 1.0
        F = np.zeros(N)
        np.add.at(F, a, i)
        np.add.at(F, b, -i)
        J = np.zeros(N * N)
        for node, sg in ((a, 1.0), (b, -1.0)):
            np.add.at(J, node * N + a, sg * kc)
            np.add.at(J, node * N + b, -sg * kc)
            np.add.at(J, node * N + br["ng"], sg * kk * k * br["cg"])
            np.add.at(J, node * N + br["nd"], sg * kk * k * br["cd"])
            np.add.at(J, node * N + br["ns"], -sg * kk * k * (br["cg"] + br["cd"]))
        return F, J.reshape(N, N)

    def _state(self, v):
        """What the BDF history stores for each branch: Q (qcv) or V (cdv)."""
        _, _, br = self._eval(v)
        vab = v[br["a"]] - v[br["b"]]
        return br["C"] * vab if self.form == "qcv" else vab

    # ------------------------------------------------------------------
    def _residual(self, v, coef, hist, gsh):
        I, G, br = self._eval(v)
        Fc, Jc = self._cap(v, br, coef, hist)
        return I + Fc + gsh * v, G + Jc

    def _newton(self, v, t, coef, hist, gsh, maxit=40):
        u = self.unk
        self._source_v(v, t)
        eye = gsh * np.eye(len(u))
        for it in range(maxit):
            F, J = self._residual(v, coef, hist, gsh)
            Fu = F[u]
            try:
                dx = np.linalg.solve(J[np.ix_(u, u)] + eye, -Fu)
            except np.linalg.LinAlgError:
                return v, False
            dx = np.clip(dx, -0.5, 0.5)
            v[u] += dx
            if np.max(np.abs(dx)) < 1e-6 and np.max(np.abs(Fu)) < 1e-9:
                return v, True
        return v, False

    def tran(self, tstop, v0=None, dv_step=0.05, h_max=None, h_min=1e-14,
             h_init=1e-9, gsh=1e-13, settle=200e-6, form="cdv", symmetric=False,
             verbose=False):
        """Transient from t=0. v0: dict node->initial guess (a .nodeset).

        The t=0 state is found by a pseudo-transient settle with all sources
        frozen at their t=0 value, so bistable nodes stay on the side of the
        nodeset that was given.
        """
        self.form = form
        self.symmetric = symmetric
        self._compile()
        N = self.N
        h_max = h_max or tstop / 400
        v = np.zeros(N)
        for k, val in (v0 or {}).items():
            v[self.nodes[k]] = val
        self._source_v(v, 0.0)
        v = self._settle(v, settle, gsh)

        bps = sorted({b for _, (_, bp) in self.sources.items() for b in bp
                      if 0 < b < tstop} | {tstop})
        src = list(self.sources)
        ts, vs = [0.0], [v.copy()]
        isrc = [self._residual(v, (0.0,), (), gsh)[0][src]]
        q1, q2 = self._state(v), None
        t, h, h_prev = 0.0, h_init, None
        bi = 0
        nrej = 0
        while t < tstop * (1 - 1e-12):
            while bps[bi] <= t * (1 + 1e-12) + 1e-18:
                bi += 1
            hh = min(h, h_max, bps[bi] - t)
            if q2 is None:
                coef, hist = (1 / hh, -1 / hh), (q1,)
            else:
                w = hh / h_prev
                coef = ((1 + 2 * w) / (hh * (1 + w)), -(1 + w) / hh,
                        w * w / (hh * (1 + w)))
                hist = (q1, q2)
            vn, ok = self._newton(v.copy(), t + hh, coef, hist, gsh)
            dv = np.max(np.abs(vn - v)) if ok else np.inf
            if not ok or dv > 3 * dv_step:
                if hh <= h_min:
                    raise RuntimeError(f"timestep too small at t={t:.6g} s")
                h = max(hh * (0.25 if not ok else 0.5 * dv_step / dv), h_min)
                nrej += 1
                continue
            isrc.append(self._residual(vn, coef, hist, gsh)[0][src])
            t += hh
            v = vn
            ts.append(t)
            vs.append(v.copy())
            h_prev = hh
            q2, q1 = q1, self._state(v)
            h = hh * float(np.clip(dv_step / max(dv, 1e-12), 0.5, 2.0))
            if abs(t - bps[bi]) <= 1e-12 * max(t, 1e-9):
                # breakpoint: restart BDF at first order with a small step
                q2 = None
                h = min(h, h_init * 10)
            if verbose and len(ts) % 2000 == 0:
                print(f"  t={t * 1e6:8.2f} us  steps={len(ts)}  rej={nrej}", flush=True)
        V = np.array(vs)
        out = {"time": np.array(ts), "rejected": nrej}
        for name, k in self.nodes.items():
            out[name] = V[:, k]
        # current each source delivers into the circuit (positive = sourcing)
        for k, arr in zip(src, np.array(isrc).T):
            out["i(" + self.names[k] + ")"] = arr
        return out

    def dc_sweep(self, node, values, v0=None, gsh=1e-13, settle=500e-6):
        """Quasi-static sweep of one source; each point settles from the last."""
        self.form = "cdv"
        self.symmetric = False
        self._compile()
        k = self.nodes[node]
        fn, bp = self.sources[k]
        v = np.zeros(self.N)
        for name, val in (v0 or {}).items():
            v[self.nodes[name]] = val
        rows = []
        for x in values:
            self.sources[k] = ((lambda xx: (lambda t: xx))(x), bp)
            self._source_v(v, 0.0)
            v = self._settle(v, settle, gsh)
            I, _, _ = self._eval(v)
            rows.append((v.copy(), I + gsh * v))
        self.sources[k] = (fn, bp)
        V = np.array([r[0] for r in rows])
        out = {name: V[:, j] for name, j in self.nodes.items()}
        for j in self.sources:
            out["i(" + self.names[j] + ")"] = np.array([r[1][j] for r in rows])
        return out

    def _settle(self, v, tau, gsh):
        """Pseudo-transient to the t=0 DC point (sources frozen)."""
        t, h = 0.0, 1e-9
        q1 = self._state(v)
        while t < tau:
            vn, ok = self._newton(v.copy(), 0.0, (1 / h, -1 / h), (q1,), gsh)
            if not ok:
                h /= 4
                if h < 1e-15:
                    raise RuntimeError("DC settle failed")
                continue
            dv = np.max(np.abs(vn - v))
            v = vn
            q1 = self._state(v)
            t += h
            h = min(h * (2.0 if dv < 0.05 else 1.0), tau / 20)
        return v


# ---------------------------------------------------------------------- sources
def pulse_fn(v0, v1, delay, tr, tf, pw, per):
    """SPICE PULSE semantics. Returns (fn, breakpoints generator for tstop)."""
    def fn(t):
        if t < delay:
            return v0
        tt = (t - delay) % per
        if tt < tr:
            return v0 + (v1 - v0) * tt / tr
        if tt < tr + pw:
            return v1
        if tt < tr + pw + tf:
            return v1 + (v0 - v1) * (tt - tr - pw) / tf
        return v0

    def bps(tstop):
        out = [delay]
        k = 0
        while delay + k * per < tstop:
            base = delay + k * per
            out += [base + tr, base + tr + pw, base + tr + pw + tf, base + per]
            k += 1
        return [b for b in out if b < tstop]
    return fn, bps


def pwl_fn(points):
    ts = np.array([p[0] for p in points])
    vs = np.array([p[1] for p in points])
    return (lambda t: float(np.interp(t, ts, vs))), list(ts)
