"""Batched DC + transient engine for unipolar a-IGZO TFT logic.

`src/circuit.py` is built for 2-7 node analog blocks: it calls the ANN once per
device per stamp, which is fine for a differential pair and hopeless for a
51-stage ring oscillator (4 devices/stage, ~10^4 time steps). Here every device
is evaluated in a single batched `TFTModel.op()` call and the Jacobian is
assembled with pre-computed flat index arrays, so one Newton iteration costs a
fixed handful of numpy calls regardless of circuit size.

Two physics fixes on top of the raw `.va`, both needed before unipolar *logic*
is meaningful (see outputs/va_test/README.md items 2 and 3):

* **D/S symmetrisation.** The `.va` is a one-way device: `V(D,S)` is min-max
  clamped at `vd_lo = 0`, so a reverse-biased device returns a meaningless
  ~1 nA and `gds = 0` exactly. A bootstrap node driven above VDD reverse-biases
  its charging device on purpose, and a logic node settling at VOL sits right on
  VDS = 0. Both are evaluated here by swapping the terminals and negating the
  current, which is what a physically symmetric TFT does.
* **ID(VDS=0) = 0.** A log10 output cannot reach zero, so the net sources
  0.3-5 nA into a shorted device. Left alone, the symmetrisation above turns
  that into a ~10 nA step discontinuity at VDS = 0 -- exactly where every logic
  node comes to rest -- and Newton stops converging. Subtracting ID(VGS, 0)
  makes the device odd-symmetric and continuous; it costs 0.01% of the on-state
  current, which is 4 orders below the model's own 0.8-decade accuracy.

Both are switchable (`symmetric`, `offset`) so the engine can be made
bit-identical to `src/circuit.py` for cross-validation -- see
`scripts/validate_logic_sim.py`.

Conventions: node 0 is ground. Currents are stamped as "current leaving the
node". Charge integration is backward Euler; `cap_mode` picks between the
`.va`'s own non-charge-conserving `ddt(C*V)` and the incremental-capacitance
form (see `CAP_MODES`).
"""
import numpy as np

from .va_model import TFTModel

_M = None


def model():
    global _M
    if _M is None:
        _M = TFTModel()
    return _M


VDS_BLEND = 0.05   # V; smooths the D/S cap swap through VDS = 0


def _amax(a):
    """max|a| that tolerates an empty array (device-less test netlists)."""
    a = np.asarray(a)
    return float(np.max(np.abs(a))) if a.size else 0.0

CAP_MODES = ("incremental", "cv")
# "cv"          Q = C(V)*V, i.e. what a simulator actually sees from this .va's
#               `I(G,D) <+ ddt(cgd*V(G,D))`. Not charge conserving: the
#               effective dQ/dV is C + V*dC/dV, up to 303% off C at the knee.
# "incremental" branch current = C(V) * dV/dt, i.e. C is used as the
#               (measured, 3.6%-accurate) incremental capacitance. Also not
#               strictly conserving, but it integrates the quantity that was
#               actually calibrated. Default; both are reported.


def eval_batch(vgs, vds, w, l, symmetric=True, offset=True, caps=True):
    """Vectorised device evaluation.

    Returns id, did/dvgs, did/dvds, cgd, cgs plus the model-box flags, all
    shaped like the broadcast of the inputs.
    """
    vgs = np.asarray(vgs, dtype=float)
    vds = np.asarray(vds, dtype=float)
    if symmetric:
        rev = vds < 0.0
        u = np.where(rev, -vds, vds)
        vg_eff = np.where(rev, vgs - vds, vgs)
    else:
        rev = np.zeros_like(vds, dtype=bool)
        u, vg_eff = vds, vgs

    m = model()
    o = m.op(vg_eff, u, w, l, caps=caps)
    if offset:
        o0 = m.op(vg_eff, np.zeros_like(u), w, l, caps=False)
        i0, gm0 = o0["id"], o0["gm"]
    else:
        i0 = gm0 = 0.0

    i = o["id"] - i0
    G = o["gm"] - gm0          # d/d vg_eff
    D = o["gds"]               # d/d u

    sgn = np.where(rev, -1.0, 1.0)
    idv = sgn * i
    didvg = np.where(rev, -G, G)
    didvd = np.where(rev, G + D, D)

    # the model's "cgd" is the gate-to-(whichever terminal it called D) cap, so
    # under a terminal swap cgd and cgs exchange. tanh blend keeps C continuous
    # through VDS = 0, where the two nets differ by up to 80% in subthreshold.
    if not caps:
        cgd = cgs = None
    elif symmetric:
        a = 0.5 * (1.0 + np.tanh(vds / VDS_BLEND))
        cgd = a * o["cgd"] + (1.0 - a) * o["cgs"]
        cgs = a * o["cgs"] + (1.0 - a) * o["cgd"]
    else:
        cgd, cgs = o["cgd"], o["cgs"]

    out_of_box = (vg_eff < m.vg_lo) | (vg_eff > m.vg_hi) | (u > m.vd_hi)
    return dict(id=idv, didvg=didvg, didvd=didvd, cgd=cgd, cgs=cgs,
                oob=out_of_box, vgs_eff=vg_eff, vds_eff=u, rev=rev)


class Net:
    """A netlist. Nodes are created by name; node 0 ('gnd') is always 0 V."""

    def __init__(self, symmetric=True, offset=True):
        self._idx = {"gnd": 0}
        self.dev = []                 # (name, d, g, s, w, l)
        self._res = []                # (a, b, R)
        self._cap = []                # (a, b, C)
        self.fixed = {0: 0.0}
        self.symmetric, self.offset = symmetric, offset
        # Global multiplier on every *device* capacitance (linear caps are not
        # touched). The CGD/CGS nets were trained at four geometries, all
        # L >= 15 um, so L = 5 um logic runs on an area-scaling extrapolation;
        # sweeping this factor turns that into a stated uncertainty band
        # instead of a hidden assumption.
        self.cap_scale = 1.0
        self._built = False

    # ---------- construction ----------
    def node(self, name):
        if name not in self._idx:
            self._idx[name] = len(self._idx)
            self._built = False
        return self._idx[name]

    def names(self):
        return {v: k for k, v in self._idx.items()}

    def tft(self, name, d, g, s, w, l):
        self.dev.append((name, self.node(d), self.node(g), self.node(s), w, l))
        self._built = False

    def res(self, a, b, r):
        self._res.append((self.node(a), self.node(b), r))
        self._built = False

    def cap(self, a, b, c):
        self._cap.append((self.node(a), self.node(b), c))
        self._built = False

    def force(self, name, value):
        self.fixed[self.node(name)] = float(value)
        self._built = False

    # ---------- compile ----------
    def build(self):
        n = len(self._idx)
        self.n = n
        self.unk = np.array([i for i in range(n) if i not in self.fixed], dtype=int)
        self.nu = len(self.unk)
        pos = -np.ones(n, dtype=int)
        pos[self.unk] = np.arange(self.nu)
        self.pos = pos

        self.nd = np.array([d[1] for d in self.dev], dtype=int)
        self.ng = np.array([d[2] for d in self.dev], dtype=int)
        self.ns = np.array([d[3] for d in self.dev], dtype=int)
        self.W = np.array([d[4] for d in self.dev], dtype=float)
        self.L = np.array([d[5] for d in self.dev], dtype=float)
        self.dnames = [d[0] for d in self.dev]

        # --- device Jacobian stamp pattern (values change, pattern does not) ---
        rows = np.concatenate([self.nd, self.nd, self.nd, self.ns, self.ns, self.ns])
        cols = np.concatenate([self.ng, self.nd, self.ns, self.ng, self.nd, self.ns])
        r, c = pos[rows], pos[cols]
        self._dmask = (r >= 0) & (c >= 0)
        self._dflat = (r[self._dmask] * self.nu + c[self._dmask])

        # --- capacitor branches: device cgd (g,d), device cgs (g,s), linear ---
        ca = np.concatenate([self.ng, self.ng,
                             np.array([x[0] for x in self._cap], dtype=int)])
        cb = np.concatenate([self.nd, self.ns,
                             np.array([x[1] for x in self._cap], dtype=int)])
        self.ca, self.cb = ca, cb
        self.clin = np.array([x[2] for x in self._cap], dtype=float)
        crows = np.concatenate([ca, ca, cb, cb])
        ccols = np.concatenate([ca, cb, ca, cb])
        r, c = pos[crows], pos[ccols]
        self._cmask = (r >= 0) & (c >= 0)
        self._cflat = (r[self._cmask] * self.nu + c[self._cmask])

        # --- resistors: static, fold straight into dense matrices ---
        Gr = np.zeros((n, n))
        for a, b, rr in self._res:
            g = 1.0 / rr
            Gr[a, a] += g
            Gr[a, b] -= g
            Gr[b, a] -= g
            Gr[b, b] += g
        self.Gr = Gr
        self.Gr_unk = Gr[np.ix_(self.unk, self.unk)]
        self._built = True
        return self

    # ---------- evaluation ----------
    def _ensure(self):
        if not self._built:
            self.build()

    def vfull(self, x):
        v = np.zeros(self.n)
        for k, val in self.fixed.items():
            v[k] = val
        v[self.unk] = x
        return v

    def devices(self, v):
        self._ensure()
        return eval_batch(v[self.ng] - v[self.ns], v[self.nd] - v[self.ns],
                          self.W, self.L, self.symmetric, self.offset)

    def residual(self, v, dv=None):
        """Static KCL (current leaving each node). `dv` caches device eval."""
        d = self.devices(v) if dv is None else dv
        f = self.Gr @ v
        np.add.at(f, self.nd, d["id"])
        np.add.at(f, self.ns, -d["id"])
        return f

    def jac(self, v, dv=None):
        d = self.devices(v) if dv is None else dv
        g, s = d["didvg"], d["didvd"]
        vals = np.concatenate([g, s, -(g + s), -g, -s, (g + s)])
        Jf = np.zeros(self.nu * self.nu)
        np.add.at(Jf, self._dflat, vals[self._dmask])
        return Jf.reshape(self.nu, self.nu) + self.Gr_unk

    def cvals(self, dv):
        k = self.cap_scale
        return np.concatenate([k * dv["cgd"], k * dv["cgs"], self.clin])

    def charge(self, v, cv):
        q = np.zeros(self.n)
        qq = cv * (v[self.ca] - v[self.cb])
        np.add.at(q, self.ca, qq)
        np.add.at(q, self.cb, -qq)
        return q

    def cap_jac(self, cv, scale):
        Jf = np.zeros(self.nu * self.nu)
        vals = np.concatenate([cv, -cv, -cv, cv]) * scale
        np.add.at(Jf, self._cflat, vals[self._cmask])
        return Jf.reshape(self.nu, self.nu)

    # ---------- DC ----------
    def solve_dc(self, guess=None, maxit=120, vlim=0.3, reltol=1e-6, abstol=1e-14):
        self._ensure()
        x = (np.full(self.nu, 0.0) if guess is None
             else np.asarray(guess, dtype=float)[self.unk].copy())
        for _ in range(maxit):
            v = self.vfull(x)
            d = self.devices(v)
            f = self.residual(v, d)[self.unk]
            tol = abstol + reltol * max(_amax(d["id"]), 1e-12)
            if np.max(np.abs(f)) < tol:
                return v, True
            J = self.jac(v, d)
            try:
                dx = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, -f, rcond=None)[0]
            dx = np.clip(dx, -vlim, vlim)
            best, bx, bn = None, None, np.inf
            for a in (1.0, 0.5, 0.25, 0.1, 0.03):
                xt = x + a * dx
                nf = np.max(np.abs(self.residual(self.vfull(xt))[self.unk]))
                if nf < bn:
                    bn, bx, best = nf, xt, a
            x = bx
        v = self.vfull(x)
        d = self.devices(v)
        f = _amax(self.residual(v, d)[self.unk])
        return v, bool(f < abstol + 1e-4 * max(_amax(d["id"]), 1e-12))

    def box_report(self, v):
        d = self.devices(v)
        return dict(oob=bool(np.any(d["oob"])),
                    n_oob=int(np.sum(d["oob"])),
                    vgs_max=_amax(d["vgs_eff"]),
                    vds_max=_amax(d["vds_eff"]))

    # ---------- transient ----------
    def transient(self, v0, t_end, dt0=None, dt_min=None, dt_max=None,
                  dvmax=0.05, cap_mode="incremental", drive=None,
                  maxit=25, vlim=0.3, reltol=1e-5, abstol=1e-14, max_steps=400000):
        """Backward Euler with adaptive dt. Returns (t, V) with V[k] = node volts.

        `drive(time, net)` may update `net.fixed` before each step.
        """
        self._ensure()
        assert cap_mode in CAP_MODES
        dt0 = dt0 or t_end / 2000.0
        dt_min = dt_min or t_end / 5e7
        dt_max = dt_max or t_end / 200.0

        v = np.asarray(v0, dtype=float).copy()
        for k, val in self.fixed.items():
            v[k] = val
        x = v[self.unk].copy()
        dv = self.devices(v)
        cv = self.cvals(dv)
        q_prev = self.charge(v, cv)[self.unk]
        dvq_prev = (v[self.ca] - v[self.cb])

        ts, Vs = [0.0], [v.copy()]
        tnow, dt, nstep, nrej = 0.0, dt0, 0, 0
        oob_steps = 0
        while tnow < t_end and nstep < max_steps:
            dt = min(dt, t_end - tnow, dt_max)
            if drive is not None:
                drive(tnow + dt, self)
            ok = False
            xt = x.copy()
            for _ in range(maxit):
                vt = self.vfull(xt)
                d = self.devices(vt)
                cvn = self.cvals(d)
                if cap_mode == "cv":
                    qn = self.charge(vt, cvn)[self.unk]
                    icap = (qn - q_prev) / dt
                else:
                    dvn = vt[self.ca] - vt[self.cb]
                    qq = cvn * (dvn - dvq_prev)
                    qn_ = np.zeros(self.n)
                    np.add.at(qn_, self.ca, qq)
                    np.add.at(qn_, self.cb, -qq)
                    icap = qn_[self.unk] / dt
                f = self.residual(vt, d)[self.unk] + icap
                scale = max(_amax(d["id"]), _amax(icap), 1e-12)
                if _amax(f) < abstol + reltol * scale:
                    ok = True
                    break
                J = self.jac(vt, d) + self.cap_jac(cvn, 1.0 / dt)
                try:
                    step = np.linalg.solve(J, -f)
                except np.linalg.LinAlgError:
                    break
                xt = xt + np.clip(step, -vlim, vlim)
            if not ok or not np.all(np.isfinite(xt)):
                if dt <= dt_min * 1.001:
                    break
                dt = max(dt / 4.0, dt_min)
                nrej += 1
                continue
            vt = self.vfull(xt)
            dmax = np.max(np.abs(vt[self.unk] - x))
            if dmax > 2.5 * dvmax and dt > dt_min * 1.001:
                dt = max(dt / 2.0, dt_min)      # too coarse: redo the step
                nrej += 1
                continue

            d = self.devices(vt)
            oob_steps += int(np.any(d["oob"]))
            cv = self.cvals(d)
            q_prev = self.charge(vt, cv)[self.unk]
            dvq_prev = vt[self.ca] - vt[self.cb]
            x = xt
            tnow += dt
            nstep += 1
            ts.append(tnow)
            Vs.append(vt.copy())
            grow = np.clip(dvmax / max(dmax, 1e-12), 0.6, 1.8)
            dt = float(np.clip(dt * grow, dt_min, dt_max))
        self.stats = dict(steps=nstep, rejects=nrej, oob_steps=oob_steps,
                          reached=tnow, complete=tnow >= t_end * 0.999)
        return np.array(ts), np.array(Vs)


# ---------------- waveform helpers ----------------
def crossings(t, y, level, rising=True):
    """Linearly interpolated level crossings."""
    s = np.sign(y - level)
    idx = np.where(np.diff(s) != 0)[0]
    out = []
    for i in idx:
        if rising is not None and ((s[i + 1] > s[i]) != rising):
            continue
        y0, y1 = y[i], y[i + 1]
        if y1 == y0:
            continue
        out.append(t[i] + (level - y0) * (t[i + 1] - t[i]) / (y1 - y0))
    return np.array(out)


def period_from_crossings(t, y, t_start=None, level=None):
    """Oscillation period from successive rising mid-level crossings."""
    m = slice(None) if t_start is None else (t >= t_start)
    tt, yy = t[m], y[m]
    if len(tt) < 10:
        return None
    lo, hi = yy.min(), yy.max()
    if hi - lo < 1e-3:
        return None
    lv = 0.5 * (lo + hi) if level is None else level
    xs = crossings(tt, yy, lv, rising=True)
    if len(xs) < 3:
        return None
    per = np.diff(xs)
    return dict(period=float(np.mean(per)), jitter=float(np.std(per)),
                n_periods=len(per), freq=1.0 / float(np.mean(per)),
                vmin=float(lo), vmax=float(hi), swing=float(hi - lo))
