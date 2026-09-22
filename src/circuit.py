"""A minimal nonlinear DC + small-signal AC engine for n-only a-IGZO TFT circuits.

Deliberately tiny: enough to solve a differential pair built from the ANN
Verilog-A model and get gain / bandwidth out of an actual nodal solve rather
than a textbook formula. Devices expose ID and the analytic gm/gds from
src.va_model, so the DC Newton iteration uses the model's own Jacobian --
the same derivatives Spectre would get from the .va.

Conventions: node 0 is ground; node voltages are the unknowns; every element
stamps a current into its nodes.
"""
import numpy as np

from .va_model import TFTModel

_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = TFTModel()
    return _MODEL


def _f(x):
    return float(np.ravel(x)[0])


class TFT:
    """One n-type IGZO TFT. Terminal names refer to node indices (or None=gnd)."""

    def __init__(self, name, d, g, s, w, l):
        self.name, self.d, self.g, self.s = name, d, g, s
        self.w, self.l = w, l

    def bias(self, v):
        vd, vg, vs = v[self.d], v[self.g], v[self.s]
        return vg - vs, vd - vs

    def op(self, v):
        vgs, vds = self.bias(v)
        o = model().op(vgs, vds, self.w, self.l)
        return {k: _f(val) if np.asarray(val).dtype != bool else bool(np.ravel(val)[0])
                for k, val in o.items()}


class Circuit:
    """Nodes 1..n are unknowns; `fixed` maps node -> forced voltage (sources)."""

    def __init__(self, n_nodes):
        self.n = n_nodes
        self.tfts = []
        self.res = []      # (na, nb, R)
        self.caps = []     # (na, nb, C)
        self.fixed = {}    # node -> volts (node 0 is always 0 V)

    def add_tft(self, *a, **k):
        t = TFT(*a, **k)
        self.tfts.append(t)
        return t

    def add_res(self, a, b, r):
        self.res.append((a, b, r))

    def add_cap(self, a, b, c):
        self.caps.append((a, b, c))

    # ---------------- DC ----------------
    def _unknowns(self):
        return [i for i in range(1, self.n + 1) if i not in self.fixed]

    def _full(self, x):
        v = np.zeros(self.n + 1)
        for k, val in self.fixed.items():
            v[k] = val
        for i, k in enumerate(self._unknowns()):
            v[k] = x[i]
        return v

    def residual(self, v):
        """KCL: net current leaving each node."""
        f = np.zeros(self.n + 1)
        for t in self.tfts:
            i = _f(model().id_(*t.bias(v), t.w, t.l))
            f[t.d] += i
            f[t.s] -= i
        for a, b, r in self.res:
            i = (v[a] - v[b]) / r
            f[a] += i
            f[b] -= i
        return f

    def jacobian(self, v):
        idx = self._unknowns()
        pos = {k: i for i, k in enumerate(idx)}
        J = np.zeros((len(idx), len(idx)))

        def stamp(node, wrt, val):
            if node in pos and wrt in pos:
                J[pos[node], pos[wrt]] += val

        for t in self.tfts:
            o = t.op(v)
            gm, gds = o["gm"], o["gds"]
            # i flows d->s; depends on vg,vd,vs
            for node, sgn in ((t.d, +1), (t.s, -1)):
                stamp(node, t.g, sgn * gm)
                stamp(node, t.d, sgn * gds)
                stamp(node, t.s, sgn * (-gm - gds))
        for a, b, r in self.res:
            for node, sgn in ((a, +1), (b, -1)):
                stamp(node, a, sgn / r)
                stamp(node, b, -sgn / r)
        return J

    def solve_dc(self, guess=None, tol=1e-12, maxit=200):
        idx = self._unknowns()
        x = np.array([guess[k] for k in idx]) if guess else np.full(len(idx), 1.0)
        lam = 1e-9
        for _ in range(maxit):
            v = self._full(x)
            f = self.residual(v)[idx]
            if np.max(np.abs(f)) < tol:
                return v, True
            J = self.jacobian(v)
            try:
                dx = np.linalg.solve(J + lam * np.eye(len(idx)), -f)
            except np.linalg.LinAlgError:
                return self._full(x), False
            # damped Newton with voltage-step limiting (helps across the knee)
            step = np.clip(dx, -0.5, 0.5)
            best, bx = np.inf, None
            for a in (1.0, 0.5, 0.25, 0.1, 0.03):
                xt = x + a * step
                nf = np.max(np.abs(self.residual(self._full(xt))[idx]))
                if nf < best:
                    best, bx = nf, xt
            x = bx
        return self._full(x), np.max(np.abs(self.residual(self._full(x))[idx])) < 1e-9

    # ---------------- AC ----------------
    def ac(self, v_dc, freqs, drive):
        """Nodal AC solve. `drive` maps fixed node -> complex stimulus amplitude.

        Returns array (len(freqs), n_nodes+1) of complex node voltages.
        """
        idx = self._unknowns()
        pos = {k: i for i, k in enumerate(idx)}
        ops = [(t, t.op(v_dc)) for t in self.tfts]
        out = np.zeros((len(freqs), self.n + 1), dtype=complex)

        for fi, fr in enumerate(freqs):
            s = 2j * np.pi * fr
            Y = np.zeros((len(idx), len(idx)), dtype=complex)
            I = np.zeros(len(idx), dtype=complex)

            def stamp(node, wrt, val):
                if node not in pos:
                    return
                if wrt in pos:
                    Y[pos[node], pos[wrt]] += val
                else:                      # known voltage -> moves to RHS
                    I[pos[node]] -= val * drive.get(wrt, 0.0)

            for t, o in ops:
                for node, sgn in ((t.d, +1), (t.s, -1)):
                    stamp(node, t.g, sgn * o["gm"])
                    stamp(node, t.d, sgn * o["gds"])
                    stamp(node, t.s, sgn * (-o["gm"] - o["gds"]))
                for (a, b, c) in ((t.g, t.d, o["cgd"]), (t.g, t.s, o["cgs"])):
                    for node, sgn in ((a, +1), (b, -1)):
                        stamp(node, a, sgn * s * c)
                        stamp(node, b, -sgn * s * c)
            for a, b, r in self.res:
                for node, sgn in ((a, +1), (b, -1)):
                    stamp(node, a, sgn / r)
                    stamp(node, b, -sgn / r)
            for a, b, c in self.caps:
                for node, sgn in ((a, +1), (b, -1)):
                    stamp(node, a, sgn * s * c)
                    stamp(node, b, -sgn * s * c)

            x = np.linalg.solve(Y, I)
            vv = np.zeros(self.n + 1, dtype=complex)
            for k, val in drive.items():
                vv[k] = val
            for k in idx:
                vv[k] = x[pos[k]]
            out[fi] = vv
        return out
