"""Waveform measurements shared by the logic scripts."""
import numpy as np


def crossings(t, y, level, direction):
    """Times where y crosses `level` going up (+1) or down (-1), interpolated."""
    s = np.sign(y - level)
    idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0] if direction > 0 else \
        np.where((s[:-1] > 0) & (s[1:] <= 0))[0]
    out = []
    for k in idx:
        f = (level - y[k]) / (y[k + 1] - y[k])
        out.append(t[k] + f * (t[k + 1] - t[k]))
    return np.array(out)


def delays(t, vin, vout, vdd, inverting=True):
    """50%-50% propagation delays (tpHL, tpLH) for each input edge."""
    mid = vdd / 2
    rin = crossings(t, vin, mid, +1)
    fin = crossings(t, vin, mid, -1)
    rout = crossings(t, vout, mid, +1)
    fout = crossings(t, vout, mid, -1)
    if inverting:
        pairs = ((rin, fout), (fin, rout))
    else:
        pairs = ((rin, rout), (fin, fout))
    res = []
    for ins, outs in pairs:
        d = []
        for ti in ins:
            later = outs[outs > ti]
            d.append(later[0] - ti if len(later) else np.nan)
        res.append(np.array(d))
    return res      # [delay after input rise, delay after input fall]


def transition(t, y, lo, hi, direction):
    """10-90% rise (direction +1) or 90-10% fall times."""
    a = lo + 0.1 * (hi - lo)
    b = lo + 0.9 * (hi - lo)
    if direction > 0:
        t1, t2 = crossings(t, y, a, +1), crossings(t, y, b, +1)
    else:
        t1, t2 = crossings(t, y, b, -1), crossings(t, y, a, -1)
    out = []
    for x in t1:
        later = t2[t2 > x]
        if len(later):
            out.append(later[0] - x)
    return np.array(out)


def sample(t, y, times):
    return np.interp(times, t, y)


def avg_power(t, i, vdd, t0, t1):
    m = (t >= t0) & (t <= t1)
    return vdd * np.trapezoid(i[m], t[m]) / (t[m][-1] - t[m][0])
