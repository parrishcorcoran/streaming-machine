"""PHOTONS family, analysis only (not a codec; float logs are used here only to count what an ideal coder would pay):
where do the bits of a developed value go, and what is the least any coder can pay for them given what it knows?
  part 1  the top photon's place (the exposure of this one value): static entropy given a context
  part 2  the sign of the value
  part 3  the places under the top photon (plain writing), each with its lit-share at that depth under the top
The sum of the three, with static per-context counts, is the floor for a coder that knows only that context (the
cost of learning the counts is not in it). Compared with the real bytes of the codings in photons_results.json.
    python3 shrink/photons_floor.py
"""
from photons_lib import *
import math

def H(counts):
    c = np.asarray([x for x in counts if x > 0], np.float64); n = c.sum()
    return float(-(c * np.log2(c / n)).sum()) if n else 0.0

def topplace(a):
    t = np.zeros(a.shape, np.int64); x = a.copy()
    while x.any(): t += x > 0; x >>= 1
    return t

class Acc:
    """accumulates counts: top place per context, signs, and (depth under top, lit) for the plain writing"""
    def __init__(self): self.top = {}; self.sign = np.zeros(2, np.int64); self.low = np.zeros((40, 2), np.int64); self.val = {}
    def add(self, v, ctx):
        v = np.asarray(v, np.int64).ravel(); ctx = np.asarray(ctx, np.int64).ravel(); a = np.abs(v); t = topplace(a)
        key = ctx * 64 + t; u, c = np.unique(key, return_counts=True)
        for k, n in zip(u.tolist(), c.tolist()): self.top[k] = self.top.get(k, 0) + n
        nz = a > 0; self.sign += np.bincount((v[nz] < 0).astype(np.int64), minlength=2)
        for d in range(1, 40):
            sel = t - 1 - d >= 0
            if not sel.any(): break
            b = (a[sel] >> (t[sel] - 1 - d)) & 1; self.low[d] += np.bincount(b, minlength=2)
        key = ctx * (1 << 20) + (v + (1 << 19)); u, c = np.unique(key, return_counts=True)
        for k, n in zip(u.tolist(), c.tolist()): self.val[k] = self.val.get(k, 0) + n
    def bits(self):
        by = {}
        for k, n in self.top.items(): by.setdefault(k // 64, []).append(n)
        top = sum(H(c) for c in by.values())
        byv = {}
        for k, n in self.val.items(): byv.setdefault(k >> 20, []).append(n)
        whole = sum(H(c) for c in byv.values())
        sign = H(self.sign); low = sum(H(self.low[d]) for d in range(1, 40))
        lowshare = {d: round(float(self.low[d, 1] / self.low[d].sum()), 4) for d in range(1, 8) if self.low[d].sum()}
        return dict(top_bits=top, sign_bits=sign, under_top_bits=low, sum_parts=top + sign + low, whole_value_bits=whole,
                    places_under_top=int(self.low.sum()), lit_share_by_depth=lowshare, contexts=len(byv))

def report(label, acc, n, pw, pp, real=None):
    b = acc.bits(); rec = {"what": f"FLOOR (analysis) {label}", "values": n}
    for k in ("top_bits", "sign_bits", "under_top_bits", "sum_parts", "whole_value_bits"):
        rec[k.replace("_bits", "_bits_per_value")] = round(b[k] / n, 4)
    rec["floor_MB_whole_value_static"] = round(b["whole_value_bits"] / 8e6, 3)
    rec["under_top_bits_per_place"] = round(b["under_top_bits"] / max(b["places_under_top"], 1), 4)
    rec["places_under_top_per_value"] = round(b["places_under_top"] / n, 4)
    rec["lit_share_by_depth_under_top"] = b["lit_share_by_depth"]; rec["contexts"] = b["contexts"]
    rec["floor_bits_per_wave_photon"] = round(b["whole_value_bits"] / pw, 4); rec["floor_bits_per_plain_photon"] = round(b["whole_value_bits"] / pp, 4)
    if real: rec.update(real)
    log(rec)

if __name__ == "__main__":
    P = read_round_one_plates(); n = sum(p["v"].size for p in P)
    pp = pw = 0
    for p in P: a, b = count_photons(p["v"]); pp += a; pw += b
    for label, ctxf in [("plates: one context per plate", lambda p: np.zeros(p["v"].shape, np.int64)),
                        ("plates: per plate x exposure residual", lambda p: exposure_residual(p["t"], p["off"], p["A"]).clip(-3, 1) + 3),
                        ("plates: per plate x neuron (each neuron its own counts)", lambda p: np.broadcast_to(np.arange(p["C"])[None, :], p["v"].shape))]:
        acc = Acc()
        for i, p in enumerate(P): acc.add(p["v"], ctxf(p) + 8 * 4096 * i)
        report(label, acc, n, pw, pp)
    del P
    for c in (32, 34):
        Q, e, h = tables_Q(c); assert h == ROUND1_TABLE_HASH[c]
        a, b = count_photons(Q)
        for label, per_line in [(f"tables c={c}: one context", False), (f"tables c={c}: per line", True)]:
            acc = Acc()
            for r in range(0, Q.shape[0], CH):
                q = Q[r:r + CH]; ctx = np.broadcast_to(np.arange(768)[None, :], q.shape) if per_line else np.zeros(q.shape, np.int64)
                acc.add(q, ctx)
            report(label, acc, Q.size, b, a)
        del Q
