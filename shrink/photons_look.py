"""PHOTONS family, look first: where are the photons of the developed values?
For the developed plates (v12 A=8) and the developed tables (c=32, c=34): photons per value in both writings, how
many values are dark, where the top photon sits, and how often each place under the top is lit (plain writing) or
holds a photon (single wave). A place lit half the time is a coin flip: it costs 1 bit however it is written.
    python3 shrink/photons_look.py
"""
from photons_lib import *

def look(label, chunks, extra=None):
    """chunks: an iterable of whole-number arrays (a plate, or 4096 words at a time) -- counted a slice at a time so
    the look stays small in memory"""
    n = 0; dark = 0; ph_p = 0; ph_w = 0; hp = np.zeros(40, np.int64); hw = np.zeros(40, np.int64)
    pl = np.zeros((9, 2), np.int64); wl = np.zeros((9, 2), np.int64)
    for V in chunks:
        V = np.asarray(V, np.int64).ravel(); a = np.abs(V); n += a.size; dark += int((a == 0).sum())
        pw, mw = naf_split(a); lit = pw | mw; del pw, mw
        ph_p += int(popcount(a).sum()); ph_w += int(popcount(lit).sum())
        top_w = np.zeros(a.size, np.int64); x = lit.copy()
        while x.any(): top_w += x > 0; x >>= 1
        top_p = np.zeros(a.size, np.int64); x = a.copy()
        while x.any(): top_p += x > 0; x >>= 1
        hp += np.bincount(top_p, minlength=40)[:40]; hw += np.bincount(top_w, minlength=40)[:40]
        for d in range(1, 9):
            sel = top_p - 1 - d >= 0
            if sel.any(): pl[d] += np.bincount((a[sel] >> (top_p[sel] - 1 - d)) & 1, minlength=2)
            sel = top_w - 1 - d >= 0
            if sel.any(): wl[d] += np.bincount((lit[sel] >> (top_w[sel] - 1 - d)) & 1, minlength=2)
    rec = {"what": label, "values": int(n), "dark_share": round(dark / n, 4),
           "photons_plain": round(ph_p / n, 4), "photons_wave": round(ph_w / n, 4),
           "top_place_plain_hist": {int(k): int(c) for k, c in enumerate(hp) if c},
           "top_place_wave_hist": {int(k): int(c) for k, c in enumerate(hw) if c},
           # plain writing: share lit at d places under the top; single wave: share holding a photon (d=1 never)
           "plain_lit_share_d_under_top": {d: round(float(pl[d, 1] / pl[d].sum()), 4) for d in range(1, 9) if pl[d].sum() >= 1000},
           "wave_photon_share_d_under_top": {d: round(float(wl[d, 1] / wl[d].sum()), 4) for d in range(1, 9) if wl[d].sum() >= 1000}}
    if extra: rec.update(extra)
    log(rec)

if __name__ == "__main__":
    t0 = time.time()
    P = read_round_one_plates()
    look("plates v12 A=8, stored whole numbers (all 48)", (p["v"] for p in P), {"coded_bytes_round_one": sum(p["coded"] for p in P),
         "side_bytes_round_one": sum(len(p["head"]) for p in P)})
    for kind in ("qkv", "proj", "fc", "mproj"):
        look(f"plates v12 A=8, {kind}", (p["v"] for p in P if p["name"].startswith(kind)))
    # exposure residual (known to the reader from t, off, A) against the top place, a plate at a time
    cnt = {}
    for p in P:
        res = exposure_residual(p["t"], p["off"], p["A"]).ravel(); a = np.abs(p["v"].astype(np.int64)).ravel()
        top = np.zeros(a.size, np.int64); x = a.copy()
        while x.any(): top += x > 0; x >>= 1
        for r in np.unique(res):
            sel = res == r; c0 = cnt.setdefault(int(r), [0, 0, 0])
            c0[0] += int(sel.sum()); c0[1] += int(top[sel].sum()); c0[2] += int((a[sel] == 0).sum())
    N = sum(c[0] for c in cnt.values())
    tab = {r: {"share": round(c[0] / N, 4), "mean_top_plain": round(c[1] / c[0], 3), "dark_share": round(c[2] / c[0], 4)}
           for r, c in sorted(cnt.items()) if c[0] >= 1000}
    log({"what": "plates v12 A=8: exposure residual H - 2s (from the stored t, off, A) vs top place", "by_residual": tab})
    del P
    for c in (32, 34):
        Q, e, h = tables_Q(c); assert h == ROUND1_TABLE_HASH[c], h
        look(f"tables c={c}, stored Q = D >> e", (Q[r:r + CH] for r in range(0, Q.shape[0], CH)), {"hash_matches_round_one": True,
             "lines_at_cap_12": int((e == 12).sum()), "lines_at_0": int((e == 0).sum())})
        del Q
    print("secs", round(time.time() - t0, 1))
