"""THE UP/HOLE FORM REBUILT FROM SCRATCH (groups2/, the session's own rebuild; shares no code with groups/updown_groups.py,
groups/gm_uh.c, groups/groupmachine.py or groups/check.py; reads only the plates, the light and the reference piles).

HIS PICTURE, AS THE SESSION RESTATES IT (the session's words, not his)
  A token's light reaches a plate. Take a RANDOM group of g light lines. Every place of every weight is a photon up or
  a hole (the weight w written as w + 2^15 in 16 places: a 1 is up, a 0 is a hole). At one place, one neuron's plate
  lays an up/hole pattern over the group's g lines, and the pile of that pattern is the light on the up lines minus the
  light on the holes. By the flip (Babinet), exchanging ups and holes gives exactly the opposite pile, so only half the
  patterns are built and the other half is read as a take-away of the same pile. The piles are built once per token;
  each neuron, at each place, adds ONE ready pile shifted to that place. The pile of the token then comes out as
  (sum of the shifted ready piles - X) / 2, X the light summed once a position (the halving is a shift).

WHAT IS REBUILT, AND HOW IT DIFFERS FROM THE BUILDER'S CODE
  - the grouping: the same rule (numpy's generator with key [seed, slab, layer, g], a random line order cut into runs
    of g, the last run short), written again;
  - the up/hole digits: a digit-by-digit expansion of w + 2^15 (binary at every place), checked to give w back;
  - the ready piles: made by the lowest-bit rule (V[p] = V[p less its lowest line] + 2 x), not by Gray steps;
  - the read photons: counted twice, (a) each group's patterns with how often the plate reads them, times the signed
    digits of the pile counted digit by digit, and (b) read by read by popcount(a ^ 3a); the two must agree;
  - the table's photons by a relation, not by a walk: in the half table (the patterns with the group's last line up),
    the group total costs the other g - 1 lines once each, and in the Gray walk over the first g - 1 lines line t turns
    2^(g-2-t) times; the full table (from all holes) turns line t 2^(g-1-t) times, plus the g - 1 adds of the total;
  - X: N - 1 adds of light values a position (the first line is the start); its take-away: X's signed digits, once
    per neuron;
  - piles: GPT-2 against base2's own run (recording/C_ref/plain_refs.npz); Qwen3-0.6B against qwen_base2's own piles'
    sha256, text by text (qwen06b/light/meta.json); the shuffled light against the plain product of the shuffled
    light (groups2/uhm.c, lines met one by one).
  Every count is then set beside the builder's record (groups/out/uh_<model>_g<g>.json), plate by plate, by
  groups2/uh_compare.py.

RUNS (all in one pass over the plates, plate by plate)
  main     seed 1, the builder's groups (to compare number for number)
  seed2    a second random grouping (seed 2)
  shuffle  each position's light shuffled across its lines (same values, the pairing with the plate broken), seed-1
           groups
  order    order-free: the seed-1 groups taken in a random order, each group's lines in a random order (this file's
           own random order, not the builder's)

PREDICTION (the session's, written before this file ran)
  - main: every count equal, number for number, to the builder's record at every plate (reads, read count, half and
    full table, X, offsets, odd = 0); every pile exact; the totals reproduce 28.49% / 21.04% (GPT-2, g = 8 / 12) and
    33.42% / 24.01% (Qwen3-0.6B) of plain photon meetings to 0.01 point.
  - seed2: within 0.02 point of main in every model total (the read pile's digits are set by the light's sizes and
    the plate's up/hole balance, not by which lines share a group).
  - shuffle: within 0.05 point of main (the piles' digits do not depend on which line meets which row of the plate).
  - order: piles identical to main, read photons and read count identical exactly (a pattern is a set of up lines);
    the half table changes by less than 0.01 point of plain (the turning line and the last line change); X identical.
  - Qwen o and down stay above their signed-digit group machine at g = 8 (the up/hole form reads all 16 places while
    their low places are mostly empty in signed digits).
RULE 13, CHECKED BEFORE THE RUN
  13.1 no count of the light's 1s is the measure; the measure is the work of the finished piles, every pile exact.
  13.2 the base is the machine's own plates and light, every part as it is.
  13.3 the shares are counts of work, and the shuffle is the test of whether the pairing of light and plate matters.
  13.4 the up/hole read count (16 a neuron a group) is fixed by the definition; the run measures the read photons, the
       table and the exactness.
  13.5 the picture above is the session's restatement, labelled so.
  13.6 every GPT-2 bank, every Qwen3-0.6B plate (196) at both g, every position; nothing cut short.

    python3 groups2/uh_rebuild.py model=gpt2|qwen g=8|12 [layers=a-b] [runs=main,seed2,shuffle,order]
    -> groups2/out/uhr_<model>_g<g>[_L<a>-<b>].json, .log
"""
import os, sys, json, time, ctypes, subprocess, hashlib, resource, numpy as np

HERE = "/Users/abundancemachine/streaming-machine"; G2 = f"{HERE}/groups2"; OUT = f"{G2}/out"
SLABS = ["qkv", "proj", "fc", "mproj", "o", "gu", "down"]                     # the slab index in the grouping key
QSEG = {"qkv": [("q", 0, 2048), ("k", 2048, 3072), ("v", 3072, 4096)], "o": [("o", 0, 1024)],
        "gu": [("gate", 0, 3072), ("up", 3072, 6144)], "down": [("down", 0, 1024)]}

def rss_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6

# ---------- whole-number helpers (written here) ----------
def sd_digits(a):
    """signed-digit count of each whole number, digit by digit (odd: digit 2 - (a mod 4), take it away; halve)"""
    a = np.abs(np.asarray(a, np.int64)).copy(); c = np.zeros(a.shape, np.int64)
    while True:
        nz = a != 0
        if not nz.any(): return c
        odd = (a & 1) == 1; c += odd
        a = np.where(odd, np.where((a & 2) != 0, a + 1, a - 1), a) >> 1
def ones_count(a):
    a = np.abs(np.asarray(a, np.int64)); c = np.zeros(a.shape, np.int64)
    for b in range(40): c += (a >> b) & 1
    return c
_chk = np.concatenate([np.arange(-70000, 70000), np.random.default_rng(5).integers(-(1 << 38), 1 << 38, 200000)])
_naf_ref = np.array([bin((abs(int(v)) * 3) ^ abs(int(v))).count("1") for v in _chk[::97]])
assert np.array_equal(sd_digits(_chk[::97]), _naf_ref), "digit-by-digit count disagrees with the 3a rule"

# ---------- the kernel ----------
_LIB = None
def lib():
    global _LIB
    if _LIB is None:
        src, so = f"{G2}/uhm.c", f"{G2}/uhm.dylib"
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
            subprocess.check_call(["cc", "-O3", "-shared", "-o", so, src])
        _LIB = ctypes.CDLL(so); V, I = ctypes.c_void_p, ctypes.c_int64
        _LIB.uh_machine.argtypes = [I, I, I, I, I, I] + [V] * 14; _LIB.uh_machine.restype = None
        _LIB.plain_product.argtypes = [I, I, I, V, V, V]; _LIB.plain_product.restype = None
        _LIB.ncnt.restype = ctypes.c_int
    return _LIB

def plain_product(x, w):
    x = np.ascontiguousarray(x, np.int32); w = np.ascontiguousarray(w, np.int16); P, N = x.shape; M = w.shape[1]
    out = np.empty((P, M), np.int64); lib().plain_product(P, N, M, x.ctypes.data, w.ctypes.data, out.ctypes.data); return out

# ---------- grouping ----------
def make_groups(N, g, key, reorder_key=None):
    """random line order from numpy's generator with the key, cut into runs of g (the last may be short);
    reorder_key: the same groups in a random order, each group's lines in a random order"""
    order = np.random.default_rng(key).permutation(N)
    runs = [order[s:s + g] for s in range(0, N, g)]
    if reorder_key is not None:
        r = np.random.default_rng(reorder_key)
        runs = [runs[i] for i in r.permutation(len(runs))]; runs = [r.permutation(u) for u in runs]
    sizes = np.array([len(u) for u in runs], np.int64)
    return np.concatenate(runs).astype(np.int32), np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)

# ---------- the plate's digits ----------
def expand(w, up):
    """each weight's digits, place by place, bottom up, with the offset C = 2^(top up/hole place) per neuron:
    at an up/hole place (bit b of up[j]) a binary digit of the running value; at every other place (and places 16, 17
    for what is left) a signed digit by the rule digit = 2 - (r mod 4) where r is odd. Returns E (the binary digits at
    up/hole places, as 16-bit words), the signed digits (line, neuron, place, sign) and kcorr = K - 2C per neuron,
    K = up[j] (the sum of 2^b over its up/hole places): 2 x.w = sum_b 2^b V + 2 sum signed + kcorr X."""
    w = np.asarray(w, np.int64); N, M = w.shape; up = np.asarray(up, np.int64)
    top = np.array([int(u).bit_length() - 1 for u in up]); C = np.where(up != 0, np.left_shift(1, np.maximum(top, 0)), 0).astype(np.int64)
    r = w + C[None, :]; E = np.zeros((N, M), np.int64); sd_parts = []
    for b in range(18):
        isup = ((up >> b) & 1).astype(bool)[None, :] if b < 16 else np.zeros((1, M), bool)
        odd = (r & 1) == 1
        d = np.where(odd, 2 - (r & 3), 0)
        d = np.where(isup, 0, d); e = np.where(isup & odd, 1, 0)
        if b < 16: E |= e << b
        nz = np.nonzero(d)
        if len(nz[0]): sd_parts.append((nz[0], nz[1], np.full(len(nz[0]), b), d[nz]))
        r = (r - e - d) >> 1
    assert not r.any(), "digits left over above place 17"
    back = E.copy() - C[None, :]
    if sd_parts:
        t, j, b, s = (np.concatenate(z) for z in zip(*sd_parts)); np.add.at(back, (t, j), s << b)
    else:
        t = j = b = s = np.zeros(0, np.int64)
    assert np.array_equal(back, w), "the digits do not give the plate back"
    kcorr = up - 2 * C
    return E.astype(np.uint16), (t, j, b, s), kcorr

def patterns(E, glines, goff):
    """(G, M, 16): bit t of pattern = the binary digit at that place of the group's line t"""
    N, M = E.shape; sizes = np.diff(goff); G = len(sizes); pat = np.zeros((G, M, 16), np.uint16)
    for gk in np.unique(sizes):
        ks = np.flatnonzero(sizes == gk); idx = np.stack([glines[goff[k]:goff[k] + gk] for k in ks])   # (n, gk)
        Eg = E[idx]                                                                                    # (n, gk, M)
        wts = (np.uint16(1) << np.arange(gk, dtype=np.uint16))[None, :, None]
        for b in range(16):
            pat[ks, :, b] = (((Eg >> b) & 1).astype(np.uint16) * wts).sum(1, dtype=np.uint16)
    return pat

def read_hist(pat, up, seg, nseg, gmax):
    G, M, _ = pat.shape; T = 1 << gmax
    use = (((up[:, None] >> np.arange(16)[None, :]) & 1) == 1)                     # (M, 16)
    flat = ((seg.astype(np.int64)[None, :, None] * G + np.arange(G)[:, None, None]) * T + pat.astype(np.int64))
    flat = flat[np.broadcast_to(use[None], pat.shape)]
    return np.bincount(flat, minlength=nseg * G * T).astype(np.int64)

# ---------- the machine on one plate (a slab: one light, one table, one or more plates as segments) ----------
def prepare(w, up, segs):
    """the plate's side, fixed before any light: digits, signed-digit lists, per-line counts of 1s and signed digits"""
    w = np.asarray(w); M = w.shape[1]; up = np.asarray(up, np.int64)
    seg = np.zeros(M, np.int8)
    for si, (_, a, b) in enumerate(segs): seg[a:b] = si
    E, (tt, tj, tb, ts), kcorr = expand(w, up)
    o = np.argsort(tj, kind="stable"); tt, tj, tb, ts = tt[o], tj[o], tb[o], ts[o]
    toff = np.concatenate([[0], np.cumsum(np.bincount(tj, minlength=M))]).astype(np.int64)
    per = []
    for si, (name, a, b) in enumerate(segs):
        ws = np.asarray(w[:, a:b], np.int64); m = (tj >= a) & (tj < b)
        per.append(dict(name=name, a=a, b=b, w1=ones_count(ws).sum(1), wsd=sd_digits(ws).sum(1),
                        lit_per_line=np.bincount(tt[m], minlength=w.shape[0]), lit_by_place=np.bincount(tb[m], minlength=18).tolist()))
    return dict(w=w, up=up, seg=seg, E=E, toff=toff, tline=np.ascontiguousarray(tt, np.int32), tplace=np.ascontiguousarray(tb, np.int8),
                tsign=np.ascontiguousarray(ts, np.int8), kcorr=np.ascontiguousarray(kcorr, np.int64), per=per, segs=segs)

def run_plate(x, prep, glines, goff):
    """x (P, N) light on a prepared plate. Returns piles (P, M), a record per segment, and the slab's table counts."""
    x = np.ascontiguousarray(x, np.int32); P, N = x.shape; w = prep["w"]; M = w.shape[1]; up = prep["up"]; segs = prep["segs"]
    sizes = np.diff(goff); G = len(sizes); gmax = int(sizes.max()); seg = prep["seg"]
    pat = np.ascontiguousarray(patterns(prep["E"], glines, goff))
    hist = read_hist(pat, up, seg, len(segs), gmax)
    up32 = np.ascontiguousarray(up, np.uint32); kc = prep["kcorr"]
    pile = np.empty((P, M), np.int64); ncnt = lib().ncnt(); cnt = np.empty((P, len(segs), ncnt), np.int64)
    lib().uh_machine(P, N, M, G, gmax, len(segs), x.ctypes.data, glines.ctypes.data, goff.ctypes.data, up32.ctypes.data,
                     pat.ctypes.data, hist.ctypes.data, seg.ctypes.data, prep["toff"].ctypes.data, prep["tline"].ctypes.data,
                     prep["tplace"].ctypes.data, prep["tsign"].ctypes.data, kc.ctypes.data, pile.ctypes.data, cnt.ctypes.data)
    tot = cnt.sum(0); del pat, hist
    # ----- the work counted by relations, from the light -----
    xs = sd_digits(x)                                                              # (P, N)
    any_up = bool(up.any()); half = full = half_n = full_n = 0
    if any_up:
        sxg = xs.sum(0)                                                            # per line, over positions
        for k in range(G):
            L = glines[goff[k]:goff[k + 1]]; gk = len(L); sx = [int(v) for v in sxg[L]]
            tot_part = sum(sx[1:])                                                 # the group total: g - 1 adds
            half += tot_part + sum((1 << (gk - 2 - t)) * sx[t] for t in range(gk - 1))
            full += tot_part + sum((1 << (gk - 1 - t)) * sx[t] for t in range(gk))
            half_n += P * ((gk - 1) + (1 << (gk - 1)) - 1); full_n += P * ((gk - 1) + (1 << gk) - 1)
    X = x.astype(np.int64).sum(1)
    xsum = int(xs[:, 1:].sum()) if any_up else 0; xsum_n = P * (N - 1) if any_up else 0
    ox = ones_count(x).sum(0); sxl = xs.sum(0); recs = {}
    for si, sp in enumerate(prep["per"]):
        a, b = sp["a"], sp["b"]; kseg = kc[a:b]
        plain = int(ox @ sp["w1"]); signed = int(sxl @ sp["wsd"])
        off = off_n = mk = mk_n = 0                                                # X's correction, per neuron
        for kv in np.unique(kseg):
            if kv == 0: continue
            nj = int((kseg == kv).sum()); off += nj * int(sd_digits(int(kv) * X).sum()); off_n += nj * P
            dk = int(sd_digits(np.array([kv]))[0])
            if dk > 1: mk += (dk - 1) * int(sd_digits(X).sum()); mk_n += (dk - 1) * P
        sdread = int(sxl @ sp["lit_per_line"]); sdread_n = int(sp["lit_per_line"].sum()) * P
        recs[sp["name"]] = dict(plain=plain, signed=signed, read=int(tot[si, 0]), read_pop=int(tot[si, 1]), read_n=int(tot[si, 2]),
                                odd=int(tot[si, 3]), offset=off, offset_n=off_n, make=mk, make_n=mk_n, sdread=sdread, sdread_n=sdread_n,
                                sd_lit_by_place=sp["lit_by_place"], up_places=int(np.bitwise_or.reduce(up[a:b])),
                                up_same=bool((up[a:b] == up[a]).all()))
        assert recs[sp["name"]]["read"] == recs[sp["name"]]["read_pop"], "the two read counts disagree"
    slab = dict(half=half, half_n=half_n, full=full, full_n=full_n, xsum=xsum, xsum_n=xsum_n)
    return pile, recs, slab

# ---------- the bases ----------
def gpt2_plates(L0, L1):
    Wz = np.load(f"{HERE}/base2_gpt2.npz"); R = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
    light = {k: np.load(f"{HERE}/groups/light_gpt2/{k}.npy", mmap_mode="r") for k in ("qkv", "proj", "fc", "mproj")}
    for l in range(L0, min(L1, 11) + 1):
        for kind in ("qkv", "proj", "fc", "mproj"):
            w = np.asarray(Wz[f"{kind}.{l}.w"]); ref = np.asarray(R[f"{kind}_pile_{l}"])
            yield l, kind, np.asarray(light[kind][l]), w, [(kind, 0, w.shape[1])], ("piles", ref)
def qwen_plates(L0, L1):
    Wz = np.load(f"{HERE}/qwen/qwen_base2.npz"); meta = json.load(open(f"{HERE}/qwen06b/light/meta.json"))
    light = {s: np.load(f"{HERE}/qwen06b/light/{s}_light.npy", mmap_mode="r") for s in QSEG}
    for l in range(L0, min(L1, 27) + 1):
        for slab in QSEG:
            w = np.asarray(Wz[f"{slab}.{l}.w"])
            yield l, slab, np.asarray(light[slab][l]), w, QSEG[slab], ("digests", (meta["rows"], meta["dig"][f"{slab}.{l}"]))
def piles_differ(pile, ref):
    kind, r = ref
    if kind == "piles": return int((pile != r).any(1).sum())
    rows, dig = r
    return sum(hashlib.sha256(np.ascontiguousarray(pile[a:b], np.int64).tobytes()).hexdigest() != dig[t] for t, (a, b) in enumerate(rows))

if __name__ == "__main__":
    A = dict(s.split("=") for s in sys.argv[1:]); MODEL = A.get("model", "gpt2"); g = int(A.get("g", 8))
    RUNS = A.get("runs", "main,seed2,shuffle,order").split(","); lay = A.get("layers")
    L0, L1 = (int(v) for v in lay.split("-")) if lay else (0, 99)
    tag = f"uhr_{MODEL}_g{g}" + (f"_L{L0}-{L1}" if lay else ""); os.makedirs(OUT, exist_ok=True)
    LOG = open(f"{OUT}/{tag}.log", "w"); t0 = time.time()
    def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
    log(f"UP/HOLE FORM REBUILT (groups2): {MODEL}, g = {g}, runs {RUNS}, layers {L0}..{L1}")
    res = {r: {} for r in RUNS}
    for l, slab, x, w, segs, ref in (gpt2_plates(L0, L1) if MODEL == "gpt2" else qwen_plates(L0, L1)):
        tl = time.time(); N, M = w.shape; key = [1, SLABS.index(slab), l, g]; main_pile = None
        prep = prepare(w, np.full(M, 0xFFFF, np.int64), segs); msg = []
        for run in RUNS:
            k = list(key); xr = x; reo = None
            if run == "seed2": k[0] = 2
            if run == "order": reo = [4242] + key
            if run == "shuffle": xr = np.random.default_rng([31337] + key).permuted(x, axis=1)
            glines, goff = make_groups(N, g, k, reo)
            pile, recs, sl = run_plate(xr, prep, glines, goff)
            if run == "shuffle": bad = int((pile != plain_product(xr, w)).any(1).sum())
            else: bad = piles_differ(pile, ref)
            assert bad == 0, (run, slab, l, bad)
            if run == "main": main_pile = pile
            if run == "order": assert np.array_equal(pile, main_pile), "order-free piles differ from main"
            for name, r in recs.items():
                assert r["odd"] == 0; r["piles_differ"] = bad; res[run][f"{name}.{l}"] = r
            res[run][f"{slab}.{l}.slab"] = sl
            msg.append(f"{run}: " + " ".join(f"{n} {100 * (recs[n]['read']) / recs[n]['plain']:.2f}" for n in recs))
            del pile
        log(f"  layer {l:2d} {slab:5s} exact (every run); reads % of plain: " + " | ".join(msg) + f"  ({time.time() - tl:.0f}s, {rss_mb()} MB)")
        json.dump(dict(model=MODEL, g=g, runs=res), open(f"{OUT}/{tag}.json", "w"))
    log(f"done ({time.time() - t0:.0f}s, max memory {rss_mb()} MB)")
