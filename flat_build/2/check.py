"""flat_build/2/check.py -- A SECOND PATH THROUGH PICK 2 (numpy only; imports nothing from machine.py, runm.c,
groups2/gbank.py, uh_rebuild or uh_perplace). The same session wrote it, so it is NOT a separate checker under rule 10.

What it builds again, its own way:
  groups   numpy's generator keyed [1, slab, layer, g], a random line order cut into runs of g (written again here)
  cells    c = the smallest c with the cell's smallest weight >= -2^c and its largest < 2^c (from the min and max,
           not from each weight's bit length)
  sheets   bit b of w + 32768 for the group's line t is bit t of the sheet's pattern
  table    the FULL table of every group (all 2^gk patterns) by doubling: start from the empty group, and for each
           line t make the patterns with that line a hole (-x_t) and up (+x_t)
  reads    places b < c their own sheet; place c the sign sheet (place 15); the pile's photons its signed digits
  entries  the distinct packets read up to the flip (a pattern and its complement are one entry), written once: 1s
  walk     the half table's Gray walk enumerated step by step (the line that turns at each step counted), plus the
           group total (lines 1..gk-1 once)
  piles    (sum of the read packets at their places - X) / 2, against recording/C_ref/plain_refs.npz
Then every counted item is set beside machine.py's run (out/machine_g<g>.json, its C machine), bank by bank.

    python3 flat_build/2/check.py g=12|R [layers=a-b]   -> flat_build/2/out/check_g<g>.json, .log
"""
import os, sys, json, time, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/flat_build/2/out"
SLAB_INDEX = {"qkv": 0, "proj": 1, "fc": 2, "mproj": 3}
G_R = {"qkv": 12, "proj": 11, "fc": 12, "mproj": 11}

def popc(a): return np.bitwise_count(np.abs(np.asarray(a, np.int64))).astype(np.int64)
def sdig(a):
    a = np.abs(np.asarray(a, np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
# checked here against the digit-by-digit rule (odd: digit 2 - (v mod 4), take it away; halve)
def _sd_slow(v):
    v = abs(int(v)); n = 0
    while v:
        if v & 1: n += 1; v = v + 1 if v & 2 else v - 1
        v >>= 1
    return n
_t = np.random.default_rng(9).integers(-(1 << 30), 1 << 30, 3000)
assert all(int(s) == _sd_slow(v) for s, v in zip(sdig(_t), _t)), "signed-digit count"

def groups_of(N, g, key):
    order = np.random.default_rng(key).permutation(N)
    return [order[s:s + g] for s in range(0, N, g)]

def cell_c(wg):
    lo, hi = wg.min(0), wg.max(0); c = np.zeros(wg.shape[1], np.int64)
    for cand in range(15, -1, -1):
        ok = (lo >= -(1 << cand)) & (hi < (1 << cand)); c = np.where(ok, cand, c)
    return c

def full_table(xg):
    """xg (P, gk): V (P, 2^gk), V[:, p] = sum_t x_t (2 bit_t(p) - 1)"""
    V = np.zeros((xg.shape[0], 1), np.int64)
    for t in range(xg.shape[1]):
        xt = xg[:, t:t + 1]; V = np.concatenate([V - xt, V + xt], axis=1)   # new bit t: 0 = hole, 1 = up
    return V

def walk_turns(gk):
    turns = np.zeros(gk, np.int64); turns[1:] += 1                            # the group total: lines 1..gk-1 once
    for i in range(1, 1 << (gk - 1)):
        t = (i & -i).bit_length() - 1; turns[t] += 1
    return turns

def check_bank(w, x, g, key, ref, chunk=81):
    w = np.asarray(w, np.int64); x = np.asarray(x, np.int64); P, N = x.shape; M = w.shape[1]
    U = w + 32768; runs = groups_of(N, g, key)
    D = np.zeros((P, M), np.int64); reads = np.zeros(P, np.int64); pwrite = np.zeros(P, np.int64); walk = np.zeros(P, np.int64)
    nreads = 0; sdx = sdig(x)
    for L in runs:
        gk = len(L); c = cell_c(w[L]); nreads += int((c + 1).sum())
        bits = (U[L][:, :, None] >> np.arange(16)[None, None, :]) & 1                 # (gk, M, 16)
        pat = (bits << np.arange(gk)[:, None, None]).sum(0)                          # (M, 16)
        rd = np.where(np.arange(16)[None, :] < c[:, None], pat, 0)
        rd[np.arange(M), c] = pat[:, 15]                                             # the sign sheet at place c
        use = np.arange(16)[None, :] <= c[:, None]                                   # (M, 16)
        full = (1 << gk) - 1
        canon = np.unique(np.minimum(rd[use], rd[use] ^ full))                       # one entry per pattern and its flip
        walk += sdx[:, L] @ walk_turns(gk)
        for p0 in range(0, P, chunk):
            V = full_table(x[p0:p0 + chunk][:, L])                                   # (Pc, 2^gk)
            pwrite[p0:p0 + chunk] += popc(V[:, canon]).sum(1)
            Vr = V[:, rd]                                                            # (Pc, M, 16)
            Vr = np.where(use[None], Vr, 0)
            reads[p0:p0 + chunk] += sdig(Vr).sum((1, 2))
            D[p0:p0 + chunk] += (Vr << np.arange(16)[None, None, :]).sum(2)
            del V, Vr
    X = x.sum(1); D -= X[:, None]
    assert not (D & 1).any()
    pile = D >> 1; bad = int((pile != ref).any(1).sum())
    cnt = dict(light=int(popc(x).sum()), table=int(walk.sum()), pwrite=int(pwrite.sum()), reads=int(reads.sum()),
               sdreads=0, xsum=int(sdx[:, 1:].sum()), xoff=int(M * sdig(X).sum()),
               plain=int((popc(x) @ popc(w).sum(1)).sum()), nreads=nreads * P)
    return bad, cnt

if __name__ == "__main__":
    A_ = dict(s.split("=") for s in sys.argv[1:]); GM = A_.get("g", "12"); lay = A_.get("layers")
    L0, L1 = (int(v) for v in lay.split("-")) if lay else (0, 11)
    tag = f"check_g{GM}" + (f"_L{L0}-{L1}" if lay else "")
    LOG = open(f"{OUT}/{tag}.log", "w"); t0 = time.time()
    def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
    MJ = json.load(open(f"{OUT}/machine_g{GM}.json"))["banks"]
    Wz = np.load(f"{HERE}/base2_gpt2.npz"); REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
    light = {k: np.load(f"{HERE}/groups/light_gpt2/{k}.npy", mmap_mode="r") for k in SLAB_INDEX}
    log(f"SECOND PATH THROUGH PICK 2 (flat_build/2/check.py): g mode {GM}, layers {L0}..{L1}")
    out = {}; allsame = True
    for l in range(L0, L1 + 1):
        for kind in SLAB_INDEX:
            tl = time.time(); g = 12 if GM == "12" else G_R[kind]
            w = np.asarray(Wz[f"{kind}.{l}.w"]); x = np.asarray(light[kind][l]); ref = np.asarray(REF[f"{kind}_pile_{l}"])
            bad, cnt = check_bank(w, x, g, [1, SLAB_INDEX[kind], l, g], ref)
            mc = MJ[f"{kind}.{l}"]["C"]
            diff = {k: (v, mc[k]) for k, v in cnt.items() if v != mc[k]}
            same = bad == 0 and not diff; allsame &= same
            out[f"{kind}.{l}"] = dict(piles_differ=bad, counts=cnt, differ=diff)
            log(f"  {kind:5s}.{l:<2d} g={g}: piles differing {bad}; items differing from machine.py: "
                f"{'none' if not diff else diff}  ({time.time() - tl:.0f}s, {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6} MB)")
            json.dump(out, open(f"{OUT}/{tag}.json", "w"))
    log(f"ALL BANKS: {'every pile exact and every count the same' if allsame else 'DIFFERENCES (above)'}  ({time.time() - t0:.0f}s)")
