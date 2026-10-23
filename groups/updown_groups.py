"""THE GROUP MACHINE WITH THE PLATE IN ITS UP/HOLE FORM: THE FLIP WHERE IT COSTS NOTHING (an addition to groupmachine.py,
run after it; the session's own step, labelled as such).

WHY (the session's reasoning, from groupmachine.py's first counts): in the subset forms (SD, SM) the flip of step 3 --
a subset's pile is the group total minus its complement's pile -- costs a take-away of a whole pile at every read
of the unbuilt half, and that is far more than the half table saves (GPT-2 g = 8: flips 8.2% of plain against a
saving of 0.16%). In the up/hole form (updown_machine.py: every place of every weight a photon up or a hole, D = 2w + 1),
the complement of a pattern (ups and holes exchanged) is exactly the opposite pile, V[p ^ mask] = -V[p]: the flip is a
take-away of the same ready pile, no extra work. No place of a group is ever empty and there is nothing to combine; each
(neuron, place, group) is one read of a pile holding all g lines' light. Once a position, X (the sum of the light) is
taken from each neuron's pile and the pile halved (a shift): x . w = (x . D - X) / 2.
The groups are groupmachine.py's own (seed 1, the same keys): the same random groups, the same light, the same plates.
EXACT: every pile against base2's own run (GPT-2, plain_refs.npz) and capture's digests (Qwen); the halving checked even.

PREDICTION (the session's, written before this file ran)
  16 reads a (neuron, group), each a signed pile of all g lines' light. From groupmachine.py's light curve, a pile of 8
  GPT-2 light values has about 5 signed digits, of 12 about 5.5. So GPT-2 at g = 8: 16 x 5 / 8 = 10 a pair, about 31% of
  plain (SD 35.4%); at g = 12: 16 x 5.5 / 12 = 7.3 a pair, about 23-25% of plain with the half table (SD 31.0%). The
  table (half, by the flip) at most half of SD's table; X and its take-away under 0.1%. Qwen at g = 12 about 22-25%.
  Adds: 16 a (neuron, group): at g = 12, 1.33 a pair against the direct machine's 4.5-4.7 (signed digits): about 30%.
    python3 groups/updown_groups.py model=gpt2|qwen g=8|12 [layers=a-b]   -> groups/out/uh_<model>_g<g>.json, .log
"""
import os, sys, json, time, ctypes, subprocess, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; GD = f"{HERE}/groups"; OUTD = f"{GD}/out"
ARG = dict(a.split("=") for a in sys.argv[1:]); MODEL, G_ = ARG.get("model", "gpt2"), int(ARG.get("g", 8))
_argv = sys.argv; sys.argv = [sys.argv[0]]; sys.path.insert(0, GD); import groupmachine as GM; sys.argv = _argv
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
_lib = None
def lib():
    global _lib
    if _lib is None:
        src, so = f"{GD}/gm_uh.c", f"{GD}/gm_uh.dylib"
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
            subprocess.check_call(["cc", "-O3", "-shared", "-o", so, src])
        _lib = ctypes.CDLL(so); P_, I_ = ctypes.c_void_p, ctypes.c_int64
        _lib.group_machine_uh.argtypes = [I_, I_, I_, I_, I_, P_, P_, P_, P_, P_, P_]; _lib.group_machine_uh.restype = None
        _lib.ncu.restype = ctypes.c_int
    return _lib
CNT = ["build_full", "build_full_n", "build_half", "build_half_n", "read", "read_n", "xsum", "xsum_n", "offset", "offset_n", "odd"]

def patterns(w, glines, goff):
    """(G, M, 16) up/hole patterns: bit t = 1 where line t's weight + 2^15 has a 1 at the place"""
    w = np.asarray(w, np.int64); assert w.min() >= -(1 << 15) and w.max() < (1 << 15)
    c = w + (1 << 15); N, M = w.shape; G = len(goff) - 1; PT = np.zeros((G, M, 16), np.uint16); sizes = np.diff(goff)
    D = sum(((2 * ((c >> b) & 1) - 1) << b) for b in range(16)); assert np.array_equal(D, 2 * w + 1)   # the up/hole form
    for gk in np.unique(sizes):
        ks = np.flatnonzero(sizes == gk); lines = np.stack([glines[goff[k]:goff[k + 1]] for k in ks])
        for t in range(gk):
            ct = c[lines[:, t]]
            for b in range(16): PT[ks, :, b] |= (((ct >> b) & 1).astype(np.uint16) << t)
    return PT
def run_uh(x, w, glines, goff):
    x = np.ascontiguousarray(x, np.int32); P, N = x.shape; M = w.shape[1]; G = len(goff) - 1; gmax = int(np.diff(goff).max())
    PT = np.ascontiguousarray(patterns(w, glines, goff)); pile = np.empty((P, M), np.int64); cnt = np.empty((P, lib().ncu()), np.int64)
    lib().group_machine_uh(P, N, M, G, gmax, x.ctypes.data, glines.ctypes.data, goff.ctypes.data, PT.ctypes.data, pile.ctypes.data, cnt.ctypes.data)
    return pile, cnt

if __name__ == "__main__":
    LAY = ARG.get("layers"); L0, L1 = (int(v) for v in LAY.split("-")) if LAY else (0, 99)
    t0 = time.time(); tag = f"uh_{MODEL}_g{G_}" + (f"_L{L0}-{L1}" if LAY else ""); LOG = open(f"{OUTD}/{tag}.log", "w")
    def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
    log(f"THE GROUP MACHINE, PLATE IN UP/HOLE FORM: {MODEL}, g = {G_}, groupmachine.py's groups (seed 1)")
    res = {}
    for l, slab, kinds, x, ws, ref in (GM.gpt2_plates() if MODEL == "gpt2" else GM.qwen_plates()):
        if not (L0 <= l <= L1): continue
        tl = time.time(); N = x.shape[1]; key = [1, ["qkv", "proj", "fc", "mproj", "o", "gu", "down"].index(slab), l, G_]
        glines, goff = GM.grouping(N, G_, key); piles = []
        for ki, (kind, w) in enumerate(zip(kinds, ws)):
            pile, cnt = run_uh(x, w, glines, goff); tot = cnt.sum(0)
            r = dict(base=GM.baselines(x, w), counts={k: int(tot[i]) for i, k in enumerate(CNT)})
            assert r["counts"]["odd"] == 0, "the halving met an odd number"
            if MODEL == "gpt2":
                r["piles_differ"] = int((pile != ref[ki]).any(1).sum()); assert r["piles_differ"] == 0, (kind, l)
            piles.append(pile); res[f"{kind}.{l}"] = r
        if MODEL == "qwen":
            rows, dig = ref; sp = np.concatenate(piles, 1)
            bad = sum(GM.digest(sp[a:b]) != dig[t] for t, (a, b) in enumerate(rows)); assert bad == 0, (slab, l, bad)
            for kind in kinds: res[f"{kind}.{l}"]["piles_differ"] = int(bad)
        del piles
        msg = " ".join(f"{k}: {100 * (res[f'{k}.{l}']['counts']['read']) / res[f'{k}.{l}']['base']['plain']:.1f}" for k in kinds)
        log(f"  layer {l:2d} {slab:5s} exact; reads, % of plain: {msg}  ({time.time() - tl:.0f}s, {rss()} MB)")
    json.dump(dict(model=MODEL, g=G_, plates=res), open(f"{OUTD}/{tag}.json", "w"))
    log(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
