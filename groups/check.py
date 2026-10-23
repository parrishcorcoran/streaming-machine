"""INDEPENDENT REBUILD OF THE GROUP MACHINE'S COUNTS (a separate checker: no gm.c, no code from groupmachine.py).

Everything is derived again by other means:
  - the grouping: the same rule written again (a random line order from numpy's generator with the same key, cut into
    runs of g);
  - the plate's digits: signed digits by the classic digit-by-digit rule (while n: if n is odd, the digit is 2 - (n mod
    4), take it away; halve), not by 3a; the sign and its 1s by testing each place;
  - the table: every subset's pile as one product of the light with a 0/1 table of the subsets (not Gray steps);
  - reads and combines: each group's distinct (up, down) subset pairs with how often the plate has them, times the
    signed digits of the ready pile at every position (counted by popcount(a ^ 3a), first checked equal to the digit-
    by-digit count on every number below 2^20 and on 2^20 random numbers below 2^40);
  - the table's photons: line t of a group is added 2^(g-1-t) times in the Gray order (half table: 2^(g-2-t) times over
    the first g-1 lines, then the last line once for the total);
  - the flips: every lookup of a subset holding the group's last line (not the whole group): the signed digits of its
    complement's pile;
  - the piles, at 24 positions: the reads summed with their shifts, against the plain product (int64).
Every count is compared with groupmachine.py's record (groups/out/<model>_g<g>_main.json) for the same plate:
they must be equal, number for number.
    python3 groups/check.py
"""
import json, time, hashlib, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUTD = f"{HERE}/groups/out"
SLAB = ["qkv", "proj", "fc", "mproj", "o", "gu", "down"]
def sdw_slow(a):                                       # signed digits of whole numbers, digit by digit
    n = np.abs(np.asarray(a, np.int64)).copy(); c = np.zeros(n.shape, np.int64)
    while n.any():
        odd = (n & 1) == 1; d = np.where(odd, 2 - (n & 3), 0); c += odd; n = (n - d) >> 1
    return c
def sdw(a):                                            # the same count by popcount(a ^ 3a), checked against sdw_slow first
    a = np.abs(np.asarray(a, np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
_t = np.concatenate([np.arange(1 << 20), np.random.default_rng(0).integers(0, 1 << 40, 1 << 20), -np.arange(1 << 16)])
assert np.array_equal(sdw(_t), sdw_slow(_t)), "popcount(a ^ 3a) is not the signed-digit count"
def sd_planes(w):                                      # (up, down) as 16 boolean planes, digit by digit
    n = np.asarray(w, np.int64).copy(); up = np.zeros((16,) + n.shape, bool); dn = np.zeros((16,) + n.shape, bool)
    for b in range(17):
        odd = (n & 1) == 1; d = np.where(odd, 2 - (n & 3), 0)                  # n & 3 on negatives: two's complement, still the rule
        if b < 16: up[b] = d == 1; dn[b] = d == -1
        else: assert not odd.any()
        n = (n - d) >> 1
    assert not n.any(); back = sum((up[b].astype(np.int64) - dn[b]) << b for b in range(16)); assert np.array_equal(back, w)
    return up, dn
def sm_planes(w):
    w = np.asarray(w, np.int64); a = np.abs(w)
    up = np.stack([(w > 0) & (((a >> b) & 1) == 1) for b in range(16)]); dn = np.stack([(w < 0) & (((a >> b) & 1) == 1) for b in range(16)])
    return up, dn

def check_plate(x, w, g, key, form, rec, npile=24):
    x = np.asarray(x, np.int64); w = np.asarray(w, np.int64); P, N = x.shape; M = w.shape[1]
    perm = np.random.default_rng(key).permutation(N); groups = [perm[i:i + g] for i in range(0, N, g)]
    up, dn = (sd_planes if form == "sd" else sm_planes)(w)                     # (16, N, M)
    got = dict(build_full=0, build_half=0, read=0, combine=0, flip=0, read_n=0, combine_n=0, flip_n=0)
    xs = sdw(x); sel = np.linspace(0, P - 1, npile).astype(int); pile = np.zeros((npile, M), np.int64)
    for lines in groups:
        gk = len(lines); S = np.arange(1 << gk); B = ((S[None, :] >> np.arange(gk)[:, None]) & 1).astype(np.int64)  # (gk, 2^gk)
        T = x[:, lines] @ B                                                    # (P, 2^gk): every subset's pile
        nT = sdw(T); mask = (1 << gk) - 1; top = 1 << (gk - 1)
        got["build_full"] += int((xs[:, lines] * (1 << (gk - 1 - np.arange(gk)))[None, :]).sum())
        got["build_half"] += int((xs[:, lines[:-1]] * (1 << (gk - 2 - np.arange(gk - 1)))[None, :]).sum() + xs[:, lines[-1]].sum())
        pw = (1 << np.arange(gk)).astype(np.int64)
        U = np.einsum("bgm,g->bm", up[:, lines, :].astype(np.int64), pw); D = np.einsum("bgm,g->bm", dn[:, lines, :].astype(np.int64), pw)   # (16, M)
        lit = (U | D) != 0
        pairs, cnt = np.unique((U[lit] << 16) | D[lit], return_counts=True); pu, pd = pairs >> 16, pairs & 0xFFFF
        V = T[:, pu] - T[:, pd]                                                # (P, distinct pairs): the ready piles
        both = (pu != 0) & (pd != 0)
        got["read"] += int((sdw(V) * cnt[None, :]).sum()); got["read_n"] += int(cnt.sum() * P)
        got["combine"] += int((nT[:, pd] * (cnt * both)[None, :]).sum()); got["combine_n"] += int((cnt * both).sum() * P)
        for s in (pu, pd):
            f = ((s & top) != 0) & (s != mask)
            got["flip"] += int((nT[:, s[f] ^ mask] * cnt[f][None, :]).sum()); got["flip_n"] += int(cnt[f].sum() * P)
        bsh = np.nonzero(lit)                                                  # the piles at the chosen positions
        for b in range(16):
            m = lit[b]
            if m.any(): pile[:, m] += (T[sel][:, U[b, m]] - T[sel][:, D[b, m]]) << b
    direct = x[sel] @ w
    piles_ok = bool(np.array_equal(pile, direct))
    c = rec["counts"]; same = {k: (got[k] == c[k]) for k in got}
    return got, same, piles_ok

if __name__ == "__main__":
    t0 = time.time(); LOG = open(f"{OUTD}/check.log", "w")
    def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
    log("INDEPENDENT REBUILD OF THE GROUP MACHINE'S COUNTS (groups/check.py)")
    allok = True
    # GPT-2
    M = np.load(f"{HERE}/base2_gpt2.npz")
    for g in (8, 12):
        rec = json.load(open(f"{OUTD}/gpt2_g{g}_main.json"))["plates"]
        for kind, l in (("qkv", 3), ("mproj", 7), ("proj", 11)):
            x = np.load(f"{HERE}/groups/light_gpt2/{kind}.npy", mmap_mode="r")[l]; w = M[f"{kind}.{l}.w"]
            for form in ("sd", "sm"):
                got, same, pok = check_plate(x, w, g, [1, SLAB.index(kind), l, g], form, rec[f"{kind}.{l}"][form])
                ok = all(same.values()) and pok; allok &= ok
                log(f"  gpt2 g={g:2d} {kind}.{l} {form}: every count equal {all(same.values())} {'' if all(same.values()) else same}; "
                    f"piles at 24 positions = the plain product {pok}  (read {got['read']:,}, combine {got['combine']:,}, table {got['build_full']:,})")
    del M
    # Qwen
    Mz = np.load(f"{HERE}/qwen/qwen_base2.npz")
    QK = {"q": ("qkv", 0, 2048), "k": ("qkv", 2048, 3072), "o": ("o", 0, 1024), "gate": ("gu", 0, 3072), "down": ("down", 0, 1024)}
    for g in (8, 12):
        rec = json.load(open(f"{OUTD}/qwen_g{g}_main.json"))["plates"]
        for kind, l in (("q", 5), ("k", 20), ("o", 13), ("gate", 14), ("down", 27)):
            slab, a, b = QK[kind]
            x = np.load(f"{HERE}/qwen06b/light/{slab}_light.npy", mmap_mode="r")[l]; w = np.asarray(Mz[f"{slab}.{l}.w"])[:, a:b]
            for form in ("sd",) if kind != "gate" else ("sd", "sm"):
                got, same, pok = check_plate(x, w, g, [1, SLAB.index(slab), l, g], form, rec[f"{kind}.{l}"][form])
                ok = all(same.values()) and pok; allok &= ok
                log(f"  qwen g={g:2d} {kind}.{l} {form}: every count equal {all(same.values())} {'' if all(same.values()) else same}; "
                    f"piles at 24 positions = the plain product {pok}  (read {got['read']:,}, combine {got['combine']:,}, table {got['build_full']:,})")
    log(f"ALL EQUAL: {allok}  ({time.time() - t0:.0f}s)")
