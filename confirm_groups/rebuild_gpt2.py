"""confirm_groups/rebuild_gpt2.py -- INDEPENDENT REBUILD OF base2_groups.py (GPT-2), by a separate checker (rule 10).

No import of, or copying from, base2_groups.py, qwen_groups.py, groups/* or groups2/*. Their docstrings were read for the
rules only. The reference is base2.py itself (its own steps, exec'd by judge5.py as judge5 does), judge5.judge and
judge5_ref.npz, refs_many.json and recording/C_ref/plain_refs.npz. The group plate and the group top read are this
checker's own (cg.py, gk.c). Whole numbers only.

THE MACHINE REBUILT (as the delivered docstring states it, every rule fixed before the run)
  - every bank product made by group piles, g = 12, the seed-1 grouping (key [1, slab, layer, 12]), each place
    up/hole or signed digits by the plate alone (L_b < G x M: signed digits);
  - layer 0 as delivered: qkv.0 from storage (the word's and the position's rounded rows with their stored products;
    only the -1/0/+1 carry meets the plate, in signed digits; position 0 reads its stored qkv pile), proj.0 by the
    whole light in the group form (rule 3 did not keep its storage), fc.0 from storage (word and position parts
    stored, the rest of the light in the group form), position 0 reads its whole stream after layer 0;
  - the knife kept at none of the 12 fc plates: every fc pile read whole, then base2's own divide, bias and bend;
  - every other step base2.py's own (water levels, gauges, divides, biases, the arrival, the bend, the stream);
  - the top read: the group table of 16 lines (key [1, 99, 768, 16]), per-place form, read from the top place down;
    after every place but the last the words with V_w <= v5 - (2 R2 + 2 mu) are set aside (V_w = the doubled read part
    + 2 beta_w, R2 = B(b) x S1, mu = 2 gamma + delta exactly as topband.py makes it); the words still in are finished
    exactly (X off, halved, the standing amount off, the divide, the bias).
  The storage's contents are made here from base2's own numbers (a stored row is the exact product of its rounded row
  and the plate; Z0 is base2's own layer 0 at position 0), as the storage would hold them.

WORK, in the delivered machine's conventions as this checker reads them (photon meetings and photons added):
  banks   the group items of cg.py (light, half table, ready piles written, up/hole reads, signed-digit reads, X, X off)
  qkv.0   stored rows read and added (their 1s; position 0: its stored pile's 1s), the split (the rounded rows' 1s),
          the carry: its 1s written and its signed digits x the plate's lit digits on each line
  fc.0    stored rows read and added (the 1s of the stored part's sum), the split, the rest in the group form
  fc      the whole form adds: divides (the pile's 1s x the gain's), biases, bends (H's 1s x the share's)
  arrival base2_fringes2's level 3: scores lit digit x lit digit, q k v written, shares' total, shares x values in
          signed digits, divides (the num's 1s x the gain's); layer 0: position 0 not counted
  rest    entry (the position rows' 1s), water levels (qkv, fc, the top), bank divides at qkv, biases at qkv, proj,
          mproj, stream adds, position 0's stream after layer 0 read
  top     table, reads into the words still in, X and S1 (N - 1 adds each), the bound's photons, the words read to the
          end, the standing amounts (written at pile scale and poured), the ready piles written, the light written
  Compares are a count, kept apart. Plain: 3,666,596,259 a token (the unit of every earlier result).

CONVENTION NOTE (added after a 48-position test of the top read, before the full run): this checker first read "the
bound's photons" as one add of S1 where B(b) = 2^b - 1 ((S1 << b) - S1); the delivered per-position record holds
B's 1s x S1's 1s at every test (the general rule that a product counts its numbers' photons multiplied). The rebuild
uses the product rule; the shortcut would be about 600 photons a position lower. Every other top item equalled the
record at all 48 positions tested without any change.

PREDICTION (written before this file ran; the checker's own)
  - EXACT: every pile (qkv, proj, fc, mproj), fc's output (out16 and its shift), the stream after every half layer and
    the light at the top identical to base2.py's own run on all 648 refs positions (plain_refs.npz) and on all 300
    judge5 windows (a fresh base2 run made here); judge5.judge 6000/6000 in the top 5 and 6000/6000 the same top word;
    the top 5 = judge5_ref.npz's (set and order) 6000/6000; refs top 5 = plain_refs 647 or 648 of 648 (position 494's
    exact tie for 5th place may fall either way), race winner 648/648; every check count 0.
  - WORK (seed 1): every bank item equal to the delivered record number for number at every layer (the layer-1 qkv
    test before this run already gave 7 of 7 equal), so qkv 132,698,796, proj 62,809,289, fc 190,757,633, mproj
    231,896,719 a token; arrival 8,020,119 and rest 4,558,811 (base2_fringes2's, unchanged); the top read 80,689,701:
    table, reads, ready piles written, light written and standing amounts equal; the bound's photons, the light sums
    and the words read to the end (under 8,000 a token together) may differ by convention by a few hundred to a few
    thousand a token. WHOLE 711,431,068 = 19.40% of plain, within 0.001 point.
  - OWN GROUPING (seed 7): the whole within 0.02 point of seed 1 (the delivered seed2 control moved under 0.003
    points); every pile identical.
  - ORDER-FREE (texts 0-2; groups, lines in each group and places at every bank in this checker's own random order;
    the top's groups and lines reordered): every pile, stream and top 5 identical; reads, signed-digit reads, ready
    piles written, X, X off and light identical; only the half table moves, within 0.01 point.

    python3 confirm_groups/rebuild_gpt2.py mode=refs [seed=1] [texts=0,1,2] [reorder=SEED]
    python3 confirm_groups/rebuild_gpt2.py mode=windows [a=0 b=300]
"""
import sys, os, json, time, resource, numpy as np
SM = "/Users/abundancemachine/streaming-machine"; HERE = f"{SM}/confirm_groups"
sys.path.insert(0, HERE); sys.path.insert(0, SM)
import judge5 as J                                           # base2.py's own steps, exec'd from its source
import cg
from cg import ones, sd

A = dict(a.split("=") for a in sys.argv[1:])
MODE = A.get("mode", "refs"); SEED = int(A.get("seed", 1)); REO = int(A["reorder"]) if "reorder" in A else None
TEXTSEL = [int(t) for t in A["texts"].split(",")] if "texts" in A else None
TAG = A.get("tag", f"{MODE}_s{SEED}" + (f"_t{'-'.join(map(str, TEXTSEL))}" if TEXTSEL else "") + (f"_order{REO}" if REO else ""))
LOGF = open(f"{HERE}/{TAG}.log", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
ES = J.ES
WTE = J.M["wte"]; WPE = J.M["wpe"]
PLATE = {}
def plate(name):
    if name not in PLATE: PLATE.clear(); PLATE[name] = (J.M[name + ".w"], J.M[name + ".e"], J.M[name + ".b"])
    return PLATE[name]
def xprod(a, w):                                              # an exact whole-number product (every sum below 2^52)
    with np.errstate(all="ignore"):
        p = np.asarray(a, np.float64) @ np.asarray(w, np.float64)
    assert np.isfinite(p).all() and np.abs(p).max(initial=0) < 2 ** 52
    return p.astype(np.int64)
def gain_ones(LS): return ones(J.EXP2[np.bitwise_and(-np.asarray(LS, np.int64), 0xFFFF)])
def water(x):
    S1, S2 = x.sum(1), (x * x).sum(1); o = ones(x)
    return (o ** 2).sum(1) + o.sum(1) + ones(S1) ** 2 + 2 * ones(S2)
def share_of(H):                                              # the table entry base2's bend reads
    idx = J.rshift(H, ES - 12) + 32768
    return np.where(idx < 0, 0, np.where(idx > 65535, 65536, J.PHI[np.clip(idx, 0, 65535)]))

class Count(dict):
    def add(self, lay, part, item, v): k = (lay, part, item); self[k] = self.get(k, 0) + int(np.sum(v))
def bank_of(kind, l):
    w, e, b = plate(f"{kind}.{l}")
    reorder = None if REO is None else [REO, 4242, cg.SLAB[kind], l]
    prng = None if REO is None else np.random.default_rng([REO, 77, cg.SLAB[kind], l])
    return cg.Bank(w, 12, [SEED, cg.SLAB[kind], l, 12], reorder=reorder, place_rng=prng), e, b
def group_read(C, lay, part, bk, x, plain_x=None, tag=None):
    pile, c = bk.read(x)
    for it in cg.ITEMS: C.add(lay, part if tag is None else tag, it, c[it])
    if tag is None: C.add(lay, "plain", part, ones(x if plain_x is None else plain_x) @ bk.wrow)
    return pile

def arrival(QKV, C, lay, first):
    """base2's attention, step for step, with base2_fringes2's level-3 count (rows from `first` on)."""
    P = QKV.shape[0]; out = np.zeros((P, 768), np.int64)
    for h in range(12):
        q, k, v = QKV[:, h*64:(h+1)*64], QKV[:, 768+h*64:768+(h+1)*64], QKV[:, 1536+h*64:1536+(h+1)*64]
        q16, eq = J.to16(q, -ES); k16, ek = J.to16(k, -ES)
        sc = J.rshift(xprod(q16, k16.T), -(eq[:, None] + ek[None, :] - 3 + J.SC))
        C.add(lay, "arrival", "scores, signed digits", np.tril(sd(q16) @ sd(k16).T).sum(1)[first:])
        C.add(lay, "arrival", "q, k and values written", ones(q16[first:]).sum() + ones(k16).sum() + ones(v).sum())
        ovn = sd(v).sum(1)
        for p in range(P):
            row = sc[p, :p+1]; d = row[J.race(row)] - row
            a = np.where(d < len(J.ARR), J.ARR[np.minimum(d, len(J.ARR) - 1)], 0)
            num = xprod(a[None, :], v[:p+1])[0]; Lp = J.log2fix([a.sum()])[0]
            out[p, h*64:(h+1)*64] = J.power(num, -Lp)
            if p >= first:
                C.add(lay, "arrival", "shares' total", ones(a)); C.add(lay, "arrival", "shares x values", sd(a) * ovn[:p+1])
                C.add(lay, "arrival", "divides", ones(num).sum() * gain_ones([Lp])[0])
    return out

def forward(texts, C, keep):
    """the group machine on every position of every text. keep(key, array, rows) receives every array for the check."""
    t0 = time.time()
    lens = [len(t) for t in texts]; off = np.concatenate([[0], np.cumsum(lens)])
    ids = np.concatenate([np.asarray(t, np.int64) for t in texts]); pos = np.concatenate([np.arange(n) for n in lens])
    NP = len(ids); f0 = np.flatnonzero(pos == 0); r1 = np.flatnonzero(pos != 0)
    A_rows, B_rows = WTE[ids], WPE[pos]; R = A_rows + B_rows
    C.add("L0", "rest", "entry: word + position", ones(B_rows))
    # ---------- qkv.0 from storage ----------
    bk, we, b = bank_of("qkv", 0); w = plate("qkv.0")[0]
    x16, ex, LS = J.gauge_log(R); s = ex + ES
    pile = np.zeros((NP, w.shape[1]), np.int64); nrow = sd(w).sum(1)
    carried = np.zeros(NP, bool)
    for sv in np.unique(s):
        m = np.flatnonzero(s == sv)
        if sv == 4:                                           # position 0: its qkv pile is stored
            assert (pos[m] == 0).all(); z = xprod(x16[m], w); pile[m] = z
            C.add("L0", "qkv", "stored rows read and added", ones(z))
        elif sv in (1, 2):
            a_, b_ = J.rshift(A_rows[m], sv), J.rshift(B_rows[m], sv); c = x16[m] - a_ - b_
            assert np.abs(c).max() <= 1
            t1, t2 = xprod(a_, w), xprod(b_, w)
            pile[m] = t1 + t2 + xprod(c, w)                    # the carry meets the plate in signed digits
            carried[m] = True
            C.add("L0", "qkv", "stored rows read and added", ones(t1).sum() + ones(t2).sum())
            C.add("L0", "qkv", "split", ones(a_).sum() + ones(b_).sum())
            C.add("L0", "qkv", "sdreads", sd(c) @ nrow); C.add("L0", "qkv", "light", ones(c))
            C.add("L0", "side", "qkv.0 rows and split", ones(t1).sum() + ones(t2).sum() + ones(a_).sum() + ones(b_).sum())
            C.add("L0", "side", "qkv.0 carry met in signed digits", (sd(c) @ nrow).sum() + ones(c).sum())
        else: raise ValueError(f"qkv.0 shift {sv}")
    cr = np.flatnonzero(carried)
    _, cw = bk.read(x16[cr]); C.add("L0", "side", "qkv.0 whole light, group form", sum(cw[it].sum() for it in cg.ITEMS))
    _, cc = bk.read(x16[cr] - J.rshift(A_rows[cr], s[cr][:, None]) - J.rshift(B_rows[cr], s[cr][:, None]))
    C.add("L0", "side", "qkv.0 carry, group form", sum(cc[it].sum() for it in cg.ITEMS))
    C.add("L0", "plain", "qkv", ones(x16) @ bk.wrow)
    keep("qkv_pile_0", pile, None)
    C.add("L0", "rest", "water levels", water(x16)); C.add("L0", "rest", "bank divides", ones(pile).sum(1) * gain_ones(LS))
    C.add("L0", "rest", "biases", NP * ones(b).sum())
    QKV = J.power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b; del pile, bk
    o = np.concatenate([arrival(QKV[off[t]:off[t + 1]], C, "L0", 1) for t in range(len(texts))])
    # ---------- position 0: the stream after layer 0, base2's own steps (what the storage holds) ----------
    R2 = np.empty_like(R)
    wp, wep, bp = plate("proj.0"); o16_0, eo_0 = J.to16(o[f0], -ES)
    att0 = J.rshift(xprod(o16_0, wp), -(wep[None, :] + eo_0[:, None] + ES)) + bp; R1_0 = R[f0] + att0
    wf, wef, bf = plate("fc.0"); xa, ea, La = J.gauge_log(R1_0); fp0 = xprod(xa, wf)
    H0 = J.power(fp0, (wef[None, :] + ea[:, None] + ES) * 65536 - La[:, None]) + bf
    g16_0, eg_0 = J.to16(J.edge(H0), -ES); wm, wem, bm = plate("mproj.0"); mp0 = xprod(g16_0, wm)
    R2[f0] = R1_0 + J.rshift(mp0, -(wem[None, :] + eg_0[:, None] + ES)) + bm
    C.add("L0", "rest", "position 0: the stream after layer 0 read", ones(R2[f0]))
    keep("proj_pile_0", xprod(o16_0, wp), f0); keep("stream_a_0", R1_0, f0); keep("fc_pile_0", fp0, f0)
    keep("mproj_pile_0", mp0, f0); keep("fc_out16_0", g16_0, f0); keep("fc_sh_0", eg_0 + ES, f0)
    # ---------- proj.0: the whole light in the group form ----------
    bk, wep, bp = bank_of("proj", 0)
    o16, eo = J.to16(o[r1], -ES); so = eo + ES
    ppile = group_read(C, "L0", "proj", bk, o16); keep("proj_pile_0", ppile, r1)
    ref = np.zeros_like(o16)                                  # the side count: proj.0's stored form (count only)
    for sv in np.unique(so):
        m = np.flatnonzero(so == sv); ref[m] = J.rshift(QKV[r1[m], 1536:], sv)
    own = xprod(ref, plate("proj.0")[0])
    _, cs_ = bk.read(o16 - ref)
    C.add("L0", "side", "proj.0 rows and split", ones(own).sum() + ones(ref).sum())
    C.add("L0", "side", "proj.0 rest light, group form", sum(cs_[it].sum() for it in cg.ITEMS))
    C.add("L0", "side", "proj.0 whole light, group form", sum(C.get(("L0", "proj", it), 0) for it in cg.ITEMS))
    att = J.rshift(ppile, -(wep[None, :] + eo[:, None] + ES)) + bp; del ppile, bk
    C.add("L0", "rest", "biases", len(r1) * ones(bp).sum()); C.add("L0", "rest", "stream adds", ones(att))
    R1 = R[r1] + att; keep("stream_a_0", R1, r1)
    # ---------- fc.0 from storage, mproj.0 ----------
    x1, e1, LS1 = J.gauge_log(R1); s1 = e1 + ES
    rest = x1.copy(); c0 = np.zeros((len(r1), 3072), np.int64); wf = plate("fc.0")[0]; BPROJ = plate("proj.0")[2]
    for sv in np.unique(s1):
        assert sv in (5, 6), f"fc.0 shift {sv} not stored"
        m = np.flatnonzero(s1 == sv)
        a_ = J.rshift(WTE[ids[r1[m]]], sv); b_ = J.rshift(WPE[pos[r1[m]]] + BPROJ[None, :], sv)
        rest[m] = x1[m] - a_ - b_; c0[m] = xprod(a_, wf) + xprod(b_, wf)
        C.add("L0", "fc", "stored rows read and added", ones(c0[m])); C.add("L0", "fc", "split", ones(a_).sum() + ones(b_).sum())
    mo = fc_mproj(C, "L0", 0, rest, c0, x1, e1, LS1, keep, r1)
    _, cw = fc_bank_cache[0].read(x1)
    C.add("L0", "side", "fc.0 whole light, group form", sum(cw[it].sum() for it in cg.ITEMS))
    C.add("L0", "side", "fc.0 rows and split", C[("L0", "fc", "stored rows read and added")] + C[("L0", "fc", "split")])
    C.add("L0", "side", "fc.0 rest light, group form", sum(C.get(("L0", "fc", it), 0) for it in cg.ITEMS))
    fc_bank_cache.clear()
    R2[r1] = R1 + mo; keep("stream_0", R2, None); R = R2
    log(f"    layer  0 done ({time.time() - t0:.0f}s, {rss()} MB)")
    for l in range(1, 12):
        lay = f"L{l}"
        bk, we, b = bank_of("qkv", l)
        x16, ex, LS = J.gauge_log(R); pile = group_read(C, lay, "qkv", bk, x16); keep(f"qkv_pile_{l}", pile, None)
        C.add(lay, "rest", "water levels", water(x16)); C.add(lay, "rest", "bank divides", ones(pile).sum(1) * gain_ones(LS))
        C.add(lay, "rest", "biases", NP * ones(b).sum())
        QKV = J.power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b; del pile, bk
        o = np.concatenate([arrival(QKV[off[t]:off[t + 1]], C, lay, 0) for t in range(len(texts))]); del QKV
        bk, wep, bp = bank_of("proj", l)
        o16, eo = J.to16(o, -ES); pile = group_read(C, lay, "proj", bk, o16); keep(f"proj_pile_{l}", pile, None)
        att = J.rshift(pile, -(wep[None, :] + eo[:, None] + ES)) + bp; del pile, bk
        C.add(lay, "rest", "biases", NP * ones(bp).sum()); C.add(lay, "rest", "stream adds", ones(att))
        R = R + att; keep(f"stream_a_{l}", R, None)
        x16, ex, LS = J.gauge_log(R)
        R = R + fc_mproj(C, lay, l, x16, None, x16, ex, LS, keep, None); keep(f"stream_{l}", R, None)
        log(f"    layer {l:2d} done ({time.time() - t0:.0f}s, {rss()} MB)")
    x16, ex, LS = J.gauge_log(R)
    C.add("top", "rest", "top water level", water(x16))
    return x16, ex, LS

fc_bank_cache = {}
def fc_mproj(C, lay, l, xlight, c0, x_plain, ex, LS, keep, rows):
    """fc.l read whole in the group form, base2's divide, bias and bend; then mproj.l in the group form."""
    bk, wef, bf = bank_of("fc", l)
    if l == 0: fc_bank_cache[l] = bk
    C.add(lay, "rest", "water levels", water(x_plain))
    pile = group_read(C, lay, "fc", bk, xlight, plain_x=x_plain)
    if c0 is not None: pile = pile + c0
    keep(f"fc_pile_{l}", pile, rows)
    F = (wef[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]
    H = J.power(pile, F) + bf
    C.add(lay, "fc", "divides", ones(pile).sum(1) * gain_ones(LS)); C.add(lay, "fc", "biases", len(pile) * ones(bf).sum())
    C.add(lay, "fc", "bends", ones(H) * ones(share_of(H)))
    G = J.edge(H); g16, eg = J.to16(G, -ES); keep(f"fc_out16_{l}", g16, rows); keep(f"fc_sh_{l}", eg + ES, rows)
    del pile, H, G
    bk, wem, bm = bank_of("mproj", l)
    mp = group_read(C, lay, "mproj", bk, g16); keep(f"mproj_pile_{l}", mp, rows)
    mo = J.rshift(mp, -(wem[None, :] + eg[:, None] + ES)) + bm
    C.add(lay, "rest", "biases", len(g16) * ones(bm).sum()); C.add(lay, "rest", "stream adds", ones(mo))
    return mo

def base2_run(texts, keep):
    """CHECK ONLY: a fresh run of base2.py's own steps (its product, its banks), every array kept."""
    lens = [len(t) for t in texts]; off = np.concatenate([[0], np.cumsum(lens)])
    ids = np.concatenate([np.asarray(t, np.int64) for t in texts]); pos = np.concatenate([np.arange(n) for n in lens])
    R = WTE[ids] + WPE[pos]
    for l in range(12):
        w, we, b = J.W[f"qkv.{l}"]; x16, ex, LS = J.gauge_log(R); pile = J.product(x16, w); keep(f"qkv_pile_{l}", pile)
        QKV = J.power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
        o = np.concatenate([J.attention(QKV[off[t]:off[t + 1]]) for t in range(len(texts))])
        w, we, b = J.W[f"proj.{l}"]; o16, eo = J.to16(o, -ES); pile = J.product(o16, w); keep(f"proj_pile_{l}", pile)
        R = R + J.rshift(pile, -(we[None, :] + eo[:, None] + ES)) + b; keep(f"stream_a_{l}", R)
        w, we, b = J.W[f"fc.{l}"]; x16, ex, LS = J.gauge_log(R); pile = J.product(x16, w); keep(f"fc_pile_{l}", pile)
        G = J.edge(J.power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b)
        g16, eg = J.to16(G, -ES); keep(f"fc_out16_{l}", g16); keep(f"fc_sh_{l}", eg + ES)
        w, we, b = J.W[f"mproj.{l}"]; pile = J.product(g16, w); keep(f"mproj_pile_{l}", pile)
        R = R + J.rshift(pile, -(we[None, :] + eg[:, None] + ES)) + b; keep(f"stream_{l}", R)
    return J.gauge_log(R)

TOP = None
def top():
    global TOP
    if TOP is None:
        t = time.time(); reorder = None if REO is None else [REO, 4242, 768, 16]
        wp = None if REO is None else np.random.default_rng([REO, 4343]).permutation(50257)
        TOP = cg.Top(J.M["head.w"], J.M["head.b"], J.M["head.e"], g=16, seed=SEED, reorder=reorder, word_perm=wp, log=log)
        log(f"  top table laid ({time.time() - t:.0f}s, {rss()} MB)")
    return TOP

def top_read_all(tp, x16, ex, LS, bad, B=48):
    """the top read in batches; the check path's direct product (base2's own) made per batch."""
    outs = []
    for c in range(0, len(x16), B):
        PILE = J.product(x16[c:c + B], J.W["head"][0])
        outs.append(tp.read(x16[c:c + B], ex[c:c + B], LS[c:c + B], J, PILE=PILE, bad=bad)); del PILE
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0] if k != "bad"}

def per_token(v, n): return (v + n // 2) // n
def pct(a, b): return f"{100 * a / b:.2f}%"
PLAIN_TOTAL = 3666596259
PLAIN_PART = dict(qkv=679699985, proj=248461379, fc=902064996, mproj=914738348, arrival=13053756, top=904017313, rest=4560481)
DELIV_PART = dict(qkv=132698796, proj=62809289, fc=190757633, mproj=231896719, arrival=8020119, top=80689701, rest=4558811)
DELIV_WHOLE = 711431068
ITEM_NAME = {"reads": "reads, up/hole places", "sdreads": "reads, signed-digit places", "pwrite": "ready piles written in signed digits",
             "table": "half table (by the flip)", "xoff": "X taken from each neuron", "light": "light written in signed digits",
             "xsum": "X (the light summed)", "stored rows read and added": "stored rows read and added"}
TOP_REC = dict(table="table", reads="ii_reads", light="ii_light", bound="ii_bound", end="ii_end", stand="ii_stand", pwrite="pwrite",
               lwrite="lwrite")

def run_refs():
    refs = json.load(open(f"{SM}/refs_many.json")); texts = [r["ids"] + r["ref"] for r in refs]
    sel = list(range(len(texts))) if TEXTSEL is None else TEXTSEL
    lens = [len(t) for t in texts]; offs = np.concatenate([[0], np.cumsum(lens)])
    rows = np.concatenate([np.arange(offs[t], offs[t + 1]) for t in sel]); T = [texts[t] for t in sel]; NP = len(rows)
    log(f"REFS: texts {sel if TEXTSEL else 'all 30'}, {NP} positions; grouping seed {SEED}" + (f"; ORDER-FREE reorder {REO}" if REO else ""))
    Z = np.load(f"{SM}/recording/C_ref/plain_refs.npz")
    diff = {}; mine = {}
    def keep(k, v, r):
        rr = rows if r is None else rows[r]
        if k.startswith("fc_out16") or k.startswith("fc_sh"):
            mine.setdefault(k, {})[tuple(rr)] = v; return      # checked below from base2's own fc pile
        ref = Z[k.replace("fc_out16", "fc_pile").replace("fc_sh", "fc_pile")][rr] if k in Z.files else None
        assert ref is not None, k
        diff[k] = diff.get(k, 0) + int((np.asarray(v) != ref).reshape(len(rr), -1).any(1).sum())
    C = Count()
    x16, ex, LS = forward(T, C, keep)
    # fc out16 and shift: base2's own steps on base2's own saved fc pile
    for l in range(12):
        w, we, b = J.W[f"fc.{l}"]
        Rb = Z["stream_a_" + str(l)][rows]; xb, eb, Lb = J.gauge_log(Rb)
        G = J.edge(J.power(Z[f"fc_pile_{l}"][rows], (we[None, :] + eb[:, None] + ES) * 65536 - Lb[:, None]) + b)
        g16, eg = J.to16(G, -ES)
        for k, ref in ((f"fc_out16_{l}", g16), (f"fc_sh_{l}", eg + ES)):
            n = 0
            for rr, v in mine[k].items():
                ix = np.searchsorted(rows, np.array(rr)); n += int((np.asarray(v).reshape(len(rr), -1) != np.asarray(ref)[ix].reshape(len(rr), -1)).any(1).sum())
            diff[k] = n
    for k, ref in (("x16", x16), ("ex", ex), ("LS", LS)):
        diff[f"top light {k}"] = int((np.asarray(ref) != Z[k][rows]).reshape(NP, -1).any(1).sum())
    nd = sum(diff.values())
    log(f"EXACT: {len(diff)} arrays checked against base2.py's own run (plain_refs.npz); position-arrays differing: {nd}")
    if nd: log("  differing: " + str({k: v for k, v in diff.items() if v}))
    # ---------- the top read ----------
    tp = top(); bad = {}
    o = top_read_all(tp, x16, ex, LS, bad)
    t5 = int((o["top5"] == Z["top5"][rows]).all(1).sum()); wn = int((o["win"] == Z["win"][rows]).sum())
    log(f"TOP READ: checks {bad}; top 5 = plain_refs {t5}/{NP}, race winner {wn}/{NP}")
    for q in np.flatnonzero(~(o["top5"] == Z["top5"][rows]).all(1)):
        sc = J.power(J.product(x16[q:q + 1], J.W["head"][0])[0], (int(J.M["head.e"][0]) + int(ex[q]) + ES) * 65536 - int(LS[q])) + J.M["head.b"]
        best = np.sort(sc)[::-1][:6]
        log(f"  position {rows[q]}: mine {o['top5'][q].tolist()} scores {sc[o['top5'][q]].tolist()}; plain_refs {Z['top5'][rows[q]].tolist()} "
            f"scores {sc[Z['top5'][rows[q]]].tolist()}; six best scores {best.tolist()}; mine are the five best in order: "
            f"{bool(np.array_equal(sc[o['top5'][q]], best[:5]))}")
    np.savez(f"{HERE}/{TAG}_top.npz", **{k: v for k, v in o.items() if k != "bad"}, rows=rows)
    report(C, o, NP, rows)

def report(C, o, NP, rows):
    rec = json.load(open(f"{SM}/groups2/machine/gpt2_refs.json"))["counts"] if TEXTSEL is None else None
    trec = np.load(f"{SM}/groups2/machine/gpt2_top_refs.npz") if TEXTSEL is None else None
    parts = {}
    for p in ("qkv", "proj", "fc", "mproj"):
        parts[p] = sum(v for (lay, pp, it), v in C.items() if pp == p)
    parts["arrival"] = sum(v for (lay, pp, it), v in C.items() if pp == "arrival")
    parts["rest"] = sum(v for (lay, pp, it), v in C.items() if pp == "rest")
    parts["top"] = sum(int(o[k].sum()) for k in TOP_REC)
    whole = sum(parts.values())
    log(f"\nWORK A TOKEN BY PART ({NP} positions; grouping seed {SEED}{'; reorder ' + str(REO) if REO else ''})")
    log(f"  {'part':10s} {'total':>16s} {'a token':>14s} {'delivered':>14s} {'same':>6s} {'of plain':>9s}")
    for p in ("qkv", "proj", "fc", "mproj", "arrival", "top", "rest"):
        pt = per_token(parts[p], NP)
        log(f"  {p:10s} {parts[p]:16,d} {pt:14,d} {DELIV_PART[p]:14,d} {str(pt == DELIV_PART[p]):>6s} {pct(pt, PLAIN_PART[p]):>9s}")
    wt = per_token(whole, NP)
    log(f"  {'WHOLE':10s} {whole:16,d} {wt:14,d} {DELIV_WHOLE:14,d} {str(wt == DELIV_WHOLE):>6s} {pct(wt, PLAIN_TOTAL):>9s}")
    pw = sum(v for (lay, pp, it), v in C.items() if it == "pwrite" and pp in ("qkv", "proj", "fc", "mproj"))
    log(f"  of it, the ready piles written: banks {per_token(pw, NP):,d}, top {per_token(int(o['pwrite'].sum()), NP):,d}; without them "
        f"{per_token(whole - pw - int(o['pwrite'].sum()), NP):,d} = {pct(per_token(whole - pw - int(o['pwrite'].sum()), NP), PLAIN_TOTAL)}")
    sd_ = {it: v for (lay, pp, it), v in C.items() if pp == "side"}
    log("\nRULE 3 SIDE COUNTS (count only; a token): " + "; ".join(f"{it} {per_token(v, NP):,d}" for it, v in sd_.items()))
    q_st = sd_["qkv.0 rows and split"] + sd_["qkv.0 carry met in signed digits"]; q_wh = sd_["qkv.0 whole light, group form"]
    p_st = sd_["proj.0 rows and split"] + sd_["proj.0 rest light, group form"]; p_wh = sd_["proj.0 whole light, group form"]
    f_st = sd_["fc.0 rows and split"] + sd_["fc.0 rest light, group form"]; f_wh = sd_["fc.0 whole light, group form"]
    for nm, st_, wh in (("qkv.0", q_st, q_wh), ("proj.0", p_st, p_wh), ("fc.0", f_st, f_wh)):
        log(f"  {nm}: stored form {per_token(st_, NP):,d}, whole light {per_token(wh, NP):,d} a token: storage "
            f"{'KEPT' if st_ < wh else 'NOT KEPT'} ({per_token(wh - st_, NP):+,d})")
    if rec is not None:
        for mine_, key in ((sd_["qkv.0 rows and split"], "side|L0|qkv|storage: the stored form (rows, split, the carry's work)"),
                           (q_wh, "side|L0|qkv|storage: the whole light in the group form: group work"),
                           (sd_["qkv.0 carry, group form"], "side|L0|qkv|the carry in the group form: group work"),
                           (sd_["proj.0 rows and split"], "side|L0|proj|storage: the stored form (rows, split)"),
                           (sd_["proj.0 rest light, group form"], None),
                           (p_wh, "side|L0|proj|storage: the whole light in the group form: group work"),
                           (sd_["fc.0 rows and split"], "side|L0|fc|storage: the stored form (rows, split)"),
                           (f_wh, "side|L0|fc|storage: the whole light in the group form: group work")):
            if key is None:
                rv = sum(rec.get(f"fr|L0|proj|{ITEM_NAME[it]}", 0) for it in cg.ITEMS); key = "the delivered fr|L0|proj group items (its stored form's rest)"
            else: rv = rec[key]
            log(f"    {key}: mine {mine_:,d} delivered {rv:,d} {'EQUAL' if mine_ == rv else 'DIFF %+d' % (mine_ - rv)}")
    log("\nTHE BANKS AND THE ARRIVAL BY ITEM, a token (mine; delivered record where it exists)")
    items = {}
    for (lay, pp, it), v in C.items():
        if pp in ("qkv", "proj", "fc", "mproj", "arrival", "rest"): items[(pp, it)] = items.get((pp, it), 0) + v
    for (pp, it), v in sorted(items.items()):
        log(f"  {pp:8s} {it:45s} {per_token(v, NP):14,d}")
    if rec is not None:
        log("\nLAYER BY LAYER AGAINST THE DELIVERED RECORD (totals over the 648 positions)")
        nsame = ndiff = 0
        for (lay, pp, it), v in sorted(C.items()):
            if pp not in ("qkv", "proj", "fc", "mproj"): continue
            if lay == "L0" and pp == "proj": continue           # proj.0 runs the whole light here; compared above with the side key
            name = ITEM_NAME.get(it, it)
            if it == "split": name = {"qkv": "split (stored rounded rows taken off)", "fc": "split (stored rounded rows taken off)"}[pp]
            cands = [f"fr|{lay}|{pp}|{name}", f"fcw|{lay}|{pp}|{name}"]
            if pp == "fc" and it in ("divides", "biases", "bends"): cands = [f"fcw|{lay}|fc|{it}"]
            rv = next((rec[k] for k in cands if k in rec), None)
            if rv is None: log(f"  {lay:4s} {pp:6s} {it:40s} mine {v:18,d}  (no delivered key)"); continue
            same = rv == v; nsame += same; ndiff += not same
            if not same: log(f"  DIFF {lay:4s} {pp:6s} {it:40s} mine {v:18,d} delivered {rv:18,d} ({v - rv:+,d})")
        for (lay, pp, it), v in sorted(C.items()):
            if pp == "plain":
                k = f"plain|{lay}|{it}|meetings, plain"
                if k in rec:
                    same = rec[k] == v; nsame += same; ndiff += not same
                    if not same: log(f"  DIFF {lay} plain {it} mine {v:,d} delivered {rec[k]:,d}")
            if pp in ("arrival", "rest"):
                name = {"scores, signed digits": "scores, signed digits", "q, k and values written": "q, k and values written in signed digits",
                        "shares x values": "shares x values, signed digits"}.get(it, it)
                k = f"fr|{lay}|{pp}|{name}"
                if k in rec:
                    same = rec[k] == v; nsame += same; ndiff += not same
                    if not same: log(f"  DIFF {lay} {pp} {it} mine {v:,d} delivered {rec[k]:,d} ({v - rec[k]:+,d})")
                else: log(f"  {lay} {pp} {it}: no delivered key {k}")
        log(f"  bank, arrival, rest and plain-meeting counts equal to the delivered record: {nsame}; differing: {ndiff}")
    if trec is not None:
        log("\nTHE TOP READ BY ITEM AGAINST THE DELIVERED PER-POSITION RECORD (gpt2_top_refs.npz), a token")
        for k, rk in TOP_REC.items():
            m, r = o[k], trec[rk]; neq = int((m != r).sum())
            log(f"  {k:8s} mine {per_token(int(m.sum()), NP):12,d}  delivered {per_token(int(r.sum()), NP):12,d}  positions differing {neq}/{NP}"
                + (f"  (total {int(m.sum()) - int(r.sum()):+,d})" if neq else ""))
        sv = trec["ii_surv"]; ms = o["surv"]
        log(f"  survivors: delivered array {sv.shape}, mine {ms.shape}")
        k = min(sv.shape[1], ms.shape[1])
        log(f"  survivors after each place equal (last {k} columns aligned at the end): {int((sv[:, -k:] == ms[:, -k:]).all(1).sum())}/{NP}")
        log(f"  top 5 = the delivered (ii)'s {int((trec['ii_top5'] == o['top5']).all(1).sum())}/{NP}, race winner {int((trec['ii_win'] == o['win']).sum())}/{NP}")
        med = np.median(ms, axis=0).astype(int).tolist(); log(f"  survivors (median) after each place {list(zip(top().places, med))}")
    json.dump({"counts": {"|".join(k): v for k, v in C.items()}, "parts": parts, "whole": whole, "npos": NP, "seed": SEED, "reorder": REO},
              open(f"{HERE}/{TAG}.json", "w"), indent=0)

def run_windows():
    a, b = int(A.get("a", 0)), int(A.get("b", 300)); CH = int(A.get("chunk", 10))
    win = json.load(open(f"{SM}/judge5_texts.json"))
    log(f"WINDOWS {a}..{b} in chunks of {CH}: the machine against a fresh run of base2.py's own steps; the top read at positions 3..22")
    tp = top(); ref5 = np.load(f"{SM}/judge5_ref.npz")["top5"]
    PICK, T5 = {}, {}; nd_all = 0; bad = {}; t0 = time.time()
    for c0 in range(a, b, CH):
        ws = list(range(c0, min(b, c0 + CH))); T = [win[i]["ids"] for i in ws]
        REF = {}
        def keep_ref(k, v): REF[k] = v
        xr, er, Lr = base2_run(T, keep_ref)
        lens = [len(t) for t in T]; offs = np.concatenate([[0], np.cumsum(lens)]); diff = {}
        def keep(k, v, r):
            ref = REF[k] if r is None else REF[k][r]
            diff[k] = diff.get(k, 0) + int((np.asarray(v) != ref).reshape(len(ref), -1).any(1).sum())
        C = Count(); x16, ex, LS = forward(T, C, keep)
        for k, m_, r_ in (("x16", x16, xr), ("ex", ex, er), ("LS", LS, Lr)):
            diff[f"top {k}"] = int((m_ != r_).reshape(len(m_), -1).any(1).sum())
        nd = sum(diff.values()); nd_all += nd
        sel = np.concatenate([offs[i] + np.arange(J.FIRST, J.LAST) for i in range(len(ws))])
        del REF
        o = top_read_all(tp, x16[sel], ex[sel], LS[sel], bad)
        for i, wi in enumerate(ws):
            PICK[wi] = o["win"][i * 20:(i + 1) * 20]; T5[wi] = o["top5"][i * 20:(i + 1) * 20]
        log(f"  windows {ws[0]}..{ws[-1]}: arrays differing {nd} ({len(diff)} arrays); top checks {bad}; ({time.time() - t0:.0f}s, {rss()} MB)")
    log(f"EXACT: position-arrays differing from base2's own run in windows {a}..{b}: {nd_all}; the top read's checks {bad}")
    np.savez(f"{HERE}/{TAG}_{a}_{b}.npz", win=np.array([PICK[i] for i in range(a, b)]), top5=np.array([T5[i] for i in range(a, b)]))
    if a == 0 and b == 300:
        it = iter(range(300))
        inn, same, tot = J.judge(forward=lambda ids: PICK[next(it)])
        t5 = int(sum((T5[i] == ref5[i]).all(1).sum() for i in range(300)))
        log(f"JUDGE5: judge5.judge in the reference top 5 {inn}/{tot}, the same top word {same}/{tot}; the machine's top 5 = judge5_ref.npz "
            f"(set and order) {t5}/{tot}")

if __name__ == "__main__":
    log(f"rebuild_gpt2.py {' '.join(sys.argv[1:])}  (checker's independent rebuild)")
    if MODE == "refs": run_refs()
    elif MODE == "windows": run_windows()
    log(f"done ({rss()} MB)")
