"""MEASURE LAYER 0 FROM STORAGE (base2_layer0.py) AGAINST base2.py's LAYER 0.

count   Photons per token at each of layer 0's plates, in the unit of every earlier count (extra/extra_layer0.py: a
        meeting is light photons x plate photons for the lines met; adds are photons added), split into stored reads
        (photons of stored rows added, bytes read), the split (rounded rows taken off) and what is met per token.
        Plain layer 0 is extra_layer0's own counted mirror (V0). Over the 648 positions of the 30 refs texts, then the
        7,200 of the 300 judge windows. Two versions: as built (OWN holds the texts' pairs), and the whole-vocabulary
        version (OWN cannot be stored: proj.0 met whole at positions 1 on). CONTROLS: the split made with another
        word's rows (fc.0) and another pair's own value (proj.0).
edges   (d) Every knife edge of layer 0 at positions 1 on (3,072 a position), its output split EXACTLY:
            out = bend(H)                                   the machine's own, H = power(pile, F) + fc bias
          (i)   bend(Hf), Hf = power(Pf, F) + fc bias       the fixed part alone (word + position + proj bias, the
                                                            stored pile Pf), at the token's own water level
          (ii)  S = (1/2) Pk E 2^-k                          the KV push (Pk = the arrival write's part met with fc.0)
                                                            carried through the straight half; through mproj.0 it is
                                                            the fixed turn T_0's piece, g x (KV light met with T_0)
          (iii) glow = out - (i) - (ii)                      the rest: needs the actual pile
        and the glow itself split exactly into the carry's straight half (1/2) Pc E 2^-k, the two roundings' straight
        half, and the change of the symmetric half e(H) - e(Hf) (e(H) = bend(H) - H/2). Every part is a whole number
        over 2^(k+1) (k = the places fc.0's gain rounds off, 24-30), kept exact in int64 (bounds checked).
        Also: the side-paired glow g(H) = H [H > 0] - bend(H) (direction 21), and the edges whose side the KV decides.
        CONTROL: the fixed part plus another text's KV push (same position).
turn    the identity behind (ii): (KV light) met with TN_0 == sum_n Pk[n] 2^(we_fc[n] - m0) mproj[n, :], exact
        (turns/turns.npz), and the T_0 route's meeting against mproj.0's own.
time    real time of layer 0 both ways, median of 5 interleaved runs: the whole text at once (as base2.py's forward
        runs), and one token at a time (the stream: the newest token alone, earlier q/k/v kept).

PREDICTIONS (the task's, written before the run): qkv.0 and the arrival's q/k/v are fully precomputed; fc.0's fixed
part saves only about 5% of fc.0 (the arrival's write dominates the light there); the glow is nonzero on more than 99%
of layer-0 edges.
The session's (before the run):
  - plain layer 0 = 219,153,343 photons a token on the refs texts (the pilot's figure); qkv.0 meeting 61.8M -> carry
    about 2.7M (step 1); proj.0 20.5M -> about 18.9M as built (-8%), 20.5M less position 0's share in the whole-
    vocabulary version; fc.0 61.3M -> about 58.3M (-5%); mproj.0 unchanged but for position 0.
  - layer 0 as built about 150M (68-70% of plain); whole-vocabulary version about 152M.
  - under the task's split, zero glow means the push changed nothing at the edge: the glow there is the change of
    the symmetric half, which outside the graded part is (|H| - |Hf|)/2, nonzero whenever the push is. So the count
    is the count of edges the KV leaves exactly unmoved, predicted under 0.5%. The side-paired glow is zero only
    where the table reads exactly 0 or 65,536, predicted under 1% at layer 0. The glow is as large as the straight
    half at every shut edge (it cancels it there): sum |glow| about the size of sum |S|.
  - the KV puts 5-15% of edges on the other side from the fixed part alone; another text's push, fewer right sides.
  - the T_0 route meets more than mproj.0 does (TN_0's entries are 36-38 places against mproj's 16).
  - time: whole text at once 0.95-1.10x base2's; one token at a time 1.0-1.3x faster (qkv.0's product gone, the
    others still dense products).
    python3 layer0/measure.py parts=count,edges,turn,time
"""
import sys, os, json, time, gc, platform, subprocess, resource, warnings, numpy as np
warnings.filterwarnings("ignore")
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); sys.path.insert(0, f"{HERE}/extra"); os.chdir(HERE)
PARTS = dict(a.split("=") for a in sys.argv[1:]).get("parts", "count,edges,turn,time").split(",")
import base2_layer0 as L
E, J = L.E, L.J; W, ES = J.W, J.ES
t0 = time.time()
def gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
def say(*a): print(*a, flush=True)
REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]
WIN = [w["ids"] for w in J.WINDOWS]
POP = np.array([bin(i).count("1") for i in range(1 << 16)], np.int64)
def ones(A):
    A = np.abs(np.asarray(A).astype(np.int64)); n = np.zeros(A.shape, np.int64)
    while A.any(): n += POP[A & 0xFFFF]; A = A >> 16
    return n
WF16 = J.M["fc.0.w"]; WP16 = J.M["proj.0.w"]; WM16 = J.M["mproj.0.w"]; WQ16 = J.M["qkv.0.w"]
WROW = {"qkv.0": ones(WQ16).sum(1), "proj.0": ones(WP16).sum(1), "fc.0": ones(WF16).sum(1), "mproj.0": ones(WM16).sum(1)}

def pieces(ids):                                      # the new machine's layer 0, every piece kept (rows: positions >= 1)
    ids = np.asarray(ids, np.int64); rec = {}; L.layer0(ids, rec=rec); return ids, rec

# ======================================================================================== count
def part_count():
    import extra_layer0 as X                          # the pilot's counted mirror: plain layer 0, every item per position
    assert X.W is J.W
    for label, texts in (("the 30 refs texts", TEXTS), ("the 300 judge windows", WIN)):
        rng = np.random.default_rng(31)
        T = {}                                        # totals over positions
        def add(k, v): T[k] = T.get(k, 0) + int(np.sum(v))
        npos = 0; nfirst = 0
        for ids in texts:
            n = len(ids); npos += n; nfirst += 1
            C = X.Count(); X.layer0(ids, "V0", C)
            bp = {k: v[:n].copy() for k, v in C.bypos.items()}
            for k, v in bp.items(): add("plain " + k, v)
            ids, rec = pieces(ids); r = rec["rows"]; A, B = L.WTE[ids], L.WPE[:n]
            for k in ("qkv meeting", "proj meeting", "fc meeting", "mproj meeting"): add("plain1 " + k, bp[k][r])
            add("plain1 all", sum(v[r] for v in bp.values())); add("plain0 all", sum(v[:1] for v in bp.values()))
            # ---- qkv.0 (step 1): every position; position 0 reads its whole pile (ZP0)
            x16, ex, LS = J.gauge_log(A + B); s = ex + ES
            for i in range(n):
                sv = int(s[i])
                if sv in (4, 0):
                    z = (E.ZP0 if sv == 4 else E.Z1023).read([ids[i]])[0]; add("qkv stored", ones(z).sum()); add("qkv bytes", 2304 * 5)
                else:
                    a_s, b_s = J.rshift(A[i], sv), J.rshift(B[i], sv); c = x16[i] - a_s - b_s
                    add("qkv stored", ones(E.T[sv].read([ids[i]])[0]).sum() + ones(E.P[sv].read([i])[0]).sum()); add("qkv bytes", 2 * 2304 * 5)
                    add("qkv split", ones(a_s).sum() + ones(b_s).sum()); add("qkv met", ones(c) @ WROW["qkv.0"])
            for k in ("entry: word + position", "qkv water level", "qkv divide", "qkv bias"): add("new " + k, bp[k])
            # ---- position 0: the whole stream after layer 0 read
            add("Z0 stored", ones(L.Z0.read(ids[:1])[0]).sum()); add("Z0 bytes", 768 * 4)
            # ---- positions 1 on: arrival, proj, fc, mproj
            for k in ("arrival scores", "arrival shares' total", "arrival mix (shares x values)", "arrival divide",
                      "proj bias", "stream add (arrival)", "fc water level", "fc divide", "fc bias", "fc bend",
                      "mproj meeting", "mproj bias", "stream add (neurons)"):
                add("new " + k, bp[k][r])
            so, o16 = rec["so"], rec["o16"]; keys = ids[r] * 1024 + r; vown = rec["QKV"][r, 1536:]
            okeys = np.roll(keys, 1)                                         # CONTROL: the previous position's pair
            for j in range(len(r)):
                sv = int(so[j]); ref = J.rshift(vown[j], sv)
                add("proj stored", ones(L.OWN[sv].read([keys[j]])[0]).sum()); add("proj bytes", 768 * 5)
                add("proj split", ones(ref).sum()); add("proj met", ones(o16[j] - ref) @ WROW["proj.0"])
                add("proj met, whole vocabulary", ones(o16[j]) @ WROW["proj.0"])
                oref = J.rshift(rec["QKV"][r[(j - 1) % len(r)], 1536:], sv)
                add("CONTROL proj met, another pair's own value", ones(o16[j] - oref) @ WROW["proj.0"])
            s1, x1, rest = rec["s1"], rec["x1"], rec["rest"]
            other = rng.integers(0, 50257, len(r))
            for j in range(len(r)):
                sv = int(s1[j]); a_, b_ = J.rshift(L.WTE[ids[r[j]]], sv), J.rshift(L.WPE[r[j]] + L.BP, sv)
                add("fc stored", ones(L.FT[sv].read([ids[r[j]]])[0]).sum() + ones(L.FP[sv].read([r[j]])[0]).sum()); add("fc bytes", 2 * 3072 * 4)
                add("fc split", ones(a_).sum() + ones(b_).sum()); add("fc met", ones(rest[j]) @ WROW["fc.0"])
                add("CONTROL fc met, another word's rows", ones(x1[j] - J.rshift(L.WTE[other[j]], sv) - b_) @ WROW["fc.0"])
                c = rest[j] - J.rshift(rec["att"][j] - L.BP, sv); assert np.abs(c).max() <= 1
                add("fc lines carrying", (c != 0).sum())
        n = npos; P = lambda k: T.get("plain " + k, 0) / n
        plain_items = {k[6:]: v / n for k, v in T.items() if k.startswith("plain ")}
        plain_total = sum(plain_items.values())
        arr_keys = ("arrival scores", "arrival shares' total", "arrival mix (shares x values)", "arrival divide")
        other_keys = [k for k in plain_items if k not in arr_keys and k not in ("qkv meeting", "proj meeting", "fc meeting", "mproj meeting")]
        g = lambda k: T.get(k, 0) / n
        new_arr = sum(g("new " + k) for k in arr_keys); new_other = sum(g("new " + k) for k in other_keys)
        rows_tab = [
            ("qkv.0", P("qkv meeting"), g("qkv stored"), g("qkv split"), g("qkv met"), g("qkv bytes")),
            ("the arrival (scores, shares, mix, divide)", sum(P(k) for k in arr_keys), 0, 0, new_arr, 0),
            ("proj.0", P("proj meeting"), g("proj stored"), g("proj split"), g("proj met"), g("proj bytes")),
            ("fc.0", P("fc meeting"), g("fc stored"), g("fc split"), g("fc met"), g("fc bytes")),
            ("mproj.0", P("mproj meeting"), 0, 0, g("new mproj meeting"), 0),
            ("water levels, gains, biases, bend, adds", sum(P(k) for k in other_keys), 0, 0, new_other, 0),
            ("position 0: the stream after layer 0 (Z0)", 0, g("Z0 stored"), 0, 0, g("Z0 bytes"))]
        say(f"\nPHOTONS PER TOKEN AT LAYER 0, {label} ({n:,} positions, {nfirst} at position 0)   ({time.time() - t0:.0f}s, {gb():.2f} GB)")
        say(f"  {'':44s} {'plain':>14s} | {'stored read':>12s} {'split':>10s} {'met':>14s} {'new total':>14s} {'of plain':>9s} | bytes read")
        tot_new = 0
        for name, pl, st, sp, me, by in rows_tab:
            nt = st + sp + me; tot_new += nt
            say(f"  {name:44s} {pl:14,.0f} | {st:12,.0f} {sp:10,.0f} {me:14,.0f} {nt:14,.0f} {100 * nt / pl if pl else 0:8.2f}% | {by:,.0f}")
        say(f"  {'LAYER 0':44s} {plain_total:14,.0f} | {'':12s} {'':10s} {'':14s} {tot_new:14,.0f} {100 * tot_new / plain_total:8.2f}%")
        st_all = g("qkv stored") + g("proj stored") + g("fc stored") + g("Z0 stored"); sp_all = g("qkv split") + g("proj split") + g("fc split")
        met_all = tot_new - st_all - sp_all
        say(f"  of the new total: stored rows read and added {st_all:,.0f} ({100 * st_all / plain_total:.2f}% of plain), "
            f"the split {sp_all:,.0f} ({100 * sp_all / plain_total:.2f}%), met per token {met_all:,.0f} ({100 * met_all / plain_total:.2f}%)")
        say(f"  precomputed (plain layer 0 work no longer done per token): {plain_total - tot_new:,.0f} = {100 * (plain_total - tot_new) / plain_total:.2f}% "
            f"of layer 0 = {100 * (plain_total - tot_new) / 3666596259:.2f}% of the whole plain machine (3,666,596,259 a token)")
        wv = tot_new - g("proj stored") - g("proj split") - g("proj met") + g("proj met, whole vocabulary")
        say(f"  whole-vocabulary version (OWN not storable: proj.0 met whole at positions 1 on): proj.0 {g('proj met, whole vocabulary'):,.0f}; "
            f"layer 0 {wv:,.0f} = {100 * wv / plain_total:.2f}% of plain; precomputed {100 * (plain_total - wv) / plain_total:.2f}% of layer 0 "
            f"= {100 * (plain_total - wv) / 3666596259:.2f}% of the whole")
        n1 = n - nfirst; h = lambda k: T.get(k, 0) / n1
        say(f"  positions 1 on only ({n1:,}), per token there: proj.0 plain {h('plain1 proj meeting'):,.0f} -> met {h('proj met'):,.0f} "
            f"+ stored {h('proj stored'):,.0f} + split {h('proj split'):,.0f} = {100 * (h('proj met') + h('proj stored') + h('proj split')) / h('plain1 proj meeting'):.2f}%;  "
            f"fc.0 plain {h('plain1 fc meeting'):,.0f} -> met {h('fc met'):,.0f} + stored {h('fc stored'):,.0f} + split {h('fc split'):,.0f} "
            f"= {100 * (h('fc met') + h('fc stored') + h('fc split')) / h('plain1 fc meeting'):.2f}%;  qkv.0 plain {h('plain1 qkv meeting'):,.0f}")
        say(f"  position 0 alone: plain layer 0 {T.get('plain0 all', 0) / nfirst:,.0f} a token there, now the step-1 qkv.0 path and one read of Z0; "
            f"positions 1 on: plain layer 0 {h('plain1 all'):,.0f} a token there")
        say(f"  fc.0: lines carrying (-1/+1) {g('fc lines carrying') * n / n1:.1f} of 768 a token past position 0")
        say(f"  CONTROLS (positions 1 on, per token there): fc.0 split with another word's rows: met {h('CONTROL fc met, another word' + chr(39) + 's rows'):,.0f} "
            f"(own rows {h('fc met'):,.0f}, plain {h('plain1 fc meeting'):,.0f}); proj.0 with another pair's own value: met "
            f"{h('CONTROL proj met, another pair' + chr(39) + 's own value'):,.0f} (own {h('proj met'):,.0f}, whole {h('proj met, whole vocabulary'):,.0f})")
        say("  plain items: " + "; ".join(f"{k} {v:,.0f}" for k, v in sorted(plain_items.items(), key=lambda kv: -kv[1])))

# ======================================================================================== edges
TURN = np.load(f"{HERE}/turns/turns.npz"); TN0 = TURN["TN"][0]; M0 = int(TURN["m0"][0])
def edge_split(ids, rec, other_push=None):
    """exact split of every edge at positions >= 1. Returns integer arrays over D = 2^(k+1)."""
    r = rec["rows"]; s1, x1, rest, LS, pile = rec["s1"], rec["x1"], rec["rest"], rec["LS"], rec["fc_pile"]
    n = len(r); Pf = np.empty((n, 3072), np.int64); a_s = np.empty((n, 768), np.int64)
    for sv in np.unique(s1):
        m = np.flatnonzero(s1 == sv)
        Pf[m] = L.FT[sv].read(ids[r[m]]) + L.FP[sv].read(r[m])
        a_s[m] = J.rshift(rec["att"][m] - L.BP[None, :], sv)
    c = rest - a_s; assert np.abs(c).max() <= 1
    Pk = J.product(a_s, L.WF); Pc = J.product(c, L.WF)
    assert np.array_equal(Pf + Pk + Pc, pile)
    F = (L.WEF[None, :] + (s1 - ES)[:, None] + ES) * 65536 - LS[:, None]
    I = F >> 16; Ef = J.EXP2[F & 0xFFFF].astype(np.int64); k = 16 - I; assert k.min() >= 1
    H = rec["H"]; out = rec["G"]
    Hf = J.power(Pf, F) + L.BF; outf = J.edge(Hf)
    big = max(np.abs(out).max(), np.abs(outf).max(), np.abs(H).max(), np.abs(Hf).max())
    assert int(big) * 2 ** (int(k.max()) + 1) < 2 ** 61 and np.abs(Pk).max() * Ef.max() < 2 ** 61 and np.abs(pile).max() * Ef.max() < 2 ** 61
    D = np.left_shift(np.int64(1), k + 1)
    OUT, OUTF = out * D, outf * D; S = Pk * Ef; GLOW = OUT - OUTF - S
    CAR = Pc * Ef; DH = (H - Hf) * np.left_shift(np.int64(1), k)
    RND = DH - (Pk + Pc) * Ef; DE = OUT - OUTF - DH
    assert np.array_equal(GLOW, CAR + RND + DE)
    res = dict(k=k, out=out, outf=outf, H=H, Hf=Hf, OUT=OUT, OUTF=OUTF, S=S, GLOW=GLOW, CAR=CAR, RND=RND, DE=DE,
               Pf=Pf, Pk=Pk, Pc=Pc, a_s=a_s)
    if other_push is not None:                        # CONTROL: the fixed part plus another text's push (same rows)
        Hx = J.power(Pf + other_push, F) + L.BF; res["Hx"] = Hx
    return res
def osum(X, k):                                      # exact sum of |X| / 2^(k+1), as a whole number over 2^32
    A = np.abs(X); tot = 0
    for kv in np.unique(k):
        a = A[k == kv]; hi = int((a >> 31).sum()); lo = int((a & ((1 << 31) - 1)).sum())
        tot += ((hi << 31) + lo) << (31 - int(kv))
    return tot
def part_edges():
    for label, texts in (("the 30 refs texts", TEXTS), ("the 300 judge windows", WIN)):
        cnt = dict(edges=0, glow0=0, de0=0, dh0=0, pk0=0, side_g0=0, side_g0_both=0, flip=0, ctrl_true=0, fixed_true=0,
                   ctrl_n=0, bright=0)
        sz = dict(out=0, outf=0, S=0, GLOW=0, CAR=0, RND=0, DE=0)
        pz = dict(Pf=0, Pk=0, Pc=0, pile=0)
        ATT = []                                       # each text's arrival write at every position >= 1, for the control
        for t in texts:
            ids, rec = pieces(t); ATT.append(rec["att"])
        rng = np.random.default_rng(41)
        for ti, t in enumerate(texts):
            ids, rec = pieces(t); r = rec["rows"]
            res = edge_split(ids, rec); n = len(r)
            # the control's push: another text's arrival write at the same position, rounded at this token's own shift
            other = np.zeros((n, 3072), np.int64)
            for j, p in enumerate(r):
                us = [u for u in range(len(texts)) if u != ti and len(texts[u]) > p]; u = int(rng.choice(us))
                oa = J.rshift(ATT[u][p - 1] - L.BP, int(rec["s1"][j]))
                other[j] = J.product(oa[None, :], L.WF)[0]
            Hx = J.power(res["Pf"] + other, (L.WEF[None, :] + (rec["s1"] - ES)[:, None] + ES) * 65536 - rec["LS"][:, None]) + L.BF
            k = res["k"]; cnt["edges"] += res["out"].size
            cnt["glow0"] += int((res["GLOW"] == 0).sum()); cnt["de0"] += int((res["DE"] == 0).sum())
            cnt["dh0"] += int((res["H"] == res["Hf"]).sum()); cnt["pk0"] += int((res["Pk"] == 0).sum())
            H, out, Hf, outf = res["H"], res["out"], res["Hf"], res["outf"]
            gH = np.where(H > 0, H, 0) - out; gHf = np.where(Hf > 0, Hf, 0) - outf
            cnt["side_g0"] += int((gH == 0).sum()); cnt["side_g0_both"] += int(((gH == 0) & (gHf == 0)).sum())
            cnt["flip"] += int(((H > 0) != (Hf > 0)).sum()); cnt["bright"] += int((H > 0).sum())
            cnt["fixed_true"] += int(((H > 0) == (Hf > 0)).sum()); cnt["ctrl_true"] += int(((H > 0) == (Hx > 0)).sum())
            for key in sz: sz[key] += osum(res[key] if key not in ("out", "outf") else res[key.upper()], k)
            pz["Pf"] += int(np.abs(res["Pf"]).sum()); pz["Pk"] += int(np.abs(res["Pk"]).sum())
            pz["Pc"] += int(np.abs(res["Pc"]).sum()); pz["pile"] += int(np.abs(rec["fc_pile"]).sum())
        N = cnt["edges"]; pc = lambda x: f"{100 * x / N:.3f}%"
        say(f"\nKNIFE EDGES OF LAYER 0, {label}, positions 1 on: {N:,} edges (3,072 a position)   ({time.time() - t0:.0f}s, {gb():.2f} GB)")
        say(f"  split exact at every edge: out = (i) bend(Hf) + (ii) S + (iii) glow, with the glow = carry's straight half + roundings' straight half + e(H) - e(Hf): yes (asserted)")
        say(f"  zero glow, task's split (out == bend(Hf) + S exactly):       {cnt['glow0']:,} = {pc(cnt['glow0'])}   nonzero {pc(N - cnt['glow0'])}")
        say(f"  zero change of the symmetric half (e(H) == e(Hf)):          {cnt['de0']:,} = {pc(cnt['de0'])}")
        say(f"  the edge's value unmoved by the push and carry (H == Hf):    {cnt['dh0']:,} = {pc(cnt['dh0'])};  KV push exactly 0 (Pk == 0): {cnt['pk0']:,}")
        say(f"  side-paired glow g(H) = H[H>0] - bend(H) zero:               {cnt['side_g0']:,} = {pc(cnt['side_g0'])};  zero at both H and Hf: {cnt['side_g0_both']:,} = {pc(cnt['side_g0_both'])}")
        say(f"  bright side: {pc(cnt['bright'])} of edges;  the KV puts the edge on the other side from the fixed part alone: {pc(cnt['flip'])}")
        say(f"  CONTROL, the fixed part plus another text's push (same position): on the true side {pc(cnt['ctrl_true'])}  (the fixed part alone: {pc(cnt['fixed_true'])})")
        tot = sz["out"]
        say(f"  sizes, sum over edges of |part| (units 2^-16 at the edge output), as a share of sum |out| = {tot / 2**32:,.0f}:")
        for key, name in (("outf", "(i) bend(Hf), the fixed part alone"), ("S", "(ii) S, the KV push's straight half (T_0)"),
                          ("GLOW", "(iii) glow, the rest"), ("CAR", "    of which the carry's straight half"),
                          ("RND", "    of which the roundings' straight half"), ("DE", "    of which e(H) - e(Hf)")):
            say(f"     {name:46s} {sz[key] / 2**32:16,.0f}  = {100 * sz[key] / tot:8.2f}% of sum |out|")
        say(f"  the pile's pushes (sum |.| in pile units): fixed part {pz['Pf']:,} ({100 * pz['Pf'] / (pz['Pf'] + pz['Pk'] + pz['Pc']):.1f}%), "
            f"KV push {pz['Pk']:,} ({100 * pz['Pk'] / (pz['Pf'] + pz['Pk'] + pz['Pc']):.1f}%), carry {pz['Pc']:,} "
            f"({100 * pz['Pc'] / (pz['Pf'] + pz['Pk'] + pz['Pc']):.2f}%); sum |pile| {pz['pile']:,} = {100 * pz['pile'] / (pz['Pf'] + pz['Pk'] + pz['Pc']):.1f}% of all pushing")
        del ATT

# ======================================================================================== turn
def limbs(X):                                        # whole numbers -> signed 16-place limbs (float64), X = sum L_i 2^(16 i)
    X = np.asarray(X, np.int64); sg = np.sign(X); A = np.abs(X); out = []
    while A.any(): out.append(((A & 0xFFFF) * sg).astype(np.float64)); A = A >> 16
    return out or [np.zeros(X.shape)]
def exact_mm(A, B):
    out = None
    for i, a in enumerate(limbs(A)):
        for j, b in enumerate(limbs(B)):
            S = a @ b; assert np.abs(S).max() < 2.0 ** 53
            S = S.astype(np.int64).astype(object) * (1 << (16 * (i + j))); out = S if out is None else out + S
    return out
def part_turn():
    ok = True; n = 0; turn_ph = 0; mproj_ph = 0; npos = 0
    TROW = ones(TN0).sum(1)                            # photons of each line of TN_0 over its 768 outputs
    for ids in TEXTS:
        ids, rec = pieces(ids); res = edge_split(ids, rec)
        lhs = exact_mm(res["a_s"], TN0)
        rhs = exact_mm(res["Pk"] * np.left_shift(np.int64(1), (L.WEF - M0).astype(np.int64))[None, :], WM16.astype(np.int64))
        ok &= bool((lhs == rhs).all()); n += lhs.size
        turn_ph += int((ones(res["a_s"]) @ TROW).sum())
        mproj_ph += int((ones(J.to16(rec["G"], -ES)[0]) @ WROW["mproj.0"]).sum()); npos += len(rec["rows"])
    say(f"\nTHE FIXED TURN T_0 (turns/turns.npz)   ({time.time() - t0:.0f}s, {gb():.2f} GB)")
    say(f"  (KV light) met with TN_0 == sum_n Pk[n] 2^(we_fc[n] - m0) mproj[n, :], exact whole numbers: {ok} at {n:,} of {n:,} entries "
        f"(30 refs texts, positions 1 on): (ii) through mproj.0 is g x (KV light 2^s1 met with T_0), the same turn for every token")
    say(f"  the T_0 route's meeting alone (KV light x TN_0, photons): {turn_ph / npos:,.0f} a token, against mproj.0's whole meeting "
        f"{mproj_ph / npos:,.0f}: {turn_ph / mproj_ph:.2f}x. The route would still need (i) + (iii) met with mproj.0 on all 3,072 lines.")

# ======================================================================================== time
def arrival_last(QKV):                                # base2.py's attention, its own lines, for the newest position only
    P = QKV.shape[0]; out = np.zeros((1, 768), dtype=np.int64)
    for h in range(12):
        q, k, v = QKV[-1:, h*64:(h+1)*64], QKV[:, 768+h*64:768+(h+1)*64], QKV[:, 1536+h*64:1536+(h+1)*64]
        q16, eq = J.to16(q, -ES); k16, ek = J.to16(k, -ES)
        row = J.rshift(J.product(q16, k16.T.astype(np.float64)), -(eq[:, None] + ek[None, :] - 3 + J.SC))[0]
        d = row[J.race(row)] - row
        a = np.where(d < len(J.ARR), J.ARR[np.minimum(d, len(J.ARR) - 1)], 0)
        num = J.product(a[None, :], v.astype(np.float64))[0]
        out[0, h*64:(h+1)*64] = J.power(num, -J.log2fix([a.sum()])[0])
    return out
def settings():
    env = {k: os.environ.get(k, "(unset)") for k in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    say(f"settings: {cpu}, macOS {platform.mac_ver()[0]}, Python {platform.python_version()}, numpy {np.__version__}; threads env {env}")
def timeall(fns, reps=5):                             # interleaved, the order reversed every other round
    for f in fns: f()
    runs = [[] for _ in fns]
    for r in range(reps):
        for k in (range(len(fns)) if r % 2 == 0 else reversed(range(len(fns)))):
            gc.collect(); gc.disable(); t = time.perf_counter_ns(); fns[k](); runs[k].append(time.perf_counter_ns() - t); gc.enable()
    return [(float(np.median(r)), r) for r in runs]
def part_time():
    settings()
    LIGHT = [(np.asarray(t, np.int64), L.WTE[t], L.WPE[:len(t)]) for t in TEXTS]; NP = sum(len(t) for t in TEXTS)
    def plain_text():
        out = []
        for ids, A, B in LIGHT:
            R = A + B; R1 = R + J.plain_bank(J.attention(J.gauged_bank(R, "qkv.0")), "proj.0")
            out.append(R1 + J.plain_bank(J.edge(J.gauged_bank(R1, "fc.0")), "mproj.0"))
        return out
    def new_text(): return [L.layer0(ids, A, B) for ids, A, B in LIGHT]
    def plain_tokens():                               # the stream: each newest token alone, earlier q/k/v kept
        out = []
        for ids, A, B in LIGHT:
            Q = np.zeros((len(ids), 2304), np.int64)
            for p in range(len(ids)):
                R = A[p:p + 1] + B[p:p + 1]; Q[p] = J.gauged_bank(R, "qkv.0")[0]
                R1 = R + J.plain_bank(arrival_last(Q[:p + 1]), "proj.0")
                out.append(R1 + J.plain_bank(J.edge(J.gauged_bank(R1, "fc.0")), "mproj.0"))
        return out
    def new_tokens():
        out = []
        for ids, A, B in LIGHT:
            Q = np.zeros((len(ids), 2304), np.int64)
            for p in range(len(ids)):
                R = A[p:p + 1] + B[p:p + 1]; Q[p] = E.entry_bank(R, A[p:p + 1], B[p:p + 1], ids[p:p + 1], [p])[0]
                if p == 0: out.append(L.Z0.read(ids[:1])); continue
                att = L.proj_bank(arrival_last(Q[:p + 1]), Q[p:p + 1, 1536:], ids[p:p + 1] * 1024 + p)[0]; R1 = R + att
                out.append(R1 + J.plain_bank(J.edge(L.fc_bank(R1, ids[p:p + 1], np.array([p]))[0]), "mproj.0"))
        return out
    a, b = plain_text(), new_text(); assert all(np.array_equal(x, y) for x, y in zip(a, b))
    c, d = plain_tokens(), new_tokens(); assert all(np.array_equal(x, y) for x, y in zip(c, d))
    assert all(np.array_equal(np.concatenate(c[sum(len(t) for t in TEXTS[:i]):sum(len(t) for t in TEXTS[:i + 1])]), a[i]) for i in range(len(TEXTS)))
    say(f"\nREAL TIME OF LAYER 0, the 30 refs texts ({NP} positions), word table already in memory; both ways give the identical stream "
        f"after layer 0 at every position, and the token-at-a-time stream equals the whole-text one")
    for label, fp, fn in (("whole text at once (base2.py's forward)", plain_text, new_text),
                          ("one token at a time (the newest token alone, earlier q/k/v kept)", plain_tokens, new_tokens)):
        (mp, rp), (mn, rn) = timeall([fp, fn])
        say(f"  {label}:")
        say(f"     base2.py layer 0:      {mp / 1e6 / NP * 1000:8.1f} us a token   (runs, ms total: {', '.join(f'{x / 1e6:.1f}' for x in rp)})")
        say(f"     base2_layer0 layer 0:  {mn / 1e6 / NP * 1000:8.1f} us a token = {mp / mn:.2f}x base2's speed   (runs: {', '.join(f'{x / 1e6:.1f}' for x in rn)})")

def part_timeparts():                                 # where layer 0's time goes, whole text at once, both ways
    T_ = []
    for t in TEXTS:
        ids = np.asarray(t, np.int64); A, B = L.WTE[ids], L.WPE[:len(ids)]; R = A + B; QKV = J.gauged_bank(R, "qkv.0")
        o = J.attention(QKV); R1 = R + J.plain_bank(o, "proj.0"); H = J.gauged_bank(R1, "fc.0"); G = J.edge(H)
        r = np.arange(1, len(ids)); T_.append((ids, A, B, R, QKV, o, R1, H, G, r))
    pos = lambda r: np.asarray(r)
    fns = {
        "qkv.0 bank, base2.py":            lambda: [J.gauged_bank(R, "qkv.0") for ids, A, B, R, QKV, o, R1, H, G, r in T_],
        "qkv.0 bank, step 1 (entry)":      lambda: [E.entry_bank(R, A, B, ids, np.arange(len(ids))) for ids, A, B, R, QKV, o, R1, H, G, r in T_],
        "the arrival, base2.py's own":     lambda: [J.attention(QKV) for ids, A, B, R, QKV, o, R1, H, G, r in T_],
        "proj.0 bank, base2.py (pos 1 on)": lambda: [J.plain_bank(o[r], "proj.0") for ids, A, B, R, QKV, o, R1, H, G, r in T_],
        "proj.0 bank, own value read":     lambda: [L.proj_bank(o[r], QKV[r, 1536:], ids[r] * 1024 + r) for ids, A, B, R, QKV, o, R1, H, G, r in T_],
        "fc.0 bank, base2.py (pos 1 on)":  lambda: [J.gauged_bank(R1[r], "fc.0") for ids, A, B, R, QKV, o, R1, H, G, r in T_],
        "fc.0 bank, word+position read":   lambda: [L.fc_bank(R1[r], ids[r], r) for ids, A, B, R, QKV, o, R1, H, G, r in T_],
        "the bend (pos 1 on)":             lambda: [J.edge(H[r]) for ids, A, B, R, QKV, o, R1, H, G, r in T_],
        "mproj.0 bank (pos 1 on)":         lambda: [J.plain_bank(G[r], "mproj.0") for ids, A, B, R, QKV, o, R1, H, G, r in T_]}
    res = timeall(list(fns.values())); NP = sum(len(t) for t in TEXTS)
    say(f"\nWHERE LAYER 0's TIME GOES, whole text at once, 30 refs texts, median of 5 interleaved (us a token over all {NP} positions):")
    for (name, _), (m, rr) in zip(fns.items(), res): say(f"  {name:36s} {m / 1e3 / NP:8.1f} us   (runs, ms: {', '.join(f'{x / 1e6:.1f}' for x in rr)})")

for part in PARTS:
    {"count": part_count, "edges": part_edges, "turn": part_turn, "time": part_time, "timeparts": part_timeparts}[part]()
say(f"\nfallbacks: {L.FALLBACK}   max memory {gb():.2f} GB, {time.time() - t0:.0f}s")
