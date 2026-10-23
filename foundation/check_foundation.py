"""CHECKER for foundation/foundation.log. Own arrival, own capture, own photon counts and work; the machine is base2.py,
loaded untouched. Cover per position p:  whole 0..p | found {0,p} | second {0,p-1,p} | alone {p}.
Control: the log's foreign first word per text (ctrl) and a second, own choice (ctrl2), arrivals over {0, p}.
    python3 foundation/check_foundation.py [texts]
"""
import sys, os, json, time, resource, numpy as np
NT_ARG = int(sys.argv[1]) if len(sys.argv) > 1 else 30
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]
src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
src = src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
sys.argv = ["x"]; exec(src)                                       # the machine, untouched

def arrive(QKV, cover):                                           # own arrival: same steps, restricted cover
    P = QKV.shape[0]; out = np.zeros((P, 768), np.int64)
    for h in range(12):
        q = QKV[:, h*64:(h+1)*64]; k = QKV[:, 768+h*64:768+(h+1)*64]; v = QKV[:, 1536+h*64:1536+(h+1)*64]
        q16, eq = to16(q, -ES); k16, ek = to16(k, -ES)
        for p in range(P):
            c = np.array(sorted(set(cover(p))), np.int64)
            s = q16[p].astype(np.int64)[None, :] @ k16[c].astype(np.int64).T          # int64 meeting, no float
            s = rshift(s[0], -(eq[p] + ek[c] - 3 + SC))
            d = s.max() - s
            a = np.where(d < len(ARR), ARR[np.minimum(d, len(ARR) - 1)], 0).astype(np.int64)
            num = (a[:, None] * v[c].astype(np.int64)).sum(0)
            out[p, h*64:(h+1)*64] = power(num, -log2fix([a.sum()])[0])
    return out

COV = {"whole": lambda p: range(p + 1), "found": lambda p: (0, p), "second": lambda p: (0, max(p - 1, 0), p),
       "alone": lambda p: (p,)}
NAMES = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]

def run(R0, cover):                                               # cover None: the machine's own attention()
    R = R0.copy(); cap = {}
    for l in range(12):
        x16, ex, _ = gauge_log(R); cap[f"qkv.{l}"] = (x16, ex)
        A = attention(gauged_bank(R, f"qkv.{l}")) if cover is None else arrive(gauged_bank(R, f"qkv.{l}"), cover)
        x16, ex = to16(A, -ES); cap[f"proj.{l}"] = (x16, ex)
        R = R + plain_bank(A, f"proj.{l}")
        x16, ex, _ = gauge_log(R); cap[f"fc.{l}"] = (x16, ex)
        H = edge(gauged_bank(R, f"fc.{l}"))
        x16, ex = to16(H, -ES); cap[f"mproj.{l}"] = (x16, ex)
        R = R + plain_bank(H, f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R); cap["head"] = (x16, ex)
    top = power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
    return cap, top

def bits(A):                                                      # photons: 1s of |value|, own way (byte table)
    A = np.abs(np.asarray(A, np.int64)); T = np.array([bin(i).count("1") for i in range(256)], np.int64)
    n = np.zeros(A.shape, np.int64)
    for s in range(0, 64, 8): n += T[(A >> s) & 255]
    return n
PL = {}
for k in NAMES:                                                   # plate photons per line in
    w = M[k + ".w"]; PL[k] = np.concatenate([bits(w[r:r+64]).sum(1) for r in range(0, w.shape[0], 64)]); del w

refs = json.load(open("refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in refs][:NT_ARG]
FOREIGN = [49488, 1026, 37906, 3666, 40, 464, 21197, 464, 7454, 464, 7571, 21197, 464, 7454, 464, 2484, 464, 464,
           19184, 464, 2484, 464, 3347, 464, 37906, 2514, 49488, 818, 818, 818]
FOREIGN2 = []
for t in range(30):                                               # own control choice: next text (t+11..) with another word
    u = (t + 11) % 30
    while refs[u]["ids"][0] == refs[t]["ids"][0]: u = (u + 1) % 30
    FOREIGN2.append(refs[u]["ids"][0])
WTE, WPE = M["wte"], M["wpe"]
BASES = ["found", "second", "ctrl", "ctrl2", "alone"]
Q = ["pc", "pb", "pp", "sc", "sb", "sp", "zero", "lines", "moved", "wc", "wp"]
NPOS = sum(len(s) for s in TEXTS); NB = len(NAMES)
D = {b: {q: np.zeros((NPOS, NB), np.int64) for q in Q} for b in BASES}
T5 = {b: np.zeros((NPOS, 3), bool) for b in BASES}
POS = np.zeros(NPOS, np.int64)
C = dict(whole=0, fwd=0, words=0, words_n=0, pos0=0, f1=0, s12=0, ident=0, ident_n=0, i64=0, i64_n=0)
def top5(v):                                                      # 5 largest, ties to the lowest index
    idx = np.argpartition(-v, 6)[:6]; idx = sorted(idx, key=lambda i: (-int(v[i]), int(i)))
    m5 = int(v[idx[4]]); cand = np.flatnonzero(v >= m5)        # include any tie at the 5th value
    return [int(i) for i in sorted(cand, key=lambda i: (-int(v[i]), int(i)))[:5]]

t0 = time.time(); row = 0
for t, seq in enumerate(TEXTS):
    P = len(seq); R0 = WTE[seq] + WPE[:P]
    cx, tx = run(R0, None)                                        # the machine
    cw, tw = run(R0, COV["whole"])
    C["whole"] += all(np.array_equal(cx[k][0], cw[k][0]) and np.array_equal(cx[k][1], cw[k][1]) for k in NAMES) and np.array_equal(tx, tw)
    del cw, tw
    C["fwd"] += race(tx[-1]) == forward(seq)
    n0 = len(refs[t]["ids"])
    for p in range(n0 - 1, P - 1): C["words"] += race(tx[p]) == seq[p + 1]; C["words_n"] += 1
    ans = [race(tx[p]) for p in range(P)]; t5x = [top5(tx[p]) for p in range(P)]
    R = {"found": run(R0, COV["found"]), "second": run(R0, COV["second"]), "alone": run(R0, COV["alone"]),
         "ctrl": run(WTE[[FOREIGN[t]] + seq[1:]] + WPE[:P], COV["found"]),
         "ctrl2": run(WTE[[FOREIGN2[t]] + seq[1:]] + WPE[:P], COV["found"])}
    def same(b, p): return all(np.array_equal(cx[k][0][p], R[b][0][k][0][p]) and cx[k][1][p] == R[b][0][k][1][p] for k in NAMES) and np.array_equal(tx[p], R[b][1][p])
    C["pos0"] += same("found", 0) and same("second", 0) and same("alone", 0)
    C["f1"] += same("found", 1); C["s12"] += same("second", 1) and same("second", 2)
    for b in BASES:
        cb, tb = R[b]
        for p in range(P):
            t5b = top5(tb[p]); T5[b][row + p] = (ans[p] in t5b, t5b[0] == ans[p], set(t5b) == set(t5x[p]))
        for bi, k in enumerate(NAMES):
            xc, ec = cx[k]; xb, eb = cb[k]; pg = xc - xb; w = W[k][0]
            if t < 12:                                            # identity on 12 texts, every base, every bank
                C["ident"] += np.array_equal(product(xc, w), product(xb, w) + product(pg, w)); C["ident_n"] += 1
            if t < 2 and b == "found":                            # and plain int64, no float
                wi = M[k + ".w"]; ok = True
                for c0 in range(0, wi.shape[1], 4096):
                    wc = wi[:, c0:c0+4096].astype(np.int64)
                    ok &= np.array_equal(xc.astype(np.int64) @ wc, xb.astype(np.int64) @ wc + pg.astype(np.int64) @ wc)
                C["i64"] += ok; C["i64_n"] += 1; del wi
            oc, ob, op = bits(xc), bits(xb), bits(pg); s = slice(row, row + P); d = D[b]
            d["pc"][s, bi] = oc.sum(1); d["pb"][s, bi] = ob.sum(1); d["pp"][s, bi] = op.sum(1)
            d["sc"][s, bi] = np.abs(xc).sum(1); d["sb"][s, bi] = np.abs(xb).sum(1); d["sp"][s, bi] = np.abs(pg).sum(1)
            d["zero"][s, bi] = (pg == 0).sum(1); d["lines"][s, bi] = xc.shape[1]; d["moved"][s, bi] = ec != eb
            d["wc"][s, bi] = oc @ PL[k]; d["wp"][s, bi] = op @ PL[k]
    POS[row:row + P] = np.arange(P); row += P
    del R, cx, tx
    if t % 5 == 4: print(f"  {t+1} texts {time.time()-t0:.0f}s {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.2f}GB {C}", flush=True)
print(f"checks {C}")
np.savez("/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/chk.npz",
         pos=POS, **{f"{b}_{q}": D[b][q] for b in BASES for q in Q}, **{f"{b}_t5": T5[b] for b in BASES})

def pc(a, b): return 100.0 * a / b
NH = list(range(48)); ALL = list(range(49)); HB = 48
def ov(b, sel, banks):
    d = D[b]; g = lambda q: d[q][sel][:, banks].sum()
    return (pc(g("pp"), g("pc")), pc(g("sp"), g("sc")), pc(g("zero"), g("lines")), pc(g("moved"), sel.sum() * len(banks)), pc(g("wp"), g("wc")))
for lab, sel in [("pos 1 on", POS >= 1), ("pos 2 on", POS >= 2), ("pos 3 on", POS >= 3)]:
    print(f"\n{lab} ({sel.sum()})")
    for b in BASES:
        a, h = ov(b, sel, NH), ov(b, sel, ALL)
        print(f"  {b:7s} photons {a[0]:6.1f} size {a[1]:6.1f} zero {a[2]:5.1f} moved {a[3]:5.1f} work {a[4]:6.1f} | with top: work {h[4]:6.1f}"
              f"  zero excl qkv.0 {pc(D[b]['zero'][sel][:, 1:48].sum(), D[b]['lines'][sel][:, 1:48].sum()):5.2f}")
        t = T5[b][sel]; print(f"          top5: in {t[:,0].sum()} ({pc(t[:,0].sum(), len(t)):.1f}%) first {t[:,1].sum()} ({pc(t[:,1].sum(), len(t)):.1f}%) set {t[:,2].sum()} ({pc(t[:,2].sum(), len(t)):.1f}%)")
    print(f"  light work per token {D['found']['wc'][sel][:, NH].sum()/sel.sum():.4g} / with top {D['found']['wc'][sel][:, ALL].sum()/sel.sum():.4g}")
for b in ["found", "second", "ctrl", "alone"]:
    sel = POS >= 1; d = D[b]
    print(f"\nper layer {b} pos 1 on: photons / size / work")
    for l in range(12):
        bb = [NAMES.index(f"{x}.{l}") for x in ("qkv", "proj", "fc", "mproj")]
        print(f"  L{l:2d} {pc(d['pp'][sel][:, bb].sum(), d['pc'][sel][:, bb].sum()):6.1f} {pc(d['sp'][sel][:, bb].sum(), d['sc'][sel][:, bb].sum()):6.1f} {pc(d['wp'][sel][:, bb].sum(), d['wc'][sel][:, bb].sum()):6.1f}")
    print(f"  top {pc(d['pp'][sel, HB].sum(), d['pc'][sel, HB].sum()):6.1f} {pc(d['sp'][sel, HB].sum(), d['sc'][sel, HB].sum()):6.1f}")
for b, ks, sel in [("found", ["qkv.1", "fc.0", "mproj.0", "qkv.5", "fc.5", "proj.0", "proj.1"], POS >= 2), ("found", ["fc.6"], POS >= 1)]:
    d = D[b]
    for k in ks:
        i = NAMES.index(k); n = sel.sum()
        print(f"  {b} {k}: size {pc(d['sp'][sel, i].sum(), d['sc'][sel, i].sum()):.1f} base size {pc(d['sb'][sel, i].sum(), d['sc'][sel, i].sum()):.1f}"
              f" photons/token ctx {d['pc'][sel, i].sum()/n:.0f} base {d['pb'][sel, i].sum()/n:.0f} page {d['pp'][sel, i].sum()/n:.0f}")
pr = [NAMES.index(f"proj.{l}") for l in range(12)]; d = D["found"]; sel = POS >= 2
print("  found proj sizes pos 2 on:", [round(pc(d['sp'][sel, i].sum(), d['sc'][sel, i].sum()), 1) for i in pr])
print("\nby position, banks of 12 layers: photons / size")
for p in [0, 1, 2, 3, 5, 10, 18]:
    s = POS == p
    print(f"  pos {p:2d}: " + "  ".join(f"{b} {pc(D[b]['pp'][s][:, NH].sum(), D[b]['pc'][s][:, NH].sum()):5.1f}/{pc(D[b]['sp'][s][:, NH].sum(), D[b]['sc'][s][:, NH].sum()):5.1f}" for b in BASES))
print("\nwater level, pos 1 on, banks of 12 layers: moved photons/size | not moved photons/size")
for b in BASES:
    d = D[b]; sel = POS >= 1; m = d["moved"][sel][:, NH] == 1
    f = lambda q, mm: d[q][sel][:, NH][mm].sum()
    print(f"  {b:7s} moved {m.sum()} {pc(f('pp', m), f('pc', m)):5.1f}/{pc(f('sp', m), f('sc', m)):5.1f}   not {(~m).sum()} {pc(f('pp', ~m), f('pc', ~m)):5.1f}/{pc(f('sp', ~m), f('sc', ~m)):5.1f}")
if NT_ARG == 30:                                                  # the alone base against thinpage/thin_page.npz
    Z = np.load("thinpage/thin_page.npz"); da = D["alone"]
    for mine, theirs in [("pc", "ph_ctx"), ("pb", "ph_own"), ("pp", "ph_thin"), ("sc", "size_ctx"), ("sp", "size_thin"),
                         ("zero", "zero"), ("moved", "moved"), ("wc", "work_ctx"), ("wp", "work_thin")]:
        print(f"  alone {mine} == thin_page rec_{theirs}: {np.array_equal(da[mine], Z['rec_' + theirs])}")
print(f"\nforeign2 {FOREIGN2}\npeak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.2f} GB, {time.time()-t0:.0f}s")
